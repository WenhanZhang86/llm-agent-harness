from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.runtime import AgentRuntime, RuntimeEvent, RuntimeEventLogger, RuntimePolicy
from agent_harness.trace import compare_run_summaries, load_run_events, load_run_summary, replay_transcript


class FinalLLM:
    def complete(self, payload: dict) -> dict:
        return {
            "thought": "No tool is needed.",
            "tool_calls": [],
            "final": "Done.",
            "provider": "test",
            "model": "final",
            "usage": {"total_tokens": 3},
            "cost_usd": 0.001,
        }


class CalculatorLLM:
    def complete(self, payload: dict) -> dict:
        tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
        if not tool_messages:
            return {
                "thought": "Use calculator.",
                "tool_calls": [{"name": "calculator", "arguments": {"expression": "2+2"}}],
                "final": None,
                "provider": "test",
                "model": "calculator",
                "usage": {"total_tokens": 5},
                "cost_usd": 0.001,
            }
        return {
            "thought": "Answer from observation.",
            "tool_calls": [],
            "final": "2 + 2 = 4.",
            "provider": "test",
            "model": "calculator",
            "usage": {"total_tokens": 7},
            "cost_usd": 0.002,
        }


class FailingLLM:
    def complete(self, payload: dict) -> dict:
        raise RuntimeError("API unavailable")


class ObservabilityTests(unittest.TestCase):
    def test_event_serialization(self) -> None:
        event = RuntimeEvent(
            event_id=1,
            run_id="run",
            timestamp="2026-01-01T00:00:00+00:00",
            step_id=None,
            event_type="agent_started",
            data={"task": "test"},
        )
        self.assertEqual(event.to_dict()["event_type"], "agent_started")
        self.assertEqual(event.to_dict()["data"]["task"], "test")

    def test_event_ordering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = RuntimeEventLogger("run", Path(tmp))
            logger.emit("agent_started")
            logger.emit("llm_request")
            logger.emit("agent_finished")
            rows = [json.loads(line) for line in (Path(tmp) / "events.jsonl").read_text().splitlines()]
            self.assertEqual([row["event_id"] for row in rows], [1, 2, 3])

    def test_summary_generation_and_replay_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            runtime = AgentRuntime(
                llm=CalculatorLLM(),
                workspace=workspace,
                policy=RuntimePolicy(max_steps=4),
                use_rag_context=False,
            )
            result = runtime.run("Use calculator to compute 2+2")
            summary = load_run_summary(workspace, run_id=result.run_id)
            events = load_run_events(workspace, run_id=result.run_id)
            replay = replay_transcript(json.loads(result.trace_path.read_text()), events)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["llm_calls"], 2)
            self.assertEqual(summary["tool_calls"], 1)
            self.assertIn("LLM Request", replay)
            self.assertIn("Tool Request", replay)
            self.assertIn("Final Answer", replay)

    def test_compare_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            first = AgentRuntime(llm=FinalLLM(), workspace=workspace, use_rag_context=False).run("First")
            second = AgentRuntime(llm=CalculatorLLM(), workspace=workspace, use_rag_context=False).run("Second")
            table = compare_run_summaries(
                load_run_summary(workspace, run_id=first.run_id),
                load_run_summary(workspace, run_id=second.run_id),
            )
            self.assertIn("| Metric | Run 1 | Run 2 |", table)
            self.assertIn("LLM calls", table)

    def test_runtime_error_logging(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            result = AgentRuntime(llm=FailingLLM(), workspace=workspace, use_rag_context=False).run("Fail")
            events = load_run_events(workspace, run_id=result.run_id)
            summary = load_run_summary(workspace, run_id=result.run_id)
            self.assertEqual(result.status, "error")
            self.assertTrue(any(event["event_type"] == "runtime_error" for event in events))
            self.assertTrue(summary["errors"])


if __name__ == "__main__":
    unittest.main()
