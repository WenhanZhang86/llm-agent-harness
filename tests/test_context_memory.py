from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from agent_harness.cli import run_show_context_command, run_show_memory_command
from agent_harness.llm import MockLLM
from agent_harness.runtime import AgentRuntime, RuntimePolicy
from agent_harness.runtime.memory import ContextStore, ShortTermMemory, retrieve_context
from agent_harness.trace import load_run_events, load_run_summary


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class ContextMemoryTests(unittest.TestCase):
    def test_retrieval_finds_readme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("Agent runtime evaluation and tool calling.", encoding="utf-8")
            matches = retrieve_context(ContextStore(workspace), "How does agent evaluation work?", top_k=2)
            self.assertTrue(matches)
            self.assertEqual(matches[0].item.source, "README.md")
            self.assertIn("matched terms", matches[0].reason)

    def test_short_term_memory_append_latest_export(self) -> None:
        memory = ShortTermMemory(max_entries=3)
        memory.append("user_task", "Task")
        memory.append("llm_response", "Thought")
        self.assertEqual(memory.latest(1)[0]["kind"], "llm_response")
        exported = memory.export()
        self.assertEqual(exported["size"], 2)
        self.assertIn("entries", exported)

    def test_rolling_window_summarizes_overflow(self) -> None:
        memory = ShortTermMemory(max_entries=2)
        memory.append("note", "one")
        memory.append("note", "two")
        memory.append("note", "three")
        self.assertEqual(memory.export()["size"], 2)
        self.assertIn("one", memory.summarize())

    def test_runtime_context_and_memory_integration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("Calculator tool and agent runtime documentation.", encoding="utf-8")
            result = AgentRuntime(
                llm=MockLLM(),
                workspace=workspace,
                policy=RuntimePolicy(max_steps=4),
            ).run("Use calculator to compute 2 + 2")
            summary = load_run_summary(workspace, run_id=result.run_id)
            events = load_run_events(workspace, run_id=result.run_id)
            event_types = {event["event_type"] for event in events}
            self.assertIn("context_retrieved", event_types)
            self.assertIn("memory_initialized", event_types)
            self.assertIn("memory_updated", event_types)
            self.assertTrue(summary["context_items"])
            self.assertTrue(summary["memory"]["entries"])

    def test_context_and_memory_inspectors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "README.md").write_text("Agent runtime context retrieval.", encoding="utf-8")
            result = AgentRuntime(llm=MockLLM(), workspace=workspace, policy=RuntimePolicy(max_steps=4)).run(
                "Inspect agent runtime context"
            )
            context_output = io.StringIO()
            with redirect_stdout(context_output):
                run_show_context_command(Args(workspace=str(workspace), run_id=result.run_id))
            self.assertIn("retrieved_documents:", context_output.getvalue())
            self.assertIn("README.md", context_output.getvalue())

            memory_output = io.StringIO()
            with redirect_stdout(memory_output):
                run_show_memory_command(Args(workspace=str(workspace), run_id=result.run_id))
            self.assertIn("memory_size:", memory_output.getvalue())
            self.assertIn("entries:", memory_output.getvalue())


if __name__ == "__main__":
    unittest.main()
