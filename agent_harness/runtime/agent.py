from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Protocol

from .events import RuntimeEventLogger, error_info, write_summary
from .memory import ContextStore, ContextMatch, ShortTermMemory, format_context, retrieve_context
from .policy import RuntimePolicy
from .state import AgentState
from .step import AgentStep
from .tool_registry import RuntimeToolRegistry, build_runtime_tools


class RuntimeLLM(Protocol):
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a dict with optional thought, tool_calls, final, model, provider, and usage."""


@dataclass
class RuntimeResult:
    run_id: str
    status: str
    final: str | None
    trace_path: Path
    state: AgentState


class AgentRuntime:
    def __init__(
        self,
        *,
        llm: RuntimeLLM,
        workspace: Path,
        policy: RuntimePolicy | None = None,
        tools: RuntimeToolRegistry | None = None,
        memory: ShortTermMemory | None = None,
        run_dir: Path = Path("runs"),
        system_prompt: str | None = None,
        use_rag_context: bool = True,
    ):
        self.llm = llm
        self.workspace = workspace.resolve()
        self.policy = policy or RuntimePolicy()
        self.tools = tools or build_runtime_tools(self.workspace, self.policy)
        self.memory = memory or ShortTermMemory()
        self.run_dir = run_dir
        self.use_rag_context = use_rag_context
        self.system_prompt = system_prompt or (
            "You are a tool-using LLM agent running inside an agent runtime. "
            "Use tools when useful, observe results, and return a final answer."
        )

    def run(self, task: str, context: str | None = None) -> RuntimeResult:
        run_started = time.perf_counter()
        state = AgentState(task=task, memory=self.memory)
        run_dir = self.workspace / self.run_dir / state.run_id
        logger = RuntimeEventLogger(state.run_id, run_dir)
        logger.emit("agent_started", data={"task": task})
        state.memory.append("user_task", task)
        logger.emit("memory_initialized", data={"memory": state.memory.export()})
        matches = self.retrieve_context(task)
        logger.emit("context_retrieved", data={"matches": [match.to_dict() for match in matches]})
        context_text = self.load_context(context, matches)
        state.messages.append({"role": "system", "content": self.system_prompt})
        if context_text:
            state.messages.append({"role": "system", "content": f"Context:\n{context_text}"})
        state.messages.append({"role": "user", "content": task})

        deadline = time.perf_counter() + self.policy.timeout_seconds
        try:
            for loop_index in range(1, self.policy.max_steps + 1):
                if time.perf_counter() > deadline:
                    raise TimeoutError(f"Runtime exceeded {self.policy.timeout_seconds} seconds.")
                if self.policy.max_cost_usd is not None and state.total_cost_usd > self.policy.max_cost_usd:
                    raise RuntimeError(f"Runtime exceeded max_cost_usd={self.policy.max_cost_usd}.")

                response = self.call_llm(state, loop_index, logger)
                final = response.get("final")
                if final:
                    state.status = "completed"
                    state.final = str(final)
                    state.memory.append("final_answer", state.final)
                    step = AgentStep(
                        step_id=state.next_step_id(),
                        step_type="final",
                        input={"task": task},
                        output=state.final,
                    )
                    state.add_step(step)
                    logger.emit("memory_updated", step_id=step.step_id, data={"memory": state.memory.export()})
                    logger.emit("final_answer", step_id=step.step_id, data={"final": state.final})
                    return self.finish(state, logger, run_started)

                tool_calls = response.get("tool_calls") or []
                if not isinstance(tool_calls, list):
                    self.add_error(state, "tool_calls must be a list", logger)
                    continue
                if not tool_calls:
                    state.status = "stopped"
                    state.final = "The model stopped without a final answer or tool call."
                    state.memory.append("final_answer", state.final)
                    logger.emit("memory_updated", data={"memory": state.memory.export()})
                    logger.emit("final_answer", data={"final": state.final})
                    return self.finish(state, logger, run_started)

                for call in tool_calls:
                    self.execute_tool_call(state, call, logger)

            state.status = "max_steps_exceeded"
            state.final = f"Stopped after {self.policy.max_steps} steps without a final answer."
            state.memory.append("final_answer", state.final)
            logger.emit("memory_updated", data={"memory": state.memory.export()})
            logger.emit("final_answer", data={"final": state.final})
            return self.finish(state, logger, run_started)
        except Exception as exc:
            state.status = "error"
            state.final = str(exc)
            self.add_error(state, exc, logger, include_stack=True)
            return self.finish(state, logger, run_started)

    def retrieve_context(self, task: str) -> list[ContextMatch]:
        if not self.use_rag_context:
            return []
        store = ContextStore(self.workspace)
        return retrieve_context(store, task, top_k=4)

    def load_context(self, context: str | None, matches: list[ContextMatch]) -> str:
        parts = []
        if self.memory.context():
            parts.append("Short-term memory:\n" + self.memory.context())
        if context:
            parts.append(context)
        retrieved = format_context(matches)
        if retrieved:
            parts.append(retrieved)
        return "\n\n".join(parts)

    def call_llm(self, state: AgentState, loop_index: int, logger: RuntimeEventLogger) -> dict[str, Any]:
        payload = {
            "messages": state.messages,
            "tools": self.tools.schemas(),
            "max_tool_calls": self.policy.max_steps - loop_index + 1,
        }
        started = time.perf_counter()
        logger.emit(
            "llm_request",
            data={
                "loop_index": loop_index,
                "message_count": len(state.messages),
                "tool_count": len(payload["tools"]),
            },
        )
        response = self.llm.complete(payload)
        latency_ms = (time.perf_counter() - started) * 1000
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        cost = response.get("cost_usd")
        if isinstance(cost, (int, float)):
            state.total_cost_usd += float(cost)
        assistant_message = {
            "role": "assistant",
            "step": loop_index,
            "thought": response.get("thought"),
            "tool_calls": response.get("tool_calls", []),
            "final": response.get("final"),
            "provider": response.get("provider"),
            "model": response.get("model"),
            "usage": usage,
        }
        state.messages.append(assistant_message)
        state.memory.append(
            "llm_response",
            json.dumps(
                {
                    "thought": response.get("thought"),
                    "tool_calls": response.get("tool_calls", []),
                    "final": response.get("final"),
                },
                ensure_ascii=False,
            ),
            metadata={"loop_index": loop_index},
        )
        step = AgentStep(
            step_id=state.next_step_id(),
            step_type="llm",
            input={"message_count": len(state.messages) - 1},
            output={
                "thought": response.get("thought"),
                "tool_calls": response.get("tool_calls", []),
                "final": response.get("final"),
                "provider": response.get("provider"),
                "model": response.get("model"),
            },
            latency_ms=latency_ms,
            tokens=usage,
            cost_usd=cost if isinstance(cost, (int, float)) else None,
        )
        state.add_step(step)
        logger.emit("memory_updated", step_id=step.step_id, data={"memory": state.memory.export()})
        logger.emit(
            "llm_response",
            step_id=step.step_id,
            data={
                "loop_index": loop_index,
                "provider": response.get("provider"),
                "model": response.get("model"),
                "usage": usage,
                "cost_usd": cost if isinstance(cost, (int, float)) else None,
                "latency_ms": latency_ms,
                "tool_calls": response.get("tool_calls", []),
                "has_final": bool(response.get("final")),
            },
        )
        return response

    def execute_tool_call(self, state: AgentState, call: dict[str, Any], logger: RuntimeEventLogger) -> None:
        name = call.get("name")
        arguments = call.get("arguments") or {}
        started = time.perf_counter()
        logger.emit("tool_request", data={"name": name, "arguments": arguments})
        tool_output = self.tools.execute(name, arguments)
        observation = json.dumps(tool_output.to_dict(), ensure_ascii=False)
        latency_ms = (time.perf_counter() - started) * 1000
        step = AgentStep(
            step_id=state.next_step_id(),
            step_type="tool",
            input={"name": name, "arguments": arguments},
            output=tool_output.to_dict(),
            latency_ms=latency_ms,
            error=tool_output.error,
        )
        state.add_step(step)
        logger.emit(
            "tool_response",
            step_id=step.step_id,
            data={
                "name": name,
                "ok": tool_output.ok,
                "output": tool_output.to_dict(),
                "latency_ms": latency_ms,
            },
        )
        state.messages.append(
            {
                "role": "tool",
                "name": name,
                "arguments": arguments,
                "content": observation,
            }
        )
        state.memory.append("tool_output", observation, metadata={"tool": name})
        logger.emit("memory_updated", step_id=step.step_id, data={"memory": state.memory.export()})
        observation_step = AgentStep(
            step_id=state.next_step_id(),
            step_type="observation",
            input={"name": name},
            output=observation,
        )
        state.add_step(observation_step)
        state.memory.append("observation", observation, metadata={"tool": name})
        if state.memory.summary:
            logger.emit("memory_summarized", data={"memory": state.memory.export()})
        logger.emit(
            "observation_added",
            step_id=observation_step.step_id,
            data={"name": name, "observation": observation[:1000]},
        )

    def add_error(
        self,
        state: AgentState,
        error: BaseException | str,
        logger: RuntimeEventLogger,
        *,
        include_stack: bool = False,
    ) -> None:
        info = error_info(error, step_id=state.next_step_id(), include_stack=include_stack)
        step = AgentStep(
            step_id=state.next_step_id(),
            step_type="error",
            output=info,
            error=info["message"],
        )
        state.add_step(step)
        logger.emit("runtime_error", step_id=step.step_id, data=info)

    def finish(self, state: AgentState, logger: RuntimeEventLogger, run_started: float) -> RuntimeResult:
        runtime_ms = (time.perf_counter() - run_started) * 1000
        logger.emit(
            "agent_finished",
            data={"status": state.status, "runtime_ms": runtime_ms, "total_steps": len(state.steps)},
        )
        path = self.write_trace(state, logger, runtime_ms)
        return RuntimeResult(
            run_id=state.run_id,
            status=state.status,
            final=state.final,
            trace_path=path,
            state=state,
        )

    def write_trace(self, state: AgentState, logger: RuntimeEventLogger, runtime_ms: float) -> Path:
        run_dir = self.workspace / self.run_dir / state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        trace_path = run_dir / "trace.json"
        legacy_path = self.workspace / self.run_dir / f"{state.run_id}.json"
        steps = [step.to_dict() for step in state.steps]
        provider, model = provider_model_from_messages(state.messages)
        payload = {
            "run_id": state.run_id,
            "task": state.task,
            "status": state.status,
            "final": state.final,
            "messages": state.messages,
            "structured_trace": steps,
            "events_path": str(logger.events_path),
            "summary_path": str(run_dir / "summary.json"),
            "runtime": {
                "max_steps": self.policy.max_steps,
                "timeout_seconds": self.policy.timeout_seconds,
                "max_cost_usd": self.policy.max_cost_usd,
                "allow_file_write": self.policy.allow_file_write,
                "allow_shell_exec": self.policy.allow_shell_exec,
                "allow_network": self.policy.allow_network,
                "allow_code_exec": self.policy.allow_code_exec,
                "total_cost_usd": state.total_cost_usd,
                "runtime_ms": runtime_ms,
            },
            "memory": state.memory.export(),
        }
        trace_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        write_summary(
            run_dir=run_dir,
            run_id=state.run_id,
            provider=provider,
            model=model,
            task=state.task,
            status=state.status,
            runtime_ms=runtime_ms,
            steps=steps,
            events=logger.events,
            estimated_cost=state.total_cost_usd,
            final_answer=state.final,
        )
        return trace_path


def provider_model_from_messages(messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("provider"), message.get("model")
    return None, None
