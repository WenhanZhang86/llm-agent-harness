from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .runtime import AgentRuntime, RuntimePolicy, RuntimeToolRegistry
from .tools import ToolRegistry


class LLM(Protocol):
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a dict with optional thought, tool_calls, and final fields."""


@dataclass
class AgentConfig:
    workspace: Path
    max_steps: int = 8
    run_dir: Path = Path("runs")
    system_prompt: str = (
        "You are an agent running in a harness. Use tools when useful. "
        "Return a final answer once the task is complete."
    )


@dataclass
class AgentRun:
    task: str
    run_id: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    messages: list[dict[str, Any]] = field(default_factory=list)
    structured_trace: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    final: str | None = None


class AgentHarness:
    def __init__(self, llm: LLM, tools: ToolRegistry, config: AgentConfig):
        self.llm = llm
        self.tools = tools
        self.config = config

    def run(self, task: str) -> AgentRun:
        runtime_policy = RuntimePolicy(
            max_steps=self.config.max_steps,
            timeout_seconds=240,
            allow_file_write=True,
            allow_shell_exec=True,
        )
        runtime_tools = RuntimeToolRegistry(self.config.workspace, policy=runtime_policy)
        runtime_tools.register_legacy_tools(self.tools)
        runtime = AgentRuntime(
            llm=self.llm,
            workspace=self.config.workspace,
            tools=runtime_tools,
            policy=runtime_policy,
            run_dir=self.config.run_dir,
            system_prompt=self.config.system_prompt,
            use_rag_context=False,
        )
        result = runtime.run(task)
        return AgentRun(
            task=task,
            run_id=result.run_id,
            messages=result.state.messages,
            structured_trace=[step.to_dict() for step in result.state.steps],
            status=result.status,
            final=result.final,
        )

    def _write_transcript(self, run: AgentRun) -> Path:
        run_dir = self.config.workspace / self.config.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{run.run_id}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": run.run_id,
                    "task": run.task,
                    "status": run.status,
                    "final": run.final,
                    "messages": run.messages,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return path
