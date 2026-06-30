from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

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
    status: str = "running"
    final: str | None = None


class AgentHarness:
    def __init__(self, llm: LLM, tools: ToolRegistry, config: AgentConfig):
        self.llm = llm
        self.tools = tools
        self.config = config

    def run(self, task: str) -> AgentRun:
        run = AgentRun(task=task)
        run.messages.append({"role": "system", "content": self.config.system_prompt})
        run.messages.append({"role": "user", "content": task})

        for step in range(1, self.config.max_steps + 1):
            response = self.llm.complete(
                {
                    "messages": run.messages,
                    "tools": self.tools.schemas(),
                    "max_tool_calls": self.config.max_steps - step + 1,
                }
            )
            assistant_message = {
                "role": "assistant",
                "step": step,
                "thought": response.get("thought"),
                "tool_calls": response.get("tool_calls", []),
                "final": response.get("final"),
                "provider": response.get("provider"),
                "model": response.get("model"),
                "usage": response.get("usage"),
            }
            run.messages.append(assistant_message)

            final = response.get("final")
            if final:
                run.status = "completed"
                run.final = str(final)
                self._write_transcript(run)
                return run

            tool_calls = response.get("tool_calls") or []
            if not isinstance(tool_calls, list):
                run.messages.append(
                    {
                        "role": "tool",
                        "name": "harness_error",
                        "content": "tool_calls must be a list",
                    }
                )
                continue

            if not tool_calls:
                run.status = "stopped"
                run.final = "The model stopped without a final answer or tool call."
                self._write_transcript(run)
                return run

            for call in tool_calls:
                name = call.get("name")
                arguments = call.get("arguments") or {}
                observation = self.tools.call(name, arguments)
                run.messages.append(
                    {
                        "role": "tool",
                        "name": name,
                        "arguments": arguments,
                        "content": observation,
                    }
                )

        run.status = "max_steps_exceeded"
        run.final = f"Stopped after {self.config.max_steps} steps without a final answer."
        self._write_transcript(run)
        return run

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
