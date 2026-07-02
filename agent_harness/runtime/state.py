from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from .memory import ShortTermMemory
from .step import AgentStep


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]


@dataclass
class AgentState:
    task: str
    run_id: str = field(default_factory=new_run_id)
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    memory: ShortTermMemory = field(default_factory=ShortTermMemory)
    status: str = "running"
    final: str | None = None
    total_cost_usd: float = 0.0

    def next_step_id(self) -> int:
        return len(self.steps) + 1

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)
