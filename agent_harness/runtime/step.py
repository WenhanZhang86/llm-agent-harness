from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


StepType = Literal["llm", "tool", "observation", "final", "error"]


@dataclass
class AgentStep:
    step_id: int
    step_type: StepType
    input: Any = None
    output: Any = None
    latency_ms: float = 0.0
    tokens: dict[str, Any] = field(default_factory=dict)
    cost_usd: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "input": self.input,
            "output": self.output,
            "latency_ms": self.latency_ms,
            "tokens": self.tokens,
            "cost_usd": self.cost_usd,
            "error": self.error,
        }
