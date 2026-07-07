from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .policy import RuntimePolicy


@dataclass
class ToolContext:
    workspace: Path
    policy: RuntimePolicy


@dataclass
class ToolOutput:
    ok: bool
    result: dict[str, Any] | list[Any] | str | int | float | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }


class Tool(Protocol):
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None
    required_permissions: set[str]

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        """Execute a tool and return a structured output."""
