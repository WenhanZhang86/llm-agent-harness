from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback
from typing import Any


EVENT_TYPES = {
    "agent_started",
    "context_retrieved",
    "llm_request",
    "llm_response",
    "memory_initialized",
    "memory_summarized",
    "memory_updated",
    "tool_request",
    "tool_response",
    "observation_added",
    "final_answer",
    "runtime_error",
    "agent_finished",
}


@dataclass
class RuntimeEvent:
    event_id: int
    run_id: str
    timestamp: str
    step_id: int | None
    event_type: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "step_id": self.step_id,
            "event_type": self.event_type,
            "data": self.data,
        }


class RuntimeEventLogger:
    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = run_dir
        self.events_path = run_dir / "events.jsonl"
        self.events: list[RuntimeEvent] = []
        self._next_event_id = 1
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path.write_text("", encoding="utf-8")

    def emit(self, event_type: str, *, step_id: int | None = None, data: dict[str, Any] | None = None) -> RuntimeEvent:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported runtime event type: {event_type}")
        event = RuntimeEvent(
            event_id=self._next_event_id,
            run_id=self.run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            step_id=step_id,
            event_type=event_type,
            data=data or {},
        )
        self._next_event_id += 1
        self.events.append(event)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return event


def error_info(
    error: BaseException | str,
    *,
    category: str | None = None,
    step_id: int | None = None,
    include_stack: bool = False,
) -> dict[str, Any]:
    message = str(error)
    selected_category = category or categorize_error(message)
    info: dict[str, Any] = {
        "category": selected_category,
        "message": message,
        "originating_step": step_id,
        "recovery_hint": recovery_hint(selected_category),
    }
    if include_stack and isinstance(error, BaseException):
        info["stack_trace"] = "".join(traceback.format_exception(error))
    return info


def categorize_error(message: str) -> str:
    text = message.lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "permission" in text or "denied" in text or "not allowed" in text:
        return "permission_denied"
    if "http error" in text or "api" in text or "rate limit" in text or "authorization" in text:
        return "api_error"
    if "tool" in text:
        return "tool_error"
    return "runtime_error"


def recovery_hint(category: str) -> str:
    hints = {
        "timeout": "Increase the runtime or LLM timeout, or reduce task complexity.",
        "permission_denied": "Enable the required runtime permission only if the task is trusted.",
        "api_error": "Check API key, model name, quota, rate limit, and provider availability.",
        "tool_error": "Inspect the tool arguments and the tool response in the trace.",
        "runtime_error": "Inspect events.jsonl and trace.json for the originating step.",
    }
    return hints.get(category, hints["runtime_error"])


def load_events(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events


def write_summary(
    *,
    run_dir: Path,
    run_id: str,
    provider: str | None,
    model: str | None,
    task: str,
    status: str,
    runtime_ms: float,
    steps: list[dict[str, Any]],
    events: list[RuntimeEvent],
    estimated_cost: float,
    final_answer: str | None,
) -> Path:
    llm_calls = sum(1 for event in events if event.event_type == "llm_request")
    tool_calls = sum(1 for event in events if event.event_type == "tool_request")
    tokens: dict[str, int | float] = {}
    errors: list[dict[str, Any]] = []
    for step in steps:
        for key, value in (step.get("tokens") or {}).items():
            if isinstance(value, (int, float)):
                tokens[key] = tokens.get(key, 0) + value
        if step.get("error"):
            errors.append(error_info(str(step.get("error")), step_id=step.get("step_id")))
    for event in events:
        if event.event_type == "runtime_error":
            errors.append(event.data)
    summary = {
        "run_id": run_id,
        "provider": provider,
        "model": model,
        "task": task,
        "status": status,
        "runtime_ms": runtime_ms,
        "total_steps": len(steps),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "tokens": tokens,
        "estimated_cost": estimated_cost,
        "errors": errors,
        "context_items": context_items_from_events(events),
        "memory": memory_from_events(events),
        "final_answer_preview": (final_answer or "")[:240],
    }
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def context_items_from_events(events: list[RuntimeEvent]) -> list[dict[str, Any]]:
    for event in events:
        if event.event_type == "context_retrieved":
            return list(event.data.get("matches") or [])
    return []


def memory_from_events(events: list[RuntimeEvent]) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for event in events:
        if event.event_type in {"memory_initialized", "memory_updated", "memory_summarized"}:
            latest = dict(event.data.get("memory") or latest)
    return latest
