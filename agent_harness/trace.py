from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_transcript(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> dict[str, Any]:
    path = resolve_transcript_path(workspace, run_id, transcript)
    return json.loads(path.read_text(encoding="utf-8"))


def load_run_events(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> list[dict[str, Any]]:
    run_dir = resolve_run_dir(workspace, run_id, transcript)
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run_summary(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> dict[str, Any]:
    run_dir = resolve_run_dir(workspace, run_id, transcript)
    path = run_dir / "summary.json"
    if not path.exists():
        data = load_transcript(workspace, run_id, transcript)
        return summary_from_transcript(data)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_transcript_path(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> Path:
    if transcript:
        return transcript.resolve()
    if not run_id:
        raise ValueError("Either run_id or transcript must be provided.")
    run_dir_trace = workspace / "runs" / run_id / "trace.json"
    if run_dir_trace.exists():
        return run_dir_trace.resolve()
    return (workspace / "runs" / f"{run_id}.json").resolve()


def resolve_run_dir(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> Path:
    path = resolve_transcript_path(workspace, run_id, transcript)
    if path.name == "trace.json":
        return path.parent.resolve()
    return (path.parent / path.stem).resolve()


def render_trace_markdown(
    transcript: dict[str, Any],
    events: list[dict[str, Any]] | None = None,
    summary: dict[str, Any] | None = None,
) -> str:
    messages = transcript.get("messages", [])
    events = events or []
    summary = summary or summary_from_transcript(transcript)
    lines = [
        "# Agent Trace",
        "",
        f"- Run ID: `{transcript.get('run_id', '')}`",
        f"- Status: {transcript.get('status', '')}",
        f"- Task: {transcript.get('task', '')}",
        f"- Provider: {summary.get('provider') or 'unknown'}",
        f"- Model: {summary.get('model') or 'unknown'}",
        f"- Runtime: {float(summary.get('runtime_ms') or 0.0):.2f} ms",
        f"- LLM calls: {summary.get('llm_calls', 0)}",
        f"- Tool calls: {summary.get('tool_calls', 0)}",
        f"- Estimated cost: {summary.get('estimated_cost', 0.0)}",
        "",
        "## Flow",
        "",
        "```mermaid",
        "flowchart TD",
        '  start["User task"]',
    ]

    node_index = 0
    previous = "start"
    for message in messages:
        role = message.get("role")
        if role not in {"assistant", "tool"}:
            continue
        node_index += 1
        node_id = f"n{node_index}"
        if role == "assistant":
            label = "LLM"
            if message.get("tool_calls"):
                tool_names = ", ".join(str(call.get("name")) for call in message.get("tool_calls", []))
                label = f"LLM: {tool_names}"
            elif message.get("final"):
                label = "LLM: final answer"
        else:
            label = f"Observation: {message.get('name', 'tool')}"
        lines.append(f'  {node_id}["{escape_mermaid(label)}"]')
        lines.append(f"  {previous} --> {node_id}")
        previous = node_id
    lines.extend(["```", "", "## Steps", ""])

    for message in messages:
        role = message.get("role")
        if role == "system":
            continue
        if role == "user":
            lines.extend(["### User", "", message.get("content", ""), ""])
        elif role == "assistant":
            lines.extend([f"### LLM Step {message.get('step', '')}", ""])
            if message.get("provider") or message.get("model"):
                lines.append(
                    f"- Provider: {message.get('provider') or 'unknown'}"
                    f" / Model: {message.get('model') or 'unknown'}"
                )
            if message.get("usage"):
                lines.append(f"- Usage: `{json.dumps(message.get('usage'), ensure_ascii=False)}`")
            if message.get("thought"):
                lines.extend(["", "**Thought**", "", str(message.get("thought")), ""])
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                lines.extend(["**Tool Calls**", ""])
                for call in tool_calls:
                    lines.append(
                        f"- `{call.get('name')}` with `{json.dumps(call.get('arguments') or {}, ensure_ascii=False)}`"
                    )
                lines.append("")
            if message.get("final"):
                lines.extend(["**Final Answer**", "", str(message.get("final")), ""])
        elif role == "tool":
            lines.extend(
                [
                    f"### Tool Observation: {message.get('name', '')}",
                    "",
                    f"- Arguments: `{json.dumps(message.get('arguments') or {}, ensure_ascii=False)}`",
                    "",
                    "```text",
                    truncate(str(message.get("content", ""))),
                    "```",
                    "",
                ]
            )

    structured = transcript.get("structured_trace") or []
    if structured:
        lines.extend(["## Timeline", ""])
        for step in structured:
            lines.append(
                f"- Step {step.get('step_id')} `{step.get('step_type')}`: "
                f"{float(step.get('latency_ms') or 0.0):.2f} ms"
                + (f" error={step.get('error')}" if step.get("error") else "")
            )
        lines.append("")

        lines.extend(["## Structured Trace", ""])
        for step in structured:
            lines.extend(
                [
                    f"### Step {step.get('step_id')}: {step.get('step_type')}",
                    "",
                    f"- Latency: {float(step.get('latency_ms') or 0.0):.2f} ms",
                    f"- Tokens: `{json.dumps(step.get('tokens') or {}, ensure_ascii=False)}`",
                    f"- Cost: {step.get('cost_usd') if step.get('cost_usd') is not None else 'n/a'}",
                    f"- Error: {step.get('error') or ''}",
                    "",
                    "```json",
                    json.dumps(
                        {"input": step.get("input"), "output": step.get("output")},
                        indent=2,
                        ensure_ascii=False,
                    ),
                    "```",
                    "",
                ]
            )

    if events:
        lines.extend(["## Runtime Events", ""])
        for event in events:
            data = event.get("data") or {}
            lines.append(
                f"- {event.get('event_id')}. `{event.get('event_type')}` "
                f"step={event.get('step_id')} time={event.get('timestamp')} "
                f"{event_brief(event.get('event_type'), data)}"
            )
        lines.append("")

    return "\n".join(lines)


def write_trace_markdown(
    workspace: Path,
    run_id: str | None = None,
    transcript: Path | None = None,
    output: Path | None = None,
) -> Path:
    transcript_path = resolve_transcript_path(workspace, run_id, transcript)
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    events = load_run_events(workspace, run_id, transcript)
    summary = load_run_summary(workspace, run_id, transcript)
    output_path = output or transcript_path.with_suffix(".trace.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_trace_markdown(data, events, summary), encoding="utf-8")
    return output_path


def replay_transcript(transcript: dict[str, Any], events: list[dict[str, Any]] | None = None) -> str:
    if events:
        return replay_events(transcript, events)
    lines = [
        f"run_id: {transcript.get('run_id', '')}",
        f"status: {transcript.get('status', '')}",
        f"task: {transcript.get('task', '')}",
        "",
    ]
    for message in transcript.get("messages", []):
        role = message.get("role")
        if role == "assistant":
            lines.append(f"LLM step {message.get('step', '')}")
            for call in message.get("tool_calls") or []:
                lines.append(f"  tool_call: {call.get('name')} {json.dumps(call.get('arguments') or {})}")
            if message.get("final"):
                lines.append(f"  final: {message.get('final')}")
        elif role == "tool":
            lines.append(f"observation: {message.get('name')}")
            lines.append(f"  {truncate(str(message.get('content', '')), limit=500)}")
    return "\n".join(lines)


def replay_events(transcript: dict[str, Any], events: list[dict[str, Any]]) -> str:
    lines = [
        f"run_id: {transcript.get('run_id', '')}",
        f"status: {transcript.get('status', '')}",
        f"task: {transcript.get('task', '')}",
        "",
    ]
    step_number = 0
    for event in events:
        event_type = event.get("event_type")
        data = event.get("data") or {}
        if event_type == "llm_request":
            step_number += 1
            lines.extend([f"Step {step_number}", "LLM Request", f"  messages: {data.get('message_count')}"])
        elif event_type == "context_retrieved":
            lines.extend(["Context Retrieved", f"  documents: {len(data.get('matches') or [])}"])
            for match in (data.get("matches") or [])[:5]:
                lines.append(f"  - {match.get('id')} ({match.get('reason')})")
        elif event_type == "memory_initialized":
            memory = data.get("memory") or {}
            lines.extend(["Memory Initialized", f"  entries: {memory.get('size', 0)}"])
        elif event_type == "memory_updated":
            memory = data.get("memory") or {}
            lines.extend(["Memory Updated", f"  entries: {memory.get('size', 0)}"])
        elif event_type == "memory_summarized":
            memory = data.get("memory") or {}
            lines.extend(["Memory Summarized", f"  summary: {truncate(str(memory.get('summary') or ''), 500)}"])
        elif event_type == "llm_response":
            lines.extend(["LLM Response", f"  provider: {data.get('provider') or 'unknown'}"])
            calls = data.get("tool_calls") or []
            if calls:
                for call in calls:
                    lines.append(f"  tool_call: {call.get('name')} {json.dumps(call.get('arguments') or {})}")
            if data.get("has_final"):
                lines.append("  final answer returned")
        elif event_type == "tool_request":
            lines.extend(["Tool Request", f"  {data.get('name')} {json.dumps(data.get('arguments') or {})}"])
        elif event_type == "tool_response":
            lines.extend(["Tool Response", f"  ok: {data.get('ok')}", f"  latency: {float(data.get('latency_ms') or 0.0):.2f} ms"])
        elif event_type == "observation_added":
            lines.extend(["Observation", f"  {truncate(str(data.get('observation') or ''), 500)}"])
        elif event_type == "runtime_error":
            lines.extend(["Runtime Error", f"  {data.get('category')}: {data.get('message')}"])
        elif event_type == "final_answer":
            lines.extend(["Final Answer", f"  {truncate(str(data.get('final') or ''), 1200)}"])
        elif event_type == "agent_finished":
            lines.extend(["Agent Finished", f"  status: {data.get('status')}", f"  runtime: {float(data.get('runtime_ms') or 0.0):.2f} ms"])
        if event_type in {
            "context_retrieved",
            "llm_response",
            "memory_initialized",
            "memory_updated",
            "memory_summarized",
            "tool_response",
            "observation_added",
            "final_answer",
            "agent_finished",
            "runtime_error",
        }:
            lines.append("")
    return "\n".join(lines)


def compare_run_summaries(left: dict[str, Any], right: dict[str, Any]) -> str:
    fields = [
        ("run_id", "Run ID"),
        ("status", "Status"),
        ("provider", "Provider"),
        ("model", "Model"),
        ("runtime_ms", "Runtime ms"),
        ("total_steps", "Total steps"),
        ("llm_calls", "LLM calls"),
        ("tool_calls", "Tool calls"),
        ("tokens", "Token usage"),
        ("estimated_cost", "Estimated cost"),
    ]
    lines = ["| Metric | Run 1 | Run 2 |", "| --- | --- | --- |"]
    for key, label in fields:
        lines.append(f"| {label} | {format_summary_value(left.get(key))} | {format_summary_value(right.get(key))} |")
    return "\n".join(lines)


def summary_from_transcript(transcript: dict[str, Any]) -> dict[str, Any]:
    messages = transcript.get("messages") or []
    provider = None
    model = None
    tokens: dict[str, int | float] = {}
    for message in messages:
        if message.get("role") == "assistant":
            provider = message.get("provider") or provider
            model = message.get("model") or model
            for key, value in (message.get("usage") or {}).items():
                if isinstance(value, (int, float)):
                    tokens[key] = tokens.get(key, 0) + value
    steps = transcript.get("structured_trace") or []
    runtime = transcript.get("runtime") or {}
    return {
        "run_id": transcript.get("run_id"),
        "provider": provider,
        "model": model,
        "task": transcript.get("task"),
        "status": transcript.get("status"),
        "runtime_ms": runtime.get("runtime_ms", 0.0),
        "total_steps": len(steps),
        "llm_calls": sum(1 for step in steps if step.get("step_type") == "llm"),
        "tool_calls": sum(1 for step in steps if step.get("step_type") == "tool"),
        "tokens": tokens,
        "estimated_cost": runtime.get("total_cost_usd", 0.0),
        "errors": [step.get("error") for step in steps if step.get("error")],
        "final_answer_preview": str(transcript.get("final") or "")[:240],
    }


def event_brief(event_type: str | None, data: dict[str, Any]) -> str:
    if event_type == "llm_request":
        return f"messages={data.get('message_count')} tools={data.get('tool_count')}"
    if event_type == "context_retrieved":
        return f"documents={len(data.get('matches') or [])}"
    if event_type in {"memory_initialized", "memory_updated", "memory_summarized"}:
        memory = data.get("memory") or {}
        return f"entries={memory.get('size', 0)}"
    if event_type == "llm_response":
        return f"provider={data.get('provider')} model={data.get('model')} latency_ms={float(data.get('latency_ms') or 0.0):.2f}"
    if event_type == "tool_request":
        return f"tool={data.get('name')}"
    if event_type == "tool_response":
        return f"tool={data.get('name')} ok={data.get('ok')} latency_ms={float(data.get('latency_ms') or 0.0):.2f}"
    if event_type == "runtime_error":
        return f"{data.get('category')}: {data.get('message')}"
    if event_type == "agent_finished":
        return f"status={data.get('status')} runtime_ms={float(data.get('runtime_ms') or 0.0):.2f}"
    return ""


def format_summary_value(value: Any) -> str:
    if isinstance(value, dict):
        return "`" + json.dumps(value, ensure_ascii=False) + "`"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value if value is not None else "n/a")


def escape_mermaid(value: str) -> str:
    return value.replace('"', "'")


def truncate(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... truncated ..."
