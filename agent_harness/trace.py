from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_transcript(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> dict[str, Any]:
    path = resolve_transcript_path(workspace, run_id, transcript)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_transcript_path(workspace: Path, run_id: str | None = None, transcript: Path | None = None) -> Path:
    if transcript:
        return transcript.resolve()
    if not run_id:
        raise ValueError("Either run_id or transcript must be provided.")
    return (workspace / "runs" / f"{run_id}.json").resolve()


def render_trace_markdown(transcript: dict[str, Any]) -> str:
    messages = transcript.get("messages", [])
    lines = [
        "# Agent Trace",
        "",
        f"- Run ID: `{transcript.get('run_id', '')}`",
        f"- Status: {transcript.get('status', '')}",
        f"- Task: {transcript.get('task', '')}",
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

    return "\n".join(lines)


def write_trace_markdown(
    workspace: Path,
    run_id: str | None = None,
    transcript: Path | None = None,
    output: Path | None = None,
) -> Path:
    transcript_path = resolve_transcript_path(workspace, run_id, transcript)
    data = json.loads(transcript_path.read_text(encoding="utf-8"))
    output_path = output or transcript_path.with_suffix(".trace.md")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_trace_markdown(data), encoding="utf-8")
    return output_path


def replay_transcript(transcript: dict[str, Any]) -> str:
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


def escape_mermaid(value: str) -> str:
    return value.replace('"', "'")


def truncate(value: str, limit: int = 2000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "\n... truncated ..."
