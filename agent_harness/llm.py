from __future__ import annotations

import json
import re
import subprocess
from typing import Any


class MockLLM:
    """A deterministic LLM stub for proving the harness loop works."""

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
        tool_names = {tool.get("name") for tool in payload.get("tools", [])}
        user_task = next(
            (message.get("content", "") for message in reversed(payload["messages"]) if message.get("role") == "user"),
            "",
        )
        if not tool_messages:
            if "calculator" in tool_names and any(word in user_task.lower() for word in ["calculate", "compute", "calculator"]):
                expression = extract_expression(user_task) or "2+2"
                return {
                    "thought": "I should use the calculator tool for arithmetic.",
                    "tool_calls": [{"name": "calculator", "arguments": {"expression": expression}}],
                    "final": None,
                }
            if "list_dir" not in tool_names and "local_file_search" in tool_names:
                return {
                    "thought": "I can inspect local files with the runtime file search tool.",
                    "tool_calls": [{"name": "local_file_search", "arguments": {"query": "LLM Agent Harness", "limit": 5}}],
                    "final": None,
                }
            return {
                "thought": "I should inspect the workspace before answering.",
                "tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}],
                "final": None,
            }
        return {
            "thought": "I have a tool observation and can summarize it.",
            "tool_calls": [],
            "final": f"Mock run completed. Last observation: {tool_messages[-1]['content']}",
        }


def extract_expression(task: str) -> str | None:
    matches = re.findall(r"[-+*/%(). 0-9]+", task)
    candidates = [match.strip() for match in matches if any(operator in match for operator in ["+", "-", "*", "/", "%"])]
    return max(candidates, key=len) if candidates else None


class SubprocessLLM:
    """Run an external LLM adapter that speaks the harness JSON protocol."""

    def __init__(self, command: str, timeout: int = 120):
        self.command = command
        self.timeout = timeout

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        completed = subprocess.run(
            self.command,
            input=json.dumps(payload, ensure_ascii=False),
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"LLM command failed: {completed.stderr.strip()}")
        try:
            response = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM command returned invalid JSON: {completed.stdout[:1000]}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("LLM command must return a JSON object")
        return response
