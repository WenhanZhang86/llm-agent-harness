from __future__ import annotations

import json
import subprocess
from typing import Any


class MockLLM:
    """A deterministic LLM stub for proving the harness loop works."""

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        tool_messages = [message for message in payload["messages"] if message.get("role") == "tool"]
        if not tool_messages:
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
