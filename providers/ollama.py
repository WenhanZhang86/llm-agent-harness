from __future__ import annotations

import json
import os
from typing import Any

from .base import HARNESS_INSTRUCTIONS, Provider, build_model_input, parse_json_response, post_json


class OllamaProvider(Provider):
    default_model = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        data = post_json(
            f"{base_url.rstrip('/')}/api/chat",
            {},
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [
                    {"role": "system", "content": HARNESS_INSTRUCTIONS},
                    {"role": "user", "content": build_model_input(payload)},
                ],
                "options": {"temperature": 0},
            },
        )
        parsed = parse_json_response(data.get("message", {}).get("content", "{}"))
        if not parsed.get("final") and not parsed.get("tool_calls"):
            fallback = self._fallback_final(base_url, payload)
            if fallback:
                return fallback
        return parsed

    def _fallback_final(self, base_url: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        latest_tool = next(
            (message for message in reversed(payload.get("messages", [])) if message.get("role") == "tool"),
            None,
        )
        if not latest_tool:
            return None
        try:
            tool_payload = json.loads(latest_tool.get("content", "{}"))
        except json.JSONDecodeError:
            tool_payload = {"result": latest_tool.get("content", "")}
        tool_result = tool_payload.get("result")
        if not tool_result:
            return None

        user_task = next(
            (message.get("content", "") for message in payload.get("messages", []) if message.get("role") == "user"),
            "",
        )
        data = post_json(
            f"{base_url.rstrip('/')}/api/chat",
            {},
            {
                "model": self.model,
                "stream": False,
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer the user task using the tool result. Return plain text only.",
                    },
                    {
                        "role": "user",
                        "content": f"User task:\n{user_task}\n\nTool result:\n{tool_result}",
                    },
                ],
                "options": {"temperature": 0},
            },
        )
        text = data.get("message", {}).get("content", "").strip()
        if not text:
            return None
        return {
            "thought": "The provider used a fallback final answer from the latest tool observation.",
            "tool_calls": [],
            "final": text,
        }


def main() -> int:
    return OllamaProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
