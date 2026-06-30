from __future__ import annotations

import os
import re
from typing import Any

from .base import HARNESS_INSTRUCTIONS, Provider, build_model_input, env_required, merge_usage, parse_json_response, post_json


class OpenAICompatibleProvider(Provider):
    default_model = os.environ.get("OPENAI_COMPATIBLE_MODEL", "gpt-4o-mini")

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        super().__init__(model)
        self.base_url = base_url or env_required("OPENAI_COMPATIBLE_BASE_URL")
        self.api_key = api_key or env_required("OPENAI_COMPATIBLE_API_KEY")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = post_json(
            self._chat_completions_url(),
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": HARNESS_INSTRUCTIONS},
                    {"role": "user", "content": build_model_input(payload)},
                ],
                "response_format": {"type": "json_object"},
            },
        )
        content = data["choices"][0]["message"].get("content") or "{}"
        parsed = parse_json_response(content)
        initial_usage = data.get("usage") or {}
        parsed.setdefault("provider", self.__class__.__name__)
        parsed.setdefault("model", data.get("model") or self.model)
        parsed.setdefault("usage", initial_usage)
        if parsed.get("final") or parsed.get("tool_calls"):
            return parsed

        tool_fallback = self._fallback_tool_call(payload)
        if tool_fallback:
            tool_fallback["provider"] = self.__class__.__name__
            tool_fallback["model"] = data.get("model") or self.model
            tool_fallback["usage"] = initial_usage
            return tool_fallback

        final_fallback = self._fallback_final(payload, initial_usage)
        if final_fallback:
            return final_fallback

        return parsed

    def _chat_completions_url(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def _fallback_tool_call(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        if any(message.get("role") == "tool" for message in payload.get("messages", [])):
            return None

        user_task = self._user_task(payload)
        tool_names = {tool.get("name") for tool in payload.get("tools", [])}

        if "read_file" in tool_names and "read_file" in user_task:
            match = re.search(r"read_file\s+tool\s+to\s+read\s+([^\s,.;]+)", user_task, re.IGNORECASE)
            path = match.group(1) if match else "README.md"
            return {
                "thought": "The provider inferred the requested read_file call from the user task.",
                "tool_calls": [{"name": "read_file", "arguments": {"path": path}}],
                "final": None,
            }

        if "list_dir" in tool_names and any(phrase in user_task.lower() for phrase in ["list files", "list the files"]):
            return {
                "thought": "The provider inferred the requested list_dir call from the user task.",
                "tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}],
                "final": None,
            }

        return None

    def _fallback_final(self, payload: dict[str, Any], initial_usage: dict[str, Any] | None) -> dict[str, Any] | None:
        latest_tool = next(
            (message for message in reversed(payload.get("messages", [])) if message.get("role") == "tool"),
            None,
        )
        if not latest_tool:
            return None

        data = post_json(
            self._chat_completions_url(),
            {"Authorization": f"Bearer {self.api_key}"},
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "Answer the user task using the latest tool result. Return plain text only.",
                    },
                    {
                        "role": "user",
                        "content": f"User task:\n{self._user_task(payload)}\n\nLatest tool result:\n{latest_tool.get('content', '')}",
                    },
                ],
            },
        )
        text = data["choices"][0]["message"].get("content", "").strip()
        if not text:
            return None
        return {
            "thought": "The provider used a fallback final answer from the latest tool observation.",
            "tool_calls": [],
            "final": text,
            "provider": self.__class__.__name__,
            "model": data.get("model") or self.model,
            "usage": merge_usage(initial_usage, data.get("usage") or {}),
        }

    def _user_task(self, payload: dict[str, Any]) -> str:
        return next(
            (message.get("content", "") for message in payload.get("messages", []) if message.get("role") == "user"),
            "",
        )


def main() -> int:
    return OpenAICompatibleProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
