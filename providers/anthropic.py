from __future__ import annotations

import os
from typing import Any

from .base import HARNESS_INSTRUCTIONS, Provider, build_model_input, env_required, parse_json_response, post_json


class AnthropicProvider(Provider):
    default_model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "x-api-key": env_required("ANTHROPIC_API_KEY"),
                "anthropic-version": os.environ.get("ANTHROPIC_VERSION", "2023-06-01"),
            },
            {
                "model": self.model,
                "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS", "2000")),
                "system": HARNESS_INSTRUCTIONS,
                "messages": [{"role": "user", "content": build_model_input(payload)}],
            },
        )
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        return parse_json_response(text)


def main() -> int:
    return AnthropicProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
