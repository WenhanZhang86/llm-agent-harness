from __future__ import annotations

import os
from typing import Any
from urllib import parse

from .base import HARNESS_INSTRUCTIONS, Provider, build_model_input, env_required, parse_json_response, post_json


class GeminiProvider(Provider):
    default_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-pro")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        key = parse.quote(env_required("GEMINI_API_KEY"))
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        data = post_json(
            url,
            {},
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": HARNESS_INSTRUCTIONS + "\n\n" + build_model_input(payload)}],
                    }
                ],
                "generationConfig": {"responseMimeType": "application/json"},
            },
        )
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return parse_json_response(text)


def main() -> int:
    return GeminiProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
