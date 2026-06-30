from __future__ import annotations

import os
from typing import Any

from openai import OpenAI

from .base import HARNESS_INSTRUCTIONS, Provider, build_model_input, parse_json_response


class OpenAIProvider(Provider):
    default_model = os.environ.get("OPENAI_MODEL", "gpt-5.4-nano")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = OpenAI()
        response = client.responses.create(
            model=self.model,
            instructions=HARNESS_INSTRUCTIONS,
            input=build_model_input(payload),
        )
        return parse_json_response(response.output_text)


def main() -> int:
    return OpenAIProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
