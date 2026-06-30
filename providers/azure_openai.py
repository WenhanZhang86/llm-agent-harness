from __future__ import annotations

import os
from typing import Any

from openai import AzureOpenAI

from .base import HARNESS_INSTRUCTIONS, Provider, build_model_input, env_required, parse_json_response


class AzureOpenAIProvider(Provider):
    default_model = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = AzureOpenAI(
            api_key=env_required("AZURE_OPENAI_API_KEY"),
            azure_endpoint=env_required("AZURE_OPENAI_ENDPOINT"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        )
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": HARNESS_INSTRUCTIONS},
                {"role": "user", "content": build_model_input(payload)},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return parse_json_response(content)


def main() -> int:
    return AzureOpenAIProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
