from __future__ import annotations

import os

from .base import env_required
from .openai_compatible import OpenAICompatibleProvider


class GroqProvider(OpenAICompatibleProvider):
    default_model = os.environ.get("GROQ_MODEL", "llama-3.1-70b-versatile")

    def __init__(self):
        super().__init__(
            model=self.default_model,
            base_url=os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            api_key=env_required("GROQ_API_KEY"),
        )


def main() -> int:
    return GroqProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
