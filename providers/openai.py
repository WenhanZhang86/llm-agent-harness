from __future__ import annotations

import os

from .base import env_required
from .openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    default_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    def __init__(self) -> None:
        super().__init__(
            model=self.default_model,
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=env_required("OPENAI_API_KEY"),
        )


def main() -> int:
    return OpenAIProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
