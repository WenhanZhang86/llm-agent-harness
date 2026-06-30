from __future__ import annotations

import os

from .base import env_required
from .openai_compatible import OpenAICompatibleProvider


class DeepSeekProvider(OpenAICompatibleProvider):
    default_model = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")

    def __init__(self):
        super().__init__(
            model=self.default_model,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            api_key=env_required("DEEPSEEK_API_KEY"),
        )


def main() -> int:
    return DeepSeekProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
