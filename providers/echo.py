from __future__ import annotations

from typing import Any

from .base import Provider


class EchoProvider(Provider):
    default_model = "echo"

    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        tool_messages = [message for message in payload.get("messages", []) if message.get("role") == "tool"]
        if not tool_messages:
            return {
                "thought": "I will list the workspace.",
                "tool_calls": [{"name": "list_dir", "arguments": {"path": "."}}],
                "final": None,
            }
        return {
            "thought": "The workspace has been listed.",
            "tool_calls": [],
            "final": "Echo provider finished with observation: " + tool_messages[-1]["content"],
        }


def main() -> int:
    return EchoProvider().run_stdin_stdout()


if __name__ == "__main__":
    raise SystemExit(main())
