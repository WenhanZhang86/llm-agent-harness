from __future__ import annotations

from abc import ABC, abstractmethod
import http.client
import json
import os
import sys
import time
from typing import Any
from urllib import error, request


HARNESS_INSTRUCTIONS = """
You are the LLM backend for an agent harness.
Return only valid JSON. Do not wrap the JSON in Markdown.
Never return an empty JSON object.

The JSON shape must be:
{
  "thought": "short planning summary",
  "tool_calls": [
    {"name": "tool_name", "arguments": {"key": "value"}}
  ],
  "final": null
}

When the task is complete, return:
{
  "thought": "short completion summary",
  "tool_calls": [],
  "final": "final answer for the user"
}

Use only the tools listed in the payload. If no tool is needed, answer in final.
If the task asks you to read a file and a tool observation already contains that file content, do not call read_file again. Return a final answer.
If the latest tool observation has "ok": true and contains enough information to answer, return final with no tool_calls.
If you are uncertain, prefer a concise final answer over repeating the same tool call.
"""


class Provider(ABC):
    default_model: str

    def __init__(self, model: str | None = None):
        self.model = model or self.default_model

    @abstractmethod
    def complete(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Return a harness response with thought, tool_calls, and final fields."""

    def run_stdin_stdout(self) -> int:
        payload = json.loads(sys.stdin.read())
        response = self.complete(payload)
        print(json.dumps(normalize_harness_response(response), ensure_ascii=False))
        return 0


def compact_messages(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "unknown")
        content = message.get("content")
        if content is None:
            content = {
                key: value
                for key, value in message.items()
                if key != "role" and value is not None
            }
        lines.append(f"{role}: {json.dumps(content, ensure_ascii=False)}")
    return "\n".join(lines)


def build_model_input(payload: dict[str, Any]) -> str:
    messages = payload.get("messages", [])
    latest_tool_observation = next(
        (message for message in reversed(messages) if message.get("role") == "tool"),
        None,
    )
    return json.dumps(
        {
            "task_context": compact_messages(messages),
            "latest_tool_observation": latest_tool_observation,
            "available_tools": payload.get("tools", []),
            "max_tool_calls_remaining": payload.get("max_tool_calls", 0),
            "instruction": (
                "Return final if latest_tool_observation contains enough information. "
                "Do not repeat an identical tool call."
            ),
        },
        ensure_ascii=False,
    )


def parse_json_response(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Provider response was not valid harness JSON: {text[:1000]}") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("Provider response must be a JSON object")
    return parsed


def normalize_harness_response(response: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "thought": response.get("thought", ""),
        "tool_calls": response.get("tool_calls") or [],
        "final": response.get("final"),
    }
    for key in ("provider", "model", "usage"):
        if key in response:
            normalized[key] = response[key]
    return normalized


def merge_usage(*items: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for item in items:
        if not item:
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            elif key not in merged:
                merged[key] = value
    return merged


def env_required(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def post_json(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: int = 120,
    retries: int = 2,
) -> dict[str, Any]:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    for attempt in range(retries + 1):
        try:
            with request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            response_body = exc.read().decode("utf-8", errors="replace")[:2000]
            if exc.code not in {408, 429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(
                    f"HTTP {exc.code} from {url}: {exc.reason}. Response body: {response_body}"
                ) from exc
        except (error.URLError, TimeoutError, http.client.RemoteDisconnected) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Request failed after {retries + 1} attempts: {exc}") from exc
        time.sleep(0.75 * (attempt + 1))
    raise RuntimeError("Request failed unexpectedly")
