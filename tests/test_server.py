from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from urllib.parse import urlsplit

from agent_harness.server.app import create_app


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        (self.workspace / "README.md").write_text("LLM Agent Harness test workspace.", encoding="utf-8")
        self.app = create_app(workspace=self.workspace)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def request(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict | list]:
        return asyncio.run(asgi_json_request(self.app, method, path, body))

    def test_health_endpoint(self) -> None:
        status, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")

    def test_tools_endpoint(self) -> None:
        status, data = self.request("GET", "/tools")
        self.assertEqual(status, 200)
        names = {tool["name"] for tool in data["tools"]}
        self.assertIn("calculator", names)
        self.assertIn("file_reader", names)

    def test_run_agent_and_fetch_artifacts(self) -> None:
        status, data = self.request(
            "POST",
            "/run-agent",
            {"task": "Use calculator to compute 2 + 2", "provider": "mock", "max_steps": 4},
        )
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "completed")
        run_id = data["run_id"]

        summary_status, summary = self.request("GET", f"/runs/{run_id}")
        self.assertEqual(summary_status, 200)
        self.assertEqual(summary["run_id"], run_id)

        trace_status, trace = self.request("GET", f"/runs/{run_id}/trace")
        self.assertEqual(trace_status, 200)
        self.assertEqual(trace["run_id"], run_id)

        events_status, events = self.request("GET", f"/runs/{run_id}/events")
        self.assertEqual(events_status, 200)
        self.assertTrue(any(event["event_type"] == "agent_started" for event in events))

        replay_status, replay = self.request("GET", f"/runs/{run_id}/replay")
        self.assertEqual(replay_status, 200)
        self.assertEqual(replay["run_id"], run_id)
        self.assertTrue(replay["events"])

    def test_safe_default_permissions(self) -> None:
        status, data = self.request(
            "POST",
            "/run-agent",
            {
                "task": "Write a file",
                "provider": "mock",
                "permissions": {"allow_file_write": True},
            },
        )
        self.assertEqual(status, 403)
        detail = data["detail"]
        self.assertEqual(detail["category"], "permission_denied")
        self.assertIn("allow_file_write", detail["details"]["denied_permissions"])


async def asgi_json_request(app, method: str, path: str, body: dict | None = None) -> tuple[int, dict | list]:
    parsed = urlsplit(path)
    body_bytes = b""
    headers = []
    if body is not None:
        body_bytes = json.dumps(body).encode("utf-8")
        headers.append((b"content-type", b"application/json"))
        headers.append((b"content-length", str(len(body_bytes)).encode("ascii")))
    messages = [
        {
            "type": "http.request",
            "body": body_bytes,
            "more_body": False,
        }
    ]
    sent = []

    async def receive():
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": parsed.path,
        "raw_path": parsed.path.encode("utf-8"),
        "query_string": parsed.query.encode("utf-8"),
        "headers": headers,
        "client": ("testclient", 50000),
        "server": ("testserver", 80),
    }
    await app(scope, receive, send)
    status = 500
    chunks = []
    for message in sent:
        if message["type"] == "http.response.start":
            status = int(message["status"])
        elif message["type"] == "http.response.body":
            chunks.append(message.get("body", b""))
    data = json.loads(b"".join(chunks).decode("utf-8"))
    return status, data


if __name__ == "__main__":
    unittest.main()
