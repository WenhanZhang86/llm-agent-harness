from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

from agent_harness.runtime import RuntimePolicy, ToolContext, build_runtime_tools, register_mcp_tools_from_config


class RuntimeToolTests(unittest.TestCase):
    def test_registry_register_get_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_runtime_tools(Path(tmp), RuntimePolicy())
            self.assertIsNotNone(registry.get("calculator"))
            names = {tool["name"] for tool in registry.list_tools()}
            self.assertIn("calculator", names)
            self.assertIn("file_reader", names)
            self.assertIn("list_dir", names)
            self.assertIn("read_file", names)
            self.assertIn("write_file", names)
            self.assertIn("run_shell", names)

    def test_calculator_works(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_runtime_tools(Path(tmp), RuntimePolicy())
            output = registry.execute("calculator", {"expression": "2 + 3 * 4"})
            self.assertTrue(output.ok)
            self.assertEqual(output.result, {"value": 14})

    def test_denied_permission_returns_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_runtime_tools(Path(tmp), RuntimePolicy())
            output = registry.execute("mock_web_search", {"query": "agent harness"})
            self.assertFalse(output.ok)
            self.assertEqual(output.metadata.get("error_type"), "permission_denied")

    def test_file_reader_blocks_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_runtime_tools(Path(tmp), RuntimePolicy())
            output = registry.execute("file_reader", {"path": "../outside.txt"})
            self.assertFalse(output.ok)
            self.assertIn("Path escapes workspace", output.error or "")

    def test_legacy_named_runtime_file_tools_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            policy = RuntimePolicy(allow_file_write=True)
            registry = build_runtime_tools(workspace, policy)
            written = registry.execute("write_file", {"path": "notes/example.txt", "content": "hello"})
            read = registry.execute("read_file", {"path": "notes/example.txt"})
            listed = registry.execute("list_dir", {"path": "notes"})

            self.assertTrue(written.ok)
            self.assertTrue(read.ok)
            self.assertEqual(read.result, "hello")
            self.assertTrue(listed.ok)
            self.assertIn("example.txt", listed.result)

    def test_write_file_policy_limits_are_runtime_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            read_only = build_runtime_tools(workspace, RuntimePolicy(allow_file_write=True, read_only=True))
            denied_read_only = read_only.execute("write_file", {"path": "x.txt", "content": "hello"})
            self.assertFalse(denied_read_only.ok)
            self.assertIn("read-only permission policy", denied_read_only.error or "")

            size_limited = build_runtime_tools(workspace, RuntimePolicy(allow_file_write=True, max_write_bytes=3))
            denied_size = size_limited.execute("write_file", {"path": "x.txt", "content": "hello"})
            self.assertFalse(denied_size.ok)
            self.assertIn("above the 3 byte limit", denied_size.error or "")

    def test_run_shell_policy_is_runtime_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            disabled_policy = RuntimePolicy(allow_shell_exec=True, shell_enabled=False)
            disabled = build_runtime_tools(workspace, disabled_policy).execute("run_shell", {"command": "pwd"})
            self.assertFalse(disabled.ok)
            self.assertIn("Shell tool is disabled", disabled.error or "")

            allowlist_policy = RuntimePolicy(allow_shell_exec=True, shell_enabled=True, shell_allowlist=["pwd"])
            blocked = build_runtime_tools(workspace, allowlist_policy).execute("run_shell", {"command": "ls"})
            self.assertFalse(blocked.ok)
            self.assertIn("not in the allowlist", blocked.error or "")

            dangerous_policy = RuntimePolicy(allow_shell_exec=True, shell_enabled=True)
            dangerous = build_runtime_tools(workspace, dangerous_policy).execute("run_shell", {"command": "curl https://example.com"})
            self.assertFalse(dangerous.ok)
            self.assertIn("Shell command blocked by safety policy", dangerous.error or "")

    def test_mock_web_search_is_deterministic_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = RuntimePolicy(allow_network=True)
            registry = build_runtime_tools(Path(tmp), policy)
            context = ToolContext(workspace=Path(tmp), policy=policy)
            first = registry.execute("mock_web_search", {"query": "agent harness"}, context)
            second = registry.execute("mock_web_search", {"query": "agent harness"}, context)
            self.assertTrue(first.ok)
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_mcp_stdio_tool_registration_and_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config_path = workspace / "mcp.json"
            fake_server = Path(__file__).with_name("fake_mcp_server.py")
            config_path.write_text(
                """
{
  "servers": {
    "fake": {
      "transport": "stdio",
      "command": ["%s", "%s"],
      "permissions": ["network"]
    }
  }
}
"""
                % (sys.executable.replace("\\", "\\\\"), str(fake_server).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            policy = RuntimePolicy(allow_network=True)
            registry = build_runtime_tools(workspace, policy)
            registered = register_mcp_tools_from_config(registry, config_path, workspace=workspace)

            self.assertIn("mcp__fake__echo", registered)
            output = registry.execute("mcp__fake__echo", {"message": "hello"})
            self.assertTrue(output.ok)
            self.assertEqual(output.result, "fake-mcp: hello")
            self.assertEqual(output.metadata.get("server"), "fake")

    def test_mcp_tool_respects_permission_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            config_path = workspace / "mcp.json"
            fake_server = Path(__file__).with_name("fake_mcp_server.py")
            config_path.write_text(
                """
{
  "servers": {
    "fake": {
      "transport": "stdio",
      "command": ["%s", "%s"],
      "permissions": ["network"]
    }
  }
}
"""
                % (sys.executable.replace("\\", "\\\\"), str(fake_server).replace("\\", "\\\\")),
                encoding="utf-8",
            )
            registry = build_runtime_tools(workspace, RuntimePolicy())
            register_mcp_tools_from_config(registry, config_path, workspace=workspace)

            output = registry.execute("mcp__fake__echo", {"message": "hello"})
            self.assertFalse(output.ok)
            self.assertEqual(output.metadata.get("error_type"), "permission_denied")


if __name__ == "__main__":
    unittest.main()
