from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_harness.runtime import RuntimePolicy, ToolContext, build_runtime_tools


class RuntimeToolTests(unittest.TestCase):
    def test_registry_register_get_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = build_runtime_tools(Path(tmp), RuntimePolicy())
            self.assertIsNotNone(registry.get("calculator"))
            names = {tool["name"] for tool in registry.list_tools()}
            self.assertIn("calculator", names)
            self.assertIn("file_reader", names)

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

    def test_mock_web_search_is_deterministic_when_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy = RuntimePolicy(allow_network=True)
            registry = build_runtime_tools(Path(tmp), policy)
            context = ToolContext(workspace=Path(tmp), policy=policy)
            first = registry.execute("mock_web_search", {"query": "agent harness"}, context)
            second = registry.execute("mock_web_search", {"query": "agent harness"}, context)
            self.assertTrue(first.ok)
            self.assertEqual(first.to_dict(), second.to_dict())


if __name__ == "__main__":
    unittest.main()
