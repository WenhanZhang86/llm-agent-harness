from __future__ import annotations

from dataclasses import dataclass
import ast
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any

from .policy import RuntimePolicy
from .tool import Tool, ToolContext, ToolOutput


TEXT_EXTENSIONS = {".md", ".txt", ".py", ".json", ".jsonl", ".toml", ".yaml", ".yml"}
SKIP_PARTS = {"__pycache__", ".git", "runs", "dashboard", "rag"}


class RuntimeToolRegistry:
    def __init__(self, workspace: Path, policy: RuntimePolicy | None = None):
        self.workspace = workspace.resolve()
        self.policy = policy or RuntimePolicy()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
                "output_schema": tool.output_schema,
                "required_permissions": sorted(tool.required_permissions),
            }
            for tool in self._tools.values()
        ]

    def tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    def schemas(self) -> list[dict[str, Any]]:
        return self.tool_schemas()

    def execute(self, name: str | None, args: dict[str, Any], context: ToolContext | None = None) -> ToolOutput:
        if not name or name not in self._tools:
            return ToolOutput(
                ok=False,
                error=f"Unknown tool: {name}",
                metadata={"error_type": "unknown_tool"},
            )
        tool = self._tools[name]
        validation_error = validate_args(args, tool.input_schema)
        if validation_error:
            return ToolOutput(
                ok=False,
                error=validation_error,
                metadata={"tool": name, "error_type": "invalid_args"},
            )
        denied = [permission for permission in tool.required_permissions if not self.policy.allows(permission)]
        if denied:
            return ToolOutput(
                ok=False,
                error=f"Permission denied for tool '{name}': {', '.join(sorted(denied))}",
                metadata={"tool": name, "error_type": "permission_denied", "denied_permissions": sorted(denied)},
            )
        context = context or ToolContext(workspace=self.workspace, policy=self.policy)
        try:
            output = tool.run(args, context)
        except Exception as exc:
            return ToolOutput(
                ok=False,
                error=str(exc),
                metadata={"tool": name, "error_type": exc.__class__.__name__},
            )
        output.metadata.setdefault("tool", name)
        return output

    def call(self, name: str | None, arguments: dict[str, Any]) -> str:
        return json.dumps(self.execute(name, arguments).to_dict(), ensure_ascii=False)

    def close(self) -> None:
        seen: set[int] = set()
        for tool in self._tools.values():
            closer = getattr(tool, "close", None)
            if not callable(closer):
                continue
            identity = id(closer)
            if identity in seen:
                continue
            seen.add(identity)
            closer()

@dataclass
class CalculatorTool:
    name: str = "calculator"
    description: str = "Evaluate a basic arithmetic expression. Supports numbers and +, -, *, /, //, %, **."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        }
        self.output_schema = {
            "type": "object",
            "properties": {"value": {"type": ["number", "integer"]}},
        }
        self.required_permissions = set()

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        value = safe_calculate(str(args.get("expression", "")))
        return ToolOutput(ok=True, result={"value": value}, metadata={"expression": args.get("expression", "")})


@dataclass
class LocalFileSearchTool:
    name: str = "local_file_search"
    description: str = "Search text files under the workspace for a query string."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string", "default": "."},
                "limit": {"type": "integer", "default": 10},
                "max_file_bytes": {"type": "integer", "default": 200000},
            },
            "required": ["query"],
        }
        self.output_schema = {"type": "object", "properties": {"matches": {"type": "array"}}}
        self.required_permissions = {"read_files"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        query = str(args.get("query", "")).strip()
        root = resolve_workspace_path(context.workspace, args.get("path") or ".")
        limit = max(1, min(int(args.get("limit", 10)), 50))
        max_file_bytes = max(1000, min(int(args.get("max_file_bytes", 200000)), 1_000_000))
        matches: list[dict[str, Any]] = []
        for path in iter_text_files(context.workspace, root):
            if len(matches) >= limit:
                break
            if path.stat().st_size > max_file_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            index = text.lower().find(query.lower())
            if query and index != -1:
                matches.append(
                    {
                        "path": str(path.relative_to(context.workspace)),
                        "snippet": text[max(0, index - 120) : index + len(query) + 120],
                    }
                )
        return ToolOutput(ok=True, result={"matches": matches}, metadata={"count": len(matches), "query": query})


@dataclass
class FileReaderTool:
    name: str = "file_reader"
    description: str = "Read a UTF-8 text file under the workspace."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
        }
        self.output_schema = {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}}
        self.required_permissions = {"read_files"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        path = resolve_workspace_path(context.workspace, args.get("path"))
        if not path.is_file():
            raise FileNotFoundError(f"No such file: {path.relative_to(context.workspace)}")
        max_bytes = max(1, min(int(args.get("max_bytes", 20000)), 1_000_000))
        data = path.read_bytes()[:max_bytes]
        content = data.decode("utf-8", errors="replace")
        return ToolOutput(
            ok=True,
            result={"path": str(path.relative_to(context.workspace)), "content": content},
            metadata={"bytes_read": len(data), "truncated": path.stat().st_size > max_bytes},
        )


@dataclass
class FileWriterTool:
    name: str = "file_writer"
    description: str = "Write a UTF-8 text file under the workspace. Disabled unless write_files permission is allowed."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        }
        self.output_schema = {"type": "object", "properties": {"path": {"type": "string"}, "bytes": {"type": "integer"}}}
        self.required_permissions = {"write_files"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        if context.policy.read_only:
            raise PermissionError("file_writer is disabled by the active read-only permission policy.")
        path = resolve_workspace_path(context.workspace, args.get("path"))
        content = str(args.get("content", ""))
        byte_count = len(content.encode("utf-8"))
        if byte_count > context.policy.max_write_bytes:
            raise PermissionError(
                f"file_writer payload is {byte_count} bytes, above the {context.policy.max_write_bytes} byte limit."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolOutput(
            ok=True,
            result={"path": str(path.relative_to(context.workspace)), "bytes": byte_count},
        )


@dataclass
class ListDirTool:
    name: str = "list_dir"
    description: str = "List files and directories under a workspace-relative path."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
        self.output_schema = {"type": "array", "items": {"type": "string"}}
        self.required_permissions = {"read_files"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        path = resolve_workspace_path(context.workspace, args.get("path"))
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        return ToolOutput(
            ok=True,
            result=sorted(child.name + ("/" if child.is_dir() else "") for child in path.iterdir()),
        )


@dataclass
class ReadFileTool:
    name: str = "read_file"
    description: str = "Read a UTF-8 text file under the workspace."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "limit": {"type": "integer", "default": 20000},
            },
            "required": ["path"],
        }
        self.output_schema = {"type": "string"}
        self.required_permissions = {"read_files"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        path = resolve_workspace_path(context.workspace, args.get("path"))
        limit = max(1, min(int(args.get("limit", 20000)), 1_000_000))
        return ToolOutput(ok=True, result=path.read_text(encoding="utf-8", errors="replace")[:limit])


@dataclass
class WriteFileTool:
    name: str = "write_file"
    description: str = "Write a UTF-8 text file under the workspace."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }
        self.output_schema = {"type": "object", "properties": {"path": {"type": "string"}, "bytes": {"type": "integer"}}}
        self.required_permissions = {"write_files"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        if context.policy.read_only:
            raise PermissionError("write_file is disabled by the active read-only permission policy.")
        path = resolve_workspace_path(context.workspace, args.get("path"))
        content = str(args.get("content", ""))
        byte_count = len(content.encode("utf-8"))
        if byte_count > context.policy.max_write_bytes:
            raise PermissionError(
                f"write_file payload is {byte_count} bytes, above the {context.policy.max_write_bytes} byte limit."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return ToolOutput(ok=True, result={"path": str(path), "bytes": byte_count})


@dataclass
class RunShellTool:
    name: str = "run_shell"
    description: str = "Run a shell command in the workspace. Disabled unless shell execution is allowed."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "default": 20},
            },
            "required": ["command"],
        }
        self.output_schema = {"type": "object", "properties": {"returncode": {"type": "integer"}, "stdout": {"type": "string"}, "stderr": {"type": "string"}}}
        self.required_permissions = {"shell_exec"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        if context.policy.shell_enabled is False:
            raise PermissionError("Shell tool is disabled. Re-run with --allow-shell to enable it.")
        command = str(args.get("command", ""))
        validate_shell_command(command, context.policy)
        timeout = max(1, min(int(args.get("timeout", 20)), context.policy.timeout_seconds))
        completed = subprocess.run(
            command,
            cwd=context.workspace,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return ToolOutput(
            ok=True,
            result={
                "returncode": completed.returncode,
                "stdout": completed.stdout[-20000:],
                "stderr": completed.stderr[-20000:],
            },
        )


@dataclass
class MockWebSearchTool:
    name: str = "mock_web_search"
    description: str = "Return deterministic mocked web-search results. This tool never makes a real network call."
    input_schema: dict[str, Any] = None  # type: ignore[assignment]
    output_schema: dict[str, Any] | None = None
    required_permissions: set[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.input_schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        self.output_schema = {"type": "object", "properties": {"results": {"type": "array"}}}
        self.required_permissions = {"network"}

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        query = str(args.get("query", ""))
        return ToolOutput(
            ok=True,
            result={
                "results": [
                    {
                        "title": "Mock search result",
                        "url": "mock://search/1",
                        "snippet": f"Deterministic mock result for: {query}",
                    }
                ]
            },
            metadata={"network_used": False, "query": query},
        )


def build_runtime_tools(workspace: Path, policy: RuntimePolicy) -> RuntimeToolRegistry:
    registry = RuntimeToolRegistry(workspace, policy=policy)
    for tool in [
        ListDirTool(),
        ReadFileTool(),
        WriteFileTool(),
        RunShellTool(),
        CalculatorTool(),
        LocalFileSearchTool(),
        FileReaderTool(),
        FileWriterTool(),
        MockWebSearchTool(),
    ]:
        registry.register(tool)
    return registry


def validate_args(args: dict[str, Any], schema: dict[str, Any]) -> str | None:
    if not isinstance(args, dict):
        return "Tool arguments must be an object."
    for name in schema.get("required", []):
        if name not in args:
            return f"Missing required argument: {name}"
    properties = schema.get("properties", {})
    for name, value in args.items():
        expected = properties.get(name, {}).get("type")
        if expected and not matches_type(value, expected):
            return f"Argument '{name}' must be {expected}."
    return None


def matches_type(value: Any, expected: Any) -> bool:
    expected_types = expected if isinstance(expected, list) else [expected]
    for item in expected_types:
        if item == "string" and isinstance(value, str):
            return True
        if item == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if item == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if item == "boolean" and isinstance(value, bool):
            return True
        if item == "object" and isinstance(value, dict):
            return True
        if item == "array" and isinstance(value, list):
            return True
    return False


def resolve_workspace_path(workspace: Path, value: Any) -> Path:
    raw = Path(str(value or "."))
    path = raw if raw.is_absolute() else workspace / raw
    resolved = path.resolve()
    if workspace != resolved and workspace not in resolved.parents:
        raise ValueError(f"Path escapes workspace: {value}")
    return resolved


def iter_text_files(workspace: Path, root: Path) -> list[Path]:
    candidates = [root] if root.is_file() else sorted(root.rglob("*"))
    files: list[Path] = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        if SKIP_PARTS.intersection(path.relative_to(workspace).parts):
            continue
        files.append(path)
    return files


def validate_shell_command(command: str, policy: RuntimePolicy) -> None:
    if not command.strip():
        raise PermissionError("Empty shell command is not allowed.")
    try:
        parts = shlex.split(command)
    except ValueError as exc:
        raise PermissionError(f"Invalid shell command: {exc}") from exc
    if not parts:
        raise PermissionError("Empty shell command is not allowed.")

    if policy.shell_allowlist is not None:
        allowed = {item.strip() for item in policy.shell_allowlist if item.strip()}
        if parts[0] not in allowed:
            raise PermissionError(f"Shell command '{parts[0]}' is not in the allowlist: {sorted(allowed)}")

    if policy.deny_dangerous_shell:
        normalized = " ".join(parts).lower()
        dangerous_patterns = [
            "rm -rf /",
            "sudo ",
            "mkfs",
            "diskutil erase",
            ":(){",
            "chmod -r 777 /",
            "chown -r",
            "curl ",
            "wget ",
        ]
        for pattern in dangerous_patterns:
            if pattern in normalized:
                raise PermissionError(f"Shell command blocked by safety policy: {pattern.strip()}")


def safe_calculate(expression: str) -> int | float:
    tree = ast.parse(expression, mode="eval")
    return eval_node(tree.body)


def eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left = eval_node(node.left)
        right = eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        return left**right
    raise ValueError("Only numeric arithmetic expressions are allowed.")
