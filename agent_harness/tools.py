from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Callable


ToolFn = Callable[[dict[str, Any]], Any]


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn


class ToolRegistry:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str | None, arguments: dict[str, Any]) -> str:
        if not name or name not in self._tools:
            return json.dumps({"ok": False, "error": f"Unknown tool: {name}"}, ensure_ascii=False)
        try:
            result = self._tools[name].fn(arguments)
            return json.dumps({"ok": True, "result": result}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)

    def resolve_path(self, value: str | None) -> Path:
        raw = Path(value or ".")
        path = raw if raw.is_absolute() else self.workspace / raw
        resolved = path.resolve()
        if self.workspace != resolved and self.workspace not in resolved.parents:
            raise ValueError(f"Path escapes workspace: {value}")
        return resolved


@dataclass
class PermissionPolicy:
    read_only: bool = False
    shell_allowlist: list[str] | None = None
    deny_dangerous_shell: bool = True
    max_write_bytes: int = 1_000_000


def build_default_tools(
    workspace: Path,
    allow_shell: bool,
    policy: PermissionPolicy | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(workspace)
    policy = policy or PermissionPolicy()

    def list_dir(args: dict[str, Any]) -> list[str]:
        path = registry.resolve_path(args.get("path"))
        if not path.is_dir():
            raise ValueError(f"Not a directory: {path}")
        return sorted(child.name + ("/" if child.is_dir() else "") for child in path.iterdir())

    def read_file(args: dict[str, Any]) -> str:
        path = registry.resolve_path(args.get("path"))
        limit = int(args.get("limit", 20000))
        return path.read_text(encoding="utf-8", errors="replace")[:limit]

    def write_file(args: dict[str, Any]) -> dict[str, Any]:
        if policy.read_only:
            raise PermissionError("write_file is disabled by the active read-only permission policy.")
        path = registry.resolve_path(args.get("path"))
        content = str(args.get("content", ""))
        byte_count = len(content.encode("utf-8"))
        if byte_count > policy.max_write_bytes:
            raise PermissionError(
                f"write_file payload is {byte_count} bytes, above the {policy.max_write_bytes} byte limit."
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": str(path), "bytes": byte_count}

    def run_shell(args: dict[str, Any]) -> dict[str, Any]:
        if not allow_shell:
            raise PermissionError("Shell tool is disabled. Re-run with --allow-shell to enable it.")
        command = str(args.get("command", ""))
        validate_shell_command(command, policy)
        timeout = int(args.get("timeout", 20))
        completed = subprocess.run(
            command,
            cwd=registry.workspace,
            shell=True,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout[-20000:],
            "stderr": completed.stderr[-20000:],
        }

    registry.register(
        Tool(
            "list_dir",
            "List files and directories under a workspace-relative path.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
            list_dir,
        )
    )
    registry.register(
        Tool(
            "read_file",
            "Read a UTF-8 text file under the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "limit": {"type": "integer", "default": 20000},
                },
                "required": ["path"],
            },
            read_file,
        )
    )
    registry.register(
        Tool(
            "write_file",
            "Write a UTF-8 text file under the workspace.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            write_file,
        )
    )
    registry.register(
        Tool(
            "run_shell",
            "Run a shell command in the workspace. Disabled unless the harness is started with --allow-shell.",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "default": 20},
                },
                "required": ["command"],
            },
            run_shell,
        )
    )
    return registry


def validate_shell_command(command: str, policy: PermissionPolicy) -> None:
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
            raise PermissionError(
                f"Shell command '{parts[0]}' is not in the allowlist: {sorted(allowed)}"
            )

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
