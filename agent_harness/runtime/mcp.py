from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shlex
import subprocess
import threading
from typing import Any

from .tool import ToolContext, ToolOutput
from .tool_registry import RuntimeToolRegistry


DEFAULT_PROTOCOL_VERSION = "2025-06-18"


@dataclass
class MCPServerConfig:
    name: str
    transport: str = "stdio"
    command: str | list[str] = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    permissions: set[str] = field(default_factory=lambda: {"network"})
    timeout_seconds: int = 30


class MCPStdioClient:
    def __init__(self, config: MCPServerConfig, workspace: Path):
        if config.transport != "stdio":
            raise ValueError(f"Unsupported MCP transport for {config.name}: {config.transport}")
        self.config = config
        self.workspace = workspace.resolve()
        self._process: subprocess.Popen[str] | None = None
        self._responses: Queue[dict[str, Any]] = Queue()
        self._next_id = 1
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._initialized = False

    def list_tools(self) -> list[dict[str, Any]]:
        self.initialize()
        response = self.request("tools/list", {})
        tools = response.get("tools") or []
        if not isinstance(tools, list):
            raise RuntimeError(f"MCP server {self.config.name} returned invalid tools/list payload.")
        return [tool for tool in tools if isinstance(tool, dict)]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.initialize()
        return self.request("tools/call", {"name": name, "arguments": arguments})

    def initialize(self) -> None:
        if self._initialized:
            return
        self.start()
        self.request(
            "initialize",
            {
                "protocolVersion": DEFAULT_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "llm-agent-harness", "version": "0.1.0"},
            },
        )
        self.notify("notifications/initialized", {})
        self._initialized = True

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        command = build_command(self.config)
        cwd = Path(self.config.cwd).expanduser() if self.config.cwd else self.workspace
        env = None
        if self.config.env:
            env = {**os.environ, **expand_env_values(self.config.env)}
        self._process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

    def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            while True:
                try:
                    message = self._responses.get(timeout=self.config.timeout_seconds)
                except Empty as exc:
                    raise TimeoutError(f"MCP request timed out: {self.config.name}.{method}") from exc
                if message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise RuntimeError(format_jsonrpc_error(message["error"]))
                result = message.get("result") or {}
                if not isinstance(result, dict):
                    raise RuntimeError(f"MCP response for {method} must be an object.")
                return result

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def close(self) -> None:
        if self._process is None:
            self._initialized = False
            return
        if self._process.poll() is None:
            self._process.terminate()
        self._process = None
        self._initialized = False

    def _write(self, message: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"MCP server is not running: {self.config.name}")
        self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self._process.stdin.flush()

    def _read_stdout(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(message, dict) and "id" in message:
                self._responses.put(message)


@dataclass
class MCPToolAdapter:
    server_name: str
    mcp_tool_name: str
    client: MCPStdioClient
    description: str
    input_schema: dict[str, Any]
    required_permissions: set[str]
    output_schema: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return f"mcp__{sanitize_name(self.server_name)}__{sanitize_name(self.mcp_tool_name)}"

    def run(self, args: dict[str, Any], context: ToolContext) -> ToolOutput:
        result = self.client.call_tool(self.mcp_tool_name, args)
        is_error = bool(result.get("isError"))
        content = result.get("content")
        normalized = normalize_mcp_content(content)
        metadata = {
            "mcp": True,
            "server": self.server_name,
            "remote_tool": self.mcp_tool_name,
        }
        if is_error:
            return ToolOutput(ok=False, error=stringify_mcp_content(normalized), metadata=metadata)
        return ToolOutput(ok=True, result=normalized, metadata=metadata)

    def close(self) -> None:
        self.client.close()


def register_mcp_tools_from_config(
    registry: RuntimeToolRegistry,
    config_path: Path | None,
    *,
    workspace: Path,
) -> list[str]:
    if config_path is None:
        return []
    path = config_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"MCP config not found: {path}")
    configs = load_mcp_config(path)
    registered: list[str] = []
    for config in configs:
        client = MCPStdioClient(config, workspace=workspace)
        for remote_tool in client.list_tools():
            name = str(remote_tool.get("name") or "").strip()
            if not name:
                continue
            schema = remote_tool.get("inputSchema") or {"type": "object", "properties": {}}
            if not isinstance(schema, dict):
                schema = {"type": "object", "properties": {}}
            adapter = MCPToolAdapter(
                server_name=config.name,
                mcp_tool_name=name,
                client=client,
                description=str(remote_tool.get("description") or f"MCP tool {config.name}.{name}"),
                input_schema=schema,
                required_permissions=set(config.permissions),
            )
            registry.register(adapter)
            registered.append(adapter.name)
    return registered


def load_mcp_config(path: Path) -> list[MCPServerConfig]:
    data = json.loads(path.read_text(encoding="utf-8"))
    servers = data.get("servers", data)
    if not isinstance(servers, dict):
        raise ValueError("MCP config must contain an object named 'servers'.")
    configs: list[MCPServerConfig] = []
    for name, raw in servers.items():
        if not isinstance(raw, dict):
            raise ValueError(f"MCP server config must be an object: {name}")
        configs.append(
            MCPServerConfig(
                name=str(name),
                transport=str(raw.get("transport", "stdio")),
                command=raw.get("command", ""),
                args=[str(item) for item in raw.get("args", [])],
                env={str(key): str(value) for key, value in (raw.get("env") or {}).items()},
                cwd=str(raw["cwd"]) if raw.get("cwd") is not None else None,
                permissions={str(item) for item in raw.get("permissions", ["network"])},
                timeout_seconds=int(raw.get("timeout_seconds", 30)),
            )
        )
    return configs


def build_command(config: MCPServerConfig) -> list[str]:
    if isinstance(config.command, list):
        command = [str(item) for item in config.command]
    else:
        command = shlex.split(str(config.command))
    command.extend(config.args)
    if not command:
        raise ValueError(f"MCP server command is required: {config.name}")
    return command


def expand_env_values(values: dict[str, str]) -> dict[str, str]:
    expanded: dict[str, str] = {}
    for key, value in values.items():
        text = str(value)
        if text.startswith("$") and len(text) > 1:
            expanded[key] = os.environ.get(text[1:], "")
        elif text.startswith("${") and text.endswith("}") and len(text) > 3:
            expanded[key] = os.environ.get(text[2:-1], "")
        else:
            expanded[key] = text
    return expanded


def sanitize_name(value: str) -> str:
    sanitized = "".join(char if char.isalnum() or char == "_" else "_" for char in value.strip())
    return sanitized.strip("_") or "tool"


def normalize_mcp_content(content: Any) -> Any:
    if isinstance(content, list):
        items: list[Any] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                items.append(item.get("text", ""))
            else:
                items.append(item)
        return items[0] if len(items) == 1 else items
    return content


def stringify_mcp_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def format_jsonrpc_error(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message") or error
        return f"MCP error: {message}"
    return f"MCP error: {error}"
