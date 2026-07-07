from __future__ import annotations

import json
import sys


TOOLS = [
    {
        "name": "echo",
        "description": "Echo a message through a fake MCP server.",
        "inputSchema": {
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
    }
]


def main() -> int:
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method = message.get("method")
        if method == "initialize":
            respond(message["id"], {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}}})
        elif method == "tools/list":
            respond(message["id"], {"tools": TOOLS})
        elif method == "tools/call":
            params = message.get("params") or {}
            if params.get("name") != "echo":
                respond_error(message["id"], f"Unknown fake tool: {params.get('name')}")
                continue
            arguments = params.get("arguments") or {}
            respond(
                message["id"],
                {
                    "content": [{"type": "text", "text": f"fake-mcp: {arguments.get('message', '')}"}],
                    "isError": False,
                },
            )
        else:
            respond_error(message["id"], f"Unsupported method: {method}")
    return 0


def respond(request_id: int, result: dict) -> None:
    print(json.dumps({"jsonrpc": "2.0", "id": request_id, "result": result}), flush=True)


def respond_error(request_id: int, message: str) -> None:
    print(
        json.dumps({"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": message}}),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
