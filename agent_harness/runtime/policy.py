from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimePolicy:
    max_steps: int = 8
    timeout_seconds: int = 240
    max_cost_usd: float | None = None
    allow_file_write: bool = False
    allow_shell_exec: bool = False
    allow_network: bool = False
    allow_code_exec: bool = False

    def allows(self, permission: str) -> bool:
        if permission == "read_files":
            return True
        if permission == "write_files":
            return self.allow_file_write
        if permission == "shell_exec":
            return self.allow_shell_exec
        if permission == "network":
            return self.allow_network
        if permission == "code_exec":
            return self.allow_code_exec
        return False
