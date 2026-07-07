from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PermissionsRequest(BaseModel):
    allow_file_write: bool = False
    allow_shell_exec: bool = False
    allow_code_exec: bool = False
    allow_network: bool = False


class RunAgentRequest(BaseModel):
    task: str = Field(..., min_length=1)
    provider: str | None = None
    model: str | None = None
    max_steps: int | None = Field(default=None, ge=1, le=20)
    timeout_seconds: int | None = Field(default=None, ge=1, le=300)
    use_rag_context: bool | None = True
    permissions: PermissionsRequest | None = None


class ErrorResponse(BaseModel):
    ok: bool = False
    error: str
    category: str
    details: dict[str, Any] = Field(default_factory=dict)


class RunAgentResponse(BaseModel):
    run_id: str
    status: str
    final_answer: str | None
    trace_path: str
    summary_path: str


class HealthResponse(BaseModel):
    status: str
    service: str


class ReplayEvent(BaseModel):
    event_id: int | None = None
    event_type: str
    step_id: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ReplayResponse(BaseModel):
    run_id: str
    status: str | None = None
    task: str | None = None
    final_answer: str | None = None
    events: list[ReplayEvent]
