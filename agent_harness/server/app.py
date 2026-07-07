from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from ..llm import MockLLM, SubprocessLLM
from ..runtime import AgentRuntime, RuntimePolicy, build_runtime_tools
from ..trace import load_run_events, load_run_summary, load_transcript
from .schemas import (
    ErrorResponse,
    HealthResponse,
    PermissionsRequest,
    ReplayResponse,
    RunAgentRequest,
    RunAgentResponse,
)


SAFE_MAX_STEPS = 8
SAFE_TIMEOUT_SECONDS = 120


def create_app(workspace: Path | None = None) -> FastAPI:
    workspace_path = (workspace or Path.cwd()).resolve()
    app = FastAPI(title="LLM Agent Harness", version="0.1.0")
    app.state.workspace = workspace_path

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", service="llm-agent-harness")

    @app.get("/tools")
    def tools() -> dict[str, Any]:
        policy = safe_policy()
        registry = build_runtime_tools(workspace_path, policy)
        return {"tools": registry.list_tools()}

    @app.post("/run-agent", response_model=RunAgentResponse, responses={403: {"model": ErrorResponse}})
    def run_agent(request: RunAgentRequest) -> RunAgentResponse:
        denied = denied_permissions(request.permissions)
        if denied:
            raise HTTPException(
                status_code=403,
                detail={
                    "ok": False,
                    "error": "Requested runtime permissions are denied by the server safety policy.",
                    "category": "permission_denied",
                    "details": {"denied_permissions": denied},
                },
            )
        policy = safe_policy(
            max_steps=request.max_steps,
            timeout_seconds=request.timeout_seconds,
        )
        llm = llm_for_provider(request.provider, timeout=policy.timeout_seconds)
        with temporary_model_env(request.provider, request.model):
            runtime = AgentRuntime(
                llm=llm,
                workspace=workspace_path,
                policy=policy,
                use_rag_context=bool(request.use_rag_context),
            )
            result = runtime.run(request.task)
        summary_path = workspace_path / "runs" / result.run_id / "summary.json"
        return RunAgentResponse(
            run_id=result.run_id,
            status=result.status,
            final_answer=result.final,
            trace_path=str(result.trace_path),
            summary_path=str(summary_path),
        )

    @app.get("/runs/{run_id}")
    def run_summary(run_id: str) -> dict[str, Any]:
        return load_or_404(lambda: load_run_summary(workspace_path, run_id=run_id), run_id)

    @app.get("/runs/{run_id}/trace")
    def run_trace(run_id: str) -> dict[str, Any]:
        return load_or_404(lambda: load_transcript(workspace_path, run_id=run_id), run_id)

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str) -> list[dict[str, Any]]:
        return load_or_404(lambda: load_run_events(workspace_path, run_id=run_id), run_id)

    @app.get("/runs/{run_id}/replay", response_model=ReplayResponse)
    def run_replay(run_id: str) -> ReplayResponse:
        trace = load_or_404(lambda: load_transcript(workspace_path, run_id=run_id), run_id)
        events = load_or_404(lambda: load_run_events(workspace_path, run_id=run_id), run_id)
        return ReplayResponse(
            run_id=run_id,
            status=trace.get("status"),
            task=trace.get("task"),
            final_answer=trace.get("final"),
            events=[
                {
                    "event_id": event.get("event_id"),
                    "event_type": str(event.get("event_type")),
                    "step_id": event.get("step_id"),
                    "data": event.get("data") or {},
                }
                for event in events
            ],
        )

    return app


def safe_policy(max_steps: int | None = None, timeout_seconds: int | None = None) -> RuntimePolicy:
    return RuntimePolicy(
        max_steps=max_steps or SAFE_MAX_STEPS,
        timeout_seconds=timeout_seconds or SAFE_TIMEOUT_SECONDS,
        allow_file_write=False,
        allow_shell_exec=False,
        allow_network=False,
        allow_code_exec=False,
    )


def denied_permissions(permissions: PermissionsRequest | None) -> list[str]:
    if permissions is None:
        return []
    denied: list[str] = []
    for name in ["allow_file_write", "allow_shell_exec", "allow_code_exec", "allow_network"]:
        if bool(getattr(permissions, name, False)):
            denied.append(name)
    return denied


def llm_for_provider(provider: str | None, timeout: int) -> MockLLM | SubprocessLLM:
    selected = (provider or "mock").strip().lower()
    if selected in {"", "mock"}:
        return MockLLM()
    if not selected.replace("_", "").replace("-", "").isalnum():
        raise HTTPException(status_code=400, detail={"ok": False, "error": "Invalid provider name.", "category": "bad_request"})
    module_name = selected.replace("-", "_")
    return SubprocessLLM(f"python3 -m providers.{module_name}", timeout=timeout)


@contextmanager
def temporary_model_env(provider: str | None, model: str | None):
    if not provider or not model:
        yield
        return
    env_name = f"{provider.strip().upper().replace('-', '_')}_MODEL"
    previous = os.environ.get(env_name)
    os.environ[env_name] = model
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = previous


def load_or_404(loader, run_id: str):
    try:
        return loader()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": f"Run not found: {run_id}", "category": "not_found"},
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404,
            detail={"ok": False, "error": str(exc), "category": "not_found"},
        )


app = create_app()
