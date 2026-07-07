from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time
from typing import Any

from .core import AgentConfig, AgentHarness
from .eval import collect_tool_calls, collect_usage
from .llm import MockLLM, SubprocessLLM
from .runtime import RuntimePolicy, build_runtime_tools, register_mcp_tools_from_config


@dataclass
class DeterminismResult:
    run_number: int
    run_id: str
    status: str
    latency_seconds: float
    tool_sequence: list[str]
    usage: dict[str, Any]
    final: str
    error: str | None = None


def run_determinism(
    *,
    task: str,
    workspace: Path,
    runs: int,
    llm_cmd: str | None,
    max_steps: int,
    llm_timeout: int,
    allow_shell: bool,
    output_dir: Path,
    policy: RuntimePolicy | None = None,
    mcp_config: Path | None = None,
) -> tuple[Path, Path, list[DeterminismResult]]:
    results: list[DeterminismResult] = []
    for index in range(1, runs + 1):
        llm = SubprocessLLM(llm_cmd, timeout=llm_timeout) if llm_cmd else MockLLM()
        base_policy = policy or RuntimePolicy()
        runtime_policy = RuntimePolicy(
            max_steps=max_steps,
            timeout_seconds=llm_timeout,
            max_cost_usd=base_policy.max_cost_usd,
            allow_file_write=True,
            allow_shell_exec=True,
            allow_network=base_policy.allow_network,
            allow_code_exec=base_policy.allow_code_exec,
            read_only=base_policy.read_only,
            shell_enabled=allow_shell,
            shell_allowlist=base_policy.shell_allowlist,
            deny_dangerous_shell=base_policy.deny_dangerous_shell,
            max_write_bytes=base_policy.max_write_bytes,
        )
        tools = build_runtime_tools(workspace, runtime_policy)
        if mcp_config is not None:
            register_mcp_tools_from_config(tools, mcp_config, workspace=workspace)
        harness = AgentHarness(
            llm=llm,
            config=AgentConfig(workspace=workspace, max_steps=max_steps),
            tools=tools,
            policy=runtime_policy,
        )
        started = time.perf_counter()
        try:
            run = harness.run(task)
            latency = time.perf_counter() - started
            tool_calls = collect_tool_calls(run)
            results.append(
                DeterminismResult(
                    run_number=index,
                    run_id=run.run_id,
                    status=run.status,
                    latency_seconds=latency,
                    tool_sequence=[str(call.get("name")) for call in tool_calls],
                    usage=collect_usage(run),
                    final=run.final or "",
                )
            )
        except Exception as exc:
            latency = time.perf_counter() - started
            results.append(
                DeterminismResult(
                    run_number=index,
                    run_id="",
                    status="error",
                    latency_seconds=latency,
                    tool_sequence=[],
                    usage={},
                    final="",
                    error=str(exc),
                )
            )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"determinism-{timestamp}.jsonl"
    report_path = output_dir / f"determinism-{timestamp}.md"
    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")
    report_path.write_text(render_determinism_report(task, results), encoding="utf-8")
    return results_path, report_path, results


def render_determinism_report(task: str, results: list[DeterminismResult]) -> str:
    total = len(results)
    completed = sum(1 for result in results if result.status == "completed")
    final_counter = Counter(normalize_text(result.final) for result in results if result.final)
    tool_counter = Counter(tuple(result.tool_sequence) for result in results)
    most_common_final_count = final_counter.most_common(1)[0][1] if final_counter else 0
    most_common_tool_count = tool_counter.most_common(1)[0][1] if tool_counter else 0
    avg_latency = sum(result.latency_seconds for result in results) / total if total else 0.0
    total_tokens = sum(
        int(result.usage.get("total_tokens") or result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0))
        for result in results
    )

    lines = [
        "# Determinism Report",
        "",
        f"- Task: {task}",
        f"- Runs: {total}",
        f"- Completed: {completed}",
        f"- Success consistency: {completed / total:.1%}" if total else "- Success consistency: n/a",
        f"- Final answer consistency: {most_common_final_count / total:.1%}" if total else "- Final answer consistency: n/a",
        f"- Tool sequence consistency: {most_common_tool_count / total:.1%}" if total else "- Tool sequence consistency: n/a",
        f"- Average latency: {avg_latency:.2f}s",
        f"- Total tokens: {total_tokens}",
        "",
        "## Tool Sequences",
        "",
    ]
    if not tool_counter:
        lines.append("No tool calls recorded.")
    for sequence, count in tool_counter.most_common():
        label = ", ".join(sequence) if sequence else "none"
        lines.append(f"- `{label}`: {count}")

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Status | Latency | Tools | Run ID | Error |",
            "| ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for result in results:
        tools = ", ".join(result.tool_sequence) or "none"
        lines.append(
            f"| {result.run_number} | {result.status} | {result.latency_seconds:.2f}s | "
            f"{tools} | `{result.run_id}` | {result.error or ''} |"
        )
    lines.append("")
    return "\n".join(lines)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())
