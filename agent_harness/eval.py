from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

from .core import AgentConfig, AgentHarness, AgentRun
from .llm import MockLLM, SubprocessLLM
from .tools import PermissionPolicy, build_default_tools


@dataclass
class EvalTask:
    id: str
    task: str
    expect_status: str = "completed"
    expect_tools: list[str] = field(default_factory=list)
    forbid_tools: list[str] = field(default_factory=list)
    expect_contains: list[str] = field(default_factory=list)
    expect_any_contains: list[str] = field(default_factory=list)
    expect_observation_contains: list[str] = field(default_factory=list)
    allow_shell: bool = False


@dataclass
class EvalResult:
    task_id: str
    task: str
    status: str
    passed: bool
    checks: dict[str, bool]
    latency_seconds: float
    tool_calls: list[dict[str, Any]]
    usage: dict[str, Any]
    cost_usd: float | None
    cost_breakdown_usd: dict[str, float | None]
    failure_category: str
    final: str | None
    run_id: str
    error: str | None = None


def load_tasks(path: Path) -> list[EvalTask]:
    tasks: list[EvalTask] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        tasks.append(
            EvalTask(
                id=str(data.get("id") or f"task_{line_number}"),
                task=str(data["task"]),
                expect_status=str(data.get("expect_status", "completed")),
                expect_tools=[str(item) for item in data.get("expect_tools", [])],
                forbid_tools=[str(item) for item in data.get("forbid_tools", [])],
                expect_contains=[str(item) for item in data.get("expect_contains", [])],
                expect_any_contains=[str(item) for item in data.get("expect_any_contains", [])],
                expect_observation_contains=[
                    str(item) for item in data.get("expect_observation_contains", [])
                ],
                allow_shell=bool(data.get("allow_shell", False)),
            )
        )
    return tasks


def collect_tool_calls(run: AgentRun) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for message in run.messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def collect_observations(run: AgentRun) -> str:
    observations: list[str] = []
    for message in run.messages:
        if message.get("role") == "tool":
            observations.append(str(message.get("content", "")))
    return "\n".join(observations)


def collect_usage(run: AgentRun) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for message in run.messages:
        if message.get("role") != "assistant":
            continue
        item = message.get("usage") or {}
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
            elif key not in usage:
                usage[key] = value
    return usage


def collect_model(run: AgentRun) -> str | None:
    for message in reversed(run.messages):
        if message.get("role") == "assistant" and message.get("model"):
            return str(message.get("model"))
    return None


def calculate_cost_breakdown_usd(
    model: str | None, usage: dict[str, Any], pricing: dict[str, Any]
) -> dict[str, float | None]:
    if not model:
        return {"prompt": None, "completion": None, "embedding": 0.0, "tool": 0.0, "total": None}
    prices = pricing.get("prices_per_1m_tokens", {}).get(model)
    if not prices:
        return {"prompt": None, "completion": None, "embedding": 0.0, "tool": 0.0, "total": None}

    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
        return {"prompt": None, "completion": None, "embedding": 0.0, "tool": 0.0, "total": None}
    prompt_cost = input_tokens * float(prices.get("input", 0.0)) / 1_000_000
    completion_cost = output_tokens * float(prices.get("output", 0.0)) / 1_000_000
    return {
        "prompt": prompt_cost,
        "completion": completion_cost,
        "embedding": 0.0,
        "tool": 0.0,
        "total": prompt_cost + completion_cost,
    }


def calculate_cost_usd(model: str | None, usage: dict[str, Any], pricing: dict[str, Any]) -> float | None:
    return calculate_cost_breakdown_usd(model, usage, pricing).get("total")


def load_pricing(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {"prices_per_1m_tokens": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def classify_failure(checks: dict[str, bool], status: str, error: str | None, observations: str) -> str:
    if all(checks.values()):
        return "success"

    text = f"{error or ''}\n{observations}".lower()
    if "timed out" in text or "timeoutexpired" in text:
        return "timeout"
    if "429" in text or "rate limit" in text:
        return "rate_limit"
    if "http error" in text or "remotedisconnected" in text or "request failed" in text:
        return "api_error"
    if "shell tool is disabled" in text or "path escapes workspace" in text or "permission" in text:
        return "permission_denied"
    if not checks.get("status", True) or status in {"stopped", "max_steps_exceeded"}:
        return "planning_error"
    if not checks.get("expected_tools", True) or not checks.get("forbidden_tools", True):
        return "planning_error"
    if not checks.get("expected_observation", True):
        return "tool_error"
    if not checks.get("expected_text", True):
        return "hallucination"
    return "failed"


def evaluate_run(task: EvalTask, run: AgentRun, latency_seconds: float, pricing: dict[str, Any]) -> EvalResult:
    tool_calls = collect_tool_calls(run)
    tool_names = [str(call.get("name")) for call in tool_calls]
    final = run.final or ""
    observations = collect_observations(run)
    usage = collect_usage(run)
    model = collect_model(run)

    checks = {
        "status": run.status == task.expect_status,
        "expected_tools": all(name in tool_names for name in task.expect_tools),
        "forbidden_tools": all(name not in tool_names for name in task.forbid_tools),
        "expected_text": all(text.lower() in final.lower() for text in task.expect_contains),
        "expected_any_text": (
            any(text.lower() in final.lower() for text in task.expect_any_contains)
            if task.expect_any_contains
            else True
        ),
        "expected_observation": all(
            text.lower() in observations.lower() for text in task.expect_observation_contains
        ),
    }
    cost_breakdown = calculate_cost_breakdown_usd(model, usage, pricing)
    failure_category = classify_failure(checks, run.status, None, observations)
    return EvalResult(
        task_id=task.id,
        task=task.task,
        status=run.status,
        passed=all(checks.values()),
        checks=checks,
        latency_seconds=latency_seconds,
        tool_calls=tool_calls,
        usage=usage,
        cost_usd=cost_breakdown.get("total"),
        cost_breakdown_usd=cost_breakdown,
        failure_category=failure_category,
        final=run.final,
        run_id=run.run_id,
    )


def run_eval(
    *,
    tasks_path: Path,
    workspace: Path,
    llm_cmd: str | None,
    max_steps: int,
    llm_timeout: int,
    results_dir: Path,
    reports_dir: Path,
    pricing_path: Path | None = None,
    policy: PermissionPolicy | None = None,
) -> tuple[Path, Path, list[EvalResult]]:
    tasks = load_tasks(tasks_path)
    pricing = load_pricing(pricing_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"eval-{timestamp}.jsonl"
    report_path = reports_dir / f"eval-{timestamp}.md"

    results: list[EvalResult] = []
    for task in tasks:
        llm = SubprocessLLM(llm_cmd, timeout=llm_timeout) if llm_cmd else MockLLM()
        tools = build_default_tools(workspace, allow_shell=task.allow_shell, policy=policy)
        harness = AgentHarness(
            llm=llm,
            tools=tools,
            config=AgentConfig(workspace=workspace, max_steps=max_steps),
        )
        started = time.perf_counter()
        latency = time.perf_counter() - started
        try:
            run = harness.run(task.task)
            latency = time.perf_counter() - started
            results.append(evaluate_run(task, run, latency, pricing))
        except Exception as exc:
            latency = time.perf_counter() - started
            results.append(
                EvalResult(
                    task_id=task.id,
                    task=task.task,
                    status="error",
                    passed=False,
                    checks={"status": False, "expected_tools": False, "expected_text": False},
                    latency_seconds=latency,
                    tool_calls=[],
                    usage={},
                    cost_usd=None,
                    cost_breakdown_usd={
                        "prompt": None,
                        "completion": None,
                        "embedding": 0.0,
                        "tool": 0.0,
                        "total": None,
                    },
                    failure_category=classify_failure(
                        {"status": False, "expected_tools": False, "expected_text": False},
                        "error",
                        str(exc),
                        "",
                    ),
                    final=None,
                    run_id="",
                    error=str(exc),
                )
            )

    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")

    report_path.write_text(render_report(tasks_path, results), encoding="utf-8")
    return results_path, report_path, results


def render_report(tasks_path: Path, results: list[EvalResult]) -> str:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    avg_latency = sum(result.latency_seconds for result in results) / total if total else 0.0
    total_cost = sum(result.cost_usd or 0.0 for result in results)
    prompt_cost = sum(result.cost_breakdown_usd.get("prompt") or 0.0 for result in results)
    completion_cost = sum(result.cost_breakdown_usd.get("completion") or 0.0 for result in results)
    embedding_cost = sum(result.cost_breakdown_usd.get("embedding") or 0.0 for result in results)
    tool_cost = sum(result.cost_breakdown_usd.get("tool") or 0.0 for result in results)
    total_tokens = sum(
        int(result.usage.get("total_tokens") or result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0))
        for result in results
    )
    categories = Counter(result.failure_category for result in results)
    tool_accuracy = (
        sum(1 for result in results if result.checks.get("expected_tools", False)) / total
        if total
        else 0.0
    )

    lines = [
        "# Evaluation Report",
        "",
        f"- Tasks file: `{tasks_path}`",
        f"- Total tasks: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Success rate: {passed / total:.1%}" if total else "- Success rate: n/a",
        f"- Tool accuracy: {tool_accuracy:.1%}" if total else "- Tool accuracy: n/a",
        f"- Average latency: {avg_latency:.2f}s",
        f"- Total tokens: {total_tokens}",
        f"- Estimated cost: ${total_cost:.6f}",
        f"- Prompt cost: ${prompt_cost:.6f}",
        f"- Completion cost: ${completion_cost:.6f}",
        f"- Embedding cost: ${embedding_cost:.6f}",
        f"- Tool cost: ${tool_cost:.6f}",
        f"- Failure categories: {format_counter(categories)}",
        "",
        "| Task | Status | Passed | Category | Latency | Tokens | Cost | Tools | Checks | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        tools = ", ".join(str(call.get("name")) for call in result.tool_calls) or "none"
        tokens = result.usage.get("total_tokens") or (
            result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
        )
        cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "n/a"
        checks = ", ".join(
            f"{name}={'pass' if value else 'fail'}" for name, value in result.checks.items()
        )
        lines.append(
            f"| `{result.task_id}` | {result.status} | {'yes' if result.passed else 'no'} | "
            f"{result.failure_category} | "
            f"{result.latency_seconds:.2f}s | {tokens} | {cost} | {tools} | {checks} | {result.error or ''} |"
        )

    lines.extend(["", "## Failed Tasks", ""])
    failed = [result for result in results if not result.passed]
    if not failed:
        lines.append("No failed tasks.")
    for result in failed:
        lines.extend(
            [
                f"### {result.task_id}",
                "",
                f"- Status: {result.status}",
                f"- Failure category: {result.failure_category}",
                f"- Run ID: `{result.run_id}`",
                f"- Checks: `{json.dumps(result.checks, ensure_ascii=False)}`",
                f"- Cost breakdown: `{json.dumps(result.cost_breakdown_usd, ensure_ascii=False)}`",
                f"- Error: {result.error or ''}",
                f"- Final: {result.final or ''}",
                "",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(counter.items()))
