from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any

from .core import AgentConfig, AgentHarness, AgentRun
from .llm import MockLLM, SubprocessLLM
from .runtime import RuntimePolicy, build_runtime_tools, register_mcp_tools_from_config


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
    read_only: bool | None = None
    shell_allow: list[str] | None = None
    max_write_bytes: int | None = None
    expected_answer: str | None = None
    rubric: str | None = None
    judge_min_score: float = 0.8


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
    structured_trace: list[dict[str, Any]] = field(default_factory=list)
    judge: dict[str, Any] | None = None
    failure_details: dict[str, Any] = field(default_factory=dict)
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
                read_only=data.get("read_only") if data.get("read_only") is None else bool(data.get("read_only")),
                shell_allow=(
                    [str(item) for item in data.get("shell_allow", [])]
                    if data.get("shell_allow") is not None
                    else None
                ),
                max_write_bytes=(
                    int(data["max_write_bytes"]) if data.get("max_write_bytes") is not None else None
                ),
                expected_answer=(
                    str(data["expected_answer"]) if data.get("expected_answer") is not None else None
                ),
                rubric=str(data["rubric"]) if data.get("rubric") is not None else None,
                judge_min_score=float(data.get("judge_min_score", 0.8)),
            )
        )
    return tasks


def dataset_metadata(path: Path, task_count: int) -> dict[str, Any]:
    content = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content).hexdigest(),
        "task_count": task_count,
        "bytes": len(content),
    }


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
    prices = resolve_model_pricing(model, pricing)
    if not prices:
        return {"prompt": None, "completion": None, "embedding": 0.0, "tool": 0.0, "total": None}

    input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    output_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    if not isinstance(input_tokens, (int, float)) or not isinstance(output_tokens, (int, float)):
        return {"prompt": None, "completion": None, "embedding": 0.0, "tool": 0.0, "total": None}
    prompt_cost = calculate_prompt_cost(input_tokens, usage, prices)
    completion_cost = output_tokens * float(prices.get("output", 0.0)) / 1_000_000
    return {
        "prompt": prompt_cost,
        "completion": completion_cost,
        "embedding": 0.0,
        "tool": 0.0,
        "total": prompt_cost + completion_cost,
    }


def resolve_model_pricing(model: str, pricing: dict[str, Any]) -> dict[str, Any] | None:
    price_table = pricing.get("prices_per_1m_tokens", {})
    if model in price_table:
        return price_table[model]
    base_model = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model)
    if base_model in price_table:
        return price_table[base_model]
    return None


def calculate_prompt_cost(input_tokens: int | float, usage: dict[str, Any], prices: dict[str, Any]) -> float:
    cache_hit_price = prices.get("cache_hit")
    cache_hit_tokens = usage.get("prompt_cache_hit_tokens")
    cache_miss_tokens = usage.get("prompt_cache_miss_tokens")
    if (
        cache_hit_price is not None
        and isinstance(cache_hit_tokens, (int, float))
        and isinstance(cache_miss_tokens, (int, float))
    ):
        return (
            cache_miss_tokens * float(prices.get("input", 0.0))
            + cache_hit_tokens * float(cache_hit_price)
        ) / 1_000_000
    return input_tokens * float(prices.get("input", 0.0)) / 1_000_000


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


def analyze_failure(
    task: EvalTask,
    checks: dict[str, bool],
    tool_names: list[str],
    final: str | None,
    observations: str,
    error: str | None = None,
) -> dict[str, Any]:
    failed_checks = [name for name, value in checks.items() if not value]
    details: dict[str, Any] = {
        "failed_checks": failed_checks,
        "expected_tools": task.expect_tools,
        "actual_tools": tool_names,
        "forbidden_tools": task.forbid_tools,
        "expected_text": task.expect_contains,
        "expected_any_text": task.expect_any_contains,
        "expected_observation": task.expect_observation_contains,
        "recommendation": recommend_fix(failed_checks, error, observations),
    }
    if error:
        details["error"] = error
    if final is not None:
        details["final_preview"] = final[:500]
    if observations:
        details["observation_preview"] = observations[:500]
    return details


def recommend_fix(failed_checks: list[str], error: str | None, observations: str) -> str:
    text = f"{error or ''}\n{observations}".lower()
    if "timed out" in text or "timeoutexpired" in text:
        return "Increase --llm-timeout, add --retries, or reduce the task size."
    if "429" in text or "rate limit" in text:
        return "Retry later, lower concurrency, or use a provider key with more quota."
    if "http error" in text or "authorization" in text:
        return "Check the provider API key, model name, and adapter endpoint."
    if "path escapes workspace" in text or "permission" in text or "disabled" in text:
        return "Review the task permission policy and expected safety behavior."
    if "expected_tools" in failed_checks or "forbidden_tools" in failed_checks:
        return "Clarify the task instruction or update the expected tool sequence."
    if "expected_observation" in failed_checks:
        return "Inspect the tool output and verify the expected observation string."
    if "expected_text" in failed_checks or "expected_any_text" in failed_checks or "judge_score" in failed_checks:
        return "Inspect the final answer against the expected answer or rubric."
    return "Inspect the transcript and failed checks."


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
    failure_details = analyze_failure(task, checks, tool_names, final, observations)
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
        structured_trace=run.structured_trace,
        failure_details=failure_details,
    )


def apply_judge(
    task: EvalTask,
    result: EvalResult,
    *,
    judge_cmd: str | None,
    judge_timeout: int,
) -> None:
    if not judge_cmd or not (task.expected_answer or task.rubric):
        return

    prompt = {
        "instruction": (
            "You are judging an agent evaluation result. Return final as strict JSON with "
            "keys: score, passed, reason. Score must be between 0 and 1."
        ),
        "task": task.task,
        "expected_answer": task.expected_answer,
        "rubric": task.rubric,
        "minimum_passing_score": task.judge_min_score,
        "actual_answer": result.final,
        "tool_calls": result.tool_calls,
        "checks": result.checks,
    }
    try:
        judge_llm = SubprocessLLM(judge_cmd, timeout=judge_timeout)
        response = judge_llm.complete(
            {
                "messages": [{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
                "tools": [],
                "max_tool_calls": 0,
            }
        )
        final = response.get("final")
        parsed = json.loads(final) if isinstance(final, str) else response
        score = float(parsed.get("score", 0.0))
        passed = bool(parsed.get("passed", score >= task.judge_min_score))
        result.judge = {
            "score": score,
            "passed": passed,
            "reason": str(parsed.get("reason", "")),
            "model": response.get("model"),
            "usage": response.get("usage", {}),
        }
        result.checks["judge_score"] = passed
    except Exception as exc:
        result.judge = {"score": None, "passed": False, "reason": f"Judge failed: {exc}"}
        result.checks["judge_score"] = False

    result.passed = all(result.checks.values())
    if not result.passed and result.failure_category == "success":
        result.failure_category = "hallucination"
    tool_names = [str(call.get("name")) for call in result.tool_calls]
    result.failure_details = analyze_failure(task, result.checks, tool_names, result.final, "")


def build_task_policy(base_policy: RuntimePolicy | None, task: EvalTask, *, max_steps: int, timeout_seconds: int) -> RuntimePolicy:
    base_policy = base_policy or RuntimePolicy()
    return RuntimePolicy(
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        max_cost_usd=base_policy.max_cost_usd,
        allow_file_write=True,
        allow_shell_exec=True,
        allow_network=base_policy.allow_network,
        allow_code_exec=base_policy.allow_code_exec,
        read_only=base_policy.read_only if task.read_only is None else bool(task.read_only),
        shell_enabled=task.allow_shell,
        shell_allowlist=base_policy.shell_allowlist if task.shell_allow is None else task.shell_allow,
        deny_dangerous_shell=base_policy.deny_dangerous_shell,
        max_write_bytes=base_policy.max_write_bytes if task.max_write_bytes is None else task.max_write_bytes,
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
    policy: RuntimePolicy | None = None,
    mcp_config: Path | None = None,
    judge_cmd: str | None = None,
    judge_timeout: int = 240,
    retries: int = 0,
    repeat: int = 1,
) -> tuple[Path, Path, list[EvalResult]]:
    tasks = load_tasks(tasks_path)
    pricing = load_pricing(pricing_path)
    metadata = dataset_metadata(tasks_path, len(tasks))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"eval-{timestamp}.jsonl"
    report_path = reports_dir / f"eval-{timestamp}.md"
    summary_path = results_dir / f"eval-{timestamp}.summary.json"
    csv_path = results_dir / f"eval-{timestamp}.csv"
    html_path = reports_dir / f"eval-{timestamp}.html"

    results: list[EvalResult] = []
    for task in tasks:
        for repeat_index in range(max(1, repeat)):
            result = run_single_eval_task(
                task=task,
                workspace=workspace,
                llm_cmd=llm_cmd,
                max_steps=max_steps,
                llm_timeout=llm_timeout,
                pricing=pricing,
                policy=policy,
                mcp_config=mcp_config,
                judge_cmd=judge_cmd,
                judge_timeout=judge_timeout,
                retries=max(0, retries),
            )
            if repeat > 1:
                result.task_id = f"{task.id}#run{repeat_index + 1}"
            results.append(result)

    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result.__dict__, ensure_ascii=False) + "\n")

    report = render_report(tasks_path, results, metadata=metadata)
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(build_summary(metadata, results), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(csv_path, results)
    html_path.write_text(render_html_report(report), encoding="utf-8")
    return results_path, report_path, results


def run_single_eval_task(
    *,
    task: EvalTask,
    workspace: Path,
    llm_cmd: str | None,
    max_steps: int,
    llm_timeout: int,
    pricing: dict[str, Any],
    policy: RuntimePolicy | None,
    mcp_config: Path | None,
    judge_cmd: str | None,
    judge_timeout: int,
    retries: int,
) -> EvalResult:
    attempts = retries + 1
    last_result: EvalResult | None = None
    for attempt in range(1, attempts + 1):
        result = execute_eval_attempt(
            task=task,
            workspace=workspace,
            llm_cmd=llm_cmd,
            max_steps=max_steps,
            llm_timeout=llm_timeout,
            pricing=pricing,
            policy=policy,
            mcp_config=mcp_config,
            judge_cmd=judge_cmd,
            judge_timeout=judge_timeout,
        )
        result.failure_details["attempt"] = attempt
        result.failure_details["max_attempts"] = attempts
        if result.passed:
            return result
        last_result = result
        if result.failure_category not in {"api_error", "timeout", "rate_limit"}:
            return result
    return last_result if last_result is not None else execute_eval_attempt(
        task=task,
        workspace=workspace,
        llm_cmd=llm_cmd,
        max_steps=max_steps,
        llm_timeout=llm_timeout,
        pricing=pricing,
        policy=policy,
        mcp_config=mcp_config,
        judge_cmd=judge_cmd,
        judge_timeout=judge_timeout,
    )


def execute_eval_attempt(
    *,
    task: EvalTask,
    workspace: Path,
    llm_cmd: str | None,
    max_steps: int,
    llm_timeout: int,
    pricing: dict[str, Any],
    policy: RuntimePolicy | None,
    mcp_config: Path | None,
    judge_cmd: str | None,
    judge_timeout: int,
) -> EvalResult:
    llm = SubprocessLLM(llm_cmd, timeout=llm_timeout) if llm_cmd else MockLLM()
    runtime_policy = build_task_policy(
        policy,
        task,
        max_steps=max_steps,
        timeout_seconds=llm_timeout,
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
        run = harness.run(task.task)
        latency = time.perf_counter() - started
        result = evaluate_run(task, run, latency, pricing)
        apply_judge(task, result, judge_cmd=judge_cmd, judge_timeout=judge_timeout)
        return result
    except Exception as exc:
        latency = time.perf_counter() - started
        checks = {"status": False, "expected_tools": False, "expected_text": False}
        category = classify_failure(checks, "error", str(exc), "")
        return EvalResult(
            task_id=task.id,
            task=task.task,
            status="error",
            passed=False,
            checks=checks,
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
            failure_category=category,
            final=None,
            run_id="",
            structured_trace=[],
            judge=None,
            failure_details=analyze_failure(task, checks, [], None, "", str(exc)),
            error=str(exc),
        )


def render_report(
    tasks_path: Path, results: list[EvalResult], metadata: dict[str, Any] | None = None
) -> str:
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
    judged = [result for result in results if result.judge]
    judge_avg = (
        sum(float(result.judge.get("score") or 0.0) for result in judged) / len(judged)
        if judged
        else None
    )
    tool_accuracy = (
        sum(1 for result in results if result.checks.get("expected_tools", False)) / total
        if total
        else 0.0
    )

    lines = [
        "# Evaluation Report",
        "",
        f"- Tasks file: `{tasks_path}`",
        f"- Dataset SHA-256: `{metadata.get('sha256')}`" if metadata else "- Dataset SHA-256: n/a",
        f"- Dataset tasks: {metadata.get('task_count')}" if metadata else "- Dataset tasks: n/a",
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
        f"- Judged tasks: {len(judged)}",
        f"- Average judge score: {judge_avg:.2f}" if judge_avg is not None else "- Average judge score: n/a",
        "",
        "| Task | Status | Passed | Category | Latency | Tokens | Cost | Judge | Tools | Failed Checks | Error |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for result in results:
        tools = ", ".join(str(call.get("name")) for call in result.tool_calls) or "none"
        tokens = result.usage.get("total_tokens") or (
            result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
        )
        cost = f"${result.cost_usd:.6f}" if result.cost_usd is not None else "n/a"
        failed_checks = ", ".join(name for name, value in result.checks.items() if not value) or "none"
        judge = (
            f"{float(result.judge.get('score') or 0.0):.2f}"
            if result.judge and result.judge.get("score") is not None
            else "n/a"
        )
        lines.append(
            f"| `{result.task_id}` | {result.status} | {'yes' if result.passed else 'no'} | "
            f"{result.failure_category} | "
            f"{result.latency_seconds:.2f}s | {tokens} | {cost} | {judge} | {tools} | "
            f"{failed_checks} | {result.error or ''} |"
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
                f"- Failure details: `{json.dumps(result.failure_details, ensure_ascii=False)}`",
                f"- Judge: `{json.dumps(result.judge, ensure_ascii=False)}`",
                f"- Cost breakdown: `{json.dumps(result.cost_breakdown_usd, ensure_ascii=False)}`",
                f"- Error: {result.error or ''}",
                f"- Final: {result.final or ''}",
                "",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def build_summary(metadata: dict[str, Any], results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    categories = Counter(result.failure_category for result in results)
    total_cost = sum(result.cost_usd or 0.0 for result in results)
    total_tokens = sum(
        int(
            result.usage.get("total_tokens")
            or result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
        )
        for result in results
    )
    return {
        "dataset": metadata,
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / total if total else None,
        "average_latency_seconds": (
            sum(result.latency_seconds for result in results) / total if total else 0.0
        ),
        "total_tokens": total_tokens,
        "total_cost_usd": total_cost,
        "failure_categories": dict(sorted(categories.items())),
    }


def write_csv(path: Path, results: list[EvalResult]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "task_id",
                "status",
                "passed",
                "failure_category",
                "latency_seconds",
                "tokens",
                "cost_usd",
                "tools",
                "failed_checks",
                "judge_score",
                "error",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "task_id": result.task_id,
                    "status": result.status,
                    "passed": result.passed,
                    "failure_category": result.failure_category,
                    "latency_seconds": f"{result.latency_seconds:.4f}",
                    "tokens": result.usage.get("total_tokens")
                    or (
                        result.usage.get("prompt_tokens", 0)
                        + result.usage.get("completion_tokens", 0)
                    ),
                    "cost_usd": "" if result.cost_usd is None else f"{result.cost_usd:.8f}",
                    "tools": ", ".join(str(call.get("name")) for call in result.tool_calls),
                    "failed_checks": ", ".join(
                        name for name, value in result.checks.items() if not value
                    ),
                    "judge_score": (
                        ""
                        if not result.judge or result.judge.get("score") is None
                        else result.judge.get("score")
                    ),
                    "error": result.error or "",
                }
            )


def render_html_report(markdown: str) -> str:
    body = (
        markdown.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>\n")
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Evaluation Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; color: #172026; }}
    code {{ background: #eef2f4; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>{body}</body>
</html>
"""


def format_counter(counter: Counter[str]) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in sorted(counter.items()))
