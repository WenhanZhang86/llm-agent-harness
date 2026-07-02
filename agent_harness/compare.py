from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .eval import EvalResult, run_eval
from .tools import PermissionPolicy


@dataclass
class ProviderSpec:
    name: str
    command: str


def parse_provider_specs(values: list[str]) -> list[ProviderSpec]:
    specs: list[ProviderSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Provider must use name=command format: {value}")
        name, command = value.split("=", 1)
        name = name.strip()
        command = command.strip()
        if not name or not command:
            raise ValueError(f"Provider must use name=command format: {value}")
        specs.append(ProviderSpec(name=name, command=command))
    if not specs:
        raise ValueError("At least one provider is required.")
    return specs


def run_compare(
    *,
    tasks_path: Path,
    workspace: Path,
    providers: list[ProviderSpec],
    max_steps: int,
    llm_timeout: int,
    results_dir: Path,
    reports_dir: Path,
    comparisons_dir: Path,
    pricing_path: Path | None = None,
    policy: PermissionPolicy | None = None,
    judge_cmd: str | None = None,
    judge_timeout: int = 240,
    retries: int = 0,
    repeat: int = 1,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    comparisons_dir.mkdir(parents=True, exist_ok=True)
    comparison_json = comparisons_dir / f"comparison-{timestamp}.json"
    comparison_report = comparisons_dir / f"comparison-{timestamp}.md"

    provider_blocks: list[dict[str, Any]] = []
    for provider in providers:
        provider_results_dir = results_dir / provider.name
        provider_reports_dir = reports_dir / provider.name
        results_path, report_path, results = run_eval(
            tasks_path=tasks_path,
            workspace=workspace,
            llm_cmd=provider.command,
            max_steps=max_steps,
            llm_timeout=llm_timeout,
            results_dir=provider_results_dir,
            reports_dir=provider_reports_dir,
            pricing_path=pricing_path,
            policy=policy,
            judge_cmd=judge_cmd,
            judge_timeout=judge_timeout,
            retries=retries,
            repeat=repeat,
        )
        provider_blocks.append(
            {
                "provider": provider.name,
                "command": provider.command,
                "results_path": str(results_path),
                "report_path": str(report_path),
                "summary": summarize_results(results),
                "results": [result.__dict__ for result in results],
            }
        )

    comparison_json.write_text(
        json.dumps(
            {
                "tasks_path": str(tasks_path),
                "providers": provider_blocks,
                "leaderboard": build_leaderboard(provider_blocks),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    comparison_report.write_text(render_comparison_report(tasks_path, provider_blocks), encoding="utf-8")
    return comparison_json, comparison_report, provider_blocks


def summarize_results(results: list[EvalResult]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for result in results if result.passed)
    tool_accuracy = (
        sum(1 for result in results if result.checks.get("expected_tools", False)) / total
        if total
        else None
    )
    avg_latency = sum(result.latency_seconds for result in results) / total if total else 0.0
    priced_results = [result for result in results if result.cost_usd is not None]
    total_cost = sum(result.cost_usd or 0.0 for result in priced_results)
    cost_known = len(priced_results) == total
    total_tokens = sum(
        int(
            result.usage.get("total_tokens")
            or result.usage.get("prompt_tokens", 0) + result.usage.get("completion_tokens", 0)
        )
        for result in results
    )
    categories = Counter(result.failure_category for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed / total if total else None,
        "tool_accuracy": tool_accuracy,
        "average_latency_seconds": avg_latency,
        "total_cost_usd": total_cost if cost_known else None,
        "known_cost_usd": total_cost,
        "cost_known": cost_known,
        "missing_cost_count": total - len(priced_results),
        "total_tokens": total_tokens,
        "failure_categories": dict(sorted(categories.items())),
    }


def build_leaderboard(provider_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in provider_blocks:
        summary = block["summary"]
        rows.append(
            {
                "provider": block["provider"],
                "success_rate": summary["success_rate"],
                "tool_accuracy": summary["tool_accuracy"],
                "average_latency_seconds": summary["average_latency_seconds"],
                "total_cost_usd": summary["total_cost_usd"],
                "known_cost_usd": summary["known_cost_usd"],
                "cost_known": summary["cost_known"],
                "missing_cost_count": summary["missing_cost_count"],
                "total_tokens": summary["total_tokens"],
                "failed": summary["failed"],
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -(row["success_rate"] or 0.0),
            -(row["tool_accuracy"] or 0.0),
            row["average_latency_seconds"],
            row["total_cost_usd"] if row["total_cost_usd"] is not None else row["known_cost_usd"],
        ),
    )


def render_comparison_report(tasks_path: Path, provider_blocks: list[dict[str, Any]]) -> str:
    leaderboard = build_leaderboard(provider_blocks)
    lines = [
        "# Provider Comparison Report",
        "",
        f"- Tasks file: `{tasks_path}`",
        f"- Providers: {', '.join(block['provider'] for block in provider_blocks)}",
        "",
        "## Leaderboard",
        "",
        "| Rank | Provider | Success Rate | Tool Accuracy | Failed | Avg Latency | Total Tokens | Cost |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, row in enumerate(leaderboard, start=1):
        lines.append(
            f"| {index} | `{row['provider']}` | {format_percent(row['success_rate'])} | "
            f"{format_percent(row['tool_accuracy'])} | {row['failed']} | "
            f"{row['average_latency_seconds']:.2f}s | {row['total_tokens']} | "
            f"{format_cost(row)} |"
        )

    lines.extend(["", "## Provider Details", ""])
    for block in provider_blocks:
        summary = block["summary"]
        lines.extend(
            [
                f"### {block['provider']}",
                "",
                f"- Results: `{block['results_path']}`",
                f"- Report: `{block['report_path']}`",
                f"- Passed: {summary['passed']} / {summary['total']}",
                f"- Success rate: {format_percent(summary['success_rate'])}",
                f"- Tool accuracy: {format_percent(summary['tool_accuracy'])}",
                f"- Average latency: {summary['average_latency_seconds']:.2f}s",
                f"- Total tokens: {summary['total_tokens']}",
                f"- Estimated cost: {format_summary_cost(summary)}",
                f"- Known cost from priced tasks: {format_money(summary['known_cost_usd'])}",
                f"- Tasks missing pricing: {summary['missing_cost_count']}",
                f"- Failure categories: {format_categories(summary['failure_categories'])}",
                "",
            ]
        )
    return "\n".join(lines)


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def format_cost(row: dict[str, Any]) -> str:
    if row.get("total_cost_usd") is not None:
        return format_money(row["total_cost_usd"])
    known = format_money(row.get("known_cost_usd"))
    missing = int(row.get("missing_cost_count") or 0)
    return f"{known} partial ({missing} missing)" if missing else known


def format_summary_cost(summary: dict[str, Any]) -> str:
    return format_cost(summary)


def format_categories(categories: dict[str, int]) -> str:
    if not categories:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in categories.items())
