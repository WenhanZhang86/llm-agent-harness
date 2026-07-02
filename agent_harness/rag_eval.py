from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .rag import build_rag_index, query_rag_index


@dataclass
class RagEvalTask:
    id: str
    question: str
    expected_sources: list[str] = field(default_factory=list)
    expected_answer_contains: list[str] = field(default_factory=list)
    forbidden_answer_contains: list[str] = field(default_factory=list)


def load_rag_tasks(path: Path) -> list[RagEvalTask]:
    tasks: list[RagEvalTask] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        tasks.append(
            RagEvalTask(
                id=str(data.get("id") or f"rag_task_{line_number}"),
                question=str(data["question"]),
                expected_sources=[str(item) for item in data.get("expected_sources", [])],
                expected_answer_contains=[str(item) for item in data.get("expected_answer_contains", [])],
                forbidden_answer_contains=[str(item) for item in data.get("forbidden_answer_contains", [])],
            )
        )
    return tasks


def run_rag_eval(
    *,
    workspace: Path,
    tasks_path: Path,
    index_path: Path,
    inputs: list[str],
    results_dir: Path,
    reports_dir: Path,
    top_k: int = 5,
) -> tuple[Path, Path, list[dict[str, Any]]]:
    if not index_path.exists():
        build_rag_index(workspace=workspace, inputs=inputs, output=index_path)
    tasks = load_rag_tasks(tasks_path)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"rag-eval-{timestamp}.jsonl"
    report_path = reports_dir / f"rag-eval-{timestamp}.md"

    results: list[dict[str, Any]] = []
    for task in tasks:
        answer = query_rag_index(index_path=index_path, question=task.question, top_k=top_k)
        checks = {
            "expected_sources": all(source in answer for source in task.expected_sources),
            "expected_answer": all(text.lower() in answer.lower() for text in task.expected_answer_contains),
            "forbidden_answer": all(text.lower() not in answer.lower() for text in task.forbidden_answer_contains),
        }
        results.append(
            {
                "task_id": task.id,
                "question": task.question,
                "passed": all(checks.values()),
                "checks": checks,
                "answer": answer,
            }
        )

    with results_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
    report_path.write_text(render_rag_report(tasks_path, results), encoding="utf-8")
    return results_path, report_path, results


def render_rag_report(tasks_path: Path, results: list[dict[str, Any]]) -> str:
    total = len(results)
    passed = sum(1 for result in results if result["passed"])
    source_hits = sum(1 for result in results if result["checks"].get("expected_sources"))
    answer_hits = sum(1 for result in results if result["checks"].get("expected_answer"))
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Tasks file: `{tasks_path}`",
        f"- Total questions: {total}",
        f"- Passed: {passed}",
        f"- Failed: {total - passed}",
        f"- Pass rate: {passed / total:.1%}" if total else "- Pass rate: n/a",
        f"- Retrieval/source hit rate: {source_hits / total:.1%}" if total else "- Retrieval/source hit rate: n/a",
        f"- Answer hit rate: {answer_hits / total:.1%}" if total else "- Answer hit rate: n/a",
        "",
        "| Question | Passed | Checks |",
        "| --- | --- | --- |",
    ]
    for result in results:
        checks = ", ".join(f"{key}={'pass' if value else 'fail'}" for key, value in result["checks"].items())
        lines.append(f"| `{result['task_id']}` | {'yes' if result['passed'] else 'no'} | {checks} |")
    failed = [result for result in results if not result["passed"]]
    if failed:
        lines.extend(["", "## Failed Questions", ""])
        for result in failed:
            lines.extend(
                [
                    f"### {result['task_id']}",
                    "",
                    f"- Question: {result['question']}",
                    f"- Checks: `{json.dumps(result['checks'], ensure_ascii=False)}`",
                    "",
                    "```text",
                    result["answer"][:2000],
                    "```",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"
