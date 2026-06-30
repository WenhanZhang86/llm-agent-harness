from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def build_dashboard(workspace: Path, output: Path | None = None) -> Path:
    output_path = output or workspace / "dashboard" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_results = load_jsonl_files(workspace / "evals" / "results")
    determinism_results = load_jsonl_files(workspace / "evals" / "determinism")
    runs = sorted((workspace / "runs").glob("*.json")) if (workspace / "runs").exists() else []
    output_path.write_text(
        render_dashboard(workspace, eval_results, determinism_results, runs),
        encoding="utf-8",
    )
    return output_path


def load_jsonl_files(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not directory.exists():
        return rows
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item["_source"] = str(path)
            rows.append(item)
    return rows


def render_dashboard(
    workspace: Path,
    eval_results: list[dict[str, Any]],
    determinism_results: list[dict[str, Any]],
    runs: list[Path],
) -> str:
    total_evals = len(eval_results)
    passed_evals = sum(1 for item in eval_results if item.get("passed"))
    completed_determinism = sum(1 for item in determinism_results if item.get("status") == "completed")
    total_cost = sum(float(item.get("cost_usd") or 0.0) for item in eval_results)
    avg_latency = (
        sum(float(item.get("latency_seconds") or 0.0) for item in eval_results) / total_evals
        if total_evals
        else 0.0
    )
    latest_eval_rows = eval_results[-20:]
    latest_det_rows = determinism_results[-20:]

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Agent Harness Dashboard</title>
  <style>
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #172026;
      background: #f5f7f8;
    }}
    header {{
      padding: 28px 36px 18px;
      background: #ffffff;
      border-bottom: 1px solid #dfe5e8;
    }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    main {{ padding: 24px 36px 40px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; }}
    .metric, section {{
      background: #ffffff;
      border: 1px solid #dfe5e8;
      border-radius: 8px;
    }}
    .metric {{ padding: 16px; }}
    .label {{ color: #66747c; font-size: 13px; }}
    .value {{ font-size: 26px; font-weight: 700; margin-top: 6px; }}
    section {{ margin-top: 18px; overflow: hidden; }}
    h2 {{ margin: 0; padding: 16px 18px; font-size: 18px; border-bottom: 1px solid #dfe5e8; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #edf1f3; text-align: left; vertical-align: top; }}
    th {{ color: #51616a; font-weight: 600; background: #fafbfb; }}
    code {{ background: #edf1f3; padding: 2px 4px; border-radius: 4px; }}
    @media (max-width: 820px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} main, header {{ padding-left: 18px; padding-right: 18px; }} }}
  </style>
</head>
<body>
  <header>
    <h1>LLM Agent Harness Dashboard</h1>
    <div class="label">{escape(str(workspace))}</div>
  </header>
  <main>
    <div class="grid">
      {metric("Eval pass rate", percent(passed_evals, total_evals))}
      {metric("Eval tasks", str(total_evals))}
      {metric("Average latency", f"{avg_latency:.2f}s")}
      {metric("Estimated cost", f"${total_cost:.6f}")}
      {metric("Transcripts", str(len(runs)))}
      {metric("Determinism runs", str(len(determinism_results)))}
      {metric("Completed stability runs", str(completed_determinism))}
      {metric("Workspace", "ready")}
    </div>
    <section>
      <h2>Latest Evaluation Results</h2>
      {render_eval_table(latest_eval_rows)}
    </section>
    <section>
      <h2>Latest Determinism Runs</h2>
      {render_determinism_table(latest_det_rows)}
    </section>
  </main>
</body>
</html>
"""


def metric(label: str, value: str) -> str:
    return f'<div class="metric"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'


def render_eval_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No evaluation results yet.</p>"
    body = []
    for item in rows:
        body.append(
            "<tr>"
            f"<td><code>{escape(str(item.get('task_id', '')))}</code></td>"
            f"<td>{escape(str(item.get('status', '')))}</td>"
            f"<td>{'yes' if item.get('passed') else 'no'}</td>"
            f"<td>{escape(str(item.get('failure_category', '')))}</td>"
            f"<td>{float(item.get('latency_seconds') or 0.0):.2f}s</td>"
            f"<td>${float(item.get('cost_usd') or 0.0):.6f}</td>"
            f"<td><code>{escape(str(item.get('run_id', '')))}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Task</th><th>Status</th><th>Passed</th><th>Category</th>"
        "<th>Latency</th><th>Cost</th><th>Run ID</th></tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def render_determinism_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>No determinism results yet.</p>"
    body = []
    for item in rows:
        tools = ", ".join(item.get("tool_sequence") or []) or "none"
        body.append(
            "<tr>"
            f"<td>{escape(str(item.get('run_number', '')))}</td>"
            f"<td>{escape(str(item.get('status', '')))}</td>"
            f"<td>{float(item.get('latency_seconds') or 0.0):.2f}s</td>"
            f"<td>{escape(tools)}</td>"
            f"<td><code>{escape(str(item.get('run_id', '')))}</code></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Run</th><th>Status</th><th>Latency</th><th>Tools</th><th>Run ID</th></tr>"
        "</thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.1%}" if denominator else "n/a"


def escape(value: str) -> str:
    return html.escape(value, quote=True)
