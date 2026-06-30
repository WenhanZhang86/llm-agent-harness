from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_dashboard(workspace: Path, output: Path | None = None) -> Path:
    output_path = output or workspace / "dashboard" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = collect_dashboard_data(workspace)
    output_path.write_text(render_dashboard(data), encoding="utf-8")
    return output_path


def collect_dashboard_data(workspace: Path) -> dict[str, Any]:
    eval_results = load_jsonl_files(workspace / "evals" / "results")
    determinism_results = load_jsonl_files(workspace / "evals" / "determinism")
    runs = load_runs(workspace / "runs")
    rag_index = load_json(workspace / "rag" / "index.json", default={"chunks": []})
    latest_eval_source = str(eval_results[-1].get("_source", "")) if eval_results else ""
    latest_eval_rows = [item for item in eval_results if item.get("_source") == latest_eval_source]
    return {
        "workspace": str(workspace),
        "evalResults": eval_results,
        "latestEvalRows": latest_eval_rows,
        "determinismResults": determinism_results,
        "runs": runs,
        "rag": rag_index,
        "summary": summarize(eval_results, latest_eval_rows, determinism_results, runs),
    }


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


def load_runs(directory: Path) -> list[dict[str, Any]]:
    if not directory.exists():
        return []
    runs: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        item["_source"] = str(path)
        item["tool_sequence"] = collect_tool_sequence(item)
        item["usage"] = collect_usage(item)
        runs.append(item)
    return runs


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def collect_tool_sequence(run: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for message in run.get("messages", []):
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            sequence.append(str(call.get("name", "")))
    return sequence


def collect_usage(run: dict[str, Any]) -> dict[str, int | float]:
    usage: dict[str, int | float] = {}
    for message in run.get("messages", []):
        if message.get("role") != "assistant":
            continue
        item = message.get("usage") or {}
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if isinstance(value, (int, float)):
                usage[key] = usage.get(key, 0) + value
    return usage


def summarize(
    eval_results: list[dict[str, Any]],
    latest_eval_rows: list[dict[str, Any]],
    determinism_results: list[dict[str, Any]],
    runs: list[dict[str, Any]],
) -> dict[str, Any]:
    total_latest = len(latest_eval_rows)
    passed_latest = sum(1 for item in latest_eval_rows if item.get("passed"))
    total_cost = sum(float(item.get("cost_usd") or 0.0) for item in latest_eval_rows)
    avg_latency = (
        sum(float(item.get("latency_seconds") or 0.0) for item in latest_eval_rows) / total_latest
        if total_latest
        else 0.0
    )
    return {
        "latestPassRate": passed_latest / total_latest if total_latest else None,
        "latestPassed": passed_latest,
        "latestTotal": total_latest,
        "latestCost": total_cost,
        "latestAvgLatency": avg_latency,
        "totalEvalRows": len(eval_results),
        "runCount": len(runs),
        "determinismCount": len(determinism_results),
    }


def render_dashboard(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>LLM Agent Harness Dashboard</title>
  <style>
    :root {{
      --bg: #f4f6f7;
      --panel: #ffffff;
      --line: #dce3e6;
      --line-soft: #edf1f3;
      --text: #172026;
      --muted: #60707a;
      --teal: #0f766e;
      --blue: #2f5d9f;
      --amber: #a15c00;
      --red: #b42318;
      --green-bg: #e8f6ef;
      --green: #0f6b3f;
      --red-bg: #fae8e6;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .app {{ min-height: 100vh; display: grid; grid-template-columns: 260px 1fr; }}
    aside {{
      background: #ffffff;
      border-right: 1px solid var(--line);
      padding: 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }}
    main {{ min-width: 0; }}
    .brand {{ font-size: 18px; font-weight: 760; margin: 4px 0 2px; }}
    .workspace {{
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      line-height: 1.35;
      margin-bottom: 18px;
    }}
    .nav {{ display: grid; gap: 6px; }}
    .nav button, .small-button {{
      border: 1px solid transparent;
      background: transparent;
      color: var(--text);
      border-radius: 7px;
      min-height: 36px;
      padding: 8px 10px;
      text-align: left;
      font: inherit;
      cursor: pointer;
    }}
    .nav button:hover, .small-button:hover {{ background: #f0f4f5; }}
    .nav button.active {{ background: #e7f2f1; border-color: #b7d5d1; color: #0b554f; font-weight: 650; }}
    header {{
      background: #ffffff;
      border-bottom: 1px solid var(--line);
      padding: 18px 28px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
    }}
    h1 {{ margin: 0; font-size: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 17px; }}
    h3 {{ margin: 0 0 8px; font-size: 15px; }}
    .page {{ padding: 20px 28px 34px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(150px, 1fr)); gap: 12px; margin-bottom: 16px; }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 14px; }}
    .metric-label {{ color: var(--muted); font-size: 12px; }}
    .metric-value {{ margin-top: 5px; font-size: 24px; font-weight: 760; }}
    .grid-2 {{ display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(320px, 0.9fr); gap: 14px; }}
    .panel {{ overflow: hidden; }}
    .panel-head {{ padding: 14px 16px; border-bottom: 1px solid var(--line-soft); display: flex; justify-content: space-between; gap: 10px; align-items: center; }}
    .panel-body {{ padding: 14px 16px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid var(--line-soft); text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-weight: 650; background: #fafbfb; }}
    tr.selectable {{ cursor: pointer; }}
    tr.selectable:hover {{ background: #f7fafb; }}
    code, pre {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    code {{ background: #eef2f4; border-radius: 4px; padding: 2px 4px; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.5;
      max-height: 260px;
      overflow: auto;
    }}
    .pill {{ display: inline-flex; align-items: center; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 650; }}
    .pass {{ background: var(--green-bg); color: var(--green); }}
    .fail {{ background: var(--red-bg); color: var(--red); }}
    .neutral {{ background: #eef2f4; color: #44535b; }}
    .tabs {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .tabs button {{ text-align: center; }}
    .timeline {{ display: grid; gap: 10px; }}
    .step {{ border-left: 3px solid #b7d5d1; background: #fbfcfc; padding: 10px 12px; border-radius: 0 7px 7px 0; }}
    .step.tool {{ border-left-color: #d7a44a; }}
    .search {{ width: 100%; min-height: 40px; border: 1px solid var(--line); border-radius: 7px; padding: 9px 10px; font: inherit; }}
    .muted {{ color: var(--muted); }}
    .empty {{ color: var(--muted); padding: 22px; text-align: center; }}
    .summary-line {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .health {{
      display: grid;
      grid-template-columns: minmax(240px, 1fr) repeat(3, minmax(130px, 180px));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .health-main {{ padding: 16px; background: #ffffff; border: 1px solid var(--line); border-radius: 8px; }}
    .health-title {{ font-size: 18px; font-weight: 760; margin-bottom: 6px; }}
    .health-copy {{ color: var(--muted); line-height: 1.45; }}
    .toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }}
    .toolbar input, .toolbar select {{
      min-height: 36px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 7px 9px;
      font: inherit;
      background: #ffffff;
    }}
    .toolbar input {{ min-width: 220px; flex: 1; }}
    .chart-list {{ display: grid; gap: 10px; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(110px, 180px) 1fr auto; gap: 10px; align-items: center; }}
    .bar-track {{ height: 10px; background: #edf1f3; border-radius: 999px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--teal); border-radius: 999px; min-width: 2px; }}
    .bar-fill.cost {{ background: var(--amber); }}
    .bar-fill.tokens {{ background: var(--blue); }}
    .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12px; }}
    .bar-value {{ color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; }}
    @media (max-width: 980px) {{
      .app {{ grid-template-columns: 1fr; }}
      aside {{ position: relative; height: auto; border-right: 0; border-bottom: 1px solid var(--line); }}
      .nav {{ grid-template-columns: repeat(3, 1fr); }}
      .metrics {{ grid-template-columns: repeat(2, minmax(150px, 1fr)); }}
      .grid-2 {{ grid-template-columns: 1fr; }}
      .health {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 620px) {{
      header, .page {{ padding-left: 14px; padding-right: 14px; }}
      .metrics {{ grid-template-columns: 1fr; }}
      .nav {{ grid-template-columns: 1fr 1fr; }}
      .health {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
      table {{ font-size: 12px; }}
      th, td {{ padding: 8px 7px; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">LLM Agent Harness</div>
      <div class="workspace" id="workspace"></div>
      <nav class="nav" id="nav"></nav>
    </aside>
    <main>
      <header>
        <div>
          <h1 id="title">Overview</h1>
          <div class="muted" id="subtitle"></div>
        </div>
        <div class="tabs" id="headerActions"></div>
      </header>
      <div class="page" id="page"></div>
    </main>
  </div>
  <script>
    const DATA = {payload};
    const state = {{
      page: "overview",
      selectedEval: null,
      selectedRun: null,
      selectedDet: null,
      evalFilter: "latest",
      evalSort: "source",
      evalSearch: ""
    }};
    const navItems = [
      ["overview", "Overview"],
      ["evaluations", "Evaluations"],
      ["runs", "Runs"],
      ["determinism", "Determinism"],
      ["rag", "RAG Search"]
    ];

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, char => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", "\\"": "&quot;", "'": "&#39;"
      }}[char]));
    }}
    function fmtMoney(value) {{ return value == null ? "n/a" : "$" + Number(value || 0).toFixed(6); }}
    function fmtSeconds(value) {{ return Number(value || 0).toFixed(2) + "s"; }}
    function fmtPercent(value) {{ return value == null ? "n/a" : (value * 100).toFixed(1) + "%"; }}
    function tokens(item) {{
      const usage = item.usage || {{}};
      return usage.total_tokens || ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0));
    }}
    function tools(item) {{
      const calls = item.tool_calls || [];
      if (calls.length) return calls.map(call => call.name).join(", ");
      const sequence = item.tool_sequence || [];
      return sequence.length ? sequence.join(", ") : "none";
    }}
    function pill(label, ok = null) {{
      const klass = ok === true ? "pass" : ok === false ? "fail" : "neutral";
      return `<span class="pill ${{klass}}">${{escapeHtml(label)}}</span>`;
    }}
    function latestEvalRows() {{ return DATA.latestEvalRows || []; }}
    function healthStatus() {{
      const rows = latestEvalRows();
      const failed = rows.filter(row => !row.passed);
      if (!rows.length) return {{ label: "No baseline", ok: null, copy: "No evaluation results are available yet." }};
      if (failed.length === 0) return {{ label: "Healthy", ok: true, copy: "All default tasks passed. No failed categories were detected in the latest evaluation." }};
      const categories = [...new Set(failed.map(row => row.failure_category || "failed"))].join(", ");
      return {{ label: "Needs attention", ok: false, copy: `${{failed.length}} task(s) failed in the latest evaluation. Categories: ${{categories}}.` }};
    }}
    function renderHealthSummary() {{
      const s = DATA.summary;
      const health = healthStatus();
      return `<div class="health">
        <div class="health-main">
          <div class="health-title">Latest Eval: ${{s.latestPassed}}/${{s.latestTotal}} passed ${{pill(health.label, health.ok)}}</div>
          <div class="health-copy">${{escapeHtml(health.copy)}}</div>
        </div>
        <div class="metric"><div class="metric-label">Tool accuracy</div><div class="metric-value">${{fmtPercent(toolAccuracy(latestEvalRows()))}}</div></div>
        <div class="metric"><div class="metric-label">Risk</div><div class="metric-value">${{health.ok === true ? "Low" : health.ok === false ? "High" : "n/a"}}</div></div>
        <div class="metric"><div class="metric-label">Failed tasks</div><div class="metric-value">${{latestEvalRows().filter(row => !row.passed).length}}</div></div>
      </div>`;
    }}
    function toolAccuracy(rows) {{
      if (!rows.length) return null;
      const accurate = rows.filter(row => row.checks && row.checks.expected_tools === true).length;
      return accurate / rows.length;
    }}
    function setPage(page) {{
      state.page = page;
      render();
    }}
    function renderNav() {{
      document.getElementById("workspace").textContent = DATA.workspace;
      document.getElementById("nav").innerHTML = navItems.map(([id, label]) =>
        `<button class="${{state.page === id ? "active" : ""}}" onclick="setPage('${{id}}')">${{label}}</button>`
      ).join("");
    }}
    function renderMetrics() {{
      const s = DATA.summary;
      const items = [
        ["Latest pass rate", fmtPercent(s.latestPassRate)],
        ["Latest tasks", `${{s.latestPassed}} / ${{s.latestTotal}}`],
        ["Average latency", fmtSeconds(s.latestAvgLatency)],
        ["Latest cost", fmtMoney(s.latestCost)],
        ["Transcripts", s.runCount],
        ["Eval rows", s.totalEvalRows],
        ["Stability runs", s.determinismCount],
        ["RAG chunks", (DATA.rag.chunks || []).length]
      ];
      return `<div class="metrics">${{items.map(([label, value]) =>
        `<div class="metric"><div class="metric-label">${{escapeHtml(label)}}</div><div class="metric-value">${{escapeHtml(value)}}</div></div>`
      ).join("")}}</div>`;
    }}
    function evalTable(rows, limit = rows.length) {{
      const view = rows.slice(-limit).reverse();
      if (!view.length) return `<div class="empty">No evaluation results found.</div>`;
      return `<table><thead><tr><th>Task</th><th>Passed</th><th>Category</th><th>Latency</th><th>Tokens</th><th>Cost</th><th>Tools</th></tr></thead><tbody>
        ${{view.map((row, index) => `<tr class="selectable" onclick="selectEval('${{escapeHtml(row._source)}}', '${{escapeHtml(row.task_id)}}')">
          <td><code>${{escapeHtml(row.task_id)}}</code></td>
          <td>${{pill(row.passed ? "yes" : "no", !!row.passed)}}</td>
          <td>${{escapeHtml(row.failure_category || "")}}</td>
          <td>${{fmtSeconds(row.latency_seconds)}}</td>
          <td>${{tokens(row)}}</td>
          <td>${{fmtMoney(row.cost_usd)}}</td>
          <td>${{escapeHtml(tools(row))}}</td>
        </tr>`).join("")}}
      </tbody></table>`;
    }}
    function filteredEvalRows() {{
      const base = state.evalFilter === "latest" ? latestEvalRows() : DATA.evalResults;
      const q = state.evalSearch.trim().toLowerCase();
      let rows = base.filter(row => {{
        if (state.evalFilter === "failed" && row.passed) return false;
        if (!q) return true;
        return [row.task_id, row.task, row.failure_category, tools(row), row.final]
          .some(value => String(value || "").toLowerCase().includes(q));
      }});
      const sorters = {{
        source: (a, b) => String(a._source || "").localeCompare(String(b._source || "")),
        latency_desc: (a, b) => Number(b.latency_seconds || 0) - Number(a.latency_seconds || 0),
        cost_desc: (a, b) => Number(b.cost_usd || 0) - Number(a.cost_usd || 0),
        tokens_desc: (a, b) => tokens(b) - tokens(a),
        task: (a, b) => String(a.task_id || "").localeCompare(String(b.task_id || ""))
      }};
      return rows.slice().sort(sorters[state.evalSort] || sorters.source);
    }}
    function renderEvalToolbar() {{
      return `<div class="toolbar">
        <input id="evalSearch" placeholder="Search task, category, tool, or answer" value="${{escapeHtml(state.evalSearch)}}" oninput="state.evalSearch=this.value; refreshEvalList();">
        <select id="evalFilter" onchange="state.evalFilter=this.value; refreshEvalList();">
          <option value="latest" ${{state.evalFilter === "latest" ? "selected" : ""}}>Latest eval</option>
          <option value="failed" ${{state.evalFilter === "failed" ? "selected" : ""}}>Failed only</option>
          <option value="all" ${{state.evalFilter === "all" ? "selected" : ""}}>All rows</option>
        </select>
        <select id="evalSort" onchange="state.evalSort=this.value; refreshEvalList();">
          <option value="source" ${{state.evalSort === "source" ? "selected" : ""}}>Source order</option>
          <option value="latency_desc" ${{state.evalSort === "latency_desc" ? "selected" : ""}}>Slowest first</option>
          <option value="cost_desc" ${{state.evalSort === "cost_desc" ? "selected" : ""}}>Costliest first</option>
          <option value="tokens_desc" ${{state.evalSort === "tokens_desc" ? "selected" : ""}}>Most tokens</option>
          <option value="task" ${{state.evalSort === "task" ? "selected" : ""}}>Task name</option>
        </select>
      </div>`;
    }}
    function refreshEvalList() {{
      const list = document.getElementById("evalList");
      const charts = document.getElementById("evalCharts");
      if (list) list.innerHTML = evalTable(filteredEvalRows(), 120);
      if (charts) charts.innerHTML = renderEvalCharts(filteredEvalRows());
    }}
    function renderEvalCharts(rows) {{
      const latest = rows.slice(0, 12);
      if (!latest.length) return `<div class="empty">No rows match the current filters.</div>`;
      return `<div class="grid-2">
        <section class="panel"><div class="panel-head"><h2>Latency by Task</h2></div><div class="panel-body">${{barChart(latest, "latency_seconds", "latency", value => fmtSeconds(value))}}</div></section>
        <section class="panel"><div class="panel-head"><h2>Cost by Task</h2></div><div class="panel-body">${{barChart(latest, "cost_usd", "cost", value => fmtMoney(value))}}</div></section>
        <section class="panel"><div class="panel-head"><h2>Tokens by Task</h2></div><div class="panel-body">${{tokenBarChart(latest)}}</div></section>
        <section class="panel"><div class="panel-head"><h2>Failure Categories</h2></div><div class="panel-body">${{categorySummary(rows)}}</div></section>
      </div>`;
    }}
    function barChart(rows, key, klass, formatter) {{
      const max = Math.max(...rows.map(row => Number(row[key] || 0)), 0.000001);
      return `<div class="chart-list">${{rows.map(row => {{
        const value = Number(row[key] || 0);
        const width = Math.max(2, (value / max) * 100);
        return `<div class="bar-row"><div class="bar-label">${{escapeHtml(row.task_id)}}</div><div class="bar-track"><div class="bar-fill ${{klass}}" style="width:${{width}}%"></div></div><div class="bar-value">${{formatter(value)}}</div></div>`;
      }}).join("")}}</div>`;
    }}
    function tokenBarChart(rows) {{
      const max = Math.max(...rows.map(row => tokens(row)), 1);
      return `<div class="chart-list">${{rows.map(row => {{
        const value = tokens(row);
        const width = Math.max(2, (value / max) * 100);
        return `<div class="bar-row"><div class="bar-label">${{escapeHtml(row.task_id)}}</div><div class="bar-track"><div class="bar-fill tokens" style="width:${{width}}%"></div></div><div class="bar-value">${{value}}</div></div>`;
      }}).join("")}}</div>`;
    }}
    function categorySummary(rows) {{
      const counts = {{}};
      for (const row of rows) counts[row.failure_category || "unknown"] = (counts[row.failure_category || "unknown"] || 0) + 1;
      return `<div class="summary-line">${{Object.entries(counts).map(([name, count]) => pill(`${{name}}: ${{count}}`, name === "success" ? true : false)).join("")}}</div>`;
    }}
    function runTable(rows, limit = rows.length) {{
      const view = rows.slice(-limit).reverse();
      if (!view.length) return `<div class="empty">No transcripts found.</div>`;
      return `<table><thead><tr><th>Run ID</th><th>Status</th><th>Tools</th><th>Tokens</th></tr></thead><tbody>
        ${{view.map(row => `<tr class="selectable" onclick="selectRun('${{escapeHtml(row.run_id)}}')">
          <td><code>${{escapeHtml(row.run_id)}}</code></td>
          <td>${{pill(row.status || "", row.status === "completed" ? true : row.status === "error" ? false : null)}}</td>
          <td>${{escapeHtml(tools(row))}}</td>
          <td>${{tokens(row)}}</td>
        </tr>`).join("")}}
      </tbody></table>`;
    }}
    function determinismTable(rows, limit = rows.length) {{
      const view = rows.slice(-limit).reverse();
      if (!view.length) return `<div class="empty">No determinism results found.</div>`;
      return `<table><thead><tr><th>Run</th><th>Status</th><th>Latency</th><th>Tools</th><th>Run ID</th></tr></thead><tbody>
        ${{view.map((row, index) => `<tr class="selectable" onclick="selectDet(${{DATA.determinismResults.indexOf(row)}})">
          <td>${{escapeHtml(row.run_number || "")}}</td>
          <td>${{pill(row.status || "", row.status === "completed" ? true : row.status === "error" ? false : null)}}</td>
          <td>${{fmtSeconds(row.latency_seconds)}}</td>
          <td>${{escapeHtml(tools(row))}}</td>
          <td><code>${{escapeHtml(row.run_id || "")}}</code></td>
        </tr>`).join("")}}
      </tbody></table>`;
    }}
    function renderOverview() {{
      return `${{renderHealthSummary()}}${{renderMetrics()}}
      <div class="grid-2">
        <section class="panel"><div class="panel-head"><h2>Latest Evaluation</h2><span class="muted">${{DATA.latestEvalRows.length}} tasks</span></div>${{evalTable(DATA.latestEvalRows, 12)}}</section>
        <section class="panel"><div class="panel-head"><h2>Recent Runs</h2><span class="muted">${{DATA.runs.length}} transcripts</span></div>${{runTable(DATA.runs, 10)}}</section>
      </div>
      <div style="margin-top:14px;">${{renderEvalCharts(DATA.latestEvalRows)}}</div>`;
    }}
    function selectEval(source, taskId) {{
      state.selectedEval = DATA.evalResults.find(row => row._source === source && row.task_id === taskId) || null;
      state.page = "evaluations";
      render();
    }}
    function renderEvalDetail(row) {{
      if (!row) return `<div class="panel"><div class="empty">Select an evaluation row to inspect details.</div></div>`;
      return `<section class="panel"><div class="panel-head"><h2>${{escapeHtml(row.task_id)}}</h2>${{pill(row.passed ? "passed" : "failed", !!row.passed)}}</div>
        <div class="panel-body">
          <p class="muted">${{escapeHtml(row.task)}}</p>
          <p class="summary-line">${{pill(row.failure_category || "unknown")}} ${{pill("status: " + (row.status || ""))}} ${{pill("tools: " + tools(row))}}</p>
          <h3>Checks</h3><pre>${{escapeHtml(JSON.stringify(row.checks || {{}}, null, 2))}}</pre>
          <h3>Cost Breakdown</h3><pre>${{escapeHtml(JSON.stringify(row.cost_breakdown_usd || {{}}, null, 2))}}</pre>
          <h3>Final Answer</h3><pre>${{escapeHtml(row.final || "")}}</pre>
          <h3>Run ID</h3><code>${{escapeHtml(row.run_id || "")}}</code>
        </div></section>`;
    }}
    function renderEvaluations() {{
      const rows = filteredEvalRows();
      return `${{renderHealthSummary()}}
      <section class="panel" style="margin-bottom:14px;"><div class="panel-head"><h2>Filters</h2></div><div class="panel-body">${{renderEvalToolbar()}}</div></section>
      <div id="evalCharts" style="margin-bottom:14px;">${{renderEvalCharts(rows)}}</div>
      <div class="grid-2">
        <section class="panel"><div class="panel-head"><h2>Evaluation Results</h2><span class="muted">${{rows.length}} rows</span></div><div id="evalList">${{evalTable(rows, 120)}}</div></section>
        ${{renderEvalDetail(state.selectedEval || DATA.latestEvalRows[DATA.latestEvalRows.length - 1])}}
      </div>`;
    }}
    function selectRun(runId) {{
      state.selectedRun = DATA.runs.find(row => row.run_id === runId) || null;
      state.page = "runs";
      render();
    }}
    function renderTimeline(run) {{
      if (!run) return "";
      return (run.messages || []).filter(msg => msg.role !== "system").map(msg => {{
        if (msg.role === "assistant") {{
          const calls = (msg.tool_calls || []).map(call => `${{call.name}} ${{JSON.stringify(call.arguments || {{}})}}`).join("\\n");
          return `<div class="step"><strong>LLM Step ${{escapeHtml(msg.step || "")}}</strong>
            ${{msg.thought ? `<p>${{escapeHtml(msg.thought)}}</p>` : ""}}
            ${{calls ? `<pre>${{escapeHtml(calls)}}</pre>` : ""}}
            ${{msg.final ? `<p><strong>Final</strong></p><pre>${{escapeHtml(msg.final)}}</pre>` : ""}}
          </div>`;
        }}
        if (msg.role === "tool") {{
          return `<div class="step tool"><strong>Tool Observation: ${{escapeHtml(msg.name || "")}}</strong>
            <pre>${{escapeHtml(JSON.stringify(msg.arguments || {{}}))}}</pre>
            <pre>${{escapeHtml(msg.content || "")}}</pre>
          </div>`;
        }}
        return `<div class="step"><strong>User</strong><pre>${{escapeHtml(msg.content || "")}}</pre></div>`;
      }}).join("");
    }}
    function renderRunDetail(run) {{
      if (!run) return `<div class="panel"><div class="empty">Select a transcript to inspect its execution trace.</div></div>`;
      return `<section class="panel"><div class="panel-head"><h2>${{escapeHtml(run.run_id)}}</h2>${{pill(run.status || "", run.status === "completed")}}</div>
        <div class="panel-body">
          <p class="muted">${{escapeHtml(run.task || "")}}</p>
          <p class="summary-line">${{pill("tools: " + tools(run))}} ${{pill("tokens: " + tokens(run))}}</p>
          <div class="timeline">${{renderTimeline(run)}}</div>
        </div></section>`;
    }}
    function renderRuns() {{
      return `<div class="grid-2">
        <section class="panel"><div class="panel-head"><h2>Transcripts</h2><span class="muted">${{DATA.runs.length}} runs</span></div>${{runTable(DATA.runs, 120)}}</section>
        ${{renderRunDetail(state.selectedRun || DATA.runs[DATA.runs.length - 1])}}
      </div>`;
    }}
    function selectDet(index) {{
      state.selectedDet = DATA.determinismResults[index] || null;
      state.page = "determinism";
      render();
    }}
    function renderDeterminismDetail(row) {{
      if (!row) return `<div class="panel"><div class="empty">Select a determinism run to inspect details.</div></div>`;
      return `<section class="panel"><div class="panel-head"><h2>Stability Run ${{escapeHtml(row.run_number || "")}}</h2>${{pill(row.status || "", row.status === "completed")}}</div>
        <div class="panel-body">
          <p class="summary-line">${{pill("latency: " + fmtSeconds(row.latency_seconds))}} ${{pill("tools: " + tools(row))}}</p>
          <h3>Usage</h3><pre>${{escapeHtml(JSON.stringify(row.usage || {{}}, null, 2))}}</pre>
          <h3>Final Answer</h3><pre>${{escapeHtml(row.final || "")}}</pre>
          <h3>Error</h3><pre>${{escapeHtml(row.error || "")}}</pre>
        </div></section>`;
    }}
    function renderDeterminism() {{
      return `<div class="grid-2">
        <section class="panel"><div class="panel-head"><h2>Determinism Runs</h2><span class="muted">${{DATA.determinismResults.length}} rows</span></div>${{determinismTable(DATA.determinismResults, 120)}}</section>
        ${{renderDeterminismDetail(state.selectedDet || DATA.determinismResults[DATA.determinismResults.length - 1])}}
      </div>`;
    }}
    const stopwords = new Set(["a","an","and","are","do","for","how","i","in","is","it","of","on","or","the","to","with"]);
    function tokenize(value) {{
      return String(value || "").toLowerCase().match(/[a-z0-9_]+/g)?.filter(item => !stopwords.has(item)) || [];
    }}
    function ragSearch(question) {{
      const terms = tokenize(question);
      if (!terms.length) return [];
      return (DATA.rag.chunks || []).map(chunk => {{
        const textTerms = tokenize(chunk.text);
        let score = 0;
        for (const term of terms) if (textTerms.includes(term)) score += 1;
        if (chunk.text.toLowerCase().includes("eval --tasks")) score += 3;
        if (chunk.source.endsWith(".py")) score -= 1;
        return {{...chunk, score}};
      }}).filter(item => item.score > 0).sort((a, b) => b.score - a.score).slice(0, 5);
    }}
    function runRag() {{
      const question = document.getElementById("ragQuestion").value;
      const rows = ragSearch(question);
      document.getElementById("ragResults").innerHTML = rows.length ? rows.map(row =>
        `<div class="step"><strong>${{escapeHtml(row.source)}}#chunk-${{row.chunk_id}}</strong> ${{pill("score: " + row.score)}}
          <pre>${{escapeHtml(row.text.slice(0, 1300))}}</pre></div>`
      ).join("") : `<div class="empty">No matching chunks found.</div>`;
    }}
    function renderRag() {{
      return `<section class="panel"><div class="panel-head"><h2>Local RAG Search</h2><span class="muted">${{(DATA.rag.chunks || []).length}} chunks</span></div>
        <div class="panel-body">
          <input class="search" id="ragQuestion" value="How do I run the DeepSeek evaluation benchmark?">
          <div style="margin:10px 0;"><button class="small-button" onclick="runRag()">Search</button></div>
          <div class="timeline" id="ragResults"></div>
        </div></section>`;
    }}
    function render() {{
      renderNav();
      const titles = {{
        overview: ["Overview", "Latest benchmark health, cost, latency, transcripts, and stability data."],
        evaluations: ["Evaluations", "Inspect benchmark rows, checks, cost, tool usage, and final answers."],
        runs: ["Runs", "Replay saved transcript steps, tool calls, observations, and final answers."],
        determinism: ["Determinism", "Review repeated-run stability and tool sequence behavior."],
        rag: ["RAG Search", "Search indexed project files and inspect cited chunks."]
      }};
      document.getElementById("title").textContent = titles[state.page][0];
      document.getElementById("subtitle").textContent = titles[state.page][1];
      document.getElementById("headerActions").innerHTML = "";
      const page = document.getElementById("page");
      page.innerHTML = {{
        overview: renderOverview,
        evaluations: renderEvaluations,
        runs: renderRuns,
        determinism: renderDeterminism,
        rag: renderRag
      }}[state.page]();
      if (state.page === "rag") runRag();
    }}
    render();
  </script>
</body>
</html>
"""
