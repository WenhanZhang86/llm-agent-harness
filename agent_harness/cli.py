from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .compare import parse_provider_specs, run_compare
from .core import AgentConfig, AgentHarness
from .dashboard import build_dashboard
from .determinism import run_determinism
from .eval import run_eval
from .llm import MockLLM, SubprocessLLM
from .model_registry import provider_specs_from_models
from .rag import build_rag_index, query_rag_index
from .rag_eval import run_rag_eval
from .runtime import AgentRuntime, RuntimePolicy, ToolContext, build_runtime_tools, register_mcp_tools_from_config
from .trace import (
    compare_run_summaries,
    load_run_events,
    load_run_summary,
    load_transcript,
    replay_transcript,
    write_trace_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a task through the LLM Agent Harness.")
    add_run_arguments(parser)
    return parser


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an evaluation task set.")
    parser.add_argument("--tasks", default="evals/tasks.jsonl", help="JSONL eval task file.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to tools.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent loop steps.")
    parser.add_argument("--llm-timeout", type=int, default=240, help="LLM subprocess timeout in seconds.")
    parser.add_argument("--llm-cmd", help="Command for a JSON-speaking external LLM adapter.")
    parser.add_argument("--results-dir", default="evals/results", help="Directory for JSONL eval results.")
    parser.add_argument("--reports-dir", default="evals/reports", help="Directory for Markdown eval reports.")
    parser.add_argument("--pricing", default="evals/pricing.json", help="JSON pricing file for cost estimates.")
    parser.add_argument("--judge-cmd", help="Optional JSON-speaking LLM adapter used to grade rubric tasks.")
    parser.add_argument("--judge-timeout", type=int, default=240, help="Judge subprocess timeout in seconds.")
    parser.add_argument("--retries", type=int, default=0, help="Retry transient API, rate-limit, or timeout failures.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each task multiple times for regression sampling.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    add_permission_arguments(parser)
    return parser


def build_run_agent_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tool-using agent through the Agent Runtime.")
    parser.add_argument("task", help="Task for the runtime agent.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to tools.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum runtime loop steps.")
    parser.add_argument("--timeout-seconds", type=int, default=240, help="Overall runtime timeout.")
    parser.add_argument("--max-cost-usd", type=float, help="Stop when tracked runtime cost exceeds this value.")
    parser.add_argument("--llm-cmd", help="Command for a JSON-speaking external LLM adapter.")
    parser.add_argument("--allow-file-write", action="store_true", help="Enable write_file in the runtime.")
    parser.add_argument("--allow-shell-exec", action="store_true", help="Enable run_shell in the runtime.")
    parser.add_argument("--allow-network", action="store_true", help="Allow tools that use network access. No network tools are enabled by default.")
    parser.add_argument("--allow-code-exec", action="store_true", help="Allow code execution tools if installed. Disabled by default.")
    parser.add_argument("--no-rag-context", action="store_true", help="Do not inject local RAG context even if rag/index.json exists.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    return parser


def build_list_tools_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List registered Agent Runtime tools.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--allow-file-write", action="store_true", help="Show tools with write permission enabled.")
    parser.add_argument("--allow-shell-exec", action="store_true", help="Show tools with shell permission enabled.")
    parser.add_argument("--allow-network", action="store_true", help="Show tools with network permission enabled.")
    parser.add_argument("--allow-code-exec", action="store_true", help="Show tools with code execution permission enabled.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    return parser


def build_run_tool_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Agent Runtime tool with JSON arguments.")
    parser.add_argument("tool", help="Tool name.")
    parser.add_argument("args_json", help="JSON object passed to the tool.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--allow-file-write", action="store_true", help="Enable write_files permission.")
    parser.add_argument("--allow-shell-exec", action="store_true", help="Enable shell_exec permission.")
    parser.add_argument("--allow-network", action="store_true", help="Enable network permission.")
    parser.add_argument("--allow-code-exec", action="store_true", help="Enable code_exec permission.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    return parser


def build_compare_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the same eval tasks across multiple providers.")
    parser.add_argument("--tasks", default="evals/tasks.jsonl", help="JSONL eval task file.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to tools.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent loop steps.")
    parser.add_argument("--llm-timeout", type=int, default=300, help="LLM subprocess timeout in seconds.")
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help='Provider spec in name=command format, for example deepseek="python3 -m providers.deepseek".',
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help="Model names from configs/models.json. Example: --models deepseek_pro openai_gpt55.",
    )
    parser.add_argument("--model-config", default="configs/models.json", help="Model registry JSON file.")
    parser.add_argument("--results-dir", default="evals/results", help="Directory for JSONL eval results.")
    parser.add_argument("--reports-dir", default="evals/reports", help="Directory for Markdown eval reports.")
    parser.add_argument(
        "--comparisons-dir",
        default="evals/comparisons",
        help="Directory for comparison JSON and Markdown reports.",
    )
    parser.add_argument("--pricing", default="evals/pricing.json", help="JSON pricing file for cost estimates.")
    parser.add_argument("--judge-cmd", help="Optional JSON-speaking LLM adapter used to grade rubric tasks.")
    parser.add_argument("--judge-timeout", type=int, default=240, help="Judge subprocess timeout in seconds.")
    parser.add_argument("--retries", type=int, default=0, help="Retry transient API, rate-limit, or timeout failures.")
    parser.add_argument("--repeat", type=int, default=1, help="Run each task multiple times for regression sampling.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    add_permission_arguments(parser)
    return parser


def build_trace_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a run transcript as a Markdown trace.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--run-id", help="Run ID from the runs directory.")
    parser.add_argument("--transcript", help="Path to a transcript JSON file.")
    parser.add_argument("--output", help="Output Markdown path. Defaults to runs/<run_id>.trace.md.")
    return parser


def build_replay_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a saved transcript without calling a model.")
    parser.add_argument("run_id_arg", nargs="?", help="Run ID from the runs directory.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--run-id", help="Run ID from the runs directory.")
    parser.add_argument("--transcript", help="Path to a transcript JSON file.")
    return parser


def build_compare_runs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two saved runtime runs.")
    parser.add_argument("run1", help="First run ID.")
    parser.add_argument("run2", help="Second run ID.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    return parser


def build_show_context_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show retrieved context for a runtime run.")
    parser.add_argument("run_id", help="Run ID from the runs directory.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    return parser


def build_show_memory_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show short-term memory for a runtime run.")
    parser.add_argument("run_id", help="Run ID from the runs directory.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    return parser


def build_serve_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the local FastAPI server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to the server.")
    return parser


def build_determinism_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the same task multiple times and measure consistency.")
    parser.add_argument("--task", required=True, help="Task to run repeatedly.")
    parser.add_argument("--runs", type=int, default=20, help="Number of repeated runs.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to tools.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent loop steps.")
    parser.add_argument("--llm-timeout", type=int, default=240, help="LLM subprocess timeout in seconds.")
    parser.add_argument("--llm-cmd", help="Command for a JSON-speaking external LLM adapter.")
    parser.add_argument("--allow-shell", action="store_true", help="Enable the run_shell tool.")
    parser.add_argument("--output-dir", default="evals/determinism", help="Directory for determinism reports.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    add_permission_arguments(parser)
    return parser


def build_dashboard_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a static HTML dashboard.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--output", default="dashboard/index.html", help="Output HTML path.")
    return parser


def build_rag_index_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a local lexical RAG index.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to index.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--output", default="rag/index.json", help="Output index JSON path.")
    parser.add_argument("--chunk-chars", type=int, default=1600, help="Approximate chunk size.")
    return parser


def build_rag_query_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query a local RAG index with citations.")
    parser.add_argument("question", help="Question to answer from indexed files.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--index", default="rag/index.json", help="RAG index JSON path.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to return.")
    return parser


def build_rag_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate local RAG retrieval and citation quality.")
    parser.add_argument("--tasks", default="evals/rag_tasks.jsonl", help="JSONL RAG eval task file.")
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--index", default="rag/index.json", help="RAG index JSON path.")
    parser.add_argument("inputs", nargs="*", default=["README.md", "evals", "agent_harness"], help="Files or directories to index if needed.")
    parser.add_argument("--inputs", nargs="+", dest="input_paths", help="Files or directories to index if needed.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve per query.")
    parser.add_argument("--results-dir", default="evals/rag_results", help="Directory for RAG eval JSONL results.")
    parser.add_argument("--reports-dir", default="evals/rag_reports", help="Directory for RAG eval reports.")
    return parser


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task", help="Task for the agent.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to tools.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent loop steps.")
    parser.add_argument("--llm", choices=["mock"], default="mock", help="Built-in LLM adapter.")
    parser.add_argument("--llm-cmd", help="Command for a JSON-speaking external LLM adapter.")
    parser.add_argument("--allow-shell", action="store_true", help="Enable the run_shell tool.")
    parser.add_argument("--llm-timeout", type=int, default=240, help="LLM subprocess timeout in seconds.")
    parser.add_argument("--mcp-config", help="JSON config file that defines MCP servers to expose as runtime tools.")
    add_permission_arguments(parser)


def add_permission_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--read-only", action="store_true", help="Disable write_file.")
    parser.add_argument(
        "--shell-allow",
        action="append",
        default=[],
        help="Allow one shell executable. Can be repeated, for example --shell-allow pwd --shell-allow ls.",
    )
    parser.add_argument(
        "--max-write-bytes",
        type=int,
        default=1_000_000,
        help="Maximum bytes write_file can write in one call.",
    )
    parser.add_argument("--allow-network", action="store_true", help="Enable network tools such as MCP API tools.")
    parser.add_argument("--allow-code-exec", action="store_true", help="Enable code execution tools if installed.")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        args = build_eval_parser().parse_args(argv[1:])
        return run_eval_command(args)
    if argv and argv[0] == "run-agent":
        args = build_run_agent_parser().parse_args(argv[1:])
        return run_agent_command(args)
    if argv and argv[0] == "list-tools":
        args = build_list_tools_parser().parse_args(argv[1:])
        return list_tools_command(args)
    if argv and argv[0] == "run-tool":
        args = build_run_tool_parser().parse_args(argv[1:])
        return run_tool_command(args)
    if argv and argv[0] == "compare":
        args = build_compare_parser().parse_args(argv[1:])
        return run_compare_command(args)
    if argv and argv[0] == "trace":
        args = build_trace_parser().parse_args(argv[1:])
        return run_trace_command(args)
    if argv and argv[0] == "replay":
        args = build_replay_parser().parse_args(argv[1:])
        return run_replay_command(args)
    if argv and argv[0] == "compare-runs":
        args = build_compare_runs_parser().parse_args(argv[1:])
        return run_compare_runs_command(args)
    if argv and argv[0] == "show-context":
        args = build_show_context_parser().parse_args(argv[1:])
        return run_show_context_command(args)
    if argv and argv[0] == "show-memory":
        args = build_show_memory_parser().parse_args(argv[1:])
        return run_show_memory_command(args)
    if argv and argv[0] == "serve":
        args = build_serve_parser().parse_args(argv[1:])
        return run_serve_command(args)
    if argv and argv[0] == "determinism":
        args = build_determinism_parser().parse_args(argv[1:])
        return run_determinism_command(args)
    if argv and argv[0] == "dashboard":
        args = build_dashboard_parser().parse_args(argv[1:])
        return run_dashboard_command(args)
    if argv and argv[0] == "rag-index":
        args = build_rag_index_parser().parse_args(argv[1:])
        return run_rag_index_command(args)
    if argv and argv[0] == "rag-query":
        args = build_rag_query_parser().parse_args(argv[1:])
        return run_rag_query_command(args)
    if argv and argv[0] == "rag-eval":
        args = build_rag_eval_parser().parse_args(argv[1:])
        return run_rag_eval_command(args)
    if argv and argv[0] == "run":
        argv = argv[1:]

    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    llm = SubprocessLLM(args.llm_cmd, timeout=args.llm_timeout) if args.llm_cmd else MockLLM()
    policy = RuntimePolicy(
        max_steps=args.max_steps,
        timeout_seconds=args.llm_timeout,
        allow_file_write=not args.read_only,
        allow_shell_exec=args.allow_shell,
        allow_network=args.allow_network,
        allow_code_exec=args.allow_code_exec,
        read_only=args.read_only,
        shell_enabled=args.allow_shell,
        shell_allowlist=list(args.shell_allow) if args.shell_allow else None,
        max_write_bytes=args.max_write_bytes,
    )
    tools = runtime_tools_from_args(workspace, policy, args)
    harness = AgentHarness(
        llm=llm,
        config=AgentConfig(workspace=workspace, max_steps=args.max_steps),
        tools=tools,
        policy=policy,
    )
    run = harness.run(args.task)

    print(f"run_id: {run.run_id}")
    print(f"status: {run.status}")
    print()
    print(run.final or "")
    return 0 if run.status in {"completed", "stopped"} else 1


def run_eval_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    results_path, report_path, results = run_eval(
        tasks_path=Path(args.tasks).resolve(),
        workspace=workspace,
        llm_cmd=args.llm_cmd,
        max_steps=args.max_steps,
        llm_timeout=args.llm_timeout,
        results_dir=(workspace / args.results_dir).resolve(),
        reports_dir=(workspace / args.reports_dir).resolve(),
        pricing_path=(workspace / args.pricing).resolve(),
        policy=build_permission_policy(args),
        mcp_config=resolve_optional_workspace_path(workspace, getattr(args, "mcp_config", None)),
        judge_cmd=args.judge_cmd,
        judge_timeout=args.judge_timeout,
        retries=args.retries,
        repeat=args.repeat,
    )
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print(f"tasks: {total}")
    print(f"passed: {passed}")
    print(f"failed: {total - passed}")
    print(f"results: {results_path}")
    print(f"report: {report_path}")
    return 0 if passed == total else 1


def run_agent_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    llm = SubprocessLLM(args.llm_cmd, timeout=args.timeout_seconds) if args.llm_cmd else MockLLM()
    policy = RuntimePolicy(
        max_steps=args.max_steps,
        timeout_seconds=args.timeout_seconds,
        max_cost_usd=args.max_cost_usd,
        allow_file_write=args.allow_file_write,
        allow_shell_exec=args.allow_shell_exec,
        allow_network=args.allow_network,
        allow_code_exec=args.allow_code_exec,
    )
    tools = runtime_tools_from_args(workspace, policy, args)
    runtime = AgentRuntime(
        llm=llm,
        workspace=workspace,
        policy=policy,
        tools=tools,
        use_rag_context=not args.no_rag_context,
    )
    result = runtime.run(args.task)
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"trace: {result.trace_path}")
    print()
    print(result.final or "")
    return 0 if result.status in {"completed", "stopped"} else 1


def list_tools_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    policy = runtime_policy_from_args(args)
    registry = runtime_tools_from_args(workspace, policy, args)
    print(json_pretty({"tools": registry.list_tools()}))
    return 0


def run_tool_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    policy = runtime_policy_from_args(args)
    registry = runtime_tools_from_args(workspace, policy, args)
    try:
        tool_args = json_loads_object(args.args_json)
    except ValueError as exc:
        print(json_pretty({"ok": False, "error": str(exc), "metadata": {"error_type": "invalid_json"}}))
        return 1
    output = registry.execute(args.tool, tool_args, ToolContext(workspace=workspace, policy=policy))
    print(json_pretty(output.to_dict()))
    return 0 if output.ok else 1


def run_compare_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    providers = parse_provider_specs(args.provider) if args.provider else []
    if args.models:
        providers.extend(
            provider_specs_from_models((workspace / args.model_config).resolve(), args.models)
        )
    if not providers:
        raise SystemExit("At least one --provider or --models entry is required.")
    comparison_json, comparison_report, provider_blocks = run_compare(
        tasks_path=Path(args.tasks).resolve(),
        workspace=workspace,
        providers=providers,
        max_steps=args.max_steps,
        llm_timeout=args.llm_timeout,
        results_dir=(workspace / args.results_dir).resolve(),
        reports_dir=(workspace / args.reports_dir).resolve(),
        comparisons_dir=(workspace / args.comparisons_dir).resolve(),
        pricing_path=(workspace / args.pricing).resolve(),
        policy=build_permission_policy(args),
        mcp_config=resolve_optional_workspace_path(workspace, getattr(args, "mcp_config", None)),
        judge_cmd=args.judge_cmd,
        judge_timeout=args.judge_timeout,
        retries=args.retries,
        repeat=args.repeat,
    )
    print(f"providers: {len(provider_blocks)}")
    for block in provider_blocks:
        summary = block["summary"]
        print(
            f"{block['provider']}: passed={summary['passed']}/{summary['total']} "
            f"success_rate={format_percent(summary['success_rate'])} "
            f"tool_accuracy={format_percent(summary['tool_accuracy'])} "
            f"latency={summary['average_latency_seconds']:.2f}s "
            f"cost={format_money(summary['total_cost_usd'])}"
        )
    print(f"comparison: {comparison_json}")
    print(f"report: {comparison_report}")
    failed = sum(block["summary"]["failed"] for block in provider_blocks)
    return 0 if failed == 0 else 1


def run_trace_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    output = Path(args.output).resolve() if args.output else None
    transcript = Path(args.transcript).resolve() if args.transcript else None
    output_path = write_trace_markdown(
        workspace=workspace,
        run_id=args.run_id,
        transcript=transcript,
        output=output,
    )
    print(f"trace: {output_path}")
    return 0


def run_replay_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    transcript = Path(args.transcript).resolve() if args.transcript else None
    run_id = args.run_id or args.run_id_arg
    data = load_transcript(workspace=workspace, run_id=run_id, transcript=transcript)
    events = load_run_events(workspace=workspace, run_id=run_id, transcript=transcript)
    print(replay_transcript(data, events))
    return 0


def run_compare_runs_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    left = load_run_summary(workspace=workspace, run_id=args.run1)
    right = load_run_summary(workspace=workspace, run_id=args.run2)
    print(compare_run_summaries(left, right))
    return 0


def run_show_context_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    summary = load_run_summary(workspace=workspace, run_id=args.run_id)
    items = summary.get("context_items") or []
    print(f"run_id: {args.run_id}")
    print(f"retrieved_documents: {len(items)}")
    print()
    for index, item in enumerate(items, start=1):
        print(f"{index}. {item.get('title') or item.get('id')}")
        print(f"   id: {item.get('id')}")
        print(f"   source: {item.get('source')}")
        print(f"   tags: {', '.join(item.get('tags') or [])}")
        print(f"   score: {item.get('score')}")
        print(f"   reason: {item.get('reason')}")
        print("   preview:")
        print(indent_text(str(item.get("preview") or ""), "     ", 500))
        print()
    return 0


def run_show_memory_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    summary = load_run_summary(workspace=workspace, run_id=args.run_id)
    memory = summary.get("memory") or {}
    entries = memory.get("entries") or []
    print(f"run_id: {args.run_id}")
    print(f"memory_size: {memory.get('size', len(entries))}")
    print(f"max_entries: {memory.get('max_entries', 'n/a')}")
    print()
    print("summary:")
    print(indent_text(str(memory.get("summary") or ""), "  ", 1000) or "  n/a")
    print()
    print("entries:")
    for entry in entries:
        print(f"- #{entry.get('entry_id')} {entry.get('kind')}")
        print(indent_text(str(entry.get("content") or ""), "  ", 500))
    return 0


def run_serve_command(args: argparse.Namespace) -> int:
    import uvicorn

    from .server.app import create_app

    workspace = Path(args.workspace).resolve()
    app = create_app(workspace=workspace)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def run_determinism_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    results_path, report_path, results = run_determinism(
        task=args.task,
        workspace=workspace,
        runs=args.runs,
        llm_cmd=args.llm_cmd,
        max_steps=args.max_steps,
        llm_timeout=args.llm_timeout,
        allow_shell=args.allow_shell,
        output_dir=(workspace / args.output_dir).resolve(),
        policy=build_permission_policy(args),
        mcp_config=resolve_optional_workspace_path(workspace, getattr(args, "mcp_config", None)),
    )
    completed = sum(1 for result in results if result.status == "completed")
    total = len(results)
    print(f"runs: {total}")
    print(f"completed: {completed}")
    print(f"results: {results_path}")
    print(f"report: {report_path}")
    return 0 if completed == total else 1


def run_dashboard_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    output = (workspace / args.output).resolve()
    output_path = build_dashboard(workspace, output=output)
    print(f"dashboard: {output_path}")
    return 0


def run_rag_index_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    output = (workspace / args.output).resolve()
    output_path = build_rag_index(
        workspace=workspace,
        inputs=args.paths,
        output=output,
        chunk_chars=args.chunk_chars,
    )
    print(f"index: {output_path}")
    return 0


def run_rag_query_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    index_path = (workspace / args.index).resolve()
    print(query_rag_index(index_path=index_path, question=args.question, top_k=args.top_k))
    return 0


def run_rag_eval_command(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).resolve()
    inputs = args.input_paths if args.input_paths is not None else args.inputs
    results_path, report_path, results = run_rag_eval(
        tasks_path=(workspace / args.tasks).resolve(),
        workspace=workspace,
        index_path=(workspace / args.index).resolve(),
        inputs=inputs,
        top_k=args.top_k,
        results_dir=(workspace / args.results_dir).resolve(),
        reports_dir=(workspace / args.reports_dir).resolve(),
    )
    passed = sum(1 for result in results if result.get("passed"))
    total = len(results)
    print(f"rag tasks: {total}")
    print(f"passed: {passed}")
    print(f"failed: {total - passed}")
    print(f"results: {results_path}")
    print(f"report: {report_path}")
    return 0 if passed == total else 1


def build_permission_policy(args: argparse.Namespace) -> RuntimePolicy:
    shell_allowlist = list(args.shell_allow) if getattr(args, "shell_allow", None) else None
    return RuntimePolicy(
        read_only=bool(getattr(args, "read_only", False)),
        shell_allowlist=shell_allowlist,
        max_write_bytes=int(getattr(args, "max_write_bytes", 1_000_000)),
        allow_network=bool(getattr(args, "allow_network", False)),
        allow_code_exec=bool(getattr(args, "allow_code_exec", False)),
    )


def runtime_policy_from_args(args: argparse.Namespace) -> RuntimePolicy:
    return RuntimePolicy(
        max_steps=int(getattr(args, "max_steps", 8)),
        timeout_seconds=int(getattr(args, "timeout_seconds", 240)),
        max_cost_usd=getattr(args, "max_cost_usd", None),
        allow_file_write=bool(getattr(args, "allow_file_write", False)),
        allow_shell_exec=bool(getattr(args, "allow_shell_exec", False)),
        allow_network=bool(getattr(args, "allow_network", False)),
        allow_code_exec=bool(getattr(args, "allow_code_exec", False)),
        read_only=bool(getattr(args, "read_only", False)),
        shell_allowlist=list(getattr(args, "shell_allow", []) or []) or None,
        max_write_bytes=int(getattr(args, "max_write_bytes", 1_000_000)),
    )


def runtime_tools_from_args(workspace: Path, policy: RuntimePolicy, args: argparse.Namespace):
    registry = build_runtime_tools(workspace, policy)
    config = getattr(args, "mcp_config", None)
    if config:
        register_mcp_tools_from_config(
            registry,
            (workspace / config).resolve() if not Path(config).is_absolute() else Path(config).resolve(),
            workspace=workspace,
        )
    return registry


def resolve_optional_workspace_path(workspace: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (workspace / path).resolve()


def json_loads_object(value: str) -> dict[str, object]:
    data = json.loads(value)
    if not isinstance(data, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return data


def json_pretty(value: object) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.6f}"


def indent_text(value: str, prefix: str, limit: int) -> str:
    text = value[:limit]
    if len(value) > limit:
        text += "\n... truncated ..."
    return "\n".join(prefix + line for line in text.splitlines())


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
