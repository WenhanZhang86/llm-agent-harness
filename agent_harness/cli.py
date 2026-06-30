from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .core import AgentConfig, AgentHarness
from .dashboard import build_dashboard
from .determinism import run_determinism
from .eval import run_eval
from .llm import MockLLM, SubprocessLLM
from .rag import build_rag_index, query_rag_index
from .tools import PermissionPolicy, build_default_tools
from .trace import load_transcript, replay_transcript, write_trace_markdown


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
    parser.add_argument("--workspace", default=".", help="Workspace root.")
    parser.add_argument("--run-id", help="Run ID from the runs directory.")
    parser.add_argument("--transcript", help="Path to a transcript JSON file.")
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


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("task", help="Task for the agent.")
    parser.add_argument("--workspace", default=".", help="Workspace root exposed to tools.")
    parser.add_argument("--max-steps", type=int, default=8, help="Maximum agent loop steps.")
    parser.add_argument("--llm", choices=["mock"], default="mock", help="Built-in LLM adapter.")
    parser.add_argument("--llm-cmd", help="Command for a JSON-speaking external LLM adapter.")
    parser.add_argument("--allow-shell", action="store_true", help="Enable the run_shell tool.")
    parser.add_argument("--llm-timeout", type=int, default=240, help="LLM subprocess timeout in seconds.")
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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "eval":
        args = build_eval_parser().parse_args(argv[1:])
        return run_eval_command(args)
    if argv and argv[0] == "trace":
        args = build_trace_parser().parse_args(argv[1:])
        return run_trace_command(args)
    if argv and argv[0] == "replay":
        args = build_replay_parser().parse_args(argv[1:])
        return run_replay_command(args)
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
    if argv and argv[0] == "run":
        argv = argv[1:]

    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    llm = SubprocessLLM(args.llm_cmd, timeout=args.llm_timeout) if args.llm_cmd else MockLLM()
    tools = build_default_tools(
        workspace,
        allow_shell=args.allow_shell,
        policy=build_permission_policy(args),
    )
    harness = AgentHarness(
        llm=llm,
        tools=tools,
        config=AgentConfig(workspace=workspace, max_steps=args.max_steps),
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
    )
    passed = sum(1 for result in results if result.passed)
    total = len(results)
    print(f"tasks: {total}")
    print(f"passed: {passed}")
    print(f"failed: {total - passed}")
    print(f"results: {results_path}")
    print(f"report: {report_path}")
    return 0 if passed == total else 1


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
    data = load_transcript(workspace=workspace, run_id=args.run_id, transcript=transcript)
    print(replay_transcript(data))
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


def build_permission_policy(args: argparse.Namespace) -> PermissionPolicy:
    shell_allowlist = list(args.shell_allow) if getattr(args, "shell_allow", None) else None
    return PermissionPolicy(
        read_only=bool(getattr(args, "read_only", False)),
        shell_allowlist=shell_allowlist,
        max_write_bytes=int(getattr(args, "max_write_bytes", 1_000_000)),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
