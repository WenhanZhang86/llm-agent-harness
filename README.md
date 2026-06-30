# LLM Agent Harness

A lightweight, locally runnable, model-agnostic harness for LLM agents.

This README assumes the project folder is on your Desktop and is named:

```text
~/Desktop/llm-agent-harness
```

If your folder is somewhere else, replace `~/Desktop/llm-agent-harness` with your actual path.

## Current Feature Status

This project is no longer just a toy CLI loop. The current version includes a working baseline for building, running, inspecting, and evaluating tool-using LLM agents.

Implemented:

- Model-agnostic agent loop with tool calling and observation feedback
- Subprocess provider protocol for swapping model adapters without changing harness code
- Provider adapters for Echo, OpenAI, OpenAI-compatible APIs, Anthropic, Gemini, Ollama, Azure OpenAI, DeepSeek, and Groq
- Built-in workspace tools: `list_dir`, `read_file`, `write_file`, and `run_shell`
- Per-run JSON transcripts under `runs/`
- Evaluation benchmark runner with JSONL task files
- Metrics for success rate, tool accuracy, latency, token usage, estimated cost, and cost breakdown
- Failure classification for `planning_error`, `tool_error`, `timeout`, `hallucination`, `permission_denied`, `api_error`, `rate_limit`, and `failed`
- Flexible eval assertions with `expect_contains`, `expect_any_contains`, `expect_tools`, `forbid_tools`, and `expect_observation_contains`
- Determinism runner for repeated-task consistency checks
- Markdown trace visualization with Mermaid flowcharts
- Transcript replay without calling a model
- Permission controls including read-only mode, shell allowlists, dangerous command blocking, and write-size limits
- Interactive static HTML dashboard with health summary, filters, sorting, charts, run details, and RAG search
- Local lexical RAG with cited source chunks

Not implemented yet:

- Browser UI for starting runs interactively
- Embedding-backed semantic RAG
- PDF ingestion
- Parallel eval execution
- Production-grade sandboxing beyond workspace path checks and command policies
- Human approval workflow before file writes or shell commands

## Latest Verified Baseline

The current DeepSeek V4 Flash benchmark baseline passed all default tasks:

```text
Total tasks: 12
Passed: 12
Failed: 0
Success rate: 100.0%
Tool accuracy: 100.0%
Average latency: 4.41s
Total tokens: 40952
Estimated cost: $0.006220
Failure categories: success=12
```

This baseline proves that the harness can reliably exercise basic agent behaviors: directory listing, provider inspection, README summarization, config reading, file writing, nested file writing, path escape denial, shell denial, allowed shell execution, multi-step read/write, avoiding unnecessary shell usage, and missing-file error handling.

## What It Provides

- Agent loop: send context, receive model actions, execute tools, and feed observations back into the run
- Tool registry: built-in `list_dir`, `read_file`, `write_file`, and `run_shell`
- Provider adapters: connect OpenAI, OpenAI-compatible APIs, Anthropic, Gemini, Ollama, Azure OpenAI, DeepSeek, Groq, or the local echo test provider
- Run transcripts: each task writes a JSON transcript under `runs/`
- Evaluation reports: success rate, tool accuracy, latency, token usage, cost, failure categories, and cost breakdown
- Trace and replay tools: render saved runs as Markdown flow traces or replay the recorded execution
- Determinism checks: run the same task repeatedly and measure consistency
- Permission policies: read-only mode, shell command allowlists, blocked dangerous commands, and write-size limits
- Interactive dashboard: browse evals, runs, determinism results, costs, latency, transcripts, and local RAG citations
- Local RAG: index workspace documents and retrieve cited chunks for document QA

## Quick Start

```bash
cd ~/Desktop/llm-agent-harness
python3 -m agent_harness.cli "List files in the workspace" --llm mock
```

You should see:

```text
status: completed
```

## Command Reference

Run one agent task:

```bash
python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.deepseek"
```

Run the default benchmark:

```bash
python3 -m agent_harness.cli eval --tasks evals/tasks.jsonl --llm-timeout 300 --llm-cmd "python3 -m providers.deepseek"
```

Run a no-API smoke test:

```bash
python3 -m agent_harness.cli eval --tasks evals/echo-smoke.jsonl --llm-cmd "python3 -m providers.echo"
```

Measure determinism:

```bash
python3 -m agent_harness.cli determinism \
  --task "Use the read_file tool to read README.md, then summarize this project in 5 bullet points." \
  --runs 20 \
  --llm-timeout 300 \
  --llm-cmd "python3 -m providers.deepseek"
```

Render a trace:

```bash
python3 -m agent_harness.cli trace --run-id RUN_ID
```

Replay a transcript:

```bash
python3 -m agent_harness.cli replay --run-id RUN_ID
```

Generate the dashboard:

```bash
python3 -m agent_harness.cli dashboard
open dashboard/index.html
```

Build and query the local RAG index:

```bash
python3 -m agent_harness.cli rag-index README.md evals agent_harness
python3 -m agent_harness.cli rag-query "How do I run the DeepSeek evaluation benchmark?"
```

## Project Structure

```text
agent_harness/
  cli.py       # Command-line entry point
  core.py      # Agent loop and transcript writing
  dashboard.py # Interactive static HTML dashboard generation
  determinism.py # Repeated-run consistency checks
  llm.py       # Mock and subprocess LLM adapters
  eval.py      # Evaluation runner and report generation
  rag.py       # Local lexical RAG index and query helpers
  trace.py     # Trace rendering and transcript replay
  tools.py     # Tool registry and built-in tools
providers/
  __init__.py
  base.py
  echo.py
  openai.py
  openai_compatible.py
  anthropic.py
  gemini.py
  ollama.py
  azure_openai.py
  deepseek.py
  groq.py
```

## How Providers Work

Every provider is a subprocess adapter. The harness sends JSON to the provider over stdin, and the provider prints JSON to stdout.

Run any provider with this pattern:

```bash
python3 -m agent_harness.cli "Your task here" --llm-cmd "python3 -m providers.PROVIDER_NAME"
```

For example:

```bash
python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.deepseek"
```

## Provider Setup

Use this table to choose a provider. Keep API keys in environment variables. Do not write real keys into project files.

| Provider | Module | Required setup | Example command |
| --- | --- | --- | --- |
| Echo test provider | `providers.echo` | No API key. Used to test the subprocess protocol. | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.echo"` |
| OpenAI | `providers.openai` | `OPENAI_API_KEY`, optional `OPENAI_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.openai"` |
| OpenAI-compatible API | `providers.openai_compatible` | `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.openai_compatible"` |
| Anthropic | `providers.anthropic` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.anthropic"` |
| Gemini | `providers.gemini` | `GEMINI_API_KEY`, optional `GEMINI_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.gemini"` |
| Ollama | `providers.ollama` | Local Ollama server, optional `OLLAMA_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.ollama"` |
| Azure OpenAI | `providers.azure_openai` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.azure_openai"` |
| DeepSeek | `providers.deepseek` | `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.deepseek"` |
| Groq | `providers.groq` | `GROQ_API_KEY`, optional `GROQ_MODEL` | `python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.groq"` |

## DeepSeek Example

DeepSeek uses an OpenAI-compatible API format. This project defaults to the lower-cost V4 Flash model.

```bash
cd ~/Desktop/llm-agent-harness

export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"

python3 -m agent_harness.cli "List files in the workspace" --llm-cmd "python3 -m providers.deepseek"
```

To use the stronger V4 Pro model:

```bash
export DEEPSEEK_MODEL="deepseek-v4-pro"
```

## Evaluation Benchmark

The project includes a small evaluation runner under `evals/`.

Run the default benchmark with DeepSeek:

```bash
cd ~/Desktop/llm-agent-harness
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"

python3 -m agent_harness.cli eval --tasks evals/tasks.jsonl --llm-cmd "python3 -m providers.deepseek"
```

If a provider is slow, increase the subprocess timeout:

```bash
python3 -m agent_harness.cli eval --tasks evals/tasks.jsonl --llm-timeout 300 --llm-cmd "python3 -m providers.deepseek"
```

The eval runner writes:

- JSONL results under `evals/results/`
- Markdown reports under `evals/reports/`
- Per-run transcripts under `runs/`

Each eval task checks:

- expected final status
- expected tool calls
- expected text in the final answer
- forbidden tool calls
- expected text in tool observations
- latency
- token usage when the provider returns usage data
- estimated cost when pricing is configured
- prompt and completion cost breakdown
- failure category classification
- provider errors and timeouts

The default task file is:

```text
evals/tasks.jsonl
```

The expanded benchmark covers:

- directory listing
- provider directory inspection
- README summarization
- pricing config reading
- flat file writing
- nested file writing
- workspace path escape denial
- shell denial without permission
- shell execution with explicit permission
- multi-step read-then-write workflows
- avoiding unnecessary shell usage
- missing-file error handling

The default pricing file is:

```text
evals/pricing.json
```

Cost estimates are calculated from token usage returned by the provider and the per-1M-token prices in `evals/pricing.json`.

## Failure Classification

Evaluation results classify failures into categories that are more useful than a generic failed status:

- `success`
- `planning_error`
- `tool_error`
- `timeout`
- `hallucination`
- `permission_denied`
- `api_error`
- `rate_limit`
- `failed`

The category is stored in JSONL results and shown in Markdown reports.

## Trace Visualization

Every normal run and eval task writes a transcript under `runs/`. Render any transcript as a Markdown trace:

```bash
python3 -m agent_harness.cli trace --run-id RUN_ID
```

This creates:

```text
runs/RUN_ID.trace.md
```

The trace includes the user task, LLM steps, tool calls, observations, final answer, usage data, and a Mermaid flowchart.

## Replay

Replay a saved run without calling a model:

```bash
python3 -m agent_harness.cli replay --run-id RUN_ID
```

Replay prints the recorded question, model steps, tool calls, observations, and final answer. It is useful for debugging and explaining how a result was produced.

## Determinism

Run the same task repeatedly to measure stability:

```bash
python3 -m agent_harness.cli determinism \
  --task "Use the read_file tool to read README.md, then summarize this project in 5 bullet points." \
  --runs 20 \
  --llm-timeout 300 \
  --llm-cmd "python3 -m providers.deepseek"
```

The determinism runner writes:

- JSONL run records under `evals/determinism/`
- Markdown consistency reports under `evals/determinism/`

The report includes success consistency, final answer consistency, tool sequence consistency, average latency, total tokens, and per-run transcript IDs.

## Environment Variables

The `.env.example` file lists the environment variables used by the provider adapters. You can copy values from it into your shell, but do not put real secrets in committed project files.

Example:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
```

## Optional SDK Dependencies

Most providers in this project use the Python standard library. The OpenAI and Azure OpenAI providers require the official OpenAI Python SDK:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install openai
```

DeepSeek does not require the OpenAI Python SDK.

## Subprocess Protocol

The harness sends this JSON payload to a provider:

```json
{
  "messages": [],
  "tools": [],
  "max_tool_calls": 8
}
```

The provider returns a JSON object like this to call a tool:

```json
{
  "thought": "I should inspect the files.",
  "tool_calls": [
    {"name": "list_dir", "arguments": {"path": "."}}
  ],
  "final": null
}
```

When the task is complete, the provider returns:

```json
{
  "thought": "I have enough information.",
  "tool_calls": [],
  "final": "Here is the answer."
}
```

## Shell Access

The `run_shell` tool is disabled by default. Enable it only for trusted tasks:

```bash
python3 -m agent_harness.cli "Run tests" --allow-shell --llm-cmd "python3 -m providers.deepseek"
```

Use shell allowlists to restrict which executables can run:

```bash
python3 -m agent_harness.cli "Run pwd and list files." \
  --allow-shell \
  --shell-allow pwd \
  --shell-allow ls \
  --llm-cmd "python3 -m providers.deepseek"
```

Use read-only mode when the agent should inspect but not write files:

```bash
python3 -m agent_harness.cli "Inspect the project structure." \
  --read-only \
  --llm-cmd "python3 -m providers.deepseek"
```

The shell tool also blocks common dangerous command patterns such as `sudo`, remote download commands, and destructive root-level deletion.

## Interactive Dashboard

Generate an interactive HTML dashboard from saved eval results, determinism reports, transcripts, and the local RAG index:

```bash
python3 -m agent_harness.cli dashboard
open dashboard/index.html
```

Open the generated file:

```bash
open dashboard/index.html
```

The dashboard includes:

- latest benchmark health summary with pass rate, failed task count, tool accuracy, and risk level
- overview metrics for pass rate, latency, cost, transcripts, and indexed chunks
- evaluation tables with pass/fail status, failure category, latency, token usage, cost, tools, checks, cost breakdown, and final answers
- evaluation filtering by latest eval, failed-only rows, or all historical rows
- evaluation search across task id, category, tool, and final answer
- evaluation sorting by source order, latency, cost, token usage, or task name
- lightweight charts for latency, cost, token usage, and failure categories
- transcript inspection with model steps, tool calls, observations, and final answers
- determinism run inspection with latency, tool sequence, usage, and final answer
- local RAG search over indexed project files with cited chunks

Generated preview images:

```text
dashboard/dashboard-v2-preview.png
dashboard/dashboard-v2-mobile-preview.png
```

## Local RAG

Build a local lexical RAG index over Markdown, text, code, JSON, JSONL, TOML, YAML, and similar project files:

```bash
python3 -m agent_harness.cli rag-index README.md evals agent_harness
```

Ask a question with citations:

```bash
python3 -m agent_harness.cli rag-query "How do I run the DeepSeek evaluation benchmark?"
```

This local RAG implementation does not require an embedding API. It uses chunking and lexical retrieval, then returns cited source chunks. For production semantic search, replace the ranking function in `agent_harness/rag.py` with an embedding model.

## Next Extensions

- Add concurrent task execution and retry handling
- Add embedding-backed RAG and PDF extraction
- Add a live run launcher to the dashboard
- Add approval prompts before write or shell actions
