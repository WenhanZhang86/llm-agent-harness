# LLM Agent Harness

A lightweight **AI Agent Harness** for running, tracing, debugging, deploying, and evaluating tool-using LLM agents across multiple model providers.

The project is intentionally small and local-dev friendly. It is not just an evaluation dashboard: evaluation is a downstream analysis layer built on top of the Agent Runtime and its structured traces.

This README assumes the project lives at:

```text
~/Desktop/llm-agent-harness
```

If your folder is somewhere else, replace that path with your actual project path.

## What This Project Does

LLM Agent Harness lets you:

- Run a multi-step tool-using agent from the command line or HTTP API
- Connect different model providers through a common subprocess adapter interface
- Register schema-based tools and enforce runtime permissions
- Retrieve project context before an agent run
- Maintain short-term working memory during a run
- Save structured traces, event logs, and summaries for every run
- Replay, inspect, and compare saved runs without calling a model again
- Evaluate agents with JSONL benchmarks, expected tools, expected answers, rubrics, and optional judge models
- Compare multiple providers on the same task set and generate leaderboards
- Measure determinism by running the same task repeatedly
- Build and query a lightweight local RAG index with citations
- Evaluate RAG retrieval quality
- Generate a static dashboard for runs, evals, comparisons, determinism, traces, memory, and RAG
- Serve the harness locally with FastAPI or Docker

## Architecture

```text
Agent Runtime
  -> Context Retrieval
  -> Short-Term Memory
  -> Tool System
  -> Observability
  -> Structured Trace
  -> Evaluation
  -> Dashboard
  -> FastAPI Service
```

The core execution path is:

```text
User task
-> retrieve context
-> initialize short-term memory
-> call LLM provider
-> execute tool call or return final answer
-> append observation
-> update memory
-> repeat until done
-> write trace, events, and summary
```

## Current Capability Summary

| Area | Implemented capability |
| --- | --- |
| Agent Runtime | Multi-step agent loop with LLM calls, tool calls, observations, final answers, max steps, timeout, and basic cost limits. |
| Context Retrieval | Keyword-based retrieval from `README.md`, `docs/`, `specs/`, `prompts/`, and existing RAG chunks. |
| Short-Term Memory | Current-run working memory for task, LLM responses, tool outputs, observations, summaries, and final answer. |
| Tool System | First-class schema-based tools with structured outputs, permission checks, and trace integration. |
| Observability | Per-run `trace.json`, `events.jsonl`, and `summary.json` files. |
| Replay and Trace | CLI replay, Markdown trace rendering, run summaries, event timelines, and run-to-run comparison. |
| Providers | Echo, OpenAI, OpenAI-compatible APIs, Anthropic, Gemini, Ollama, Azure OpenAI, DeepSeek, and Groq adapters. |
| Evaluation | JSONL benchmark runner with success, tool accuracy, expected text, expected answers, rubrics, judge scoring, latency, tokens, cost, and failure categories. |
| Comparison | Multi-provider benchmark comparison with leaderboard reports. |
| Determinism | Repeated-run consistency checks for success, final answers, and tool sequences. |
| RAG | Local lexical RAG indexing and cited query answers. |
| RAG Evaluation | Source-hit checks, answer text checks, and forbidden-claim checks. |
| Dashboard | Static HTML dashboard for overview, leaderboard, evaluations, runs, determinism, RAG search, traces, context, memory, and events. |
| Deployment | FastAPI server, API schemas, CLI `serve` command, Dockerfile, and `.dockerignore`. |
| Safety | Safe default runtime policy: no file writes, no shell/code execution, no real network tools unless explicitly allowed. |

## Quick Start

Install the project in editable mode:

```bash
cd ~/Desktop/llm-agent-harness
python3 -m pip install -e .
```

Run the Agent Runtime with the built-in mock provider:

```bash
python3 -m agent_harness.cli run-agent "Use calculator to compute 123 * 456"
```

List available runtime tools:

```bash
python3 -m agent_harness.cli list-tools
```

Run one tool directly:

```bash
python3 -m agent_harness.cli run-tool calculator '{"expression":"123 * 456"}'
```

Generate the dashboard:

```bash
python3 -m agent_harness.cli dashboard
open dashboard/index.html
```

## Agent Runtime

The runtime module lives under:

```text
agent_harness/runtime/
```

Important files:

```text
agent.py          # Agent Runtime loop
state.py          # Runtime state
step.py           # Structured AgentStep records
events.py         # Runtime events, event logger, and run summaries
policy.py         # Runtime permission and budget policy
tool.py           # Base Tool interface and ToolOutput
tool_registry.py  # Tool registry and built-in runtime tools
memory/           # Context retrieval and short-term memory
```

Each runtime run writes:

```text
runs/RUN_ID/
  trace.json
  events.jsonl
  summary.json
```

`trace.json` stores the structured run. `events.jsonl` stores append-only runtime events. `summary.json` stores a compact overview for replay, dashboard, and API responses.

## Structured Agent Steps

Each `AgentStep` can include:

- `step_id`
- `step_type`: `llm`, `tool`, `observation`, `final`, or `error`
- `input`
- `output`
- `latency_ms`
- `tokens`
- `cost_usd`
- `error`

This makes evaluation and debugging downstream consumers of the runtime, instead of separate one-off scripts.

## Runtime Events

Runtime events are JSON-serializable and append-only.

Supported event types include:

- `agent_started`
- `context_retrieved`
- `memory_initialized`
- `llm_request`
- `llm_response`
- `tool_request`
- `tool_response`
- `observation_added`
- `memory_updated`
- `memory_summarized`
- `final_answer`
- `runtime_error`
- `agent_finished`

## Tool System

Runtime tools are schema-based, permission-aware, and traceable.

Each tool defines:

- `name`
- `description`
- `input_schema`
- optional `output_schema`
- `required_permissions`
- `run(args, context)`

Tool output is always structured:

```json
{
  "ok": true,
  "result": {},
  "error": null,
  "metadata": {}
}
```

Built-in runtime tools:

| Tool | Purpose | Permission |
| --- | --- | --- |
| `calculator` | Safe arithmetic without unsafe `eval` | none |
| `local_file_search` | Search text files under the workspace | `read_files` |
| `file_reader` | Read a UTF-8 text file under the workspace | `read_files` |
| `file_writer` | Write a UTF-8 text file under the workspace | `write_files` |
| `mock_web_search` | Deterministic fake search results, no real network call | `network` |

Safe defaults deny file writes, shell execution, code execution, and network permissions.

## Context & Memory

The memory layer is for agent execution only. It is not chatbot memory, personal memory, or autonomous long-term memory.

It has two parts:

- Context Store: lightweight repository of project context
- Short-Term Memory: rolling current-run working memory

Context sources:

- `README.md`
- `docs/`
- `specs/`
- `prompts/`
- `rag/index.json` if available

Inspect context and memory for a run:

```bash
python3 -m agent_harness.cli show-context RUN_ID
python3 -m agent_harness.cli show-memory RUN_ID
```

## Observability and Debugging

Replay a run without calling the model:

```bash
python3 -m agent_harness.cli replay RUN_ID
```

Render a trace:

```bash
python3 -m agent_harness.cli trace --run-id RUN_ID
```

Compare two saved runs:

```bash
python3 -m agent_harness.cli compare-runs RUN_ID_1 RUN_ID_2
```

The replay and trace commands read from saved run files only.

## Provider Adapters

Providers are subprocess adapters. The harness sends JSON through stdin and expects JSON through stdout.

Run any provider with this pattern:

```bash
python3 -m agent_harness.cli run-agent "Your task here" --llm-cmd "python3 -m providers.PROVIDER_NAME"
```

Available provider modules:

| Provider | Module | Required setup |
| --- | --- | --- |
| Echo | `providers.echo` | No API key. Local protocol smoke test. |
| OpenAI | `providers.openai` | `OPENAI_API_KEY`, optional `OPENAI_MODEL` |
| OpenAI-compatible | `providers.openai_compatible` | `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, `OPENAI_COMPATIBLE_MODEL` |
| Anthropic | `providers.anthropic` | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` |
| Gemini | `providers.gemini` | `GEMINI_API_KEY`, optional `GEMINI_MODEL` |
| Ollama | `providers.ollama` | Local Ollama server, optional `OLLAMA_MODEL` |
| Azure OpenAI | `providers.azure_openai` | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`; requires the optional `openai` Python SDK |
| DeepSeek | `providers.deepseek` | `DEEPSEEK_API_KEY`, optional `DEEPSEEK_MODEL` |
| Groq | `providers.groq` | `GROQ_API_KEY`, optional `GROQ_MODEL` |

Do not write real API keys into project files. Use environment variables.

## DeepSeek Example

```bash
cd ~/Desktop/llm-agent-harness

export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"

python3 -m agent_harness.cli run-agent \
  "List files and summarize README" \
  --llm-cmd "python3 -m providers.deepseek"
```

## OpenAI Example

```bash
cd ~/Desktop/llm-agent-harness

export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"

python3 -m agent_harness.cli run-agent \
  "List files and summarize README" \
  --llm-cmd "python3 -m providers.openai"
```

## Evaluation

Evaluation consumes runtime behavior and saved traces.

Run the default benchmark:

```bash
python3 -m agent_harness.cli eval \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --llm-cmd "python3 -m providers.deepseek"
```

Run a no-API smoke test:

```bash
python3 -m agent_harness.cli eval \
  --tasks evals/echo-smoke.jsonl \
  --llm-cmd "python3 -m providers.echo"
```

Evaluation reports include:

- total tasks
- passed and failed tasks
- success rate
- tool accuracy
- average latency
- token usage
- estimated cost
- prompt, completion, embedding, and tool cost breakdown
- failure categories
- failed checks
- actual tool sequences
- error details
- recommended fixes when available

Task assertions can include:

- expected status
- expected tools
- forbidden tools
- expected final-answer text
- expected observation text
- expected answer
- rubric
- optional judge-model scoring
- per-task permission policy

Outputs are written under:

```text
evals/results/
evals/reports/
```

## Multi-Provider Comparison

Compare provider commands directly:

```bash
python3 -m agent_harness.cli compare \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --provider deepseek="python3 -m providers.deepseek" \
  --provider openai="python3 -m providers.openai"
```

Compare named models from `configs/models.json`:

```bash
python3 -m agent_harness.cli compare \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --models deepseek_pro openai_gpt55
```

Comparison outputs are written under:

```text
evals/comparisons/
```

## Determinism

Run the same task multiple times and measure consistency:

```bash
python3 -m agent_harness.cli determinism \
  --task "Use the read_file tool to read README.md, then summarize this project in 5 bullet points." \
  --runs 20 \
  --llm-timeout 300 \
  --llm-cmd "python3 -m providers.deepseek"
```

Reports include:

- completed runs
- success consistency
- final answer consistency
- tool sequence consistency
- average latency
- total tokens
- per-run status and tool sequence

## RAG

Build a local lexical RAG index:

```bash
python3 -m agent_harness.cli rag-index README.md evals agent_harness
```

Ask a cited question:

```bash
python3 -m agent_harness.cli rag-query "How do I run the DeepSeek evaluation benchmark?"
```

Evaluate RAG retrieval:

```bash
python3 -m agent_harness.cli rag-eval \
  --tasks evals/rag_tasks.jsonl \
  --inputs README.md evals agent_harness
```

This RAG implementation is intentionally lightweight. It uses lexical retrieval, not embeddings or a vector database.

## Dashboard

Generate the static dashboard:

```bash
python3 -m agent_harness.cli dashboard
open dashboard/index.html
```

The dashboard reads local project artifacts:

- run summaries
- event logs
- traces
- evaluation results
- provider comparisons
- determinism reports
- RAG index and RAG eval reports

It is a static HTML dashboard. It does not require a database.

Preview images:

```text
dashboard/dashboard-preview.png
dashboard/dashboard-mobile-preview.png
dashboard/dashboard-v2-preview.png
dashboard/dashboard-v2-mobile-preview.png
```

## FastAPI Service

Start the local server:

```bash
python3 -m agent_harness.cli serve --host 127.0.0.1 --port 8000
```

Or run with uvicorn:

```bash
uvicorn agent_harness.server.app:app --host 127.0.0.1 --port 8000
```

Endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `GET` | `/tools` | Registered tool names, descriptions, schemas, and permissions |
| `POST` | `/run-agent` | Run the Agent Runtime |
| `GET` | `/runs/{run_id}` | Return `summary.json` |
| `GET` | `/runs/{run_id}/trace` | Return `trace.json` |
| `GET` | `/runs/{run_id}/events` | Return parsed `events.jsonl` |
| `GET` | `/runs/{run_id}/replay` | Return structured replay data |

Example:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl http://127.0.0.1:8000/tools
```

```bash
curl -X POST http://127.0.0.1:8000/run-agent \
  -H "Content-Type: application/json" \
  -d '{"task":"Use calculator to compute 123 * 456","provider":"mock","max_steps":4}'
```

Server defaults are safe:

- file writes disabled
- shell execution disabled
- code execution disabled
- network tools disabled
- limited max steps
- limited timeout

Requests that try to enable denied permissions return a structured permission error.

## Docker

Build and run locally:

```bash
docker build -t llm-agent-harness .
docker run -p 8000:8000 llm-agent-harness
```

This Docker setup is for local development. It does not add a database, queue, worker system, authentication, Kubernetes, or cloud deployment code.

## Project Structure

```text
.
├── agent_harness/
│   ├── cli.py
│   ├── core.py
│   ├── llm.py
│   ├── eval.py
│   ├── compare.py
│   ├── determinism.py
│   ├── rag.py
│   ├── rag_eval.py
│   ├── dashboard.py
│   ├── trace.py
│   ├── model_registry.py
│   ├── runtime/
│   ├── server/
│   └── tools/
├── providers/
├── evals/
├── configs/
├── dashboard/
├── rag/
├── runs/
├── tests/
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── README.md
```

## Testing

Run Python compilation checks:

```bash
python3 -m compileall agent_harness providers tests
```

Run unit tests:

```bash
python3 -m unittest discover -s tests
```

Useful smoke checks:

```bash
python3 -m agent_harness.cli run-tool calculator '{"expression":"2+2"}'
python3 -m agent_harness.cli run-agent "Use calculator to compute 123 * 456"
python3 -m agent_harness.cli eval --tasks evals/echo-smoke.jsonl --llm-cmd "python3 -m providers.echo"
python3 -m agent_harness.cli dashboard
```

## Safety Model

The project includes lightweight local safety controls:

- workspace path checks
- path traversal blocking
- read-only mode for legacy tools
- runtime permission policy
- shell allowlists for legacy eval commands
- dangerous command blocking
- write-size limits
- server-side denial of dangerous runtime permissions

This is not a production-grade sandbox. Do not expose it directly to untrusted users or the public internet.

## Current Limitations

Not implemented yet:

- browser UI for starting new runs interactively
- embedding-backed semantic RAG
- PDF ingestion
- parallel eval execution
- human approval workflow before file writes or shell commands
- production authentication and authorization
- production sandboxing beyond local path and command policies
- hosted cloud deployment templates

## Why This Project Exists

Most simple LLM demos stop at a chatbot or a single tool call. This project is meant to show the engineering layer around agents:

- runtime control
- tool execution
- context management
- memory within a run
- observability
- replay
- evaluation
- model comparison
- cost and latency analysis
- local deployment

In short: it is a small but complete harness for understanding how tool-using LLM agents behave across providers.
