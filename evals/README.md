# Evaluations

This directory contains benchmark tasks for the LLM Agent Harness.

Run the default eval set from the project root:

```bash
python3 -m agent_harness.cli eval --tasks evals/tasks.jsonl --llm-cmd "python3 -m providers.deepseek"
```

Increase timeout for slower providers:

```bash
python3 -m agent_harness.cli eval --tasks evals/tasks.jsonl --llm-timeout 300 --llm-cmd "python3 -m providers.deepseek"
```

Run a no-API smoke test with the echo provider:

```bash
python3 -m agent_harness.cli eval --tasks evals/echo-smoke.jsonl --llm-cmd "python3 -m providers.echo"
```

The eval runner writes:

- JSONL results under `evals/results/`
- A Markdown report under `evals/reports/`

It reports:

- success rate
- tool accuracy
- latency
- token usage when available
- estimated cost when pricing is configured
- prompt, completion, embedding, and tool cost breakdown
- failure categories for debugging failed tasks
- transcript run IDs
- provider errors and timeouts

Each task is one JSON object per line:

```json
{
  "id": "read_readme_summary",
  "task": "Use the read_file tool to read README.md, then summarize this project in 5 bullet points.",
  "expect_status": "completed",
  "expect_tools": ["read_file"],
  "forbid_tools": [],
  "expect_contains": ["provider", "tool"],
  "expect_any_contains": [],
  "expect_observation_contains": [],
  "allow_shell": false
}
```

Use `expect_contains` when every phrase must appear in the final answer. Use `expect_any_contains` when any one of several acceptable phrases is enough.

Token pricing lives in `evals/pricing.json`. Prices are expressed per 1 million tokens.

## Failure Categories

Each result includes a `failure_category` field:

- `success`
- `planning_error`
- `tool_error`
- `timeout`
- `hallucination`
- `permission_denied`
- `api_error`
- `rate_limit`
- `failed`

The Markdown report summarizes category counts and shows each task category in the main table.

## Trace and Replay

Render a transcript as a Markdown trace:

```bash
python3 -m agent_harness.cli trace --run-id RUN_ID
```

Replay a transcript in the terminal:

```bash
python3 -m agent_harness.cli replay --run-id RUN_ID
```

## Determinism

Run one task repeatedly and measure stability:

```bash
python3 -m agent_harness.cli determinism \
  --task "Use the read_file tool to read README.md, then summarize this project in 5 bullet points." \
  --runs 20 \
  --llm-timeout 300 \
  --llm-cmd "python3 -m providers.deepseek"
```

The determinism report includes success consistency, final answer consistency, tool sequence consistency, latency, token usage, and run IDs for replay.

## Permission Controls

Run the agent in read-only mode:

```bash
python3 -m agent_harness.cli "Inspect the project." --read-only --llm-cmd "python3 -m providers.deepseek"
```

Restrict shell execution to specific executables:

```bash
python3 -m agent_harness.cli "Run pwd." \
  --allow-shell \
  --shell-allow pwd \
  --llm-cmd "python3 -m providers.deepseek"
```

## Dashboard

Generate a static HTML dashboard:

```bash
python3 -m agent_harness.cli dashboard
```

The output is `dashboard/index.html`.

## Local RAG

Build an index:

```bash
python3 -m agent_harness.cli rag-index README.md evals agent_harness
```

Query it with citations:

```bash
python3 -m agent_harness.cli rag-query "How do I run the DeepSeek evaluation benchmark?"
```
