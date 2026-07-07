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

Compare DeepSeek and OpenAI on the same eval set:

```bash
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export DEEPSEEK_MODEL="deepseek-v4-flash"
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_MODEL="gpt-4o-mini"

python3 -m agent_harness.cli compare \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --provider deepseek="python3 -m providers.deepseek" \
  --provider openai="python3 -m providers.openai"
```

Compare named models from the registry:

```bash
python3 -m agent_harness.cli compare \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --models deepseek_pro openai_gpt55
```

Retry transient failures or repeat each task for regression sampling:

```bash
python3 -m agent_harness.cli eval \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --retries 2 \
  --repeat 3 \
  --llm-cmd "python3 -m providers.deepseek"
```

The eval runner writes:

- JSONL results under `evals/results/`
- A Markdown report under `evals/reports/`
- Summary JSON and CSV exports under `evals/results/`
- HTML report exports under `evals/reports/`

The compare runner also writes combined provider comparison files under `evals/comparisons/`.

It reports:

- success rate
- tool accuracy
- latency
- token usage when available
- estimated cost when pricing is configured
- prompt, completion, embedding, and tool cost breakdown
- failure categories for debugging failed tasks
- optional judge score and reason for rubric tasks
- dataset SHA-256 for versioned regression history
- failed checks, failure details, and recommended fixes
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
  "allow_shell": false,
  "read_only": false,
  "shell_allow": ["pwd"],
  "max_write_bytes": 1000,
  "expected_answer": "A concise summary of the expected answer.",
  "rubric": "Score 1.0 for a complete and grounded answer.",
  "judge_min_score": 0.8
}
```

Use `expect_contains` when every phrase must appear in the final answer. Use `expect_any_contains` when any one of several acceptable phrases is enough.

The default task file currently contains 41 tasks covering basic tool use, multi-step workflows, tool-choice ambiguity, safety and permissions, error recovery, hallucination detection, formatting, longer-context inspection, cost-sensitive behavior, and regression cases.

Run rubric grading with an optional judge model:

```bash
python3 -m agent_harness.cli eval \
  --tasks evals/tasks.jsonl \
  --llm-timeout 300 \
  --llm-cmd "python3 -m providers.deepseek" \
  --judge-cmd "python3 -m providers.deepseek"
```

Token pricing lives in `evals/pricing.json`. Prices are expressed per 1 million tokens.

## RAG Evaluation

Run RAG evaluation with the default RAG task file:

```bash
python3 -m agent_harness.cli rag-eval \
  --tasks evals/rag_tasks.jsonl \
  --inputs README.md evals agent_harness
```

RAG eval reports check whether retrieval cites expected sources, whether the answer contains required facts, and whether it avoids forbidden claims.

## Model Registry

Named model configs live in `configs/models.json`. The file stores provider commands and model names, but not API keys.

Example:

```bash
python3 -m agent_harness.cli compare \
  --tasks evals/tasks.jsonl \
  --models deepseek_flash openai_mini
```

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
