from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


DEFAULT_EXTENSIONS = {".md", ".txt", ".py", ".json", ".jsonl", ".toml", ".yaml", ".yml"}
SKIP_PARTS = {
    "__pycache__",
    ".git",
    "runs",
    "dashboard",
    "rag",
    "results",
    "reports",
    "determinism",
    "comparisons",
    "rag_results",
    "rag_reports",
}
SKIP_FILES = {"rag_tasks.jsonl"}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass
class Chunk:
    source: str
    chunk_id: int
    text: str


def build_rag_index(
    workspace: Path,
    inputs: list[str],
    output: Path,
    chunk_chars: int = 1600,
) -> Path:
    chunks: list[Chunk] = []
    for file_path in discover_files(workspace, inputs):
        text = file_path.read_text(encoding="utf-8", errors="replace")
        text = remove_rag_query_examples(text)
        relative = str(file_path.relative_to(workspace))
        for index, chunk_text in enumerate(split_text(text, chunk_chars=chunk_chars), start=1):
            chunks.append(Chunk(source=relative, chunk_id=index, text=chunk_text))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "workspace": str(workspace),
                "chunk_count": len(chunks),
                "chunks": [chunk.__dict__ for chunk in chunks],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def query_rag_index(index_path: Path, question: str, top_k: int = 5) -> str:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in data.get("chunks", [])]
    ranked = rank_chunks(chunks, question)[:top_k]

    lines = [
        "# RAG Answer",
        "",
        f"Question: {question}",
        "",
    ]
    if not ranked:
        lines.append("No relevant chunks were found.")
        return "\n".join(lines)

    lines.extend(["## Answer", ""])
    lines.append(build_extractive_answer(ranked, question))
    lines.extend(["", "## Citations", ""])
    for score, chunk in ranked:
        snippet = summarize_chunk(chunk.text, question)
        lines.extend(
            [
                f"### [{chunk.source}#chunk-{chunk.chunk_id}]",
                "",
                f"- Score: {score:.2f}",
                "",
                "```text",
                snippet,
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def discover_files(workspace: Path, inputs: list[str]) -> list[Path]:
    files: list[Path] = []
    for item in inputs or ["."]:
        path = (workspace / item).resolve()
        if workspace != path and workspace not in path.parents:
            raise ValueError(f"Path escapes workspace: {item}")
        if path.is_file() and path.suffix.lower() in DEFAULT_EXTENSIONS and path.name not in SKIP_FILES:
            files.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in DEFAULT_EXTENSIONS:
                    if child.name in SKIP_FILES:
                        continue
                    if not SKIP_PARTS.intersection(child.relative_to(workspace).parts):
                        files.append(child)
    return sorted(set(files))


def split_text(text: str, chunk_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > chunk_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        if len(paragraph) > chunk_chars:
            chunks.extend(paragraph[i : i + chunk_chars] for i in range(0, len(paragraph), chunk_chars))
            continue
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def rank_chunks(chunks: list[Chunk], question: str) -> list[tuple[float, Chunk]]:
    query_terms = tokenize(question)
    if not query_terms:
        return []
    query_counts = Counter(query_terms)
    scored: list[tuple[float, Chunk]] = []
    for chunk in chunks:
        terms = tokenize(chunk.text)
        counts = Counter(terms)
        overlap = sum(min(counts[term], query_counts[term]) for term in query_counts)
        if overlap == 0:
            continue
        score = overlap / max(len(query_terms), 1)
        score += keyword_boost(chunk, question)
        scored.append((score, chunk))
    return sorted(scored, key=lambda item: item[0], reverse=True)


def summarize_chunk(text: str, question: str, limit: int = 1500) -> str:
    command_block = extract_relevant_command_block(text, question)
    if command_block:
        return command_block[:limit]
    terms = set(tokenize(question))
    sentences = re.split(r"(?<=[.!?])\s+|\n", text)
    selected = [sentence.strip() for sentence in sentences if terms.intersection(tokenize(sentence))]
    snippet = "\n".join(selected[:20]) if selected else text[:limit]
    return snippet[:limit]


def build_extractive_answer(ranked: list[tuple[float, Chunk]], question: str) -> str:
    snippets = []
    preferred = [item for item in ranked if not item[1].source.endswith(".py")]
    selected = preferred[:3] or ranked[:3]
    for _, chunk in selected:
        snippet = summarize_chunk(chunk.text, question, limit=350)
        snippets.append(f"{snippet} [{chunk.source}#chunk-{chunk.chunk_id}]")
    return "\n\n".join(snippets)


def tokenize(value: str) -> list[str]:
    return [
        item.lower()
        for item in re.findall(r"[A-Za-z0-9_]+", value)
        if item.lower() not in STOPWORDS
    ]


def keyword_boost(chunk: Chunk, question: str) -> float:
    haystack = chunk.text.lower()
    query = question.lower()
    score = 0.0
    if chunk.source.endswith(".py"):
        score -= 0.5
    if chunk.source.lower() in {"readme.md", "evals/readme.md"}:
        score += 0.25
    if "deepseek" in query and "deepseek" in haystack:
        score += 0.25
    if "compare" in query or "same benchmark" in query:
        if "compare deepseek and openai" in haystack:
            score += 1.2
        if "agent_harness.cli compare" in haystack:
            score += 1.0
        if "providers.openai" in haystack and "providers.deepseek" in haystack:
            score += 0.8
    if "dashboard" in query:
        if "agent_harness.cli dashboard" in haystack:
            score += 1.2
        if "open dashboard/index.html" in haystack:
            score += 1.0
        if "generate the dashboard" in haystack:
            score += 0.6
    if "evaluation" in query or "benchmark" in query:
        if "eval --tasks" in haystack:
            score += 0.45
        if "evals/tasks.jsonl" in haystack:
            score += 0.35
        if "evaluation benchmark" in haystack:
            score += 0.25
    return score


def remove_rag_query_examples(text: str) -> str:
    cleaned_blocks: list[str] = []
    position = 0
    for match in re.finditer(r"```.*?```", text, flags=re.DOTALL):
        cleaned_blocks.append(text[position : match.start()])
        block = match.group(0)
        if "rag-query" not in block:
            cleaned_blocks.append(block)
        position = match.end()
    cleaned_blocks.append(text[position:])
    return "".join(cleaned_blocks)


def extract_relevant_command_block(text: str, question: str) -> str | None:
    query_terms = set(tokenize(question))
    blocks = re.findall(r"```(?:bash|sh|text)?\n(.*?)```", text, flags=re.DOTALL)
    best: tuple[int, str] | None = None
    for block in blocks:
        if "rag-query" in block:
            continue
        block_terms = set(tokenize(block))
        score = len(query_terms.intersection(block_terms))
        if "eval --tasks" in block:
            score += 5
        if "providers.deepseek" in block:
            score += 3
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, block.strip())
    return best[1] if best else None
