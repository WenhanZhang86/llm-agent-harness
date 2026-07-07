from __future__ import annotations

import re

from .base import ContextItem, ContextMatch
from .context_store import ContextStore


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
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


def retrieve_context(store: ContextStore, task: str, top_k: int = 4) -> list[ContextMatch]:
    terms = tokenize(task)
    if not terms:
        return []
    matches = []
    for item in store.load_items():
        score, matched_terms = score_item(item, terms)
        if score <= 0:
            continue
        reason = "matched terms: " + ", ".join(matched_terms[:8])
        matches.append(ContextMatch(item=item, score=score, reason=reason))
    matches.sort(key=lambda match: (-match.score, match.item.source, match.item.id))
    return matches[:top_k]


def format_context(matches: list[ContextMatch], max_chars: int = 5000) -> str:
    if not matches:
        return ""
    sections = []
    budget = max_chars
    for match in matches:
        header = f"[{match.item.id}] {match.item.title} ({match.reason})"
        content = match.item.content[: max(200, min(1200, budget))]
        section = header + "\n" + content
        sections.append(section)
        budget -= len(section)
        if budget <= 0:
            break
    return "Retrieved context:\n\n" + "\n\n---\n\n".join(sections)


def score_item(item: ContextItem, terms: list[str]) -> tuple[float, list[str]]:
    haystack = " ".join([item.title, item.source, " ".join(item.tags), item.content]).lower()
    matched: list[str] = []
    score = 0.0
    for term in terms:
        count = haystack.count(term)
        if count:
            matched.append(term)
            score += min(count, 5)
            if term in item.title.lower():
                score += 3
            if term in item.source.lower():
                score += 2
            if term in item.tags:
                score += 1
    if "readme" in item.tags:
        score += 0.5
    return score, matched


def tokenize(value: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9_]+", value.lower())
    return [token for token in tokens if len(token) > 2 and token not in STOPWORDS]
