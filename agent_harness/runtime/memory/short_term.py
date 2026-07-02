from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import MemoryEntry


@dataclass
class ShortTermMemory:
    max_entries: int = 24
    max_entry_chars: int = 1200
    summary: str = ""
    entries: list[MemoryEntry] = field(default_factory=list)

    def append(self, kind: str, content: str, metadata: dict[str, Any] | None = None) -> MemoryEntry | None:
        text = str(content or "").strip()
        if not text:
            return None
        if len(text) > self.max_entry_chars:
            text = text[: self.max_entry_chars] + "\n... truncated ..."
        entry = MemoryEntry(
            entry_id=self.next_entry_id(),
            kind=kind,
            content=text,
            metadata=metadata or {},
        )
        self.entries.append(entry)
        self.trim()
        return entry

    def add(self, value: str) -> None:
        self.append("note", value)

    def latest(self, limit: int = 5) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries[-limit:]]

    def summarize(self) -> str:
        if self.summary:
            prefix = self.summary
        else:
            prefix = ""
        recent = []
        for entry in self.entries[-8:]:
            recent.append(f"{entry.kind}: {entry.content[:180]}")
        combined = "\n".join(item for item in [prefix, *recent] if item)
        self.summary = combined[:2000]
        return self.summary

    def export(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "size": len(self.entries),
            "max_entries": self.max_entries,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def context(self) -> str:
        parts = []
        if self.summary:
            parts.append("Working memory summary:\n" + self.summary)
        if self.entries:
            lines = [f"- {entry.kind}: {entry.content}" for entry in self.entries[-8:]]
            parts.append("Recent working memory:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def trim(self) -> None:
        if len(self.entries) <= self.max_entries:
            return
        overflow = self.entries[: len(self.entries) - self.max_entries]
        overflow_summary = "\n".join(f"{entry.kind}: {entry.content[:180]}" for entry in overflow)
        if overflow_summary:
            self.summary = "\n".join(item for item in [self.summary, overflow_summary] if item)[-2000:]
        self.entries = self.entries[-self.max_entries :]

    def next_entry_id(self) -> int:
        if not self.entries:
            return 1
        return self.entries[-1].entry_id + 1
