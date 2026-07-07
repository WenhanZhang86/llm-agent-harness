from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class MemoryEntry:
    entry_id: int
    kind: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


@dataclass
class ContextItem:
    id: str
    title: str
    source: str
    tags: list[str]
    content: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "tags": self.tags,
            "content": self.content,
        }


@dataclass
class ContextMatch:
    item: ContextItem
    score: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        data = self.item.to_dict()
        data["score"] = self.score
        data["reason"] = self.reason
        data["preview"] = self.item.content[:500]
        return data
