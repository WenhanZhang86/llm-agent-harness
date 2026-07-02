from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .base import ContextItem


DEFAULT_CONTEXT_DIRS = ("docs", "specs", "prompts")
TEXT_SUFFIXES = {".md", ".txt", ".rst", ".prompt", ".json", ".yaml", ".yml"}


class ContextStore:
    def __init__(self, workspace: Path):
        self.workspace = workspace.resolve()

    def load_items(self) -> list[ContextItem]:
        items: list[ContextItem] = []
        items.extend(self.load_file(self.workspace / "README.md", tags=["readme", "project"]))
        for folder in DEFAULT_CONTEXT_DIRS:
            root = self.workspace / folder
            if root.exists():
                for path in sorted(root.rglob("*")):
                    if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                        items.extend(self.load_file(path, tags=[folder]))
        items.extend(self.load_rag_items())
        return items

    def load_file(self, path: Path, tags: Iterable[str]) -> list[ContextItem]:
        if not path.exists() or not path.is_file():
            return []
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []
        rel = self.safe_relative(path)
        return [
            ContextItem(
                id=rel,
                title=path.stem or path.name,
                source=rel,
                tags=list(tags),
                content=content[:8000],
            )
        ]

    def load_rag_items(self) -> list[ContextItem]:
        index_path = self.workspace / "rag" / "index.json"
        if not index_path.exists():
            return []
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        items: list[ContextItem] = []
        for chunk in data.get("chunks", []):
            source = str(chunk.get("source") or "rag")
            chunk_id = str(chunk.get("chunk_id") or len(items))
            text = str(chunk.get("text") or "")
            if not text:
                continue
            items.append(
                ContextItem(
                    id=f"{source}#chunk-{chunk_id}",
                    title=Path(source).name,
                    source=source,
                    tags=["rag", "indexed"],
                    content=text[:4000],
                )
            )
        return items

    def safe_relative(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.workspace))
        except ValueError:
            return path.name
