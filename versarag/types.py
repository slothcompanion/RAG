"""Core data types shared across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A unit of loaded content (a file, a PDF page, a CSV row...)."""

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A piece of a document that gets embedded and indexed."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(id=d["id"], text=d["text"], metadata=d.get("metadata", {}))


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {**self.chunk.to_dict(), "score": self.score}


@dataclass
class Answer:
    text: str
    sources: list[ScoredChunk]

    def to_dict(self) -> dict[str, Any]:
        return {"answer": self.text, "sources": [s.to_dict() for s in self.sources]}
