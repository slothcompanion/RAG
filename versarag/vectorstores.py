"""Vector stores. ``numpy`` is dependency-free and persists to a folder;
``chroma`` scales further. Implement :class:`VectorStore` to add others
(Qdrant, pgvector, LanceDB, ...)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from .registry import register
from .types import Chunk, ScoredChunk


def matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    """Equality filter; a list value means "any of"."""
    if not filters:
        return True
    for k, v in filters.items():
        actual = metadata.get(k)
        if isinstance(v, (list, tuple, set)):
            if actual not in v:
                return False
        elif actual != v:
            return False
    return True


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        """Add chunks, skipping ids already present. Returns number added."""

    @abstractmethod
    def search(
        self, query: np.ndarray, k: int, filters: dict[str, Any] | None = None
    ) -> list[ScoredChunk]: ...

    @abstractmethod
    def all_chunks(self) -> list[Chunk]: ...

    @abstractmethod
    def has(self, ids: list[str]) -> set[str]:
        """Return the subset of ``ids`` already stored."""

    @abstractmethod
    def delete(self, filters: dict[str, Any]) -> int: ...

    @abstractmethod
    def count(self) -> int: ...

    def persist(self) -> None:  # noqa: B027 - optional hook
        pass

    def get_meta(self) -> dict[str, Any]:
        return {}

    def set_meta(self, meta: dict[str, Any]) -> None:  # noqa: B027
        pass


@register("vectorstore", "numpy")
class NumpyVectorStore(VectorStore):
    """Exact cosine search in memory; saved as ``embeddings.npy`` + ``chunks.jsonl``.

    Comfortable up to a few hundred thousand chunks.
    """

    def __init__(self, path: str | None = ".rag_index") -> None:
        self.path = Path(path) if path else None
        self._chunks: list[Chunk] = []
        self._ids: dict[str, int] = {}
        self._emb: np.ndarray | None = None
        self._meta: dict[str, Any] = {}
        if self.path and (self.path / "chunks.jsonl").exists():
            self._load()

    def _load(self) -> None:
        assert self.path is not None
        with open(self.path / "chunks.jsonl", encoding="utf-8") as f:
            self._chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        self._ids = {c.id: i for i, c in enumerate(self._chunks)}
        emb_file = self.path / "embeddings.npy"
        self._emb = np.load(emb_file) if emb_file.exists() and self._chunks else None
        meta_file = self.path / "meta.json"
        if meta_file.exists():
            self._meta = json.loads(meta_file.read_text())

    def persist(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        if self._emb is not None:
            np.save(self.path / "embeddings.npy", self._emb)
        elif (self.path / "embeddings.npy").exists():
            (self.path / "embeddings.npy").unlink()
        (self.path / "meta.json").write_text(json.dumps(self._meta, indent=2))

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        keep = [i for i, c in enumerate(chunks) if c.id not in self._ids]
        if not keep:
            return 0
        new = np.asarray(embeddings, dtype=np.float32)[keep]
        for i in keep:
            self._ids[chunks[i].id] = len(self._chunks)
            self._chunks.append(chunks[i])
        self._emb = new if self._emb is None else np.vstack([self._emb, new])
        return len(keep)

    def search(self, query: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        if self._emb is None or not self._chunks:
            return []
        scores = self._emb @ np.asarray(query, dtype=np.float32)
        if filters:
            mask = np.array([matches(c.metadata, filters) for c in self._chunks])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [ScoredChunk(self._chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i])]

    def all_chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def has(self, ids: list[str]) -> set[str]:
        return {i for i in ids if i in self._ids}

    def delete(self, filters: dict[str, Any]) -> int:
        keep = [i for i, c in enumerate(self._chunks) if not matches(c.metadata, filters)]
        removed = len(self._chunks) - len(keep)
        if removed:
            self._chunks = [self._chunks[i] for i in keep]
            self._emb = self._emb[keep] if self._emb is not None and keep else None
            self._ids = {c.id: i for i, c in enumerate(self._chunks)}
        return removed

    def count(self) -> int:
        return len(self._chunks)

    def get_meta(self) -> dict[str, Any]:
        return dict(self._meta)

    def set_meta(self, meta: dict[str, Any]) -> None:
        self._meta = dict(meta)


@register("vectorstore", "chroma")
class ChromaVectorStore(VectorStore):
    def __init__(self, path: str = ".rag_index_chroma", collection: str = "versarag") -> None:
        try:
            import chromadb
        except ImportError as e:  # pragma: no cover
            raise ImportError("Chroma store needs `pip install chromadb` (or versarag[chroma])") from e
        self.meta_file = Path(path) / "versarag_meta.json"
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def _where(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = [{k: {"$in": list(v)}} if isinstance(v, (list, tuple, set)) else {k: v} for k, v in filters.items()]
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    @staticmethod
    def _clean(meta: dict[str, Any]) -> dict[str, Any]:
        return {k: v if isinstance(v, (str, int, float, bool)) else json.dumps(v) for k, v in meta.items() if v is not None}

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> int:
        existing = self.has([c.id for c in chunks])
        keep = [i for i, c in enumerate(chunks) if c.id not in existing]
        if not keep:
            return 0
        self.col.add(
            ids=[chunks[i].id for i in keep],
            embeddings=np.asarray(embeddings)[keep].tolist(),
            documents=[chunks[i].text for i in keep],
            metadatas=[self._clean(chunks[i].metadata) or {"_": 0} for i in keep],
        )
        return len(keep)

    def search(self, query: np.ndarray, k: int, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        n = self.count()
        if not n:
            return []
        res = self.col.query(
            query_embeddings=[np.asarray(query).tolist()], n_results=min(k, n), where=self._where(filters)
        )
        return [
            ScoredChunk(Chunk(id=i, text=d, metadata=m or {}), 1.0 - float(dist))
            for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def all_chunks(self) -> list[Chunk]:
        res = self.col.get(include=["documents", "metadatas"])
        return [Chunk(id=i, text=d, metadata=m or {}) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])]

    def has(self, ids: list[str]) -> set[str]:
        return set(self.col.get(ids=ids, include=[])["ids"]) if ids else set()

    def delete(self, filters: dict[str, Any]) -> int:
        ids = self.col.get(where=self._where(filters), include=[])["ids"]
        if ids:
            self.col.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.col.count()

    def get_meta(self) -> dict[str, Any]:
        return json.loads(self.meta_file.read_text()) if self.meta_file.exists() else {}

    def set_meta(self, meta: dict[str, Any]) -> None:
        self.meta_file.parent.mkdir(parents=True, exist_ok=True)
        self.meta_file.write_text(json.dumps(meta, indent=2))
