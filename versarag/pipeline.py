"""The :class:`RAG` facade: ingest documents, retrieve, and answer."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import embeddings as _embeddings  # noqa: F401  (registers providers)
from . import llms as _llms  # noqa: F401
from . import retrieval as _retrieval  # noqa: F401
from . import vectorstores as _vectorstores  # noqa: F401
from .chunking import chunk_documents
from .config import load_config
from .embeddings import Embedder
from .llms import LLM
from .loaders import iter_files, load_file
from .prompts import build_messages
from .registry import create
from .retrieval import Retriever
from .types import Answer, Document, ScoredChunk
from .vectorstores import VectorStore

log = logging.getLogger("versarag")


class RAG:
    """Configurable retrieval-augmented generation pipeline.

    >>> rag = RAG.from_config("configs/qwen-ollama.yaml")
    >>> rag.ingest(["./docs"])
    >>> print(rag.ask("What is our refund policy?").text)

    The LLM is created lazily, so ingesting/retrieving never loads a model.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.embedder: Embedder = create("embeddings", config["embeddings"])
        self.store: VectorStore = create("vectorstore", config["vectorstore"])
        self._check_index_compat()
        r = config.get("retrieval", {})
        self.retriever = Retriever(
            self.store,
            self.embedder,
            mode=r.get("mode", "hybrid"),
            top_k=r.get("top_k", 5),
            candidate_k=r.get("candidate_k", 25),
            rrf_k=r.get("rrf_k", 60),
            min_score=r.get("min_score"),
            reranker=create("reranker", config.get("reranker")),
        )
        self._llm: LLM | None = None

    @classmethod
    def from_config(cls, path: str | Path | dict[str, Any] | None = None, overrides: list[str] | None = None) -> "RAG":
        return cls(load_config(path, overrides))

    # -- components ---------------------------------------------------------
    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = create("llm", self.config["llm"])
            if self._llm is None:
                raise ValueError("No LLM configured (llm.provider is 'none')")
        return self._llm

    @llm.setter
    def llm(self, value: LLM) -> None:
        self._llm = value

    def _check_index_compat(self) -> None:
        meta = self.store.get_meta()
        name = getattr(self.embedder, "name", None)
        if self.store.count() and meta.get("embedder") and meta["embedder"] != name:
            raise ValueError(
                f"Index at {self.config['vectorstore']} was built with embeddings {meta['embedder']!r} "
                f"but config uses {name!r}. Use a different vectorstore.path or re-ingest with --reset."
            )

    # -- ingestion ----------------------------------------------------------
    def ingest_documents(self, docs: list[Document], replace: bool = True) -> dict[str, int]:
        c = self.config.get("chunking", {})
        chunks = chunk_documents(docs, c.get("chunk_size", 1000), c.get("chunk_overlap", 150), c.get("separators"))
        if replace:
            # Drop stale chunks of re-ingested sources whose content changed.
            new_ids_by_source: dict[str, set[str]] = {}
            for ch in chunks:
                new_ids_by_source.setdefault(str(ch.metadata.get("source", "")), set()).add(ch.id)
            for source, ids in new_ids_by_source.items():
                if source and len(self.store.has(list(ids))) != len(ids):
                    self.store.delete({"source": source})
        existing = self.store.has([ch.id for ch in chunks])
        todo = [ch for ch in chunks if ch.id not in existing]
        added = 0
        batch = int(c.get("embed_batch", 256))
        for i in range(0, len(todo), batch):
            part = todo[i : i + batch]
            added += self.store.add(part, self.embedder.embed_documents([ch.text for ch in part]))
        self.store.set_meta({**self.store.get_meta(), "embedder": getattr(self.embedder, "name", "unknown")})
        self.store.persist()
        self.retriever.invalidate()
        return {"documents": len(docs), "chunks": len(chunks), "added": added, "skipped": len(chunks) - len(todo)}

    def ingest(self, paths: Iterable[str | Path], recursive: bool = True, replace: bool = True) -> dict[str, int]:
        docs: list[Document] = []
        files = 0
        for f in iter_files(paths, recursive=recursive):
            try:
                docs.extend(load_file(f))
                files += 1
            except Exception as e:  # keep going on a bad file
                log.warning("Skipping %s: %s", f, e)
        return {"files": files, **self.ingest_documents(docs, replace=replace)}

    def ingest_texts(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> dict[str, int]:
        metadatas = metadatas or [{} for _ in texts]
        docs = [Document(t, {"source": m.get("source", f"text-{i}"), **m}) for i, (t, m) in enumerate(zip(texts, metadatas))]
        return self.ingest_documents(docs)

    def delete(self, filters: dict[str, Any]) -> int:
        n = self.store.delete(filters)
        self.store.persist()
        self.retriever.invalidate()
        return n

    # -- querying -----------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        return self.retriever.retrieve(question, top_k=top_k, filters=filters)

    def _messages(self, question: str, sources: list[ScoredChunk], history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        p = self.config.get("prompt", {})
        return build_messages(
            question,
            sources,
            system=p.get("system"),
            template=p.get("template"),
            history=history,
            max_context_chars=p.get("max_context_chars", 12000),
            history_turns=p.get("history_turns", 6),
        )

    def ask(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **gen_kwargs: Any,
    ) -> Answer:
        sources = self.retrieve(question, top_k, filters)
        text = self.llm.generate(self._messages(question, sources, history), **gen_kwargs)
        return Answer(text=text, sources=sources)

    def stream(
        self,
        question: str,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        **gen_kwargs: Any,
    ) -> tuple[Iterator[str], list[ScoredChunk]]:
        """Returns ``(token_iterator, sources)`` so callers can show sources up front."""
        sources = self.retrieve(question, top_k, filters)
        return self.llm.stream(self._messages(question, sources, history), **gen_kwargs), sources
