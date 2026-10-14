"""Dense, sparse (BM25) and hybrid retrieval with optional reranking."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from .embeddings import Embedder
from .registry import register
from .types import Chunk, ScoredChunk
from .vectorstores import VectorStore, matches

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Okapi BM25 over the chunks in the store (built in memory, rebuilt on change)."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(c.text)) for c in chunks]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        q = [t for t in tokenize(query) if t in self.idf]
        if not q:
            return []
        scored: list[ScoredChunk] = []
        for chunk, tf, dl in zip(self.chunks, self.tfs, self.lens):
            if filters and not matches(chunk.metadata, filters):
                continue
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            if s > 0:
                scored.append(ScoredChunk(chunk, s))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:k]


def reciprocal_rank_fusion(result_lists: list[list[ScoredChunk]], k: int = 60) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for results in result_lists:
        for rank, sc in enumerate(results):
            by_id[sc.chunk.id] = sc.chunk
            scores[sc.chunk.id] = scores.get(sc.chunk.id, 0.0) + 1.0 / (k + rank + 1)
    fused = [ScoredChunk(by_id[i], s) for i, s in scores.items()]
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused


class Reranker:
    def rerank(self, query: str, results: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        raise NotImplementedError


@register("reranker", "cross_encoder")
class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker, e.g. ``BAAI/bge-reranker-v2-m3`` or ``BAAI/bge-reranker-base``."""

    def __init__(self, model: str = "BAAI/bge-reranker-base", device: str | None = None,
                 batch_size: int = 16, max_length: int = 512, trust_remote_code: bool = False) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:  # pragma: no cover
            raise ImportError("cross_encoder reranker needs `pip install sentence-transformers`") from e
        self.model = CrossEncoder(model, device=device, max_length=max_length, trust_remote_code=trust_remote_code)
        self.batch_size = batch_size

    def rerank(self, query: str, results: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        if not results:
            return []
        scores = self.model.predict([(query, r.chunk.text) for r in results], batch_size=self.batch_size)
        out = [ScoredChunk(r.chunk, float(s)) for r, s in zip(results, scores)]
        out.sort(key=lambda x: x.score, reverse=True)
        return out[:top_k]


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        mode: str = "hybrid",
        top_k: int = 5,
        candidate_k: int = 25,
        rrf_k: int = 60,
        min_score: float | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        if mode not in ("dense", "sparse", "hybrid"):
            raise ValueError("retrieval.mode must be one of: dense, sparse, hybrid")
        self.store, self.embedder, self.mode = store, embedder, mode
        self.top_k, self.candidate_k, self.rrf_k = top_k, candidate_k, rrf_k
        self.min_score, self.reranker = min_score, reranker
        self._bm25: BM25 | None = None
        self._bm25_count = -1

    def invalidate(self) -> None:
        self._bm25 = None

    def _sparse(self) -> BM25:
        n = self.store.count()
        if self._bm25 is None or self._bm25_count != n:
            self._bm25 = BM25(self.store.all_chunks())
            self._bm25_count = n
        return self._bm25

    def retrieve(self, query: str, top_k: int | None = None, filters: dict[str, Any] | None = None) -> list[ScoredChunk]:
        top_k = top_k or self.top_k
        # Pull a wider pool when we will fuse or rerank, then cut down.
        pool = max(self.candidate_k, top_k) if (self.mode == "hybrid" or self.reranker) else top_k
        lists: list[list[ScoredChunk]] = []
        if self.mode in ("dense", "hybrid"):
            dense = self.store.search(self.embedder.embed_query(query), pool, filters)
            if self.min_score is not None:
                dense = [d for d in dense if d.score >= self.min_score]
            lists.append(dense)
        if self.mode in ("sparse", "hybrid"):
            lists.append(self._sparse().search(query, pool, filters))
        results = reciprocal_rank_fusion(lists, self.rrf_k) if len(lists) > 1 else lists[0]
        if self.reranker:
            return self.reranker.rerank(query, results[:pool], top_k)
        return results[:top_k]
