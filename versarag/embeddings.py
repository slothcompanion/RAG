"""Embedding backends. All return L2-normalised float32 arrays."""

from __future__ import annotations

import re
import zlib
from abc import ABC, abstractmethod
from typing import Any

import httpx
import numpy as np

from .registry import register


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class Embedder(ABC):
    #: A string identifying the model; stored with the index to catch mismatches.
    name: str = "embedder"

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray: ...


@register("embeddings", "sentence_transformers")
class SentenceTransformerEmbedder(Embedder):
    """Any sentence-transformers model from the HF Hub.

    Good open choices: ``BAAI/bge-small-en-v1.5`` (fast, CPU-friendly),
    ``BAAI/bge-m3`` (multilingual), ``Qwen/Qwen3-Embedding-0.6B`` (use
    ``query_prompt_name: query``), ``nomic-ai/nomic-embed-text-v1.5`` (use
    ``query_prefix: "search_query: "`` / ``document_prefix: "search_document: "``,
    ``trust_remote_code: true``).
    """

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
        batch_size: int = 32,
        query_prefix: str = "",
        document_prefix: str = "",
        query_prompt_name: str | None = None,
        trust_remote_code: bool = False,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence_transformers embeddings need `pip install sentence-transformers` "
                "(or versarag[embeddings])"
            ) from e
        self.name = f"sentence_transformers:{model}"
        self.model = SentenceTransformer(
            model, device=device, trust_remote_code=trust_remote_code, model_kwargs=model_kwargs
        )
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.query_prompt_name = query_prompt_name

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        texts = [self.document_prefix + t for t in texts]
        vecs = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 256
        )
        return _normalize(vecs)

    def embed_query(self, text: str) -> np.ndarray:
        kwargs = {"prompt_name": self.query_prompt_name} if self.query_prompt_name else {}
        vec = self.model.encode([self.query_prefix + text], normalize_embeddings=True, **kwargs)
        return _normalize(vec)[0]


@register("embeddings", "openai_compatible")
class OpenAICompatibleEmbedder(Embedder):
    """Any server exposing ``POST /v1/embeddings``: Ollama, vLLM, LM Studio,
    llama.cpp server, HF text-embeddings-inference, LocalAI, ..."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        batch_size: int = 64,
        query_prefix: str = "",
        document_prefix: str = "",
        timeout: float = 120.0,
    ) -> None:
        self.name = f"openai_compatible:{model}"
        self.model = model
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)

    def _embed(self, texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            r = self.client.post("/embeddings", json={"model": self.model, "input": batch})
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d.get("index", 0))
            out.extend(d["embedding"] for d in data)
        return _normalize(np.array(out))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([self.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_prefix + text])[0]


@register("embeddings", "hashing")
class HashingEmbedder(Embedder):
    """Dependency-free lexical embedder (word + char-trigram feature hashing).

    No semantic understanding — useful for tests, CI, and trying the pipeline
    without downloading a model.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing:{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = re.findall(r"\w+", text.lower())
        feats = words + [w[i : i + 3] for w in words for i in range(max(1, len(w) - 2))]
        for f in feats:
            h = zlib.crc32(f.encode())
            v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalize(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalize(self._vec(text))
