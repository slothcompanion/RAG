"""REST API (FastAPI). Start with ``versarag -c <config> serve``.

Endpoints:
  GET  /health
  POST /ingest        {"paths": ["./docs"]}              (paths on the server's disk)
  POST /ingest/texts  {"texts": [...], "metadatas": [...]}
  POST /retrieve      {"question": "...", "top_k": 5, "filters": {...}}
  POST /query         {"question": "...", "history": [...], "stream": false}
                       stream=true -> NDJSON lines: {"type":"sources",...} then {"type":"token","text":...}
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .pipeline import RAG
from .prompts import source_label


class IngestPaths(BaseModel):
    paths: list[str]
    recursive: bool = True


class IngestTexts(BaseModel):
    texts: list[str]
    metadatas: list[dict[str, Any]] | None = None


class Query(BaseModel):
    question: str
    top_k: int | None = None
    filters: dict[str, Any] | None = None
    history: list[dict[str, str]] = Field(default_factory=list)
    stream: bool = False


def _sources(results: list[Any]) -> list[dict[str, Any]]:
    return [{**r.to_dict(), "label": source_label(r.chunk.metadata)} for r in results]


def create_app(rag: RAG) -> FastAPI:
    app = FastAPI(title="versarag", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "chunks": rag.store.count(), "llm": rag.config["llm"].get("model"), "embeddings": rag.embedder.name}

    @app.post("/ingest")
    def ingest(req: IngestPaths) -> dict[str, int]:
        try:
            return rag.ingest(req.paths, recursive=req.recursive)
        except FileNotFoundError as e:
            raise HTTPException(404, f"Not found: {e}") from e

    @app.post("/ingest/texts")
    def ingest_texts(req: IngestTexts) -> dict[str, int]:
        return rag.ingest_texts(req.texts, req.metadatas)

    @app.post("/retrieve")
    def retrieve(req: Query) -> dict[str, Any]:
        return {"sources": _sources(rag.retrieve(req.question, req.top_k, req.filters))}

    @app.post("/query")
    def query(req: Query) -> Any:
        if not req.stream:
            ans = rag.ask(req.question, history=req.history, top_k=req.top_k, filters=req.filters)
            return {"answer": ans.text, "sources": _sources(ans.sources)}
        tokens, sources = rag.stream(req.question, history=req.history, top_k=req.top_k, filters=req.filters)

        def gen() -> Iterator[str]:
            yield json.dumps({"type": "sources", "sources": _sources(sources)}) + "\n"
            for t in tokens:
                yield json.dumps({"type": "token", "text": t}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app
