"""REST API and web UI (FastAPI). Start with ``versarag ui`` (or ``versarag -c <config> serve``).

Web UI:
  GET  /                      single-page chat interface (versarag/static/index.html)

Workspace (the folder being chatted with):
  GET  /api/workspace         active folder, model, chunk count, indexed files
  POST /api/workspace         {"folder": "C:/docs"}  make a folder active and index it
  POST /api/workspace/refresh re-index the active folder (picks up new, changed, deleted files)

REST API:
  GET  /health
  POST /ingest        {"paths": ["./docs"]}              (paths on the server's disk)
  POST /ingest/texts  {"texts": [...], "metadatas": [...]}
  POST /retrieve      {"question": "...", "top_k": 5, "filters": {...}}
  POST /query         {"question": "...", "history": [...], "stream": false}
                       stream=true -> NDJSON lines: {"type":"sources",...} then {"type":"token","text":...}
"""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import workspace
from .pipeline import RAG
from .prompts import source_label

STATIC_DIR = Path(__file__).parent / "static"

# The page sends a heartbeat every few seconds. Browsers slow timers in background tabs
# to about once a minute, so the silence limit is generous. A closing page sends a
# "bye" first, which shuts the server down much sooner.
HEARTBEAT_TIMEOUT = 120.0
BYE_GRACE = 3.0


def should_shut_down(last: float | None, bye: float | None, now: float) -> bool:
    """True once a browser has connected and then gone away (closed, or silent for too long)."""
    if last is None:
        return False  # never opened in a browser, so nothing to wait for
    if bye is not None and now - bye >= BYE_GRACE:
        return True  # the page said goodbye and did not come back (not just a reload)
    return now - last >= HEARTBEAT_TIMEOUT


def watch_for_close(app: FastAPI, server: Any, interval: float = 1.0) -> threading.Thread:
    """Stop `server` when the UI goes away. Runs in a daemon thread."""

    def loop() -> None:
        while not server.should_exit:
            time.sleep(interval)
            seen = app.state.ui
            if should_shut_down(seen["last"], seen["bye"], time.monotonic()):
                server.should_exit = True

    thread = threading.Thread(target=loop, name="versarag-idle-watch", daemon=True)
    thread.start()
    return thread


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


class OpenFolder(BaseModel):
    folder: str


def _sources(results: list[Any]) -> list[dict[str, Any]]:
    return [{**r.to_dict(), "label": source_label(r.chunk.metadata)} for r in results]


def _display_name(source: str, folder: Path | None) -> str:
    if folder is not None:
        try:
            return str(Path(source).relative_to(folder))
        except ValueError:
            pass
    return source


def create_app(rag: RAG, folder: Path | None = None) -> FastAPI:
    app = FastAPI(title="versarag", version="0.1.0")
    # Opening a folder builds a new RAG from this config with a per-folder index.
    base_config = copy.deepcopy(rag.config)
    state: dict[str, Any] = {"rag": rag, "folder": folder}
    app.state.ui = {"last": None, "bye": None}  # monotonic times of the page's last heartbeat / goodbye

    def current() -> RAG:
        return state["rag"]

    @app.post("/api/heartbeat")
    def heartbeat() -> dict[str, bool]:
        app.state.ui.update(last=time.monotonic(), bye=None)
        return {"ok": True}

    @app.post("/api/bye")
    def bye() -> dict[str, bool]:
        app.state.ui["bye"] = time.monotonic()
        return {"ok": True}

    def workspace_info() -> dict[str, Any]:
        r = current()
        folder_path: Path | None = state["folder"]
        counts: dict[str, int] = {}
        for c in r.store.all_chunks():
            src = str(c.metadata.get("source", "?"))
            counts[src] = counts.get(src, 0) + 1
        files = [
            {"source": src, "name": _display_name(src, folder_path), "chunks": n}
            for src, n in sorted(counts.items(), key=lambda kv: _display_name(kv[0], folder_path).lower())
        ]
        llm = r.config["llm"]
        return {
            "folder": str(folder_path) if folder_path else None,
            "llm": llm.get("model") or llm.get("provider"),
            "embeddings": r.embedder.name,
            "chunks": r.store.count(),
            "files": files,
        }

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/workspace")
    def get_workspace() -> dict[str, Any]:
        return workspace_info()

    @app.post("/api/workspace")
    def open_workspace(req: OpenFolder) -> dict[str, Any]:
        if not req.folder.strip():
            raise HTTPException(400, "Enter a folder path")
        path = Path(req.folder).expanduser().resolve()
        if not path.is_dir():
            raise HTTPException(400, f"Not a folder: {path}")
        cfg = copy.deepcopy(base_config)
        cfg["vectorstore"]["path"] = str(workspace.index_path(path))
        try:
            new_rag = RAG(cfg)
        except ValueError as e:  # e.g. the index was built with a different embedder
            raise HTTPException(400, str(e)) from e
        stats = new_rag.ingest([path])
        workspace.set_active(path)  # remembered for `versarag chat` too
        state.update(rag=new_rag, folder=path)
        return {**workspace_info(), "ingest": stats}

    @app.post("/api/workspace/refresh")
    def refresh_workspace() -> dict[str, Any]:
        folder_path: Path | None = state["folder"]
        if folder_path is None:
            raise HTTPException(400, "No folder is open")
        r = current()
        stats = r.ingest([folder_path])
        # Drop chunks of files that were deleted or moved out of the folder.
        for src in {str(c.metadata.get("source", "")) for c in r.store.all_chunks()}:
            if src and Path(src).is_absolute() and folder_path in Path(src).parents and not Path(src).exists():
                r.store.delete({"source": src})
        r.store.persist()
        r.retriever.invalidate()
        return {**workspace_info(), "ingest": stats}

    @app.get("/health")
    def health() -> dict[str, Any]:
        r = current()
        return {"status": "ok", "chunks": r.store.count(), "llm": r.config["llm"].get("model"), "embeddings": r.embedder.name}

    @app.post("/ingest")
    def ingest(req: IngestPaths) -> dict[str, int]:
        try:
            return current().ingest(req.paths, recursive=req.recursive)
        except FileNotFoundError as e:
            raise HTTPException(404, f"Not found: {e}") from e

    @app.post("/ingest/texts")
    def ingest_texts(req: IngestTexts) -> dict[str, int]:
        return current().ingest_texts(req.texts, req.metadatas)

    @app.post("/retrieve")
    def retrieve(req: Query) -> dict[str, Any]:
        return {"sources": _sources(current().retrieve(req.question, req.top_k, req.filters))}

    @app.post("/query")
    def query(req: Query) -> Any:
        r = current()
        if not req.stream:
            ans = r.ask(req.question, history=req.history, top_k=req.top_k, filters=req.filters)
            return {"answer": ans.text, "sources": _sources(ans.sources)}
        tokens, sources = r.stream(req.question, history=req.history, top_k=req.top_k, filters=req.filters)

        def gen() -> Iterator[str]:
            yield json.dumps({"type": "sources", "sources": _sources(sources)}) + "\n"
            for t in tokens:
                yield json.dumps({"type": "token", "text": t}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"

        return StreamingResponse(gen(), media_type="application/x-ndjson")

    return app
