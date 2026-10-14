"""Command line interface.

    versarag -c configs/qwen-ollama.yaml open ./my-contracts   # remember folder + config, index it
    versarag chat                                              # no path needed from here on
    versarag ask "What does the contract say about termination?"
    versarag                                                   # same as `versarag chat`
    versarag ui                                                # web UI in your browser
    versarag -c configs/qwen-vllm.yaml serve --port 8000
    versarag providers

Any config value can be overridden: ``--set llm.model=qwen3:14b --set retrieval.top_k=8``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from . import workspace
from .config import load_config
from .prompts import source_label


def _rag(args: argparse.Namespace, config: dict[str, Any] | None = None):  # noqa: ANN202
    from .pipeline import RAG

    return RAG(config or _config(args))


def _config(args: argparse.Namespace) -> dict[str, Any]:
    overrides = list(args.set or [])
    if args.model:
        overrides.append(f"llm.model={args.model}")
    # An explicit -c wins; otherwise fall back to the config remembered by `open`.
    return load_config(args.config or workspace.active_config(), overrides)


def _workspace_config(args: argparse.Namespace) -> tuple[dict[str, Any], Path | None]:
    """Config plus the active folder. The active folder's index replaces the config's index path."""
    config = _config(args)
    folder = workspace.active_folder()
    if folder is not None:
        config["vectorstore"]["path"] = str(workspace.index_path(folder))
    return config, folder


def _print_sources(sources: list[Any]) -> None:
    if not sources:
        return
    print("\nSources:")
    for i, s in enumerate(sources, start=1):
        print(f"  [{i}] {source_label(s.chunk.metadata)}  (score {s.score:.3f})")


def cmd_open(args: argparse.Namespace) -> None:
    try:
        folder = workspace.set_active(args.folder, args.config)
    except NotADirectoryError as e:
        sys.exit(str(e))
    print(f"Active folder: {folder}")
    config, _ = _workspace_config(args)
    rag = _rag(args, config)
    stats = rag.ingest([folder])
    print(f"Indexed {stats['files']} files, {rag.store.count()} chunks.")


def cmd_where(args: argparse.Namespace) -> None:
    folder = workspace.active_folder()
    if folder is None:
        print("No active folder. Run: versarag open <folder>")
        return
    print(f"Active folder: {folder}")
    if config_file := workspace.active_config():
        print(f"Config: {config_file}")


def cmd_close(args: argparse.Namespace) -> None:
    workspace.clear_active()
    print("Closed the active folder. Its index is kept in <folder>/.versarag and is reused on the next open.")


def cmd_ingest(args: argparse.Namespace) -> None:
    if args.paths:
        config = _config(args)
    else:
        config, folder = _workspace_config(args)
        if folder is None:
            sys.exit("No paths given and no active folder. Run: versarag open <folder>")
        args.paths = [str(folder)]
    if args.reset:
        path = config["vectorstore"].get("path")
        if path and Path(path).exists():
            shutil.rmtree(path)
            print(f"Removed index at {path}")
    rag = _rag(args, config)
    stats = rag.ingest(args.paths, recursive=not args.no_recursive, replace=not args.no_replace)
    print(json.dumps(stats, indent=2))
    print(f"Index now holds {rag.store.count()} chunks.")


def _parse_filters(items: list[str] | None) -> dict[str, Any] | None:
    if not items:
        return None
    out: dict[str, Any] = {}
    for item in items:
        k, _, v = item.partition("=")
        out[k] = v
    return out


def cmd_ask(args: argparse.Namespace) -> None:
    config, _ = _workspace_config(args)
    rag = _rag(args, config)
    filters = _parse_filters(args.filter)
    if args.retrieve_only:
        for i, s in enumerate(rag.retrieve(args.question, args.top_k, filters), start=1):
            print(f"[{i}] {source_label(s.chunk.metadata)} (score {s.score:.3f})\n{s.chunk.text}\n")
        return
    if args.json:
        print(json.dumps(rag.ask(args.question, top_k=args.top_k, filters=filters).to_dict(), indent=2))
        return
    tokens, sources = rag.stream(args.question, top_k=args.top_k, filters=filters)
    for t in tokens:
        print(t, end="", flush=True)
    print()
    _print_sources(sources)


def cmd_chat(args: argparse.Namespace) -> None:
    config, folder = _workspace_config(args)
    rag = _rag(args, config)
    if folder is not None and not args.no_refresh:
        # Pick up files added or edited since the last session; unchanged files cost no embedding calls.
        stats = rag.ingest([folder])
        print(f"Refreshed {folder}: {stats['files']} files, {stats['added']} new chunks.")
    history: list[dict[str, str]] = []
    llm_cfg = rag.config["llm"]
    where = f" in {folder}" if folder is not None else ""
    print(f"Chatting with {llm_cfg.get('model') or llm_cfg.get('repo_id') or llm_cfg.get('provider')} "
          f"over {rag.store.count()} chunks{where}. Commands: /reset, /sources, /quit")
    last_sources: list[Any] = []
    while True:
        try:
            q = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/reset":
            history.clear()
            print("(history cleared)")
            continue
        if q == "/sources":
            _print_sources(last_sources)
            continue
        tokens, last_sources = rag.stream(q, history=history, top_k=args.top_k)
        print("bot> ", end="", flush=True)
        answer = []
        for t in tokens:
            answer.append(t)
            print(t, end="", flush=True)
        print()
        # Keep plain Q/A in history (not the stuffed context) to save tokens.
        history += [{"role": "user", "content": q}, {"role": "assistant", "content": "".join(answer)}]


def cmd_serve(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        sys.exit("`serve`/`ui` need `pip install versarag[server]`")
    from .server import create_app, watch_for_close

    config, folder = _workspace_config(args)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Warning: the UI and API have no login. Anyone who can reach {args.host} can read your documents.")
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"versarag UI: {url}" + (f"  (folder: {folder})" if folder else "  (no folder open yet)"))
    app = create_app(_rag(args, config), folder=folder)
    server = uvicorn.Server(uvicorn.Config(app, host=args.host, port=args.port))
    if not args.keep_running:
        watch_for_close(app, server)  # stop once the browser tab is closed
    if args.open_browser:
        import webbrowser

        webbrowser.open(url)
    server.run()


def cmd_stats(args: argparse.Namespace) -> None:
    rag = _rag(args)
    sources: dict[str, int] = {}
    for c in rag.store.all_chunks():
        s = str(c.metadata.get("source", "?"))
        sources[s] = sources.get(s, 0) + 1
    print(json.dumps({"chunks": rag.store.count(), "meta": rag.store.get_meta(), "sources": sources}, indent=2))


def cmd_providers(args: argparse.Namespace) -> None:
    from . import embeddings, llms, retrieval, vectorstores  # noqa: F401
    from .loaders import supported_extensions
    from .registry import available

    for kind in ("llm", "embeddings", "vectorstore", "reranker"):
        print(f"{kind:12s} {', '.join(available(kind))}")
    print(f"{'loaders':12s} {' '.join(supported_extensions())}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="versarag", description="Versatile RAG for open-source LLMs")
    p.add_argument("-c", "--config", help="YAML config file")
    p.add_argument("--set", action="append", metavar="KEY=VALUE", help="override a config value (repeatable)")
    p.add_argument("-m", "--model", help="shortcut for --set llm.model=...")
    p.add_argument("-v", "--verbose", action="store_true")
    # Bare `versarag` starts chat in the active folder.
    p.set_defaults(func=cmd_chat, top_k=None, no_refresh=False)
    sub = p.add_subparsers(dest="command")

    s = sub.add_parser("open", help="make a folder active: remember it, index it, and use it for chat/ask")
    s.add_argument("folder")
    s.set_defaults(func=cmd_open)

    sub.add_parser("where", help="show the active folder").set_defaults(func=cmd_where)
    sub.add_parser("close", help="forget the active folder (its index is kept)").set_defaults(func=cmd_close)

    s = sub.add_parser("ingest", help="index files or folders (default: the active folder)")
    s.add_argument("paths", nargs="*")
    s.add_argument("--reset", action="store_true", help="delete the existing index first")
    s.add_argument("--no-recursive", action="store_true")
    s.add_argument("--no-replace", action="store_true", help="keep stale chunks of changed files")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("ask", help="ask a single question")
    s.add_argument("question")
    s.add_argument("-k", "--top-k", type=int)
    s.add_argument("-f", "--filter", action="append", metavar="KEY=VALUE", help="metadata filter, e.g. filename=report.pdf")
    s.add_argument("--json", action="store_true", help="print answer + sources as JSON")
    s.add_argument("--retrieve-only", action="store_true", help="show retrieved chunks, skip the LLM")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("chat", help="interactive multi-turn chat (in the active folder)")
    s.add_argument("-k", "--top-k", type=int)
    s.add_argument("--no-refresh", action="store_true", help="don't re-index the active folder before chatting")
    s.set_defaults(func=cmd_chat)

    s = sub.add_parser("serve", help="run the web UI and REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--open", dest="open_browser", action="store_true", help="open the UI in your browser")
    s.add_argument("--keep-running", action="store_true", help="don't stop when the browser tab closes")
    s.set_defaults(func=cmd_serve)

    s = sub.add_parser("ui", help="open the web UI in your browser (same as serve --open)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--keep-running", action="store_true", help="don't stop when the browser tab closes")
    s.set_defaults(func=cmd_serve, open_browser=True)

    sub.add_parser("stats", help="show index contents").set_defaults(func=cmd_stats)
    sub.add_parser("providers", help="list available components").set_defaults(func=cmd_providers)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    args.func(args)


if __name__ == "__main__":
    main()
