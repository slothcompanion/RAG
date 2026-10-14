"""Configuration loading: YAML file + defaults + ``key.path=value`` overrides.

String values may reference environment variables as ``${VAR}`` or
``${VAR:-default}``.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model": "llama3.1:8b",
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
        "num_ctx": 8192,
    },
    "embeddings": {
        "provider": "sentence_transformers",
        "model": "BAAI/bge-small-en-v1.5",
    },
    "vectorstore": {
        "provider": "numpy",
        "path": ".rag_index",
    },
    "chunking": {
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
    "retrieval": {
        "mode": "hybrid",  # dense | sparse | hybrid
        "top_k": 5,
        "candidate_k": 25,
        "rrf_k": 60,
        "min_score": None,
    },
    "reranker": {"provider": "none"},
    "prompt": {
        "system": None,  # None -> built-in default
        "template": None,
        "max_context_chars": 12000,
        "history_turns": 6,
    },
}

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        def repl(m: re.Match[str]) -> str:
            return os.environ.get(m.group(1), m.group(2) if m.group(2) is not None else "")

        return _ENV_RE.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        # Switching provider means the old provider's kwargs no longer apply.
        if (
            isinstance(v, dict)
            and isinstance(out.get(k), dict)
            and not ("provider" in v and v.get("provider") != out[k].get("provider"))
        ):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply CLI-style overrides, e.g. ``["llm.model=qwen3:8b", "retrieval.top_k=8"]``."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must look like key.path=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw) if raw else ""
        node = config
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        if parts[-1] == "provider" and node.get("provider") != value:
            # A new provider starts from a clean block; later overrides refill it.
            node.clear()
        node[parts[-1]] = value
    return config


def load_config(
    path: str | Path | dict[str, Any] | None = None,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    if isinstance(path, dict):
        user = path
    elif path:
        with open(path, encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
    else:
        user = {}
    config = deep_merge(DEFAULT_CONFIG, user)
    config = apply_overrides(config, overrides)
    return _expand_env(config)
