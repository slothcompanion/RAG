"""Remember one active folder (and config) so commands don't need them on every call.

The state lives in ``~/.versarag/workspace.json`` (override the directory with the
``VERSARAG_HOME`` environment variable). Each folder keeps its own index in
``<folder>/.versarag/index``, so switching folders never mixes documents.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

HOME_ENV = "VERSARAG_HOME"
INDEX_DIRNAME = ".versarag"


def _state_file() -> Path:
    home = os.environ.get(HOME_ENV)
    base = Path(home) if home else Path.home() / ".versarag"
    return base / "workspace.json"


def get_state() -> dict[str, Any]:
    try:
        data = json.loads(_state_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def active_folder() -> Path | None:
    folder = get_state().get("folder")
    return Path(folder) if folder else None


def active_config() -> str | None:
    return get_state().get("config") or None


def set_active(folder: str | Path, config: str | Path | None = None) -> Path:
    """Make ``folder`` the active workspace. A ``config`` given here replaces the remembered one."""
    path = Path(folder).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"Not a folder: {path}")
    state = {**get_state(), "folder": str(path)}
    if config:
        state["config"] = str(Path(config).expanduser().resolve())
    _state_file().parent.mkdir(parents=True, exist_ok=True)
    _state_file().write_text(json.dumps(state, indent=2), encoding="utf-8")
    return path


def clear_active() -> None:
    state = get_state()
    state.pop("folder", None)
    if state:
        _state_file().write_text(json.dumps(state, indent=2), encoding="utf-8")
    else:
        _state_file().unlink(missing_ok=True)


def index_path(folder: str | Path) -> Path:
    return Path(folder) / INDEX_DIRNAME / "index"
