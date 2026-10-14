"""A tiny plugin registry so every component can be swapped from config.

Components register themselves under a *kind* (``llm``, ``embeddings``,
``vectorstore``, ``reranker``) and a *provider* name. A config block like::

    llm:
      provider: ollama
      model: qwen3:8b

is turned into ``OllamaLLM(model="qwen3:8b")``. A provider can also be a dotted
import path (``my_pkg.my_module:MyLLM``) to plug in your own class without
touching this package.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

_REGISTRY: dict[str, dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def deco(factory: Callable[..., Any]) -> Callable[..., Any]:
        _REGISTRY.setdefault(kind, {})[name] = factory
        return factory

    return deco


def available(kind: str) -> list[str]:
    return sorted(_REGISTRY.get(kind, {}))


def _import_path(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.partition(":")
    if not attr:
        module_name, _, attr = path.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def create(kind: str, spec: dict[str, Any] | None) -> Any:
    """Instantiate a component from a config block (``provider`` + kwargs)."""
    if not spec:
        return None
    spec = dict(spec)
    provider = spec.pop("provider", None)
    if provider in (None, "none", "null", False):
        return None
    factory = _REGISTRY.get(kind, {}).get(provider)
    if factory is None:
        if ":" in provider or "." in provider:
            factory = _import_path(provider)
        else:
            raise ValueError(
                f"Unknown {kind} provider {provider!r}. Available: {available(kind)} "
                "(or use a 'module.path:ClassName' import path)."
            )
    return factory(**spec)
