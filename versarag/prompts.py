"""Prompt construction. Model-agnostic: we produce chat messages and each
backend applies the model's own chat template."""

from __future__ import annotations

from pathlib import Path

from .types import ScoredChunk

DEFAULT_SYSTEM = """You are a careful assistant that answers questions using the provided context passages.
- Answer only from the context. If it does not contain the answer, say you don't know rather than guessing.
- Cite the passages you use with bracketed numbers, e.g. [1] or [2][3].
- Be concise and precise. Quote numbers, names, and definitions exactly as they appear."""

DEFAULT_TEMPLATE = """Context passages:

{context}

---
Question: {question}"""


def source_label(chunk_meta: dict[str, object]) -> str:
    src = str(chunk_meta.get("filename") or Path(str(chunk_meta.get("source", "unknown"))).name or "unknown")
    for key in ("page", "row", "item"):
        if chunk_meta.get(key) is not None:
            return f"{src}, {key} {chunk_meta[key]}"
    return src


def format_context(results: list[ScoredChunk], max_chars: int | None = 12000) -> str:
    blocks: list[str] = []
    used = 0
    for i, r in enumerate(results, start=1):
        block = f"[{i}] ({source_label(r.chunk.metadata)})\n{r.chunk.text.strip()}"
        if max_chars and used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def build_messages(
    question: str,
    results: list[ScoredChunk],
    system: str | None = None,
    template: str | None = None,
    history: list[dict[str, str]] | None = None,
    max_context_chars: int | None = 12000,
    history_turns: int = 6,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system or DEFAULT_SYSTEM}]
    if history:
        messages.extend(history[-history_turns * 2 :] if history_turns else [])
    context = format_context(results, max_context_chars) if results else "(no relevant passages found)"
    messages.append({"role": "user", "content": (template or DEFAULT_TEMPLATE).format(context=context, question=question)})
    return messages
