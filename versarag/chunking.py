"""Recursive text splitter: tries paragraph, line, sentence, then word breaks.

Sizes are in characters (roughly 4 characters per token for English).
"""

from __future__ import annotations

import hashlib

from .types import Chunk, Document

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    sep = next((s for s in separators if s and s in text), "")
    if sep == "":
        return [text[i : i + size] for i in range(0, len(text), size)]
    rest = separators[separators.index(sep) + 1 :]
    raw = text.split(sep)
    parts = [p + sep for p in raw[:-1]] + [raw[-1]]
    out: list[str] = []
    for p in parts:
        if not p:
            continue
        out.extend([p] if len(p) <= size else _split_recursive(p, size, rest))
    return out


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in pieces:
        if cur and cur_len + len(p) > size:
            chunks.append("".join(cur).strip())
            # Keep a tail of the previous chunk as overlap context.
            while cur and (cur_len > overlap or cur_len + len(p) > size):
                cur_len -= len(cur.pop(0))
        cur.append(p)
        cur_len += len(p)
    if cur:
        chunks.append("".join(cur).strip())
    return [c for c in chunks if c]


def split_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[str]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    pieces = _split_recursive(text, chunk_size, separators or DEFAULT_SEPARATORS)
    return _merge(pieces, chunk_size, chunk_overlap)


def chunk_id(source: str, text: str) -> str:
    return hashlib.sha1(f"{source}\x00{text}".encode()).hexdigest()[:20]


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    separators: list[str] | None = None,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        source = str(doc.metadata.get("source", ""))
        for i, piece in enumerate(split_text(doc.text, chunk_size, chunk_overlap, separators)):
            meta = {**doc.metadata, "chunk": i}
            loc = f"{source}|{doc.metadata.get('page', '')}|{doc.metadata.get('row', '')}|{doc.metadata.get('item', '')}"
            chunks.append(Chunk(id=chunk_id(loc, piece), text=piece, metadata=meta))
    return chunks
