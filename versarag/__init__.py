"""versarag — a versatile RAG toolkit for open-source LLMs."""

from .config import load_config
from .pipeline import RAG
from .registry import available, create, register
from .types import Answer, Chunk, Document, ScoredChunk

__all__ = ["RAG", "load_config", "register", "create", "available", "Answer", "Chunk", "Document", "ScoredChunk"]
__version__ = "0.1.0"
