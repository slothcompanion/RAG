from pathlib import Path

import pytest

from versarag import RAG, load_config
from versarag.config import apply_overrides
from versarag.retrieval import BM25, reciprocal_rank_fusion
from versarag.types import Chunk, ScoredChunk

DOCS = Path(__file__).parent.parent / "examples" / "docs"


def test_ingest_and_retrieve(rag):
    stats = rag.ingest([DOCS])
    assert stats["files"] == 2 and stats["added"] > 0
    hits = rag.retrieve("refund within 30 days", top_k=3)
    assert "refund" in hits[0].chunk.text.lower()


def test_reingest_is_idempotent(rag):
    rag.ingest([DOCS])
    n = rag.store.count()
    stats = rag.ingest([DOCS])
    assert stats["added"] == 0 and rag.store.count() == n


def test_changed_file_replaces_old_chunks(rag, tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("The launch code is alpha.")
    rag.ingest([f])
    f.write_text("The launch code is bravo.")
    rag.ingest([f])
    texts = [c.text for c in rag.store.all_chunks()]
    assert texts == ["The launch code is bravo."]


def test_persistence_roundtrip(rag):
    rag.ingest([DOCS])
    again = RAG(rag.config)
    assert again.store.count() == rag.store.count()
    assert again.retrieve("warranty hammock")[0].chunk.metadata["filename"] == "products.csv"


def test_embedder_mismatch_detected(rag):
    rag.ingest([DOCS])
    cfg = apply_overrides(rag.config, ["embeddings.dim=256"])
    with pytest.raises(ValueError, match="built with embeddings"):
        RAG(cfg)


def test_filters(rag):
    rag.ingest([DOCS])
    hits = rag.retrieve("price", filters={"filename": "products.csv"})
    assert hits and all(h.chunk.metadata["filename"] == "products.csv" for h in hits)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_modes(rag, mode):
    rag.ingest([DOCS])
    rag.retriever.mode = mode
    assert rag.retrieve("support hours Pacific")


def test_ask_includes_context_and_sources(rag):
    rag.ingest([DOCS])
    ans = rag.ask("When is support available?")
    assert ans.text.startswith("[echo]") and "Support" in ans.text
    assert ans.sources


def test_stream(rag):
    rag.ingest([DOCS])
    tokens, sources = rag.stream("data retention")
    assert "".join(tokens).startswith("[echo]") and sources


def test_ingest_texts_and_delete(rag):
    rag.ingest_texts(["Sloths sleep 15 hours a day."], [{"source": "facts", "topic": "sloths"}])
    assert rag.retrieve("how long do sloths sleep")[0].chunk.metadata["topic"] == "sloths"
    assert rag.delete({"source": "facts"}) == 1 and rag.store.count() == 0


def test_bm25_and_rrf():
    chunks = [Chunk(str(i), t) for i, t in enumerate(["apple pie recipe", "car engine repair", "apple orchard"])]
    res = BM25(chunks).search("apple", 5)
    assert {r.chunk.id for r in res} == {"0", "2"}
    fused = reciprocal_rank_fusion([[ScoredChunk(chunks[0], 1)], [ScoredChunk(chunks[1], 1), ScoredChunk(chunks[0], 1)]])
    assert fused[0].chunk.id == "0"


def test_config_overrides_and_provider_switch():
    cfg = load_config(None, ["llm.model=qwen3:8b", "retrieval.top_k=9"])
    assert cfg["llm"]["model"] == "qwen3:8b" and cfg["retrieval"]["top_k"] == 9
    cfg = load_config(None, ["llm.provider=openai_compatible", "llm.model=Qwen/Qwen3-8B"])
    assert cfg["llm"] == {"provider": "openai_compatible", "model": "Qwen/Qwen3-8B"}
    cfg = load_config({"llm": {"provider": "echo"}})
    assert cfg["llm"] == {"provider": "echo"}


def test_env_expansion(monkeypatch):
    monkeypatch.setenv("MY_HOST", "http://gpu-box:11434")
    cfg = load_config({"llm": {"base_url": "${MY_HOST}", "x": "${NOPE:-fallback}"}})
    assert cfg["llm"]["base_url"] == "http://gpu-box:11434" and cfg["llm"]["x"] == "fallback"


def test_custom_provider_by_import_path(tmp_path):
    cfg = {
        "llm": {"provider": "versarag.llms:EchoLLM"},
        "embeddings": {"provider": "hashing"},
        "vectorstore": {"provider": "numpy", "path": None},
    }
    rag = RAG.from_config(cfg)
    rag.ingest_texts(["hello"])
    assert rag.ask("hello").text.startswith("[echo]")


@pytest.mark.parametrize("name", sorted(p.name for p in (Path(__file__).parent.parent / "configs").glob("*.yaml")))
def test_shipped_configs_parse(name):
    cfg = load_config(Path(__file__).parent.parent / "configs" / name)
    assert cfg["llm"]["provider"] and cfg["embeddings"]["provider"]
