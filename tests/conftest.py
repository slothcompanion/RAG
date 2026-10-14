import pytest

from versarag import RAG


@pytest.fixture
def rag(tmp_path):
    return RAG.from_config(
        {
            "llm": {"provider": "echo"},
            "embeddings": {"provider": "hashing", "dim": 512},
            "vectorstore": {"provider": "numpy", "path": str(tmp_path / "idx")},
            "chunking": {"chunk_size": 300, "chunk_overlap": 50},
        }
    )
