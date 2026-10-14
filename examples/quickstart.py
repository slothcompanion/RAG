"""Minimal Python usage. Run from the repo root:

    python examples/quickstart.py configs/qwen-ollama.yaml
"""

import sys

from versarag import RAG

config = sys.argv[1] if len(sys.argv) > 1 else "configs/offline-demo.yaml"
rag = RAG.from_config(config)
print(rag.ingest(["examples/docs"]))

answer = rag.ask("How long do customers have to get a full refund?")
print(answer.text)
for i, s in enumerate(answer.sources, 1):
    print(f"[{i}] {s.chunk.metadata.get('filename')}  score={s.score:.3f}")

# Swap the model without touching the index (same embeddings):
#   rag = RAG.from_config("configs/qwen-ollama.yaml", overrides=["llm.model=llama3.1:8b"])
