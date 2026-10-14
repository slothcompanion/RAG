# versarag

A versatile, model-agnostic RAG (retrieval-augmented generation) toolkit for **open-source LLMs**. Built and tested with **Llama** and **Qwen** in mind, and works with any chat model: Mistral, Gemma, Phi, DeepSeek, and others.

Every component is swappable from a YAML file. You don't need to change any code to move between them:

| Component | Built-in options |
|---|---|
| **LLM runtime** | `ollama` · `openai_compatible` (vLLM, llama.cpp server, LM Studio, SGLang, TGI, LocalAI, OpenRouter…) · `huggingface` (transformers, optional 4/8-bit) · `llama_cpp` (in-process GGUF) · `echo` (no model) |
| **Embeddings** | `sentence_transformers` (BGE, E5, nomic, Qwen3-Embedding…) · `openai_compatible` (Ollama, vLLM, TEI…) · `hashing` (offline) |
| **Vector store** | `numpy` (zero-dependency, persisted to disk) · `chroma` |
| **Retrieval** | `dense` · `sparse` (BM25) · `hybrid` (Reciprocal Rank Fusion of both, the default) |
| **Reranker** | `cross_encoder` (e.g. `BAAI/bge-reranker-v2-m3`) |
| **Loaders** | txt/md/rst, code, R/Rmd/qmd, PDF (per page), DOCX, HTML, CSV/TSV (per row), JSON/JSONL |

You can also plug in your own class by setting `provider: my_module:MyLLM` in the config.

## Install

```bash
git clone <this repo> && cd RAG
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # core: numpy, pyyaml, httpx
pip install -e ".[docs,server]"        # + PDF/DOCX loaders and the REST API
# Optional, depending on backend:
#   ".[embeddings]"   sentence-transformers embeddings and rerankers
#   ".[transformers]" run HF models in-process   (+ ".[quant]" for 4/8-bit)
#   ".[llamacpp]"     run GGUF files in-process
#   ".[chroma]"       Chroma vector store
```

## Quick start (Ollama, the easiest route for Llama and Qwen)

```bash
ollama pull qwen3:8b && ollama pull llama3.1:8b && ollama pull nomic-embed-text

versarag -c configs/qwen-ollama.yaml ingest ./my_docs
versarag -c configs/qwen-ollama.yaml ask "What is the refund policy?"
versarag -c configs/llama-ollama.yaml chat          # same index, different model
```

Both Ollama configs use the same embedder, so they share one index. Ingest once, then compare Llama and Qwen answers side by side. You can switch models on the fly:

```bash
versarag -c configs/qwen-ollama.yaml -m qwen3:14b ask "..."
versarag -c configs/qwen-ollama.yaml --set retrieval.top_k=10 --set llm.think=true ask "..."
```

To try the pipeline with no models at all, run `versarag -c configs/offline-demo.yaml ingest examples/docs`. That config uses the `echo` LLM, which prints the exact prompt a real model would receive.

## Shipped configs

| Config | LLM | Embeddings | Use when |
|---|---|---|---|
| `llama-ollama.yaml` | Llama 3.1 8B via Ollama | nomic-embed-text (Ollama) | Laptop/desktop. No Python ML dependencies |
| `qwen-ollama.yaml` | Qwen3 8B via Ollama (thinking off) | nomic-embed-text (Ollama) | Same as above |
| `qwen-vllm.yaml` | Qwen3 8B on vLLM | Qwen3-Embedding-0.6B + bge-reranker-v2-m3 | GPU server, best quality and throughput |
| `llama-transformers.yaml` | Llama 3.1 8B Instruct, 4-bit, in-process | bge-small-en-v1.5 | Single GPU, no server process |
| `qwen-transformers.yaml` | Qwen3 4B in-process | Qwen3-Embedding-0.6B | Same as above |
| `gguf-llamacpp.yaml` | Any GGUF (Qwen3 8B Q4_K_M by default) | bge-small-en-v1.5 | CPU or Apple Silicon |
| `offline-demo.yaml` | `echo` | `hashing` | Tests and dry runs |

## Python API

```python
from versarag import RAG

rag = RAG.from_config("configs/qwen-ollama.yaml")
rag.ingest(["./contracts", "notes.md"])          # re-ingesting is incremental

ans = rag.ask("What is the termination notice period?", filters={"filename": "msa.pdf"})
print(ans.text)                                  # cites passages as [1], [2]…
for s in ans.sources:
    print(s.chunk.metadata["source"], s.chunk.metadata.get("page"), s.score)

tokens, sources = rag.stream("Summarize section 4", history=[...])
for t in tokens: print(t, end="")

rag.ingest_texts(["free text..."], [{"source": "crm", "account": "ACME"}])
rag.delete({"source": "crm"})
```

## REST API (and calling it from R)

```bash
versarag -c configs/qwen-ollama.yaml serve --port 8000
```

| Endpoint | Body |
|---|---|
| `GET /health` | |
| `POST /ingest` | `{"paths": ["./docs"]}` (paths on the server) |
| `POST /ingest/texts` | `{"texts": [...], "metadatas": [...]}` |
| `POST /retrieve` | `{"question": "...", "top_k": 5, "filters": {...}}` |
| `POST /query` | `{"question": "...", "history": [...], "stream": false}`. With `stream: true` it returns NDJSON events (`sources`, then `token`s, then `done`) |

From R / RStudio with `httr2`:

```r
library(httr2)

ask_rag <- function(question, top_k = 5, filters = NULL) {
  request("http://127.0.0.1:8000/query") |>
    req_body_json(list(question = question, top_k = top_k, filters = filters)) |>
    req_perform() |>
    resp_body_json()
}

res <- ask_rag("What is the refund policy?")
cat(res$answer)
do.call(rbind, lapply(res$sources, \(s) data.frame(source = s$label, score = s$score)))
```

## Configuration reference

Any key can be set in YAML or with `--set key.path=value`. `${ENV_VAR:-default}` is expanded. Defaults are in `versarag/config.py`.

```yaml
llm:          { provider: ollama, model: llama3.1:8b, num_ctx: 8192, temperature: 0.2 }
embeddings:   { provider: sentence_transformers, model: BAAI/bge-small-en-v1.5 }
vectorstore:  { provider: numpy, path: .rag_index }
chunking:     { chunk_size: 1000, chunk_overlap: 150 }      # characters (~4 per token)
retrieval:    { mode: hybrid, top_k: 5, candidate_k: 25, rrf_k: 60, min_score: null }
reranker:     { provider: none }                             # or cross_encoder
prompt:       { system: null, template: null, max_context_chars: 12000, history_turns: 6 }
```

If you set `--set llm.provider=...`, the rest of that block is cleared, so add the new provider's options after it.

## Notes for Llama and Qwen

- **Chat templates are handled for you.** Every backend sends role-based messages, and the server or tokenizer applies the model's own template. Moving from Llama to Qwen never requires prompt-format changes.
- **Qwen3 thinking.** `<think>…</think>` blocks are stripped from answers, including mid-stream. To turn thinking off at the source: `think: false` (Ollama), `extra_body.chat_template_kwargs.enable_thinking: false` (vLLM/SGLang), or `chat_template_kwargs: {enable_thinking: false}` (transformers). If a model's template pre-opens `<think>` in the prompt, set `thinking_prefilled: true`.
- **Ollama context length.** Ollama's default context window is small and silently truncates long RAG prompts. The `ollama` provider sets `num_ctx` (8–16k in the shipped configs).
- **Llama weights on Hugging Face are gated.** Accept the license on the model page, then run `huggingface-cli login` or set `HF_TOKEN`.
- **Changing embedding models** requires a new index. versarag records the embedder in the index and refuses to mix them. Use a different `vectorstore.path`, or run `ingest --reset`.

## Extending

```python
from versarag import register
from versarag.llms import LLM

@register("llm", "my_backend")
class MyLLM(LLM):
    def __init__(self, model, **kw): ...
    def _generate(self, messages, **kw) -> str: ...
    def _stream(self, messages, **kw): yield ...
```

Register embedders (`Embedder`), vector stores (`VectorStore`), rerankers, and file loaders (`@register_loader(".ext")`) the same way.

## Development

```bash
pip install -e ".[dev,server]" && pytest     # runs fully offline with mocked model servers
```
