"""HTTP backends tested against mocked servers (no real model needed)."""

import json

import httpx

from versarag.embeddings import OpenAICompatibleEmbedder
from versarag.llms import OllamaLLM, OpenAICompatibleLLM

MSGS = [{"role": "user", "content": "hi"}]


def _mock(llm, handler):
    llm.client = httpx.Client(base_url=str(llm.client.base_url), transport=httpx.MockTransport(handler))
    return llm


def test_openai_compatible_generate_and_stream():
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen.update(body)
        if body["stream"]:
            lines = [
                'data: {"choices":[{"delta":{"content":"<think>x</think>"}}]}',
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                "data: [DONE]",
            ]
            return httpx.Response(200, text="\n\n".join(lines))
        return httpx.Response(200, json={"choices": [{"message": {"content": "<think>x</think>Hello"}}]})

    llm = _mock(
        OpenAICompatibleLLM(model="Qwen/Qwen3-8B", extra_body={"chat_template_kwargs": {"enable_thinking": False}}),
        handler,
    )
    assert llm.generate(MSGS) == "Hello"
    assert seen["chat_template_kwargs"] == {"enable_thinking": False}
    assert "".join(llm.stream(MSGS)) == "Hello"


def test_ollama_native():
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen.update(body)
        if body["stream"]:
            lines = [json.dumps({"message": {"content": c}, "done": False}) for c in ["Lla", "ma"]]
            lines.append(json.dumps({"done": True}))
            return httpx.Response(200, text="\n".join(lines))
        return httpx.Response(200, json={"message": {"content": "Llama"}})

    llm = _mock(OllamaLLM(model="llama3.1:8b", num_ctx=16384, think=False), handler)
    assert llm.generate(MSGS) == "Llama"
    assert seen["options"]["num_ctx"] == 16384 and seen["think"] is False
    assert "".join(llm.stream(MSGS)) == "Llama"


def test_openai_compatible_embeddings():
    def handler(req):
        inputs = json.loads(req.content)["input"]
        return httpx.Response(200, json={"data": [{"index": i, "embedding": [3.0, 4.0]} for i in range(len(inputs))]})

    emb = _mock(OpenAICompatibleEmbedder(model="nomic-embed-text", query_prefix="search_query: "), handler)
    vecs = emb.embed_documents(["a", "b", "c"])
    assert vecs.shape == (3, 2) and abs(vecs[0][0] - 0.6) < 1e-6
