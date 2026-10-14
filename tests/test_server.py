import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from versarag.server import create_app  # noqa: E402


def test_server_roundtrip(rag):
    client = TestClient(create_app(rag))
    r = client.post("/ingest/texts", json={"texts": ["Qwen and Llama are open-weight LLMs."]})
    assert r.json()["added"] == 1
    r = client.post("/query", json={"question": "Which LLMs are open-weight?"})
    assert r.status_code == 200 and r.json()["sources"]
    r = client.post("/query", json={"question": "open-weight", "stream": True})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert events[0]["type"] == "sources" and events[-1]["type"] == "done"
    assert client.get("/health").json()["chunks"] == 1
