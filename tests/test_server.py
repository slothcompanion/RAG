import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from versarag import workspace  # noqa: E402
from versarag.server import create_app  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    # Opening a folder writes the workspace state; keep that out of the real home directory.
    monkeypatch.setenv(workspace.HOME_ENV, str(tmp_path / "home"))


def test_ui_page_is_served(rag):
    r = TestClient(create_app(rag)).get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "versarag" in r.text


def test_workspace_without_folder(rag):
    ws = TestClient(create_app(rag)).get("/api/workspace").json()
    assert ws["folder"] is None and ws["files"] == []


def test_open_folder_indexes_and_refresh_prunes_deleted(rag, tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.txt").write_text("The launch code is alpha.")
    (folder / "b.txt").write_text("The budget is ten thousand dollars.")
    client = TestClient(create_app(rag))

    r = client.post("/api/workspace", json={"folder": str(folder)})
    assert r.status_code == 200
    ws = r.json()
    assert ws["folder"] == str(folder.resolve()) and ws["ingest"]["files"] == 2
    assert {f["name"] for f in ws["files"]} == {"a.txt", "b.txt"}

    (folder / "b.txt").unlink()
    ws = client.post("/api/workspace/refresh").json()
    assert [f["name"] for f in ws["files"]] == ["a.txt"]
    assert ws["chunks"] == 1


def test_shutdown_rules():
    from versarag.server import BYE_GRACE, HEARTBEAT_TIMEOUT, should_shut_down

    assert not should_shut_down(None, None, 1000.0)  # never opened in a browser
    assert not should_shut_down(100.0, None, 100.0 + HEARTBEAT_TIMEOUT - 1)  # page still alive
    assert should_shut_down(100.0, None, 100.0 + HEARTBEAT_TIMEOUT)  # silent too long
    assert should_shut_down(100.0, 100.0, 100.0 + BYE_GRACE)  # said goodbye and did not return
    assert not should_shut_down(100.0, 100.0, 100.0 + BYE_GRACE - 0.5)  # reload grace period


def test_heartbeat_and_bye_update_state(rag):
    app = create_app(rag)
    client = TestClient(app)
    assert client.post("/api/heartbeat").json() == {"ok": True}
    assert app.state.ui["last"] is not None and app.state.ui["bye"] is None
    client.post("/api/bye")
    assert app.state.ui["bye"] is not None
    client.post("/api/heartbeat")  # the page came back (reload), so the goodbye is cancelled
    assert app.state.ui["bye"] is None


def test_open_rejects_missing_folder(rag, tmp_path):
    r = TestClient(create_app(rag)).post("/api/workspace", json={"folder": str(tmp_path / "nope")})
    assert r.status_code == 400


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
