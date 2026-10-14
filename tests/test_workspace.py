from pathlib import Path

import pytest

from versarag import cli, workspace


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv(workspace.HOME_ENV, str(tmp_path / "home"))
    return tmp_path / "home"


def test_set_and_get_active(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    workspace.set_active(folder, "configs/offline-demo.yaml")
    assert workspace.active_folder() == folder.resolve()
    assert Path(workspace.active_config()).is_absolute()


def test_set_active_rejects_missing_folder(tmp_path):
    with pytest.raises(NotADirectoryError):
        workspace.set_active(tmp_path / "nope")


def test_open_without_config_keeps_remembered_config(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    workspace.set_active(folder, "configs/offline-demo.yaml")
    workspace.set_active(folder)
    assert workspace.active_config() is not None


def test_clear_active_keeps_config_and_index(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    workspace.set_active(folder, "configs/offline-demo.yaml")
    workspace.clear_active()
    assert workspace.active_folder() is None
    assert workspace.active_config() is not None


def test_index_lives_inside_folder(tmp_path):
    assert workspace.index_path(tmp_path) == tmp_path / ".versarag" / "index"


def test_workspace_config_points_index_at_folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    workspace.set_active(folder)
    args = cli.build_parser().parse_args(["chat"])
    config, active = cli._workspace_config(args)
    assert active == folder.resolve()
    assert config["vectorstore"]["path"] == str(workspace.index_path(folder.resolve()))


def test_bare_command_routes_to_chat():
    args = cli.build_parser().parse_args([])
    assert args.func is cli.cmd_chat
    assert args.command is None


def test_ingest_without_paths_or_workspace_exits(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["ingest"])
    assert "open" in str(exc.value)
