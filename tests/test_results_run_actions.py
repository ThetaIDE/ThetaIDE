"""Run archives and deletion cover all results roots and reject unsafe paths."""
import zipfile

import pytest

from src.app.services import catalog


@pytest.mark.parametrize("run_id", ["run", "path/dir/run"])
def test_archive_and_delete_run(tmp_path, monkeypatch, run_id):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    for root in ("logs", "plots", "checkpoints"):
        folder = tmp_path / root / "group" / run_id
        folder.mkdir(parents=True)
        (folder / "data.txt").write_text(root)
    other = tmp_path / "logs" / "group" / "other"
    other.mkdir()
    archive = tmp_path / "run.zip"
    catalog.archive_run("group", run_id, str(archive))
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {f"{root}/group/{run_id}/data.txt" for root in ("logs", "plots", "checkpoints")}
        assert zipped.read(f"plots/group/{run_id}/data.txt") == b"plots"
    catalog.delete_run("group", run_id)
    assert all(not (tmp_path / root / "group" / run_id).exists() for root in ("logs", "plots", "checkpoints"))
    assert other.exists() and archive.exists()


def test_run_actions_reject_unsafe_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "logs" / "group").mkdir(parents=True)
    (tmp_path / "logs" / "group" / "run").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        catalog.delete_run("group", "run")
    with pytest.raises(ValueError):
        catalog.archive_run("group", "run", str(tmp_path / "run.zip"))
    with pytest.raises(ValueError):
        catalog.delete_run("..", "run")
    assert outside.exists()


def test_group_archive_and_delete_include_all_results(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    for root, run in (("logs", "run1"), ("logs", "unlisted"), ("plots", "run2")):
        folder = tmp_path / root / "group" / run
        folder.mkdir(parents=True)
        (folder / "data.txt").write_text(run)
    other = tmp_path / "logs" / "other" / "run"
    other.mkdir(parents=True)
    archive = tmp_path / "group.zip"
    catalog.archive_group("group", str(archive))
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"logs/group/run1/data.txt", "logs/group/unlisted/data.txt",
                                        "plots/group/run2/data.txt"}
    catalog.delete_group("group")
    assert not (tmp_path / "logs" / "group").exists()
    assert not (tmp_path / "plots" / "group").exists()
    assert other.exists() and archive.exists()


def test_group_actions_reject_symlinks_and_invalid_names(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "group").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        catalog.archive_group("group", str(tmp_path / "group.zip"))
    with pytest.raises(ValueError):
        catalog.delete_group("group")
    with pytest.raises(ValueError):
        catalog.delete_group("..")
    assert outside.exists()
