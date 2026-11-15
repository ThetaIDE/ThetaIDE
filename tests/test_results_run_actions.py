"""Run archives and deletion cover all results roots and reject unsafe paths."""
import zipfile

import pytest

from src.app.services import catalog


@pytest.mark.parametrize("run_id", ["run", "path/dir/run"])
def test_archive_and_delete_run(tmp_path, monkeypatch, run_id, workspace_results):
    results = workspace_results
    for root in ("logs", "plots", "checkpoints"):
        folder = results / root / "group" / run_id
        folder.mkdir(parents=True)
        (folder / "data.txt").write_text(root)
    other = results / "logs" / "group" / "other"
    other.mkdir()
    archive = tmp_path / "run.zip"
    catalog.archive_run("group", run_id, str(archive))
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {f"{root}/group/{run_id}/data.txt" for root in ("logs", "plots", "checkpoints")}
        assert zipped.read(f"plots/group/{run_id}/data.txt") == b"plots"
    catalog.delete_run("group", run_id)
    assert all(not (results / root / "group" / run_id).exists() for root in ("logs", "plots", "checkpoints"))
    assert other.exists() and archive.exists()


def test_run_actions_reject_unsafe_paths(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    outside = tmp_path / "outside"
    outside.mkdir()
    (results / "logs" / "group").mkdir(parents=True)
    (results / "logs" / "group" / "run").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        catalog.delete_run("group", "run")
    with pytest.raises(ValueError):
        catalog.archive_run("group", "run", str(tmp_path / "run.zip"))
    with pytest.raises(ValueError):
        catalog.delete_run("..", "run")
    assert outside.exists()


def test_group_archive_and_delete_include_all_results(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    for root, run in (("logs", "run1"), ("logs", "unlisted"), ("plots", "run2")):
        folder = results / root / "group" / run
        folder.mkdir(parents=True)
        (folder / "data.txt").write_text(run)
    other = results / "logs" / "other" / "run"
    other.mkdir(parents=True)
    archive = tmp_path / "group.zip"
    catalog.archive_group("group", str(archive))
    with zipfile.ZipFile(archive) as zipped:
        assert set(zipped.namelist()) == {"logs/group/run1/data.txt", "logs/group/unlisted/data.txt",
                                        "plots/group/run2/data.txt"}
    catalog.delete_group("group")
    assert not (results / "logs" / "group").exists()
    assert not (results / "plots" / "group").exists()
    assert other.exists() and archive.exists()


def test_group_actions_reject_symlinks_and_invalid_names(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    outside = tmp_path / "outside"
    outside.mkdir()
    (results / "logs").mkdir(parents=True)
    (results / "logs" / "group").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        catalog.archive_group("group", str(tmp_path / "group.zip"))
    with pytest.raises(ValueError):
        catalog.delete_group("group")
    with pytest.raises(ValueError):
        catalog.delete_group("..")
    assert outside.exists()


@pytest.mark.parametrize("root", ["logs", "plots", "checkpoints"])
@pytest.mark.parametrize("run_id", ["run", "nested/run"])
def test_rename_run_file(tmp_path, monkeypatch, root, run_id, workspace_results):
    results = workspace_results
    relative = f"{root}/group/{run_id}/subfolder/report.txt"
    source = results / relative
    source.parent.mkdir(parents=True)
    source.write_text("report contents")
    renamed = catalog.rename_run_file("group", run_id, relative, "new report.txt")
    assert renamed == f"{root}/group/{run_id}/subfolder/new report.txt"
    assert not source.exists()
    assert catalog.read_run_file("group", run_id, renamed) == b"report contents"
    assert renamed in {artifact["path"] for artifact in catalog.run_manifest("group", run_id)["artifacts"]}


def test_rename_run_file_rejects_conflicts_and_unsafe_paths(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    folder = results / "logs/group/run"
    folder.mkdir(parents=True)
    source = folder / "report.txt"
    source.write_text("keep")
    (folder / "existing.txt").write_text("existing")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (folder / "link.txt").symlink_to(outside)
    (folder / "linked").symlink_to(tmp_path, target_is_directory=True)
    for name in ("", ".", "..", "../outside.txt", "nested/name", "bad\\name", "bad\x00name"):
        with pytest.raises(ValueError):
            catalog.rename_run_file("group", "run", "logs/group/run/report.txt", name)
    with pytest.raises(FileExistsError):
        catalog.rename_run_file("group", "run", "logs/group/run/report.txt", "existing.txt")
    for path in ("logs/group/run/link.txt", "logs/group/run/linked/outside.txt"):
        with pytest.raises(ValueError):
            catalog.rename_run_file("group", "run", path, "renamed.txt")
    with pytest.raises(PermissionError):
        catalog.rename_run_file("group", "run", "outside.txt", "renamed.txt")
    with pytest.raises(ValueError):
        catalog.rename_run_file("group", "run", "logs/group/run/../../../outside.txt", "renamed.txt")
    with pytest.raises(FileNotFoundError):
        catalog.rename_run_file("group", "run", "logs/group/run/missing.txt", "renamed.txt")
    assert source.read_text() == "keep"
    assert (folder / "existing.txt").read_text() == "existing"
    assert outside.read_text() == "outside"
