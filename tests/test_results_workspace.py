"""Results are workspace output, never authored source or another workspace's data."""
from pathlib import Path

import pytest

from src.app.workspace import WorkspacePaths


def test_results_do_not_change_source_inventory_snapshot_or_compose_key(tmp_path, monkeypatch):
    from src.app.services.config_service import _inputs_fingerprint
    from src.app.sites.snapshot import source_files
    from src.app.workspace_inventory import inventory

    paths = WorkspacePaths(tmp_path, application=tmp_path / "application")
    source = paths.research / "example.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    fingerprint = _inputs_fingerprint(paths)
    files = inventory(paths)["files"]
    snapshot = source_files(tmp_path)
    output = paths.results / "logs/g/run/source.py"
    output.parent.mkdir(parents=True)
    output.write_text("value = 2\n")
    (output.parent / "plugin.json").write_text("{}")
    (output.parent / "config.yaml").write_text("seed: 2\n")

    # A pruning regression must fail even if the discovered files are filtered
    # later: a real results tree can contain millions of entries.
    import os
    scandir = os.scandir

    def scan(path):
        assert Path(path) != paths.results, "walked into run output"
        return scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    assert _inputs_fingerprint(paths) == fingerprint
    assert inventory(paths)["files"] == files
    assert source_files(tmp_path) == snapshot
    source.write_text("value = 3\n")
    assert _inputs_fingerprint(paths) != fingerprint


def test_result_edits_do_not_acquire_the_implementation_lock(tmp_path, monkeypatch):
    from src.app.components import transactions

    paths = WorkspacePaths(tmp_path)

    def busy(_paths):
        raise RuntimeError("workspace busy")

    monkeypatch.setattr(transactions, "assert_idle", busy)
    with transactions.implementation_edit(paths, paths.results / "logs/g/run/notes.md"):
        pass
    with pytest.raises(RuntimeError, match="workspace busy"):
        with transactions.implementation_edit(paths, paths.research / "example.py"):
            pass


def test_results_sdk_uses_explicit_workspace_and_confines_artifact_reads(tmp_path, monkeypatch):
    from src.app.api import results

    selected = WorkspacePaths(tmp_path / "selected")
    other = WorkspacePaths(tmp_path / "other")
    monkeypatch.setenv("THETA_WORKSPACE", str(other.root))
    version = selected.results / "logs/g/run/method/version_0"
    version.mkdir(parents=True)
    (version.parent.parent / "config.yaml").write_text("group: g\nexperiment_id: run\n")
    (version / "metrics.csv").write_text("step,score\n1,2\n")
    assert results.list_runs() == []
    assert results.list_runs(workspace=selected)[0]["experiment_id"] == "run"
    assert results.run_metrics("g", "run", "method", workspace=selected)["total_rows"] == 1
    assert results.run_manifest("g", "run", workspace=selected)
    assert results.read_run_file("g", "run", "logs/g/run/method/version_0/metrics.csv",
                                 workspace=selected) == b"step,score\n1,2\n"
    secret = selected.root / "secret.txt"
    secret.write_text("outside")
    (version / "escape.txt").symlink_to(secret)
    for path in ("../secret.txt", "logs/g/other/secret.txt", "logs/g/run/method/version_0/escape.txt"):
        with pytest.raises(PermissionError):
            results.read_run_file("g", "run", path, workspace=selected)
