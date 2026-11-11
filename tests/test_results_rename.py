"""Results tree renames preserve all artifact roots and execution references."""
import json
import os
from pathlib import Path

import pytest

from src.app.services import catalog
from src.app.services.result_rename import rename_results


@pytest.mark.parametrize("kind", ["group", "experiment", "run"])
def test_rename_results_across_roots(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    runs = ["nested/run", "nested/run_20261110-120000", "nested/other"]
    roots = (*catalog.RESULT_ROOTS, "executions")
    for root in roots:
        for run in runs:
            path = tmp_path / root / "group" / run / "config.yaml"
            path.parent.mkdir(parents=True)
            path.write_text("original: config")
    data = rename_results(kind, "group", "nested/run" if kind != "group" else "", "renamed")
    affected = runs if kind == "group" else runs[:2] if kind == "experiment" else runs[:1]
    for run in runs:
        new_group = "renamed" if kind == "group" else "group"
        replacement = run if kind == "group" else run.replace("nested/run", "nested/renamed", 1)
        if run not in affected:
            new_group, replacement = "group", run
        for root in roots:
            assert (tmp_path / root / new_group / replacement / "config.yaml").read_text() == "original: config"
            if run in affected:
                assert not (tmp_path / root / "group" / run).exists()
    assert data["runs"] == {f"group/{run}": f"{'renamed' if kind == 'group' else 'group'}/"
                            + (run if kind == "group" else run.replace("nested/run", "nested/renamed", 1))
                            for run in affected}


def test_rename_updates_execution_identity_paths_and_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.chdir(tmp_path.parent)
    logs = tmp_path / "logs/group/run"
    logs.mkdir(parents=True)
    (logs / "config.yaml").write_text("experiment_id: run")
    execution = tmp_path / "executions/group/run/job/manifest.json"
    execution.parent.mkdir(parents=True)
    events = logs / "agent/version_0/evaluation_events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text("evaluation data")
    document = {"group": "group", "experiment_id": "run", "config": {"group": "group", "experiment_id": "run"},
                "runs": [{"experiment_id": "run", "status": "completed", "version": "version_0", "agent": "agent",
                          "logs": str(logs), "event_path": os.path.relpath(events, Path.cwd())}]}
    execution.write_text(json.dumps(document))
    pointer = logs / "agent/version_0/execution.json"
    pointer.write_text(json.dumps({"manifest": os.path.relpath(execution, pointer.parent)}))
    rename_results("run", "group", "run", "new")
    manifest = tmp_path / "executions/group/new/job/manifest.json"
    updated = json.loads(manifest.read_text())
    assert updated["experiment_id"] == updated["runs"][0]["experiment_id"] == "new"
    assert updated["config"] == document["config"]
    assert Path(updated["runs"][0]["event_path"]).read_text() == "evaluation data"
    assert Path(updated["runs"][0]["logs"]) == tmp_path / "logs/group/new"
    new_pointer = tmp_path / "logs/group/new/agent/version_0/execution.json"
    assert (new_pointer.parent / json.loads(new_pointer.read_text())["manifest"]).resolve() == manifest
    assert ("agent", "version_0") in catalog._result_replicates(tmp_path / "logs/group/new")


def test_conflict_in_later_root_does_not_partially_rename(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    for root in ("logs", "plots"):
        path = tmp_path / root / "group/run"
        path.mkdir(parents=True)
        (path / "data.txt").write_text(root)
    (tmp_path / "plots/group/new").mkdir()
    with pytest.raises(FileExistsError):
        rename_results("run", "group", "run", "new")
    assert (tmp_path / "logs/group/run/data.txt").read_text() == "logs"
    assert not (tmp_path / "logs/group/new").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "nested/name", "bad\\name"])
def test_rename_rejects_invalid_names(tmp_path, monkeypatch, name):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    with pytest.raises(ValueError):
        rename_results("run", "group", "run", name)


def test_rename_rejects_symlinks(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    (tmp_path / "logs/group").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "logs/group/run").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        rename_results("run", "group", "run", "new")
    assert outside.exists()


def test_move_failure_rolls_back_previous_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog, "LOGS_DIR", tmp_path / "logs")
    for root in ("logs", "plots"):
        (tmp_path / root / "group/run").mkdir(parents=True)
    original = Path.rename

    def fail(self, target):
        if self == tmp_path / "plots/group/run":
            raise OSError("Cannot move plots")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="Cannot move plots"):
        rename_results("run", "group", "run", "new")
    assert (tmp_path / "logs/group/run").is_dir()
    assert not (tmp_path / "logs/group/new").exists()
