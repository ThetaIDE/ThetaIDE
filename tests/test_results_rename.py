"""Results tree renames preserve all artifact roots and execution references."""
import json
import os
from pathlib import Path

import pytest

from src.app.services import catalog
from src.app.services.result_rename import rename_results


@pytest.mark.parametrize("kind", ["group", "experiment", "run"])
def test_rename_results_across_roots(tmp_path, monkeypatch, kind, workspace_results):
    results = workspace_results
    runs = ["nested/run", "nested/run_20261110-120000", "nested/other"]
    roots = (*catalog.RESULT_ROOTS, "executions")
    for root in roots:
        for run in runs:
            path = results / root / "group" / run / "config.yaml"
            path.parent.mkdir(parents=True)
            path.write_text("original: config")
    data = rename_results(kind, "group", "nested/run" if kind != "group" else "", "renamed")
    affected = runs if kind == "group" else runs[:1]
    for run in runs:
        new_group = "renamed" if kind == "group" else "group"
        replacement = run if kind == "group" else run.replace("nested/run", "nested/renamed", 1)
        if run not in affected:
            new_group, replacement = "group", run
        for root in roots:
            assert (results / root / new_group / replacement / "config.yaml").read_text() == "original: config"
            if run in affected:
                assert not (results / root / "group" / run).exists()
    assert data["runs"] == {f"group/{run}": f"{'renamed' if kind == 'group' else 'group'}/"
                            + (run if kind == "group" else run.replace("nested/run", "nested/renamed", 1))
                            for run in affected}


def test_rename_updates_execution_identity_paths_and_pointer(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    monkeypatch.chdir(tmp_path)
    logs = results / "logs/group/run"
    logs.mkdir(parents=True)
    (logs / "config.yaml").write_text("experiment_id: run")
    execution = results / "executions/group/run/job/manifest.json"
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
    manifest = results / "executions/group/new/job/manifest.json"
    updated = json.loads(manifest.read_text())
    assert updated["experiment_id"] == updated["runs"][0]["experiment_id"] == "new"
    assert updated["config"] == document["config"]
    assert Path(updated["runs"][0]["event_path"]).read_text() == "evaluation data"
    assert Path(updated["runs"][0]["logs"]) == results / "logs/group/new"
    new_pointer = results / "logs/group/new/agent/version_0/execution.json"
    assert (new_pointer.parent / json.loads(new_pointer.read_text())["manifest"]).resolve() == manifest
    assert ("agent", "version_0") in catalog._result_replicates(results / "logs/group/new")


def test_conflict_in_later_root_does_not_partially_rename(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    for root in ("logs", "plots"):
        path = results / root / "group/run"
        path.mkdir(parents=True)
        (path / "data.txt").write_text(root)
    (results / "plots/group/new").mkdir()
    with pytest.raises(FileExistsError):
        rename_results("run", "group", "run", "new")
    assert (results / "logs/group/run/data.txt").read_text() == "logs"
    assert not (results / "logs/group/new").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../outside", "nested/name", "bad\\name"])
def test_rename_rejects_invalid_names(tmp_path, monkeypatch, name, workspace_results):
    with pytest.raises(ValueError):
        rename_results("run", "group", "run", name)


def test_rename_rejects_symlinks(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    (results / "logs/group").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (results / "logs/group/run").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        rename_results("run", "group", "run", "new")
    assert outside.exists()


def test_move_failure_rolls_back_previous_roots(tmp_path, monkeypatch, workspace_results):
    results = workspace_results
    for root in ("logs", "plots"):
        (results / root / "group/run").mkdir(parents=True)
    original = Path.rename

    def fail(self, target):
        if self == results / "plots/group/run":
            raise OSError("Cannot move plots")
        return original(self, target)

    monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="Cannot move plots"):
        rename_results("run", "group", "run", "new")
    assert (results / "logs/group/run").is_dir()
    assert not (results / "logs/group/new").exists()
