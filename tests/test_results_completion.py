"""Only finalized versions are published to the Results browser."""
import json

import pytest

from src.app.services import catalog


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


@pytest.fixture
def logs(tmp_path, monkeypatch):
    root = tmp_path / "results" / "logs"
    root.mkdir(parents=True)
    monkeypatch.setattr(catalog, "LOGS_DIR", root)
    return root


def version(logs, name, *, finished=False, agent="agent", reward=1):
    directory = logs / "group" / "run" / agent / name
    directory.mkdir(parents=True)
    (directory / "metrics.csv").write_text(f"transitions,eval/reward\n10,{reward}\n")
    write_json(directory / "runtime.json", {"end_time_iso": "2026-11-03T10:00:00Z"} if finished else {})
    return directory


def test_completed_results_exclude_active_versions_without_hiding_finished_siblings(logs):
    version(logs, "version_0", finished=True, reward=12)
    version(logs, "version_1", reward=99)
    version(logs, "version_0", agent="still_training")
    entries = catalog.list_runs(completed_only=True)
    assert len(entries) == 1
    assert entries[0]["finished"]
    assert entries[0]["completed_versions"] == {"agent": ["version_0"]}
    assert entries[0]["agents"][0]["latest_reward"] == 12
    assert catalog.run_metrics("group", "run", "agent")["version"] == "version_1"
    completed = catalog.run_metrics("group", "run", "agent", completed_only=True)
    assert completed["version"] == "version_0"
    assert completed["versions"] == ["version_0"]
    with pytest.raises(FileNotFoundError):
        catalog.run_metrics("group", "run", "agent", "version_1", completed_only=True)
    manifest = catalog.run_manifest("group", "run", completed_only=True)
    assert manifest["agents"] == {"agent": ["version_0"]}
    assert all("version_1" not in item["path"] and "still_training" not in item["path"]
               for item in manifest["artifacts"])


def test_running_only_run_is_not_published_and_contract_completion_is_supported(logs):
    directory = version(logs, "version_0")
    assert catalog.list_runs(completed_only=True) == []
    with pytest.raises(FileNotFoundError):
        catalog.run_manifest("group", "run", completed_only=True)
    write_json(directory / "runtime.json", {"completed_at": "2026-11-03T10:00:00Z"})
    assert catalog.list_runs(completed_only=True)[0]["finished"]


def test_sweep_trials_publish_only_after_their_terminal_state(logs):
    from src.app.pipeline.trial_artifacts import trial_checkpoint_dir

    version(logs, "version_sweep_study_trial_0", finished=True)
    version(logs, "version_sweep_study_trial_1", finished=True)
    checkpoint_root = logs.parent / "checkpoints" / "group" / "run" / "agent"
    summary = {"study": "study", "log_method": "agent", "stop_reason": "running", "trials": [
        {"id": 0, "state": "complete", "log_version": "version_sweep_study_trial_0"},
        {"id": 1, "state": "running", "log_version": "version_sweep_study_trial_1"},
    ]}
    write_json(checkpoint_root / "tuning_summary.json", summary)
    for number in (0, 1):
        directory = trial_checkpoint_dir(checkpoint_root, "study", number)
        directory.mkdir(parents=True)
        (directory / "best_model.ckpt").write_bytes(b"weights")
    manifest = catalog.run_manifest("group", "run", completed_only=True)
    assert manifest["agents"] == {"agent": ["version_sweep_study_trial_0"]}
    assert any("trial_0/best_model.ckpt" in item["path"] for item in manifest["artifacts"])
    assert not any("trial_1" in item["path"] for item in manifest["artifacts"])
    summary["trials"][1]["state"] = "pruned"
    write_json(checkpoint_root / "tuning_summary.json", summary)
    assert len(catalog.run_manifest("group", "run", completed_only=True)["agents"]["agent"]) == 2


def test_finished_sweep_summary_survives_other_unfinished_studies(logs):
    checkpoint_root = logs.parent / "checkpoints" / "group" / "run" / "agent"
    old = {"study": "old", "stop_reason": "running", "trials": [
        {"id": 0, "state": "running", "log_version": "version_sweep_old_trial_0"},
    ]}
    finished = {"study": "finished", "stop_reason": "trial_limit", "best_trial_id": 35,
                "trials": [{"id": i, "state": "complete", "value": i,
                            "log_version": f"version_sweep_finished_trial_{i}"} for i in range(48)]}
    version(logs, "version_sweep_old_trial_0", finished=True)
    for trial in finished["trials"]:
        version(logs, trial["log_version"], finished=True)
    write_json(checkpoint_root / "sweeps/old/tuning_summary.json", old)
    saved = checkpoint_root / "sweeps/finished/tuning_summary.json"
    write_json(saved, finished)
    latest = checkpoint_root / "tuning_summary.json"
    write_json(latest, finished)

    def summary_paths():
        manifest = catalog.run_manifest("group", "run", completed_only=True)
        assert len(manifest["agents"]["agent"]) == 48
        return {item["path"] for item in manifest["artifacts"] if item["path"].endswith("tuning_summary.json")}

    saved_path = saved.relative_to(logs.parent).as_posix()
    latest_path = latest.relative_to(logs.parent).as_posix()
    assert summary_paths() == {saved_path, latest_path}
    assert json.loads(catalog.read_run_file("group", "run", saved_path))["best_trial_id"] == 35

    # Starting another sweep hides its live snapshot, while the completed study stays available.
    write_json(latest, dict(old, study="new"))
    assert summary_paths() == {saved_path}
    # A non-running stop reason alone must not publish unfinished trial data.
    write_json(latest, dict(old, stop_reason="interrupted"))
    assert summary_paths() == {saved_path}


def test_execution_results_do_not_read_running_replicates(tmp_path, monkeypatch):
    from src.app.services.executions import list_executions, read_execution

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "results" / "executions" / "group" / "run" / "execution" / "manifest.json"
    runs = [
        {"run_id": "done", "status": "completed", "method": "agent", "logs": str(tmp_path / "done")},
        {"run_id": "live", "status": "running", "method": "agent", "logs": str(tmp_path / "live")},
    ]
    write_json(path, {"schema_version": 1, "runs": runs[1:]})
    assert list_executions(completed_only=True) == []
    write_json(path, {"schema_version": 1, "runs": runs})
    assert len(list_executions(completed_only=True)) == 1
    data = read_execution(path, completed_only=True)
    assert [run["run_id"] for run in data["manifest"]["runs"]] == ["done"]
    assert set(data["events"]) == {"done"}
