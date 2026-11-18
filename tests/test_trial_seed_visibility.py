"""Execution, queue and monitor share seed identities before the objective is scored."""
import json
import os
import time

from types import SimpleNamespace
from unittest.mock import Mock

from omegaconf import OmegaConf


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def manager_for(root, job):
    from frontend.runner_manager import JobManager
    manager = SimpleNamespace(project_root=root, results_root=root / "src/usr/results", logs_dir=root / "src/usr/results/logs", jobs={job["job_id"]: job},
                              _feeds={}, _streamed_hardware=set(), job_records=Mock(), job_metrics=Mock(),
                              job_hardware=Mock(), queue_changed=Mock(), job=lambda _: job)
    for name in ("_execution_manifests", "_replicate_state", "_run_sources", "_run_dir", "_sweep_summaries", "_execution_queue_items", "_scan_metrics"):
        setattr(manager, name, getattr(JobManager, name).__get__(manager))
    return manager


def test_new_launch_does_not_follow_an_earlier_study_snapshot(tmp_path):
    job = dict(job_id="new", group="g", experiment_id="e", status="running", agents=["m"], started=time.time())
    path = tmp_path / "src/usr/results/checkpoints/g/e/m/tuning_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"component": "search", "study": "old", "stop_reason": "running", "trials": []}))
    os.utime(path, (job["started"] - 60, job["started"] - 60))
    assert manager_for(tmp_path, job)._sweep_summaries(job) == []


def test_cancelled_seed_is_not_scored_as_a_failed_or_completed_sample(tmp_path, monkeypatch):
    from src.app.pipeline.variability import run_execution
    from src.app.services.cancellation import TrainingCancelled
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "m"}})
    def dispatch(cfg, run, on_start, on_wait):
        if cfg.seed == 1:
            raise TrainingCancelled()
        on_start(123)
    path = run_execution(cfg, {"methods": {"m": {}}}, {"training_seeds": [1, 2], "evaluation_seed": None, "parallel_runs": 1},
                         dispatch=dispatch, parent={"kind": "trial", "method": "m", "study": "s", "trial_number": 0,
                                                    "log_version": "version_sweep_0"}, raise_on_failure=False)
    assert [run["status"] for run in json.loads(path.read_text())["runs"]] == ["cancelled", "completed"]


def test_stopped_trial_closes_only_its_unfinished_runs(tmp_path, monkeypatch):
    from src.app.pipeline.variability import finish_trial_runs
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    for trial in ("one", "two"):
        path = tmp_path / "src/usr/results/executions/g/e" / trial / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"parent": {"log_version": trial},
                                    "runs": [{"status": "completed"}, {"status": "running"}, {"status": "pending"}]}))
    finish_trial_runs("g", "e", "one", "cancelled")
    statuses = lambda trial: [r["status"] for r in json.loads((tmp_path / "src/usr/results/executions/g/e" / trial / "manifest.json").read_text())["runs"]]
    assert statuses("one") == ["completed", "cancelled", "cancelled"]
    assert statuses("two") == ["completed", "running", "pending"]


