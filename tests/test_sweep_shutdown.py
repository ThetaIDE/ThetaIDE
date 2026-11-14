"""Allocation thread bounds, accounting detail and interrupted producer finalization."""
import json
from pathlib import Path
from types import SimpleNamespace

import optuna
import pytest

from src.app.sites import execution
from src.app.sites.storage import write_json
from component_fixtures import optional_module
_optuna_component = optional_module('optuna', 'component.py')
OptunaTuner = _optuna_component.OptunaTuner
from component_fixtures import optional_module
_optuna_utils = optional_module('optuna', 'utils.py')
write_study_summary = _optuna_utils.write_study_summary


def test_threads_do_not_expand_to_allocation_size():
    from component_fixtures import optional_module
    worker_threads = optional_module('slurm', 'execution.py').worker_threads
    assert worker_threads({}, {"SLURM_CPUS_PER_TASK": "16"}) == 1
    assert worker_threads({"site": {"threads_per_worker": 2}}, {"SLURM_CPUS_PER_TASK": "16"}) == 2
    assert worker_threads({"site": {"threads_per_worker": 8}}, {"SLURM_CPUS_PER_TASK": "2"}) == 2
    assert worker_threads({"site": {"extra_env_vars": {"OMP_NUM_THREADS": "1"}}}, {"SLURM_CPUS_PER_TASK": "16"}) == 1


@pytest.mark.parametrize("batched", [False, True])
def test_scheduler_preserves_canceller_reason_and_time(tmp_path, monkeypatch, batched):
    from component_fixtures import install_optional
    install_optional(tmp_path, "slurm", tmp_path.parent / (tmp_path.name + "-archives"))
    record = {"execution_id": "a" * 32, "job_id": "123", "state": "running", "project_id": "test",
              "scheduler_component": "slurm"}
    write_json(execution.receipt_path(tmp_path, record["execution_id"]), record)
    replies = iter(["", "123|CANCELLED by 42|0:15|TimeLimit|00:28:00|00:30:00|start|end\n"])
    monkeypatch.setattr(execution.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=next(replies)))
    if batched:
        execution.refresh_many(tmp_path, [record], minimum_seconds=0)
    else:
        record = execution.refresh_status(tmp_path, record)
    assert record["scheduler_state"] == "cancelled"
    assert record["scheduler_state_raw"] == "CANCELLED by 42"
    assert record["scheduler_reason"] == "TimeLimit"
    assert record["scheduler_time_limit"] == "00:30:00"


def test_shutdown_closes_seeds_and_trials_without_changing_completed_work(tmp_path, monkeypatch):
    record = {"execution_id": "a" * 32, "group": "g", "result_experiment_id": "nested/e", "config": {"tuning": {}}}
    manifest = tmp_path / "results/executions/g/nested/e/x/manifest.json"
    write_json(manifest, {"runs": [{"status": s} for s in ("completed", "running", "queued")]})
    database = tmp_path / "results/optuna" / record["execution_id"] / "study.sqlite"
    database.parent.mkdir(parents=True)
    storage = "sqlite:///" + str(database)
    study = optuna.create_study(storage=storage, study_name="test")
    trial = study.ask()
    study.tell(trial, 1.0)
    study.ask()
    study.set_user_attr("theta_last_stop_reason", "running")
    study.set_user_attr("theta_requested_trials", 64)
    checkpoint = tmp_path / "results/checkpoints/g/nested/e/method"
    checkpoint.mkdir(parents=True)
    write_study_summary(study, checkpoint)
    def selected_tuner(cfg, *, paths):
        assert paths.root == tmp_path
        return OptunaTuner()
    monkeypatch.setattr("src.app.pipeline.tuners.get_tuner", selected_tuner)
    execution.close_unfinished_runs(tmp_path, record, "cancelled")
    runs = json.loads(manifest.read_text())["runs"]
    assert [r["status"] for r in runs] == ["completed", "cancelled", "cancelled"]
    summary = json.loads((checkpoint / "tuning_summary.json").read_text())
    assert [t["state"] for t in summary["trials"]] == ["complete", "fail"]
    assert summary["pending_trials"] == 0
    assert summary["stop_reason"] == "interrupted"
    assert [t.state for t in study.get_trials()] == [optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL]
