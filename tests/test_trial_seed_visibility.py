"""Execution, queue and monitor share seed identities before the objective is scored."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def manager_for(root, job):
    from frontend.runner_manager import JobManager
    manager = SimpleNamespace(project_root=root, logs_dir=root / "results/logs", jobs={job["job_id"]: job},
                              _feeds={}, _streamed_hardware=set(), job_records=Mock(), job_metrics=Mock(),
                              job_hardware=Mock(), queue_changed=Mock(), job=lambda _: job)
    for name in ("_execution_manifests", "_replicate_state", "_run_sources", "_run_dir", "_sweep_summaries", "_execution_queue_items", "_scan_metrics"):
        setattr(manager, name, getattr(JobManager, name).__get__(manager))
    return manager


def test_earlier_running_panels_are_adapted_without_modifying_artifacts(tmp_path):
    job = dict(job_id="j", group="g", experiment_id="e", status="running", agents=["m"])
    manager = manager_for(tmp_path, job)
    logs = tmp_path / "results/logs/g/e/m"
    directory = logs / "version_sweep_trial"
    directory.mkdir(parents=True)
    sample = logs / "version_sweep_trial_sample_seed_0"
    sample.mkdir()
    (sample / "training.log").write_text("training")
    (sample / "metrics.csv").write_text("step,transitions,eval/reward\n1,10,123\n")
    payload = {"config": {"theta_tuning": {"study_name": "s", "trial_number": 0}},
               "runs": [{"seed": 11, "status": "running", "artifacts": {"logs": str(sample)}}]}
    path = directory / "objective.json"
    path.write_text(json.dumps(payload))
    before = path.read_bytes()
    summary = tmp_path / "results/checkpoints/g/e/m/tuning_summary.json"
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({"component": "optuna", "study": "s", "stop_reason": "running",
        "requested_trials": 2, "trials": [{"id": 0, "state": "running", "log_version": directory.name}]}))
    sources = manager._run_sources(job)
    assert sources == [("m · s · trial 0 · seed 11", logs.parent, "m", sample.name, "running")]
    items = manager._execution_queue_items(job)
    assert next(item for item in items if item.get("sweep_trial"))["status"] == "running"
    assert path.read_bytes() == before


def test_new_launch_does_not_follow_an_earlier_study_snapshot(tmp_path):
    job = dict(job_id="new", group="g", experiment_id="e", status="running", agents=["m"], started=time.time())
    path = tmp_path / "results/checkpoints/g/e/m/tuning_summary.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"component": "optuna", "study": "old", "stop_reason": "running", "trials": []}))
    os.utime(path, (job["started"] - 60, job["started"] - 60))
    assert manager_for(tmp_path, job)._sweep_summaries(job) == []


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("custom", [True, False])
def test_real_seed_training_is_visible_before_scoring_and_keeps_resource_ownership(tmp_path, managed, custom):
    from PyQt6.QtWidgets import QApplication
    from frontend.live_run import LiveRun
    from frontend.app import Window
    from frontend.monitor.panel import MonitorPanel
    from frontend.queue_panel import QueuePanel
    from frontend.model import Config, new_run

    app = QApplication.instance() or QApplication([])
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "training.py"
    script.write_text('''import json, os, sqlite3, time
from pathlib import Path
from src.app import telemetry
def main(cfg):
    if os.environ.get("THETA_RESOURCE_POOL"):
        with sqlite3.connect(os.environ["THETA_RESOURCE_POOL"]) as db:
            assert db.execute("SELECT COUNT(*) FROM tasks WHERE pid=?", (os.getpid(),)).fetchone()[0] == 1
            assert db.execute("SELECT COUNT(*) FROM tasks WHERE pid=?", (os.getppid(),)).fetchone()[0] == 0
    run = telemetry.init(cfg.group, cfg.experiment_id, cfg.agent.name, version=cfg.variability_version, system_interval=0)
    events = []
    for step in range(5):
        run.log({"eval/reward": cfg.seed * 10}, step=step, transitions=step)
        if step == 0:
            while not Path(cfg.release_file).exists():
                time.sleep(.05)
        events.append({"schema_version": 1, "run_id": cfg.variability_run_id, "metric": "eval/reward",
                       "training_step": step, "episode_outcomes": [cfg.seed * 10]})
        time.sleep(.3)
    (run.version_dir / "evaluation_events.jsonl").write_text("\\n".join(map(json.dumps, events)))
    run.finish()
    return 1.0
''')
    release = tmp_path / "release_training"
    cfg = {"group": "g", "experiment_id": "e", "paradigm": "online_rl", "seed": 1, "release_file": str(release),
           "agent": {"name": "m", "algorithm": "sb3/ppo"}, "total_timesteps": 4, "eval_episodes": 1,
           "objective": {"component": "learning_curve"}, "theta_execution": {"script": str(script)},
           "variability": {"training_seeds": [11, 22], "parallel_runs": 2},
           "hydra": {"searchpath": [f"file://{project / 'in/config'}"]}}
    if not custom:
        cfg.pop("objective")
        cfg["tuning"] = {"metric": "eval/reward", "direction": "maximize"}
    OmegaConf.save(OmegaConf.create(cfg), tmp_path / "task.yaml")
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    command = [sys.executable, "-m", "src.app.pipeline.repeated_task", "--config-path", str(tmp_path), "--config-name", "task",
               "--multirun", "hydra/sweeper=optuna_online", "hydra.sweeper.n_trials=2",
               f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=s", "+x=choice(1,2)"]
    env = {**os.environ, "PYTHONPATH": str(project), "HYDRA_FULL_ERROR": "1"}
    env.pop("THETA_PIPELINE_PID", None)
    if managed:
        settings = tmp_path / "settings.toml"
        settings.write_text("[training]\ncpu_percent=100\nmemory_percent=95\n")
        env.update(THETA_RESOURCE_POOL=str(tmp_path / "resources.db"), THETAIDE_SETTINGS_FILE=str(settings), THETA_MANAGED_SERIAL="1")
    output_path = tmp_path / "pipeline.log"
    started = time.time()
    with output_path.open("w") as output:
        process = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=output, stderr=subprocess.STDOUT)
    job = dict(job_id="j", group="g", experiment_id="e", status="running", pid=process.pid, started=started, agents=["m"])
    manager = manager_for(tmp_path, job)
    lives = {}
    def records(_job, label, data, reset):
        lives.setdefault(label, LiveRun(budget=4)).add(data, reset=reset)
    manager.job_records.emit.side_effect = records
    run = new_run(Config(), simulated=False)
    run.update(status="running", backend={"job_id": "j", "group": "g", "experiment_id": "e", "agents": ["m"]})
    host = SimpleNamespace(selected=run, runs=[run], jobs=manager, live_runs_for_job=lambda _: lives)
    host.live_for_run = lambda run, load=True: Window.live_for_run(host, run, load)
    monitor = MonitorPanel(host)
    queue = QueuePanel(lambda *_: None, split_methods=True, lives_for_job=lambda _: host.live_for_run(run))
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline and process.poll() is None:
            manager._scan_metrics("j")
            if any(live.rows and not live.finish for live in lives.values()):
                break
            app.processEvents()
            time.sleep(.05)
        active = host.live_for_run(run)
        assert active, (str({"job": job, "sources": manager._run_sources(job),
                             "lives": {key: (len(live.rows), bool(live.finish)) for key, live in lives.items()}})
                        + "\n" + output_path.read_text())
        assert all(" · trial " in label and " · seed " in label for label in active)
        items = manager._execution_queue_items(job)
        queue.set_data({"active": items})
        assert any(card.status.text() == "Running" for card in queue.cards.values()), output_path.read_text()
        monitor.refresh_live()
        assert monitor.charts_view.points
        assert {action.text() for action in monitor.method_button.menu().actions()} == {"m"}
        assert any("Seed" in action.text() for action in monitor.trial_button.menu().actions())
        assert not any(p.exists() for p in (tmp_path / "results/logs/g/e/m").glob("version_sweep_*/objective.json"))
        release.touch()
        process.wait(timeout=45)
        assert process.returncode == 0, output_path.read_text()
        manager._scan_metrics("j")
        assert host.live_for_run(run) == {}
        manifests = list(tmp_path.glob("results/executions/g/e/*/manifest.json"))
        assert len(manifests) == 2
        assert all(r["status"] == "completed" for p in manifests for r in json.loads(p.read_text())["runs"])
    finally:
        release.touch()
        monitor.close()
        queue.close()
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


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


def test_cancelled_panel_never_calls_the_objective(tmp_path, monkeypatch):
    from src.app.pipeline.repeated_task import run_objective
    from src.app.services.cancellation import TrainingCancelled
    monkeypatch.chdir(tmp_path)
    score = Mock(side_effect=AssertionError("cancelled panels must not be scored"))
    monkeypatch.setattr("src.app.objectives.evaluation.score_execution", score)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "m"},
        "objective": {"component": "final_metric"}, "theta_execution": {"script": "src/app/train.py"},
        "variability": {"training_seeds": [1, 2]}})
    def cancel(*_):
        raise TrainingCancelled()
    with pytest.raises(TrainingCancelled):
        run_objective(cfg, execute=cancel)
    score.assert_not_called()


def test_stopped_trial_closes_only_its_unfinished_runs(tmp_path, monkeypatch):
    from src.app.pipeline.variability import finish_trial_runs
    monkeypatch.chdir(tmp_path)
    for trial in ("one", "two"):
        path = tmp_path / "results/executions/g/e" / trial / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"parent": {"log_version": trial},
                                    "runs": [{"status": "completed"}, {"status": "running"}, {"status": "pending"}]}))
    finish_trial_runs("g", "e", "one", "cancelled")
    statuses = lambda trial: [r["status"] for r in json.loads((tmp_path / "results/executions/g/e" / trial / "manifest.json").read_text())["runs"]]
    assert statuses("one") == ["completed", "cancelled", "cancelled"]
    assert statuses("two") == ["completed", "running", "pending"]


def test_cancelled_method_closes_only_current_launch_trials_and_seed_records(tmp_path, monkeypatch):
    import optuna
    from src.app.pipeline.variability import finish_trial_runs
    from component_fixtures import optional_module
    _optuna_component = optional_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner
    monkeypatch.chdir(tmp_path)
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="s", storage=storage)
    study.ask()  # An earlier launch's outstanding trial is not owned by this cancellation.
    study.set_user_attr("theta_initial_trials", 1)
    study.ask()
    for key, pid, method in [("current", 11, "m"), ("sibling", 11, "other"), ("earlier", 22, "m")]:
        path = tmp_path / "results/executions/g/e" / key / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"owner_pid": pid, "parent": {"method": method}, "runs": [{"status": "running"}]}))
    finish_trial_runs("g", "e", None, "cancelled", owner_pid=11, method="m")
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e"})
    OptunaTuner().cancel(cfg, "m", storage, "s")
    trials = optuna.load_study(study_name="s", storage=storage).trials
    assert [trial.state.name for trial in trials] == ["RUNNING", "PRUNED"]
    for key, expected in [("current", "cancelled"), ("sibling", "running"), ("earlier", "running")]:
        path = tmp_path / "results/executions/g/e" / key / "manifest.json"
        assert json.loads(path.read_text())["runs"][0]["status"] == expected
