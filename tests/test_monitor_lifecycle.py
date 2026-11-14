"""A monitor follows the current launch and drops completed training data."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

from PyQt6.QtWidgets import QApplication

from frontend.app import Window
from frontend.live_run import LiveRun
from frontend.model import Config, new_run
from frontend.monitor.panel import MonitorPanel
from frontend.queue_panel import ProcessCard
from frontend.runner_manager import JobManager


def test_reused_experiment_streams_only_current_launch_and_all_parallel_seeds(tmp_path):
    app = QApplication.instance() or QApplication([])
    # Avoid starting a runner: these methods only need source paths and signals.
    manager = SimpleNamespace(project_root=tmp_path, logs_dir=tmp_path / "results/logs",
                              _feeds={}, _streamed_hardware=set(), job_records=Mock(),
                              job_metrics=Mock(), job_hardware=Mock())
    for name in ("_execution_manifests", "_replicate_state", "_run_sources", "_run_dir", "_sweep_summaries"):
        setattr(manager, name, getattr(JobManager, name).__get__(manager))
    job = dict(job_id="job", group="g", experiment_id="variance", status="running",
               pid=20, started=100, agents=["ppo"])
    manager.jobs = {"job": job}
    for execution, pid, created, seeds in (("old", 10, 50, [1, 2]), ("current", 20, 110, [1, 3])):
        root = tmp_path / "results/executions/g/variance" / execution
        root.mkdir(parents=True)
        runs = []
        for index, seed in enumerate(seeds):
            logs = tmp_path / "results/logs/g" / execution
            version = f"version_{index}"
            directory = logs / "ppo" / version
            directory.mkdir(parents=True)
            (directory / "events.jsonl").write_text(json.dumps({
                "type": "metrics", "step": (index + 1) * 10, "data": {"eval/reward": seed}}) + "\n")
            output = root / f"{seed}.log"
            output.touch()
            runs.append(dict(run_id=str(seed), method="ppo", training_seed=seed, logs=str(logs),
                             version=version, output_log=str(output), status="running"))
        (root / "manifest.json").write_text(json.dumps(dict(owner_pid=pid,
            created_at=datetime.fromtimestamp(created, timezone.utc).isoformat(), runs=runs)))
    assert [source[0] for source in manager._run_sources(job)] == ["ppo · seed 1", "ppo · seed 3"]
    JobManager._scan_metrics(manager, "job")
    lives = {}
    for call in manager.job_records.emit.call_args_list:
        _, label, records, reset = call.args
        lives[label] = LiveRun(budget=100)
        lives[label].add(records, reset)
    for label, expected in (("ppo · seed 1", .1), ("ppo · seed 3", .2)):
        card = ProcessCard(job, label, show_progress=True)
        card.update_progress(lives)
        assert card.progress_fraction == expected
        card.close()
    # A resource wait is pending until the worker starts, independently of other seeds.
    (tmp_path / "results/executions/g/variance/current/3.log").unlink()
    assert [item["status"] for item in JobManager._execution_queue_items(manager, job)] == ["running", "pending"]
    assert [source[4] for source in manager._run_sources(job)] == ["running", "pending"]
    # The client must correct a stale running label from a detached older runner.
    from frontend.job_manager import JobManager as DesktopManager
    manager._execution_queue_items = JobManager._execution_queue_items.__get__(manager)
    manager.public = lambda job: dict(job)
    manager._rpc = lambda *_: {"jobs": [job], "queue": {
        "running": True, "active": [{**job, "agents": ["ppo · seed 3"]}], "queued": []}, "max_parallel": None}
    DesktopManager._refresh(manager, emit=False)
    assert [item["status"] for item in manager._queue["active"]] == ["running", "pending"]
    assert JobManager.replicate(manager, "job", "ppo · seed 1")["logs"].endswith("current")
    path = tmp_path / "results/executions/g/variance/current/manifest.json"
    manifest = json.loads(path.read_text())
    for run in manifest["runs"]:
        run["status"] = "completed"
    path.write_text(json.dumps(manifest))
    assert JobManager._execution_queue_items(manager, job) == []
    assert app is not None


def test_live_lookup_excludes_old_and_completed_seeds_without_disk_fallback():
    active, finished, ended_without_finish, old = (LiveRun() for _ in range(4))
    finished.finish = {"status": "completed"}
    active.start = {"type": "start"}
    job = {"job_id": "j"}
    host = SimpleNamespace(
        jobs=SimpleNamespace(job=lambda _: job, _run_sources=lambda _: [
            ("active", None, None, None, "running"),
            ("finished", None, None, None, "running"),
            ("ended", None, None, None, "completed")]),
        live_runs_for_job=lambda _: dict(active=active, finished=finished, ended=ended_without_finish, old=old))
    run = dict(status="running", backend={"job_id": "j"})
    assert Window.live_for_run(host, run) == {"active": active}
    assert finished.finish  # Results history is preserved.


def test_pending_and_empty_feeds_never_enter_monitor_legend_or_dropdown():
    app = QApplication.instance() or QApplication([])
    active, pending, empty = (LiveRun() for _ in range(3))
    active.add([{"type": "start"}])
    pending.add([{"type": "start"}])  # source state takes precedence over cached telemetry
    job = {"job_id": "j"}
    run = new_run(Config(), simulated=False)
    run["backend"] = {"job_id": "j", "group": "g", "experiment_id": "e"}
    host = SimpleNamespace(selected=run, runs=[run],
        jobs=SimpleNamespace(job=lambda _: job, queue_changed=Mock(), _run_sources=lambda _: [
            ("active", None, None, None, "running"),
            ("pending", None, None, None, "pending"),
            ("empty", None, None, None, "running")]),
        live_runs_for_job=lambda _: dict(active=active, pending=pending, empty=empty))
    host.live_for_run = lambda run, load=True: Window.live_for_run(host, run, load)
    monitor = MonitorPanel(host)
    monitor.refresh_live()
    assert [action.text() for action in monitor.method_button.menu().actions()] == ["active"]
    assert "pending" not in monitor.method_legend.text()
    assert "empty" not in monitor.method_legend.text()
    monitor.close()
    assert app is not None


def test_queue_labels_method_without_started_feed_as_pending():
    from frontend.queue_panel import QueuePanel

    app = QApplication.instance() or QApplication([])
    active = LiveRun()
    active.add([{"type": "start"}])
    panel = QueuePanel(lambda *_: None, split_methods=True,
                       lives_for_job=lambda _: {"active": active, "pending": LiveRun()})
    panel.set_data({"active": [{"job_id": "j", "status": "running", "agents": ["active", "pending"]}]})
    assert panel.cards["j:active"].status.text() == "Running"
    assert panel.cards["j:pending"].status.text() == "Pending"
    panel.close()
    assert app is not None


def test_desktop_does_not_create_empty_feeds_for_pending_workers(tmp_path):
    from PyQt6.QtCore import QObject
    from frontend.job_manager import JobManager as DesktopManager

    manager = DesktopManager.__new__(DesktopManager)
    QObject.__init__(manager)
    manager.jobs = {"j": {"job_id": "j"}}
    manager._feeds = {}
    manager._run_sources = lambda _: [("pending", tmp_path, "m", "version_0", "pending")]
    received = []
    manager.job_records.connect(lambda *args: received.append(args))
    manager._scan_metrics("j")
    assert received == []
    assert manager._feeds == {}


def test_monitor_removes_finished_curves_and_clears_after_parent_finishes():
    app = QApplication.instance() or QApplication([])
    run = new_run(Config(), simulated=False)
    run["backend"] = dict(group="g", experiment_id="e", agents=["a", "b"])
    lives = {name: LiveRun() for name in ("a", "b")}
    for live in lives.values():
        live.add([{"type": "metrics", "step": 1, "data": {"eval/reward": 2}}])
    host = SimpleNamespace(selected=run, runs=[run], live_for_run=lambda *a, **kw: lives)
    monitor = MonitorPanel(host)
    monitor.refresh_live()
    assert set(monitor.charts_view.points) == {"a", "b"}
    lives["a"].finish = {"status": "completed"}
    monitor.refresh_live()
    assert set(monitor.charts_view.points) == {"b"}
    assert [a.text() for a in monitor.method_button.menu().actions()] == ["b"]
    run["status"] = "completed"
    monitor.refresh_live()
    assert monitor.charts_view.points == {}
    assert monitor.method_legend.text() == ""
    assert monitor._monitor_runs() == []
    assert lives["a"].rows  # Completion changes visibility, never persisted training data.
    monitor.close()
    assert app is not None


def test_live_sweep_dashboard_keeps_completed_trials_in_results(tmp_path, monkeypatch, tuner_monitor_workspace):
    from src.app.pipeline import tuning_monitor as sweep_view

    app = QApplication.instance() or QApplication([])
    run = new_run(Config(), simulated=False)
    study = {"study": "s", "stop_reason": "running", "best_trial_id": 0, "trials": [
        {"id": 0, "state": "complete", "value": 3},
        {"id": 1, "state": "running", "execution_started": 1}]}
    monkeypatch.setattr(sweep_view, "study_summaries", lambda *_, **kw: [("m", study)])
    host = SimpleNamespace(selected=run, runs=[run], live_for_run=lambda *a, **kw: {},
                           jobs=SimpleNamespace(project_root=tmp_path, queue_changed=Mock()))
    host.workspace_paths, run['config'] = tuner_monitor_workspace
    monitor = MonitorPanel(host)
    monitor.show_view("Sweep")
    monitor.refresh_live()
    assert [t["id"] for _, s in monitor.sweep_view.studies for t in s["trials"]] == [0, 1]
    assert "Best 3" in monitor.sweep_view.summary.text()
    assert [t["id"] for t in study["trials"]] == [0, 1]
    study["trials"][1].pop("execution_started")
    monitor.refresh_live()
    assert monitor.sweep_view.studies == []
    assert monitor.method_button.isHidden()
    study["trials"][1]["execution_started"] = 1
    study["stop_reason"] = "trial_limit"
    monitor.refresh_live()
    assert monitor.sweep_view.studies == []
    assert monitor.method_button.isHidden()
    monitor.close()
    assert app is not None
