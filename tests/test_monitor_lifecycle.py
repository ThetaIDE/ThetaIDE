"""A monitor follows the current launch and drops completed training data."""
from types import SimpleNamespace
from unittest.mock import Mock

from PyQt6.QtWidgets import QApplication

from frontend.app import Window
from frontend.live_run import LiveRun
from frontend.model import Config, new_run
from frontend.monitor.panel import MonitorPanel


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
