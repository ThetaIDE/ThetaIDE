"""Several experiments training at once: the queue's Run up to N, the window's active runs, the monitor's switcher.

Jobs here never start a real pipeline: JobManager._start is replaced, so the tests need no training stack.
"""

import sys
import time

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)  # the window embeds WebEngine
app = QApplication.instance() or QApplication(sys.argv[:1])  # WebEngine needs a program name

from frontend.job_manager import JobManager  # noqa: E402


@pytest.fixture
def manager(tmp_path, monkeypatch):
    manager = JobManager(store_path=tmp_path / "jobs.db")
    started = []
    monkeypatch.setattr(manager, "_start", lambda job: (job.update(status="running", started=time.time()),
                                                      started.append(job["job_id"])))
    monkeypatch.setattr(manager, "_scan_metrics", lambda job_id: None)
    monkeypatch.setattr(manager, "_unwatch", lambda job: None)
    manager.started = started
    return manager


def queue(manager, *ids):
    for job_id in ids:
        manager.jobs[job_id] = {"job_id": job_id, "status": "queued", "experiment_id": job_id, "group": "g",
                                "agents": ["ppo"], "created": time.time()}
        manager.queue_order.append(job_id)


def test_the_queue_runs_up_to_n_at_once_in_order(manager):
    manager.max_parallel = 2
    queue(manager, "a", "b", "c")
    manager.start_queue()
    assert manager.started == ["a", "b"] and manager.queue_order == ["c"]
    manager._finish("a", 0)  # one finishes: the next queued one takes its place
    assert manager.started == ["a", "b", "c"]
    assert manager.queue_running  # nothing queued, but "b" and "c" are still training
    manager._finish("b", 0)
    manager._finish("c", 0)
    assert not manager.queue_running  # drained and the last one finished: it pauses itself


def test_raising_the_limit_starts_more_right_away(manager):
    queue(manager, "a", "b", "c")
    manager.start_queue()
    assert manager.started == ["a"]  # one at a time by default
    manager.set_max_parallel(3)
    assert manager.started == ["a", "b", "c"]
    assert manager.running_count() == 3


def test_lowering_the_limit_stops_nothing_that_is_running(manager):
    manager.max_parallel = 3
    queue(manager, "a", "b", "c", "d")
    manager.start_queue()
    manager.set_max_parallel(1)
    assert manager.running_count() == 3 and manager.queue_order == ["d"]
    manager._finish("a", 0)
    manager._finish("b", 0)
    assert manager.started == ["a", "b", "c"]  # "d" waits: one at a time now, and "c" is still training
    manager._finish("c", 0)
    assert manager.started == ["a", "b", "c", "d"]


# ── The window ───────────────────────────────────────────────────────────────


@pytest.fixture
def window(tmp_path, monkeypatch):
    from frontend.app import Window

    class HeadlessTerminalPanel(QWidget):
        def __init__(self, *, parent=None, **_kwargs):
            super().__init__(parent)
            self.terminals = []

        def configure(self, **_kwargs):
            pass

        def set_font_size(self, _size):
            pass

        def apply_theme(self, _theme):
            pass

    class InertBackend:
        def __init__(self, *_args, **_kwargs):
            pass

        def config_schema(self, *_args, **_kwargs):
            pass

        def compose(self, *_args, **_kwargs):
            pass

        def list_runs(self, *_args, **_kwargs):
            pass

        def run_metrics(self, *_args, **_kwargs):
            pass

    monkeypatch.setattr("frontend.app.TerminalPanel", HeadlessTerminalPanel)
    monkeypatch.setattr("frontend.app.LocalBackend", InertBackend)
    monkeypatch.setattr(Window, "init_default_experiment", lambda _self: None)
    win = Window(tmp_path / "runs")
    yield win
    win._quitting = True
    win.close()


def demo_like(window, name):
    from frontend.model import Config, new_run

    run = new_run(Config(name=name), simulated=False)
    run.update(status="running", backend={"job_id": f"job_{name}", "experiment_id": name, "group": "g"})
    window.runs.insert(0, run)
    return run


def test_the_window_follows_several_active_runs_and_finishes_each_alone(window):
    first, second = demo_like(window, "first"), demo_like(window, "second")
    window.activate(first)
    window.activate(second)
    assert window.active_runs == [first, second] and window.active is second  # the newest is shown
    window.select_run(first)
    assert window.active is first  # Stop acts on the run you are looking at
    window.finish("completed", run=first)
    assert window.active_runs == [second] and first["status"] == "completed" and second["status"] == "running"


def test_launching_waits_for_room(window):
    window.activate(demo_like(window, "first"))
    window.compose_valid, window.current_experiment = True, "csc510/cartpole_demo"
    window.update_launch_state()
    assert not window.queue_panel.launch_button.isEnabled()  # one at a time by default
    window.set_max_parallel(2)
    assert window.queue_panel.launch_button.isEnabled()
    assert window.settings_manager.get("queue", "max_parallel") == 2  # remembered
    window.activate(demo_like(window, "second"))
    assert not window.queue_panel.launch_button.isEnabled()
    assert "2 runs are training" in window.queue_panel.launch_button.toolTip()


def test_the_monitor_switches_between_the_runs_training_now(window):
    first, second = demo_like(window, "first"), demo_like(window, "second")
    switcher = window.monitor.switcher
    window.activate(first)
    assert switcher.isHidden()  # one run: no switcher
    window.activate(second)
    assert not switcher.isHidden() and list(switcher.chips) == [first["id"], second["id"]]
    assert switcher.chips[second["id"]].isChecked()  # the newest run is shown
    switcher.chips[first["id"]].click()
    assert window.selected is first and switcher.chips[first["id"]].isChecked()
    window.finish("completed", run=second)
    assert switcher.isHidden()  # one left training


def test_the_switcher_keeps_one_chip_per_run_as_runs_start(window):
    runs = [demo_like(window, name) for name in ("a", "b", "c")]
    window.set_max_parallel(3)
    assert window.queue_panel.parallel.value() == 3  # the Queue tab shows the limit however it was set
    for run in runs:
        window.activate(run)
    switcher = window.monitor.switcher
    from PyQt6.QtWidgets import QPushButton

    visible = [b for b in switcher.findChildren(QPushButton) if b.isVisibleTo(switcher)]
    assert len(visible) == 3 and sum(b.isChecked() for b in visible) == 1
