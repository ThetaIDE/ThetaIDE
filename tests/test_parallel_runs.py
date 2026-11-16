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


