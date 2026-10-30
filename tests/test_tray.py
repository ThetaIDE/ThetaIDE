"""System tray: closing the window while training hides ThetaIDE to the tray, so runs keep going."""

import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.app import Window  # noqa: E402
from frontend.tray import BADGE_COLORS, TrayController  # noqa: E402

RUNNING = {"job_id": "j1", "experiment_id": "cartpole_long", "status": "running"}


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: True))
    quits = []
    monkeypatch.setattr(QApplication, "quit", staticmethod(lambda: quits.append(True)))
    temp = tempfile.TemporaryDirectory()
    win = Window(Path(temp.name) / "runs")
    win.tray = TrayController(win)
    messages = []
    win.tray.icon.showMessage = lambda title, body, *args: messages.append((title, body))
    shutdowns = []
    real_shutdown = win.jobs.shutdown
    win.jobs.shutdown = lambda: (shutdowns.append(True), real_shutdown())
    win.show()
    app.processEvents()
    win.test_quits, win.test_messages, win.test_shutdowns = quits, messages, shutdowns
    yield win
    # Teardown runs before monkeypatch is undone: make the jobs idle so closing doesn't ask
    win.jobs.is_busy = lambda: False
    win.jobs.active_jobs = lambda: []
    win.tray = None
    win._quitting = True
    win.close()
    temp.cleanup()


def training(win, monkeypatch, busy=True):
    monkeypatch.setattr(win.jobs, "is_busy", lambda: busy)
    monkeypatch.setattr(win.jobs, "active_jobs", lambda: [dict(RUNNING)] if busy else [])


def test_closing_while_training_hides_to_the_tray(window, monkeypatch):
    training(window, monkeypatch)
    window.close()
    assert not window.isVisible()
    assert window.test_shutdowns == []  # training was left running
    assert window.test_quits == []
    assert window.test_messages and "cartpole_long" in window.test_messages[-1][1]


def test_closing_while_idle_quits(window, monkeypatch):
    training(window, monkeypatch, busy=False)
    window.close()
    assert window.test_quits and window.test_shutdowns


def test_clicking_the_tray_icon_reopens_the_window(window, monkeypatch):
    training(window, monkeypatch)
    window.close()
    window.tray._on_activated(QSystemTrayIcon.ActivationReason.Trigger)
    assert window.isVisible()


def test_quit_while_training_asks_and_no_keeps_it_running(window, monkeypatch):
    training(window, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.No))
    window.quit_app()
    assert window.test_quits == [] and window.test_shutdowns == []
    assert window._quitting is False


def test_quit_while_training_yes_stops_training_and_quits(window, monkeypatch):
    training(window, monkeypatch)
    monkeypatch.setattr(QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))
    window.quit_app()
    assert window.test_shutdowns and window.test_quits


def test_without_a_tray_closing_falls_back_to_asking(window, monkeypatch):
    training(window, monkeypatch)
    monkeypatch.setattr(QSystemTrayIcon, "isSystemTrayAvailable", staticmethod(lambda: False))
    asked = []
    monkeypatch.setattr(QMessageBox, "question",
                        staticmethod(lambda *a, **k: asked.append(True) or QMessageBox.StandardButton.No))
    window.close()
    assert asked and window.isVisible()


def test_tray_menu_shows_what_is_training(window, monkeypatch):
    training(window, monkeypatch)
    window.tray.refresh()
    assert window.tray.status_action.text() == "Training: cartpole_long"
    assert window.tray.stop_action.isEnabled()
    assert "cartpole_long" in window.tray.icon.toolTip()


def badge_pixel(win, size=32):
    """Colour at the centre of the badge (bottom-right quarter) of the tray icon at `size`."""
    image = win.tray.icon.icon().pixmap(size, size).toImage()
    scale = image.width() / size
    return image.pixelColor(int(size * 0.75 * scale), int(size * 0.75 * scale)).name()


def test_icon_shows_a_badge_while_training(window, monkeypatch):
    assert window.tray.indicator == "idle"
    training(window, monkeypatch)
    window.tray.refresh()
    assert window.tray.indicator == "training"
    assert badge_pixel(window) == BADGE_COLORS["training"]
    training(window, monkeypatch, busy=False)
    window.tray.refresh()
    assert window.tray.indicator == "idle"
    assert window.tray.icon.icon().cacheKey() == window.tray._icons["idle"].cacheKey()


def test_icon_shows_a_queued_badge_while_the_queue_waits(window, monkeypatch):
    training(window, monkeypatch, busy=False)
    window.jobs.queue_order = ["j2"]
    window.jobs.queue_running = False
    window.tray.refresh()
    assert window.tray.indicator == "idle"  # paused queue: nothing will start on its own
    window.jobs.queue_running = True
    window.tray.refresh()
    assert window.tray.indicator == "queued"
    assert badge_pixel(window) == BADGE_COLORS["queued"]
    window.jobs.queue_running = False
    window.jobs.queue_order = []


def test_stop_all_cancels_running_jobs_and_pauses_the_queue(window, monkeypatch):
    training(window, monkeypatch)
    cancelled = []
    monkeypatch.setattr(window.jobs, "cancel", lambda job_id: cancelled.append(job_id))
    window.jobs.queue_running = True
    window.tray.stop_all()
    assert cancelled == ["j1"] and window.jobs.queue_running is False


def test_tray_job_updates_do_not_send_independent_notifications(window, monkeypatch):
    done = {"job_id": "j1", "experiment_id": "cartpole_long", "status": "completed", "returncode": 0}
    window.tray._on_job_changed(done)
    assert window.test_messages == []  # window visible: the monitor already shows it
    training(window, monkeypatch)
    window.close()
    window.tray._on_job_changed(done)
    assert len(window.test_messages) == 1  # only the hide-to-tray message
    window.tray._on_job_changed(dict(done, status="failed", returncode=1))
    assert len(window.test_messages) == 1
