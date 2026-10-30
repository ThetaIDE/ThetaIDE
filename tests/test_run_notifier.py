"""When a run ends: a toast (a system notification when ThetaIDE is not in front) and a chime."""

import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication, QWidget  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)  # the window embeds WebEngine
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.run_notifier import SOUNDS, SOUNDS_DIR, RunNotifier, describe, duration_text  # noqa: E402
from frontend.toast import MAX_TOASTS, Toast  # noqa: E402


def test_the_duration_reads_like_a_clock_you_would_say_aloud():
    assert [duration_text(s) for s in (None, 45, 134, 3900)] == ["", "45s", "2m 14s", "1h 05m"]


def test_each_ending_has_its_own_mark_colour_and_sound():
    done = describe("completed", "cartpole_demo", 134)
    assert done == {"kind": "completed", "mark": "✓", "title": "cartpole_demo finished",
                    "detail": "Completed · 2m 14s", "color": "primary"}
    failed = describe("failed", "cartpole_demo", 20, exit_code=1)
    assert (failed["kind"], failed["title"], failed["detail"]) == ("failed", "cartpole_demo failed",
                                                                    "Failed · exit code 1 · 20s")
    assert describe("cancelled", "x")["kind"] == describe("stopped", "x")["kind"] == "stopped"
    assert describe("error", "x")["detail"] == "Could not run"
    for kind in ("completed", "failed", "stopped"):
        assert (SOUNDS_DIR / SOUNDS[kind]).is_file()


# ── The window ───────────────────────────────────────────────────────────────


@pytest.fixture
def window(monkeypatch):
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

    temp = tempfile.TemporaryDirectory()
    win = Window(Path(temp.name) / "runs")
    win.resize(1000, 700)
    win.show()
    app.processEvents()
    played, notified = [], []
    monkeypatch.setattr(win.run_notifier, "play", played.append)
    monkeypatch.setattr(win.run_notifier, "in_front", lambda: True)
    win.test_played, win.test_notified = played, notified
    yield win
    win._quitting = True
    win.close()
    temp.cleanup()


def finish_demo(win, status="completed"):
    win.start_demo()
    run = win.active
    win.finish(status)
    return run


def visible_toasts(win):
    return [t for t in win.run_notifier.toasts.toasts if t.isVisible()]


def test_a_finished_run_shows_a_toast_and_plays_its_chime(window):
    run = finish_demo(window)
    [toast] = visible_toasts(window)
    assert toast.mark.text() == "✓" and toast.title.text().endswith(" finished")
    assert toast.detail.text().startswith("Completed · ")
    assert toast.action_button.text() == "View results"
    assert window.test_played == ["completed"]
    window.run_notifier.run_finished(run, "completed")  # the same run again: no second notice
    assert len(visible_toasts(window)) == 1 and window.test_played == ["completed"]


def test_stopping_a_run_notifies_too(window):
    finish_demo(window, "stopped")
    assert visible_toasts(window)[0].mark.text() == "■"
    assert window.test_played == ["stopped"]


def test_view_results_plots_the_run_in_the_plot_viewer(window):
    run = finish_demo(window)
    visible_toasts(window)[0].action_button.click()
    assert window.tabs.currentWidget() is window.plot_viewer
    assert run["id"] in window.monitor.sidebar.compared
    assert visible_toasts(window) == []  # the toast goes once used


def test_away_from_thetaide_the_toast_is_a_system_notification(window, monkeypatch):
    class Tray:
        def available(self):
            return True

        def notify(self, title, body, warning=False):
            window.test_notified.append((title, body, warning))

    window.tray = Tray()
    monkeypatch.setattr(window.run_notifier, "in_front", lambda: False)
    finish_demo(window, "failed")
    assert visible_toasts(window) == []
    [(title, body, warning)] = window.test_notified
    assert title.endswith(" failed") and body.startswith("Failed") and warning is True
    assert window.test_played == ["failed"]  # the chime plays either way
    window.tray = None


def test_settings_switch_the_toast_and_the_sound_off(window):
    window.settings_manager.set("notifications", "toast", False)
    finish_demo(window)
    assert visible_toasts(window) == [] and window.test_played == ["completed"]
    window.settings_manager.set("notifications", "sound", False)
    finish_demo(window)
    assert window.test_played == ["completed"]  # no second chime
    window.settings_manager.set("notifications", "toast", True)
    window.settings_manager.set("notifications", "sound", True)


def test_queued_job_that_fails_before_activation_notifies_once(window):
    from frontend.model import new_run

    run = new_run(window.config(), simulated=False)
    run.update(status="queued", backend={"job_id": "early-failure", "experiment_id": "example"})
    window.runs.insert(0, run)
    job = {"job_id": "early-failure", "status": "error"}
    window.on_job_changed(job)
    window.on_job_changed(job)
    assert run["status"] == "error"
    assert window.test_played == ["failed"]
    assert len(visible_toasts(window)) == 1


def test_the_settings_card_switches_and_tries_them(window):
    view = window.settings_view
    card = view._build_run_notifications_card()  # kept: the card owns the switches
    toggles = view.notification_toggles
    assert toggles["toast"].isChecked() and toggles["sound"].isChecked()  # on by default
    toggles["sound"].setChecked(False)
    assert window.settings_manager.get("notifications", "sound") is False
    view._try_run_notification()
    [toast] = visible_toasts(window)
    assert toast.title.text() == "example_run finished" and window.test_played == []  # sound is off
    toggles["sound"].setChecked(True)
    view._try_run_notification()
    assert window.test_played == ["completed"]
    card.deleteLater()


def test_toasts_stack_upward_from_the_bottom_right_corner(window):
    stack = window.run_notifier.toasts
    first = stack.show(Toast("✓", "first"))
    second = stack.show(Toast("✓", "second"))
    app.processEvents()
    assert second.geometry().bottom() < window.height() and second.geometry().right() < window.width()
    assert first.geometry().bottom() < second.geometry().top()  # newest at the bottom, older above
    for i in range(MAX_TOASTS):
        stack.show(Toast("✓", f"more {i}"))
    assert len(stack.toasts) == MAX_TOASTS and first not in stack.toasts  # the oldest made room
    window.resize(1200, 800)
    app.processEvents()
    newest = stack.toasts[-1]
    assert window.width() - newest.geometry().right() < 40  # still in the corner after a resize
    for toast in list(stack.toasts):
        toast.close_button.click()
    assert stack.toasts == []


def test_a_toast_dismisses_itself_unless_the_pointer_is_on_it(window):
    toast = window.run_notifier.toasts.show(Toast("✓", "brief", timeout_ms=10))
    assert toast.timer.isActive()
    toast.enterEvent(None)
    assert not toast.timer.isActive()  # being read
    toast.leaveEvent(None)
    toast.timer.timeout.emit()
    assert toast not in window.run_notifier.toasts.toasts


def test_the_chimes_load_with_qt_multimedia(window):
    try:
        from PyQt6.QtMultimedia import QSoundEffect  # noqa: F401
    except (ImportError, OSError) as exc:
        pytest.skip(f"Qt Multimedia is unavailable: {exc}")
    notifier = RunNotifier(window)
    for kind in ("completed", "failed", "stopped"):
        effect = notifier._effect(kind)
        assert effect is not None and effect.source().toLocalFile().endswith(SOUNDS[kind])
