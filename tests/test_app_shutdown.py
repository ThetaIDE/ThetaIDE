"""Desktop shutdown must finish closing widgets before terminating Qt."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

from frontend.app import Window
from frontend.tray import TrayController


def closing_window():
    return SimpleNamespace(
        _shutting_down=False,
        plugin_manager=SimpleNamespace(can_shutdown=Mock(return_value=True), shutdown=Mock()),
        save_session=Mock(),
        findChildren=Mock(return_value=[]),
        active_runs=[],
        jobs=SimpleNamespace(shutdown=Mock()),
        tray=SimpleNamespace(remove=Mock()),
    )


def test_close_disconnects_without_reentering_quit(monkeypatch):
    window = closing_window()
    quit_app = Mock()
    monkeypatch.setattr(QApplication, "quit", quit_app)
    event = QCloseEvent()

    Window.closeEvent(window, event)

    assert event.isAccepted()
    assert window._shutting_down
    window.jobs.shutdown.assert_called_once_with()
    window.plugin_manager.shutdown.assert_called_once_with(check=False)
    window.tray.remove.assert_called_once_with()
    quit_app.assert_not_called()

    # Qt may deliver another close while processing an application quit.
    Window.closeEvent(window, QCloseEvent())
    window.plugin_manager.can_shutdown.assert_called_once_with()
    window.jobs.shutdown.assert_called_once_with()


def test_cancel_unsaved_work_keeps_desktop_connected():
    window = closing_window()
    window.plugin_manager.can_shutdown.return_value = False
    event = QCloseEvent()

    Window.closeEvent(window, event)

    assert not event.isAccepted()
    assert not window._shutting_down
    window.jobs.shutdown.assert_not_called()
    window.tray.remove.assert_not_called()


def test_removing_tray_does_not_pump_events_during_close(monkeypatch):
    process_events = Mock()
    monkeypatch.setattr(QApplication, "processEvents", process_events)
    tray = SimpleNamespace(icon=SimpleNamespace(hide=Mock()))

    TrayController.remove(tray)

    tray.icon.hide.assert_called_once_with()
    process_events.assert_not_called()


@pytest.mark.parametrize("exit_path", ["close", "quit_app", "application_quit"])
def test_webengine_deferred_deletion_finishes_before_process_exit(exit_path):
    # A separate process exercises Qt's real event loop and destruction, including
    # the styled QStackedWidget/WebEngine hierarchy from the macOS crash report.
    script = r'''
import sys
from types import SimpleNamespace
from unittest.mock import Mock
from PyQt6 import sip
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QVBoxLayout, QWidget
from frontend.app import Window
from frontend.theme import STYLE

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication(["shutdown-test"])
app.setStyle("Fusion")
app.setStyleSheet(STYLE)
app.setQuitOnLastWindowClosed(True)

class ShutdownWindow(QMainWindow):
    closeEvent = Window.closeEvent
    quit_app = Window.quit_app

window = ShutdownWindow()
window._quitting = window._shutting_down = False
window.save_session = Mock()
window.active_runs = []
window.jobs = SimpleNamespace(shutdown=Mock())
window.tray = SimpleNamespace(remove=Mock())
panel = QWidget(window)
stack = QStackedWidget(panel)
layout = QVBoxLayout(panel)
layout.addWidget(stack)
web = QWebEngineView(stack)
stack.addWidget(web)
web.setHtml("<html><body contenteditable>Editor buffer</body></html>")
window.setCentralWidget(panel)
window.plugin_manager = SimpleNamespace(
    can_shutdown=lambda: True,
    shutdown=lambda **kwargs: panel.deleteLater(),
)
quitting = []
app.aboutToQuit.connect(lambda: quitting.append(True))
window.show()
action = {"close": window.close, "quit_app": window.quit_app, "application_quit": app.quit}[sys.argv[1]]
QTimer.singleShot(500, action)
QTimer.singleShot(15000, lambda: sys.exit("Shutdown did not finish"))
assert app.exec() == 0
assert quitting
assert window._shutting_down
assert sip.isdeleted(web), "WebEngine destruction was deferred until interpreter exit"
window.jobs.shutdown.assert_called_once_with()
'''
    env = dict(
        os.environ,
        QT_QPA_PLATFORM=os.environ.get("THETA_SHUTDOWN_TEST_PLATFORM", "offscreen"),
        QTWEBENGINE_CHROMIUM_FLAGS="--disable-gpu",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, exit_path],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
