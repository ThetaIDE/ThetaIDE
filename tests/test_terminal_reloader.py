"""Unit tests for terminal-based hot-reloading (Ctrl+R) in Theta-IDE."""
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"
if sys.platform == "darwin" and "QTWEBENGINE_CHROMIUM_FLAGS" not in os.environ:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.terminal_reloader import (
        CTRL_C_BYTE,
        CTRL_R_BYTE,
        TerminalManager,
        TerminalReloader,
        build_reload_command,
        reload_frontend,
    )
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestBuildReloadCommand(unittest.TestCase):
    def test_main_module_uses_absolute_entry_point(self):
        cmd = build_reload_command(
            executable="/path/to/python",
            argv=["/Users/user/project/frontend/__main__.py", "--data-dir", "/tmp/runs"],
        )
        self.assertEqual(cmd, ["/path/to/python", str(Path(__file__).resolve().parents[1] / "frontend/__main__.py"), "--data-dir", "/tmp/runs"])

    def test_direct_script_preservation(self):
        cmd = build_reload_command(
            executable="/path/to/python",
            argv=["frontend/app.py", "--data-dir", "/tmp/theta-runs"],
        )
        self.assertEqual(cmd, ["/path/to/python", str(Path(__file__).resolve().parents[1] / "frontend/__main__.py"), "--data-dir", "/tmp/theta-runs"])

    def test_empty_argv_fallback(self):
        cmd = build_reload_command(
            executable="/path/to/python",
            argv=[],
        )
        self.assertEqual(cmd, ["/path/to/python", str(Path(__file__).resolve().parents[1] / "frontend/__main__.py")])

    def test_preserves_interpreter_workspace_and_paths_with_spaces(self):
        executable = str(Path(tempfile.gettempdir()) / "Other User" / "venv" / "python")
        flags = ["--workspace", str(Path(tempfile.gettempdir()) / "Research Workspace"),
                 "--data-dir", str(Path(tempfile.gettempdir()) / "IDE State")]
        cmd = build_reload_command(executable=executable, argv=["frontend/__main__.py", *flags])
        self.assertEqual(cmd[0], executable)
        self.assertTrue(Path(cmd[1]).is_absolute())
        self.assertTrue(Path(cmd[1]).is_file())
        self.assertEqual(cmd[2:], flags)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTerminalManager(unittest.TestCase):
    def test_non_tty_setup_fails_safely(self):
        manager = TerminalManager(fd=0)
        with patch.object(sys.stdin, "isatty", return_value=False):
            success = manager.setup_cbreak()
            self.assertFalse(success)
            self.assertFalse(manager._is_cbreak)

    def test_restore_idempotent(self):
        manager = TerminalManager(fd=0)
        # Should not raise even if cbreak was never set
        manager.restore()
        self.assertFalse(manager._is_cbreak)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTerminalReloader(unittest.TestCase):
    def test_non_tty_start_returns_false(self):
        reloader = TerminalReloader(on_reload=lambda: None)
        with patch.object(sys.stdin, "isatty", return_value=False):
            started = reloader.start()
            self.assertFalse(started)
        reloader.stop()

    def test_trigger_reload_signal_and_debouncing(self):
        reloader = TerminalReloader(on_reload=lambda: None)
        emitted = []
        reloader.signaler.reload_requested.connect(lambda: emitted.append(True))

        reloader.trigger_reload()
        app.processEvents()
        self.assertEqual(len(emitted), 1)

        # Second trigger should be debounced and not emit again
        reloader.trigger_reload()
        app.processEvents()
        self.assertEqual(len(emitted), 1)

        reloader.stop()

    def test_trigger_quit_signal(self):
        reloader = TerminalReloader(on_reload=lambda: None)
        emitted = []
        reloader.signaler.quit_requested.connect(lambda: emitted.append(True))

        reloader.trigger_quit()
        app.processEvents()
        self.assertEqual(len(emitted), 1)
        reloader.stop()

    @unittest.skipIf(sys.platform == "win32", "select() on pipes is POSIX-only; Windows uses the msvcrt listener")
    def test_posix_listener_with_pipe_ctrl_r(self):
        """Simulate Ctrl+R arriving on stdin pipe."""
        r_fd, w_fd = os.pipe()
        mock_reload = MagicMock()
        reloader = TerminalReloader(on_reload=mock_reload)
        reloader.term_manager.fd = r_fd

        reload_calls = []
        reloader.signaler.reload_requested.connect(lambda: reload_calls.append(True))

        # Start thread reading from r_fd
        reloader._stop_event.clear()
        import threading
        t = threading.Thread(target=reloader._posix_listener, daemon=True)
        reloader._thread = t
        t.start()

        # Write Ctrl+R byte
        os.write(w_fd, CTRL_R_BYTE)
        t.join(timeout=1.0)

        app.processEvents()
        self.assertEqual(len(reload_calls), 1)
        mock_reload.assert_called_once()

        os.close(w_fd)
        os.close(r_fd)
        reloader.stop()

    @unittest.skipIf(sys.platform == "win32", "select() on pipes is POSIX-only; Windows uses the msvcrt listener")
    def test_posix_listener_with_pipe_ctrl_c(self):
        """Simulate Ctrl+C arriving on stdin pipe."""
        r_fd, w_fd = os.pipe()
        reloader = TerminalReloader()
        reloader.term_manager.fd = r_fd

        quit_calls = []
        reloader.signaler.quit_requested.connect(lambda: quit_calls.append(True))

        # Start thread reading from r_fd
        reloader._stop_event.clear()
        import threading
        t = threading.Thread(target=reloader._posix_listener, daemon=True)
        reloader._thread = t
        t.start()

        # Write Ctrl+C byte
        os.write(w_fd, CTRL_C_BYTE)
        t.join(timeout=1.0)

        app.processEvents()
        self.assertEqual(len(quit_calls), 1)

        os.close(w_fd)
        os.close(r_fd)
        reloader.stop()

    @unittest.skipIf(sys.platform == "win32", "select() on pipes is POSIX-only; Windows uses the msvcrt listener")
    def test_posix_listener_with_normal_input_ignored(self):
        """Simulate normal typing (e.g. 'hello') on stdin pipe without reload."""
        r_fd, w_fd = os.pipe()
        reloader = TerminalReloader()
        reloader.term_manager.fd = r_fd

        reload_calls = []
        reloader.signaler.reload_requested.connect(lambda: reload_calls.append(True))

        reloader._stop_event.clear()
        import threading
        t = threading.Thread(target=reloader._posix_listener, daemon=True)
        reloader._thread = t
        t.start()

        os.write(w_fd, b"hello\n")
        # Should not trigger reload
        app.processEvents()
        self.assertEqual(len(reload_calls), 0)

        # Closing pipe signals EOF to exit the listener thread
        os.close(w_fd)
        t.join(timeout=1.0)
        self.assertFalse(t.is_alive())

        os.close(r_fd)
        reloader.stop()


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestReloadFrontendAction(unittest.TestCase):
    @patch("os.execv")
    def test_cancelled_close_does_not_reload_or_stop_listener(self, mock_execv):
        window, application, reloader = MagicMock(), MagicMock(), MagicMock()
        window._quitting = False
        window.close.return_value = False
        reloader._reloading = True

        reload_frontend(window=window, app=application, reloader=reloader)

        self.assertFalse(window._quitting)
        self.assertFalse(window._frontend_reload_pending)
        self.assertFalse(reloader._reloading)
        window.windowHandle.assert_not_called()
        reloader.stop.assert_not_called()
        application.processEvents.assert_not_called()
        mock_execv.assert_not_called()

    @patch("os.execv")
    def test_failed_close_does_not_reload(self, mock_execv):
        window = MagicMock()
        window._quitting = False
        window.close.side_effect = RuntimeError("Cannot save")

        reload_frontend(window=window)

        self.assertFalse(window._quitting)
        self.assertFalse(window._frontend_reload_pending)
        window.windowHandle.assert_not_called()
        mock_execv.assert_not_called()

    @patch("os.execv")
    def test_native_window_is_destroyed_before_reexec_and_reentrant_reload_is_ignored(self, mock_execv):
        window, application = MagicMock(), MagicMock()
        events = []
        window.close.side_effect = lambda: events.append("close") or True
        window.windowHandle.return_value.destroy.side_effect = lambda: events.append("destroy")

        def process_events():
            events.append("events")
            reload_frontend(window=window, app=application)

        application.processEvents.side_effect = process_events
        mock_execv.side_effect = lambda *args: events.append("exec")
        reload_frontend(window=window, app=application)

        self.assertEqual(events, ["close", "destroy", "events", "exec"])
        mock_execv.assert_called_once()

    @patch("os.execv")
    def test_fullscreen_native_surface_is_released_before_reexec(self, mock_execv):
        from PyQt6.QtCore import QEvent, QObject
        from PyQt6.QtGui import QPlatformSurfaceEvent
        from PyQt6.QtWidgets import QMainWindow

        events = []

        class SurfaceObserver(QObject):
            def eventFilter(self, watched, event):
                if (event.type() == QEvent.Type.PlatformSurface
                        and event.surfaceEventType() == QPlatformSurfaceEvent.SurfaceEventType.SurfaceAboutToBeDestroyed):
                    events.append("destroy")
                return False

        window = QMainWindow()
        window.showFullScreen()
        app.processEvents()
        observer = SurfaceObserver()
        window.windowHandle().installEventFilter(observer)
        mock_execv.side_effect = lambda *args: events.append("exec")

        reload_frontend(window=window, app=app)

        self.assertEqual(events, ["destroy", "exec"])
        window.deleteLater()

    @patch("os.execv")
    def test_reload_frontend_executes_process(self, mock_execv):
        mock_window = MagicMock()
        mock_app = MagicMock()
        mock_reloader = MagicMock()

        reload_frontend(
            window=mock_window,
            app=mock_app,
            reloader=mock_reloader,
            argv=["frontend/app.py", "--data-dir", "/tmp/data"],
        )

        mock_reloader.stop.assert_called_once()
        mock_window.close.assert_called_once()
        mock_app.processEvents.assert_called_once()
        mock_execv.assert_called_once()

        args = mock_execv.call_args[0]
        self.assertEqual(args[0], sys.executable)
        self.assertEqual(args[1], [sys.executable, str(Path(__file__).resolve().parents[1] / "frontend/__main__.py"), "--data-dir", "/tmp/data"])

    def test_real_qt_reload_reexecutes_from_outside_checkout(self):
        root = Path(__file__).resolve().parents[1]
        # Exercise the real close, event processing, and execv; the replacement
        # exits via --help so the test cannot open a second IDE or touch state.
        script = f'''
import sys
sys.path.insert(0, {str(root)!r})
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow
from frontend.terminal_reloader import reload_frontend
app = QApplication([])
window = QMainWindow()
window.show()
QTimer.singleShot(0, lambda: reload_frontend(window=window, app=app,
    argv=[{str(root / "frontend/__main__.py")!r}, "--help"]))
app.exec()
raise SystemExit("Reload exited without restarting")
'''
        with tempfile.TemporaryDirectory(prefix="theta reload ") as outside_checkout:
            result = subprocess.run([sys.executable, "-c", script], cwd=outside_checkout,
                                    env=dict(os.environ, QT_QPA_PLATFORM="offscreen"),
                                    capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--workspace", result.stdout)


if __name__ == "__main__":
    unittest.main()
