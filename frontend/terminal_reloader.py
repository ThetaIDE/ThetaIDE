"""Terminal listener for hot-reloading Theta-IDE on Ctrl+R.

When the frontend is launched from a terminal pane (e.g. ``python -m frontend``
or ``python frontend/app.py``), this module monitors ``sys.stdin`` for the
``Ctrl+R`` keystroke (ASCII 0x12). Upon receiving ``Ctrl+R``, it cleanly closes
open windows and re-executes the process via :func:`os.execv`, reloading all
code and state changes without requiring the user to stop and restart the
command manually.
"""
from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
import select
import sys
import threading
from typing import TYPE_CHECKING, Callable, Optional

from PyQt6.QtCore import QObject, pyqtSignal

if TYPE_CHECKING:
    from PyQt6.QtWidgets import QApplication, QMainWindow

logger = logging.getLogger(__name__)

# Key codes
CTRL_R_BYTE = b"\x12"  # ASCII 18 = Ctrl+R
CTRL_C_BYTE = b"\x03"  # ASCII 3  = Ctrl+C (fallback if ISIG bypassed)


def build_reload_command(
    executable: Optional[str] = None,
    argv: Optional[list[str]] = None,
) -> list[str]:
    """Construct the replacement command-line arguments for :func:`os.execv`.

    Use the absolute, directly executable entry point: macOS app bundles can
    start with a working directory outside the checkout, where ``-m frontend``
    cannot find the package. Preserve custom CLI flags and the interpreter.
    """
    exe = executable or sys.executable
    args = argv if argv is not None else list(sys.argv)

    entry_point = Path(__file__).resolve().with_name("__main__.py")
    return [exe, str(entry_point), *args[1:]]


def _stdin_fd() -> int:
    """File descriptor of stdin, or 0 when stdin has been replaced (e.g. by pytest's capture)."""
    try:
        return sys.stdin.fileno()
    except (AttributeError, OSError, ValueError):
        return 0


class TerminalManager:
    """Manages terminal raw/cbreak mode on POSIX systems with safe cleanup."""

    def __init__(self, fd: Optional[int] = None) -> None:
        self.fd = fd if fd is not None else _stdin_fd()
        self._old_settings = None
        self._is_cbreak = False
        self._lock = threading.Lock()

    def setup_cbreak(self) -> bool:
        """Place terminal into cbreak mode so single keystrokes are received immediately.

        Crucially keeps ``ISIG`` enabled so Ctrl+C (SIGINT) and Ctrl+Z (SIGTSTP)
        continue to be handled normally by the kernel and Python signal handlers.
        """
        if not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            return False

        try:
            import termios
            import tty

            with self._lock:
                if self._is_cbreak:
                    return True
                self._old_settings = termios.tcgetattr(self.fd)
                # tty.setcbreak is universally supported across Python versions
                tty.setcbreak(self.fd, termios.TCSANOW)
                self._is_cbreak = True
                atexit.register(self.restore)
                return True
        except Exception as exc:
            logger.warning("Failed to set cbreak mode on terminal: %s", exc)
            return False

    def restore(self) -> None:
        """Restore original terminal settings."""
        with self._lock:
            if not self._is_cbreak or self._old_settings is None:
                return
            try:
                import termios
                termios.tcsetattr(self.fd, termios.TCSANOW, self._old_settings)
            except Exception as exc:
                logger.debug("Failed to restore terminal mode: %s", exc)
            finally:
                self._is_cbreak = False
                try:
                    atexit.unregister(self.restore)
                except Exception:
                    pass


class ReloadSignaler(QObject):
    """Qt signal bridge for cross-thread reload and quit notifications."""
    reload_requested = pyqtSignal()
    quit_requested = pyqtSignal()


class TerminalReloader:
    """Listens for Ctrl+R on stdin and triggers frontend reload."""

    def __init__(
        self,
        window: Optional["QMainWindow"] = None,
        app: Optional["QApplication"] = None,
        on_reload: Optional[Callable[[], None]] = None,
    ) -> None:
        self.window = window
        self.app = app
        self.on_reload = on_reload
        self.term_manager = TerminalManager()
        self.signaler = ReloadSignaler()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reloading = False

        # Wire Qt signals to main thread handlers
        self.signaler.reload_requested.connect(self._handle_reload)
        self.signaler.quit_requested.connect(self._handle_quit)

    def start(self) -> bool:
        """Initialize terminal mode and start background listener thread."""
        if not hasattr(sys.stdin, "isatty") or not sys.stdin.isatty():
            return False

        # If on Windows, termios won't exist; handle accordingly
        if sys.platform == "win32":
            return self._start_windows()

        success = self.term_manager.setup_cbreak()
        if not success:
            return False

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._posix_listener,
            name="ThetaTerminalReloader",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop background listener and restore terminal mode."""
        self._stop_event.set()
        self.term_manager.restore()
        if self._thread is not None and self._thread.is_alive():
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=0.3)
            self._thread = None

    def trigger_reload(self) -> None:
        """Programmatically trigger reload via Qt signal."""
        if self._reloading:
            return
        self._reloading = True
        self.signaler.reload_requested.emit()

    def trigger_quit(self) -> None:
        """Programmatically trigger quit via Qt signal."""
        self.signaler.quit_requested.emit()

    def _posix_listener(self) -> None:
        """Background thread loop for POSIX stdin reading."""
        fd = self.term_manager.fd
        while not self._stop_event.is_set():
            try:
                r, _, _ = select.select([fd], [], [], 0.2)
                if not r:
                    continue
                chunk = os.read(fd, 1024)
                if not chunk:
                    # Stdin reached EOF (terminal closed)
                    break
                if CTRL_R_BYTE in chunk:
                    self.trigger_reload()
                    break
                elif CTRL_C_BYTE in chunk:
                    self.trigger_quit()
                    break
            except (OSError, select.error):
                break

    def _start_windows(self) -> bool:
        """Background listener thread for Windows console input."""
        try:
            import msvcrt
        except ImportError:
            return False

        def _win_listener():
            while not self._stop_event.is_set():
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        if ch == CTRL_R_BYTE:
                            self.trigger_reload()
                            break
                        elif ch == CTRL_C_BYTE:
                            self.trigger_quit()
                            break
                    self._stop_event.wait(0.2)
                except Exception:
                    break

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=_win_listener,
            name="ThetaTerminalReloaderWin",
            daemon=True,
        )
        self._thread.start()
        return True

    def _handle_reload(self) -> None:
        """Invoked on the Qt main GUI thread when reload is triggered."""
        if self.on_reload is not None:
            self.on_reload()
        else:
            reload_frontend(window=self.window, app=self.app, reloader=self)

    def _handle_quit(self) -> None:
        """Invoked on the Qt main GUI thread if Ctrl+C was received via stdin."""
        self.stop()
        if self.app is not None:
            self.app.quit()
        elif self.window is not None:
            self.window.close()


def reload_frontend(
    window: Optional["QMainWindow"] = None,
    app: Optional["QApplication"] = None,
    reloader: Optional[TerminalReloader] = None,
    argv: Optional[list[str]] = None,
) -> None:
    """Perform a clean shutdown of the current frontend and re-exec the process."""
    if window is not None:
        if getattr(window, "_frontend_reload_pending", False) is True:
            return
        window._frontend_reload_pending = True

    try:
        sys.stdout.write("\r\n\033[36m[Theta-IDE] Reloading frontend (Ctrl+R)...\033[0m\r\n")
        sys.stdout.flush()
    except Exception:
        pass

    if window is not None:
        was_quitting = getattr(window, "_quitting", False)
        try:
            if hasattr(window, "_quitting"):
                window._quitting = True  # a reload closes for real rather than hiding to the tray
            if not window.close():
                window._quitting = was_quitting
                window._frontend_reload_pending = False
                if reloader is not None:
                    reloader._reloading = False
                return
        except Exception as exc:
            window._quitting = was_quitting
            window._frontend_reload_pending = False
            if reloader is not None:
                reloader._reloading = False
            logger.exception("Error while closing window for reload: %s", exc)
            return

    if reloader is not None:
        reloader.stop()

    # QWidget.close() normally only hides the widget. In particular, hiding a
    # fullscreen Cocoa window leaves its dedicated desktop alive. execv skips
    # Qt's destructors, so explicitly release the native window before replacing
    # the process. Do this after close accepts (and saves fullscreen geometry),
    # without deleting the widget while its restart action is still on the stack.
    if window is not None:
        native_window = window.windowHandle()
        if native_window is not None:
            native_window.destroy()

    if app is not None:
        try:
            app.processEvents()
        except Exception:
            pass

    cmd = build_reload_command(argv=argv)
    try:
        os.execv(sys.executable, cmd)
    except Exception as exc:
        try:
            sys.stderr.write(f"\r\n[Theta-IDE] Reload failed: {exc}\r\n")
            sys.stderr.flush()
        except Exception:
            pass
        if app is not None:
            app.quit()
