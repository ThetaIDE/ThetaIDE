"""PyQt6 QWebEngineView wrapper for xterm.js terminal emulator with PTY integration."""
import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QObject, QTimer, QPoint
from PyQt6.QtGui import QAction, QColor, QCursor
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QPlainTextEdit, QFrame, QSizePolicy, QMenu, QApplication,
    QListWidget, QListWidgetItem, QStackedWidget, QToolButton,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel

from ..theme import current_theme
from .pty_session import PtySession, available_shells, get_default_shell, parse_shell_command


HTML_PATH = Path(__file__).resolve().parent / "terminal.html"

# Authentic terminal palettes for built-in themes
NATIVE_TERMINAL_PALETTES = {
    "catppuccin": {
        "background": "#1e1e2e",
        "foreground": "#cdd6f4",
        "cursor": "#f5e0dc",
        "cursorAccent": "#1e1e2e",
        "selectionBackground": "rgba(88, 91, 112, 0.45)",
        "selectionForeground": "#cdd6f4",
        "black": "#45475a",
        "red": "#f38ba8",
        "green": "#a6e3a1",
        "yellow": "#f9e2af",
        "blue": "#89b4fa",
        "magenta": "#f5c2e7",
        "cyan": "#94e2d5",
        "white": "#bac2de",
        "brightBlack": "#585b70",
        "brightRed": "#f38ba8",
        "brightGreen": "#a6e3a1",
        "brightYellow": "#f9e2af",
        "brightBlue": "#89b4fa",
        "brightMagenta": "#cba6f7",
        "brightCyan": "#94e2d5",
        "brightWhite": "#a6adc8",
    },
    "catppuccin macchiato": {
        "background": "#24273a",
        "foreground": "#cad3f5",
        "cursor": "#f4dbd6",
        "cursorAccent": "#24273a",
        "selectionBackground": "rgba(110, 115, 141, 0.45)",
        "selectionForeground": "#cad3f5",
        "black": "#494d64",
        "red": "#ed8796",
        "green": "#a6da95",
        "yellow": "#eed49f",
        "blue": "#8aadf4",
        "magenta": "#f5bde6",
        "cyan": "#8bd5ca",
        "white": "#b8c0e0",
        "brightBlack": "#5b6078",
        "brightRed": "#ed8796",
        "brightGreen": "#a6da95",
        "brightYellow": "#eed49f",
        "brightBlue": "#8aadf4",
        "brightMagenta": "#c6a0f6",
        "brightCyan": "#8bd5ca",
        "brightWhite": "#cad3f5",
    },
    "catppuccin latte": {
        "background": "#eff1f5",
        "foreground": "#4c4f69",
        "cursor": "#dc8a78",
        "cursorAccent": "#eff1f5",
        "selectionBackground": "rgba(172, 176, 190, 0.45)",
        "selectionForeground": "#4c4f69",
        "black": "#5c5f77",
        "red": "#d20f39",
        "green": "#40a02b",
        "yellow": "#df8e1d",
        "blue": "#1e66f5",
        "magenta": "#ea76cb",
        "cyan": "#179299",
        "white": "#acb0be",
        "brightBlack": "#6c6f85",
        "brightRed": "#d20f39",
        "brightGreen": "#40a02b",
        "brightYellow": "#df8e1d",
        "brightBlue": "#1e66f5",
        "brightMagenta": "#8839ef",
        "brightCyan": "#179299",
        "brightWhite": "#bcc0cc",
    },
    "dracula": {
        "background": "#282a36",
        "foreground": "#f8f8f2",
        "cursor": "#f8f8f2",
        "cursorAccent": "#282a36",
        "selectionBackground": "rgba(68, 71, 90, 0.5)",
        "black": "#21222c",
        "red": "#ff5555",
        "green": "#50fa7b",
        "yellow": "#f1fa8c",
        "blue": "#bd93f9",
        "magenta": "#ff79c6",
        "cyan": "#8be9fd",
        "white": "#f8f8f2",
        "brightBlack": "#6272a4",
        "brightRed": "#ff6e6e",
        "brightGreen": "#69ff94",
        "brightYellow": "#ffffa5",
        "brightBlue": "#d6acff",
        "brightMagenta": "#ff92df",
        "brightCyan": "#a4ffff",
        "brightWhite": "#ffffff",
    },
    "nord": {
        "background": "#2e3440",
        "foreground": "#d8dee9",
        "cursor": "#d8dee9",
        "cursorAccent": "#2e3440",
        "selectionBackground": "rgba(76, 86, 106, 0.5)",
        "black": "#3b4252",
        "red": "#bf616a",
        "green": "#a3be8c",
        "yellow": "#ebcb8b",
        "blue": "#81a1c1",
        "magenta": "#b48ead",
        "cyan": "#88c0d0",
        "white": "#e5e9f0",
        "brightBlack": "#4c566a",
        "brightRed": "#bf616a",
        "brightGreen": "#a3be8c",
        "brightYellow": "#ebcb8b",
        "brightBlue": "#81a1c1",
        "brightMagenta": "#b48ead",
        "brightCyan": "#8fbcbb",
        "brightWhite": "#eceff4",
    },
    "gruvbox light": {
        "background": "#fbf1c7",
        "foreground": "#3c3836",
        "cursor": "#9d6000",
        "cursorAccent": "#fbf1c7",
        "selectionBackground": "rgba(213, 196, 161, 0.6)",
        "selectionForeground": "#3c3836",
        "black": "#282828",
        "red": "#cc241d",
        "green": "#79740e",
        "yellow": "#b57614",
        "blue": "#076678",
        "magenta": "#8f3f71",
        "cyan": "#427b58",
        "white": "#7c6f64",
        "brightBlack": "#928374",
        "brightRed": "#9d0006",
        "brightGreen": "#79740e",
        "brightYellow": "#b57614",
        "brightBlue": "#076678",
        "brightMagenta": "#8f3f71",
        "brightCyan": "#427b58",
        "brightWhite": "#3c3836",
    },
    "paper": {
        "background": "#ffffff",
        "foreground": "#202b3b",
        "cursor": "#6246b5",
        "cursorAccent": "#ffffff",
        "selectionBackground": "rgba(226, 231, 239, 0.7)",
        "selectionForeground": "#202b3b",
        "black": "#202b3b",
        "red": "#d32f2f",
        "green": "#287448",
        "yellow": "#c77700",
        "blue": "#176b91",
        "magenta": "#8851c5",
        "cyan": "#00838f",
        "white": "#526176",
        "brightBlack": "#65748a",
        "brightRed": "#c62828",
        "brightGreen": "#2e7d32",
        "brightYellow": "#e65100",
        "brightBlue": "#1565c0",
        "brightMagenta": "#6a1b9a",
        "brightCyan": "#00695c",
        "brightWhite": "#101620",
    },
    "apollo": {
        "background": "#f5f0e6",
        "foreground": "#292722",
        "cursor": "#8e4a32",
        "cursorAccent": "#f5f0e6",
        "selectionBackground": "rgba(217, 200, 169, 0.65)",
        "selectionForeground": "#292722",
        "black": "#292722",
        "red": "#a34832",
        "green": "#637447",
        "yellow": "#9a741e",
        "blue": "#496a8a",
        "magenta": "#765b7a",
        "cyan": "#397d78",
        "white": "#817a6d",
        "brightBlack": "#918a76",
        "brightRed": "#b56d32",
        "brightGreen": "#748754",
        "brightYellow": "#9a741e",
        "brightBlue": "#496a8a",
        "brightMagenta": "#765b7a",
        "brightCyan": "#397d78",
        "brightWhite": "#eae2d3",
    },
    "athena": {
        "background": "#f1f3f0",
        "foreground": "#252b2c",
        "cursor": "#456e72",
        "cursorAccent": "#f1f3f0",
        "selectionBackground": "rgba(203, 218, 217, 0.65)",
        "selectionForeground": "#252b2c",
        "black": "#252b2c",
        "red": "#a34d4a",
        "green": "#55745f",
        "yellow": "#8a741f",
        "blue": "#496a8c",
        "magenta": "#6d617f",
        "cyan": "#397b7d",
        "white": "#737d7d",
        "brightBlack": "#7b8580",
        "brightRed": "#a96e3f",
        "brightGreen": "#658770",
        "brightYellow": "#8a741f",
        "brightBlue": "#496a8c",
        "brightMagenta": "#6d617f",
        "brightCyan": "#397b7d",
        "brightWhite": "#e4e9e7",
    },
    "dionysus": {
        "background": "#211820",
        "foreground": "#e5d8c8",
        "cursor": "#c9946a",
        "cursorAccent": "#211820",
        "selectionBackground": "rgba(74, 48, 64, 0.6)",
        "selectionForeground": "#e5d8c8",
        "black": "#2d2029",
        "red": "#c45a5a",
        "green": "#71875a",
        "yellow": "#c5a05a",
        "blue": "#687a9b",
        "magenta": "#a66a91",
        "cyan": "#65958d",
        "white": "#e5d8c8",
        "brightBlack": "#71656b",
        "brightRed": "#c17a45",
        "brightGreen": "#839d69",
        "brightYellow": "#c5a05a",
        "brightBlue": "#687a9b",
        "brightMagenta": "#a66a91",
        "brightCyan": "#65958d",
        "brightWhite": "#ffffff",
    },
    "ares": {
        "background": "#171514",
        "foreground": "#e4ddd2",
        "cursor": "#c65a45",
        "cursorAccent": "#171514",
        "selectionBackground": "rgba(71, 37, 34, 0.6)",
        "selectionForeground": "#e4ddd2",
        "black": "#211c1a",
        "red": "#d04a3e",
        "green": "#697354",
        "yellow": "#c09a55",
        "blue": "#596b7a",
        "magenta": "#765866",
        "cyan": "#557a78",
        "white": "#e4ddd2",
        "brightBlack": "#6f6861",
        "brightRed": "#c16a3a",
        "brightGreen": "#7b8663",
        "brightYellow": "#c09a55",
        "brightBlue": "#596b7a",
        "brightMagenta": "#765866",
        "brightCyan": "#557a78",
        "brightWhite": "#ffffff",
    },
    "poseidon": {
        "background": "#080f14",
        "foreground": "#d3dedc",
        "cursor": "#5ba8a5",
        "cursorAccent": "#080f14",
        "selectionBackground": "rgba(24, 52, 60, 0.6)",
        "selectionForeground": "#d3dedc",
        "black": "#0d171d",
        "red": "#c95b59",
        "green": "#527d69",
        "yellow": "#b6a263",
        "blue": "#527ba3",
        "magenta": "#716b8c",
        "cyan": "#3f9698",
        "white": "#d3dedc",
        "brightBlack": "#526369",
        "brightRed": "#b77d4e",
        "brightGreen": "#63947e",
        "brightYellow": "#b6a263",
        "brightBlue": "#527ba3",
        "brightMagenta": "#716b8c",
        "brightCyan": "#3f9698",
        "brightWhite": "#ffffff",
    },
}



def _is_light_hex(hex_code: str) -> bool:
    """Determine whether a hex color has light luminance."""
    try:
        c = hex_code.lstrip("#")
        if len(c) == 6:
            r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
            return (0.299 * r + 0.587 * g + 0.114 * b) > 135
    except Exception:
        pass
    return False


class TerminalBridge(QObject):
    """Bridge object exposed to JavaScript via QWebChannel."""

    data_received = pyqtSignal(str)
    theme_received = pyqtSignal(str)
    font_size_received = pyqtSignal(int)
    clear_requested = pyqtSignal()

    def __init__(self, terminal_widget: "TerminalWidget"):
        super().__init__()
        self.terminal_widget = terminal_widget

    @pyqtSlot(str)
    def send_input(self, data: str):
        """Called from xterm.js on user keyboard input."""
        if not self.terminal_widget.pty.is_alive():
            # If the process exited, pressing Enter or newline restarts the session
            if "\r" in data or "\n" in data or data == " ":
                self.terminal_widget.restart_session()
                return
            return
        self.terminal_widget.pty.write(data)

    @pyqtSlot(int, int)
    def resize_pty(self, cols: int, rows: int):
        """Called from xterm.js when container or window resizes."""
        if cols > 0 and rows > 0:
            self.terminal_widget._current_cols = cols
            self.terminal_widget._current_rows = rows
        self.terminal_widget.pty.resize(cols, rows)

    @pyqtSlot(int, int, str)
    def show_context_menu(self, x: int, y: int, selected_text: str = ""):
        """Called from JavaScript on right-click."""
        self.terminal_widget.show_custom_context_menu(x, y, selected_text)

    @pyqtSlot(str)
    def set_clipboard(self, text: str):
        """Called from JavaScript when tmux or an inner program sets the clipboard via OSC 52."""
        if text:
            QApplication.clipboard().setText(text)

    @pyqtSlot()
    def terminal_ready(self):
        """Called when xterm.js is initialized and mounted in DOM."""
        self.terminal_widget._on_terminal_ready()


class TerminalWidget(QWidget):
    """Interactive xterm.js terminal widget backed by an OS pseudo-terminal."""

    session_started = pyqtSignal()
    session_exited = pyqtSignal(int)

    def __init__(self, cwd: Optional[str] = None, parent: Optional[QWidget] = None,
                 command: Optional[list[str]] = None, env: Optional[dict[str, str]] = None):
        super().__init__(parent)
        self.cwd = cwd or str(Path.cwd())
        self.pty = PtySession(cwd=self.cwd, command=command, parent=self)
        self.pty.extra_env = dict(env or {})
        self.bridge = TerminalBridge(self)
        self._is_ready = False
        self._closed = False
        self._pending_theme: Optional[dict] = None
        self._current_cols = 80
        self._current_rows = 24

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.web_view = QWebEngineView(self)
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Configure WebEngine settings
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, False)

        # Attach WebChannel
        self.channel = QWebChannel(self.web_view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        layout.addWidget(self.web_view)

        # Connect PTY signals
        self.pty.data_ready.connect(self._on_pty_data)
        self.pty.process_exited.connect(self._on_pty_exit)

        # Load terminal HTML
        self.web_view.load(QUrl.fromLocalFile(str(HTML_PATH)))

    def _on_terminal_ready(self):
        """xterm.js has loaded, mounted, and reported ready."""
        # WebChannel can deliver readiness after the overlay was dismissed.
        # A closed terminal must never launch a hidden upload or SSH session.
        if self._closed:
            return
        self._is_ready = True
        if not self.pty.is_alive():
            self.pty.start(cols=self._current_cols, rows=self._current_rows)
            self.session_started.emit()

        if self._pending_theme:
            self.apply_theme(self._pending_theme)
            self._pending_theme = None

        if hasattr(self, "_font_size") and self._font_size:
            self.bridge.font_size_received.emit(self._font_size)

        # Only claim focus if the terminal is currently visible to the user
        if self.isVisible():
            self.focus_terminal()

    def _on_pty_data(self, text: str):
        """Send raw data from PTY process to xterm.js."""
        self.bridge.data_received.emit(text)

    def _on_pty_exit(self, exit_code: int):
        """Child shell exited; notify xterm and signal."""
        msg = f"\r\n\x1b[90m[Process completed with exit code {exit_code} — Press '+ New shell' or Enter to restart]\x1b[0m\r\n"
        self.bridge.data_received.emit(msg)
        self.session_exited.emit(exit_code)

    def send_command(self, cmd: str):
        """Write a command string followed by Enter to the shell."""
        if not self.pty.is_alive():
            self.restart_session()
        self.pty.write(f"{cmd}\n")
        self.focus_terminal()

    def restart_session(self):
        """Terminate current process and start a fresh shell session."""
        if self._closed:
            return
        self.pty.close()
        self.bridge.clear_requested.emit()
        self.pty.start(cols=self._current_cols, rows=self._current_rows)
        self.session_started.emit()
        self.fit_terminal()
        self.focus_terminal()

    def configure(self, command: Optional[list[str]] = None, cwd: Optional[str] = None,
                  env: Optional[dict[str, str]] = None):
        """Set the shell, start folder and extra environment. Applies from the next shell
        started (New Shell / restart); the running one is left alone."""
        self.pty.command = command
        if cwd:
            self.cwd = cwd
            self.pty.cwd = cwd
        self.pty.extra_env = dict(env or {})

    def clear(self):
        """Clear and reset xterm terminal viewport."""
        self.bridge.clear_requested.emit()

    def fit_terminal(self):
        """Trigger fitAddon.fit() in xterm.js."""
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof fitTerminal === 'function') fitTerminal();")

    def set_font_size(self, size: int):
        """Update terminal font size and refit."""
        self._font_size = size
        if self._is_ready:
            self.bridge.font_size_received.emit(size)

    def focus_terminal(self):
        """Focus keyboard events on the terminal."""
        self.web_view.setFocus()
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.focus();")

    def showEvent(self, event):
        """Handle showing after tab switch to ensure layout coordinates are settled."""
        super().showEvent(event)
        QTimer.singleShot(60, self.fit_terminal)
        QTimer.singleShot(60, self.focus_terminal)

    def blur_terminal(self):
        """Blur terminal to trigger DEC 1004 FocusLost in tmux/vim."""
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.blur();")

    def hideEvent(self, event):
        super().hideEvent(event)
        self.blur_terminal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self.fit_terminal)

    def show_custom_context_menu(self, x: int, y: int, selected_text: str = ""):
        """Display native context menu with standard terminal actions."""
        menu = QMenu(self)

        copy_act = QAction("Copy", menu)
        copy_act.setEnabled(bool(selected_text.strip()))
        def do_copy():
            if selected_text:
                QApplication.clipboard().setText(selected_text)
        copy_act.triggered.connect(do_copy)
        menu.addAction(copy_act)

        paste_act = QAction("Paste", menu)
        clip_text = QApplication.clipboard().text()
        paste_act.setEnabled(bool(clip_text))
        def do_paste():
            text = QApplication.clipboard().text()
            if text:
                if not self.pty.is_alive():
                    self.restart_session()
                self.pty.write(text)
        paste_act.triggered.connect(do_paste)
        menu.addAction(paste_act)

        select_all_act = QAction("Select All", menu)
        select_all_act.triggered.connect(lambda: self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.selectAll();"))
        menu.addAction(select_all_act)

        menu.addSeparator()

        clear_act = QAction("Clear Terminal", menu)
        clear_act.triggered.connect(self.clear)
        menu.addAction(clear_act)

        restart_act = QAction("Restart Shell", menu)
        restart_act.triggered.connect(self.restart_session)
        menu.addAction(restart_act)

        global_pos = self.web_view.mapToGlobal(QPoint(x, y))
        menu.exec(global_pos)

    def resolve_theme_palette(self, theme_dict: dict) -> dict:
        """Map Theta-IDE palette to xterm.js theme object."""
        name_key = (theme_dict.get("name") or "").strip().casefold()
        if name_key in NATIVE_TERMINAL_PALETTES:
            return NATIVE_TERMINAL_PALETTES[name_key]
        elif "catppuccin" in name_key and "latte" in name_key:
            return NATIVE_TERMINAL_PALETTES["catppuccin latte"]
        elif "catppuccin" in name_key and "mocha" in name_key:
            return NATIVE_TERMINAL_PALETTES["catppuccin"]
        elif "catppuccin" in name_key:
            return NATIVE_TERMINAL_PALETTES["catppuccin macchiato"]
        elif "dracula" in name_key:
            return NATIVE_TERMINAL_PALETTES["dracula"]
        elif "nord" in name_key:
            return NATIVE_TERMINAL_PALETTES["nord"]
        elif "gruvbox" in name_key and "light" in name_key:
            return NATIVE_TERMINAL_PALETTES["gruvbox light"]
        elif "paper" in name_key:
            return NATIVE_TERMINAL_PALETTES["paper"]
        elif "apollo" in name_key:
            return NATIVE_TERMINAL_PALETTES["apollo"]
        elif "athena" in name_key:
            return NATIVE_TERMINAL_PALETTES["athena"]
        elif "ares" in name_key:
            return NATIVE_TERMINAL_PALETTES["ares"]
        elif "dionysus" in name_key:
            return NATIVE_TERMINAL_PALETTES["dionysus"]
        elif "poseidon" in name_key:
            return NATIVE_TERMINAL_PALETTES["poseidon"]

        colors = theme_dict.get("colors", {})
        if not colors:
            return {}

        base = colors.get("base", "#1d2021")
        text = colors.get("text", "#ebdbb2")
        accent = colors.get("accent", "#fabd2f")
        primary = colors.get("primary", "#b8bb26")
        secondary = colors.get("secondary", "#83a598")
        number = colors.get("number", "#d3869b")
        comment = colors.get("comment", "#928374")
        border = colors.get("border", "#504945")

        is_light = _is_light_hex(base)
        if is_light:
            return {
                "background": base,
                "foreground": text,
                "cursor": accent,
                "cursorAccent": base,
                "selectionBackground": border,
                "black": text,
                "red": "#cc241d",
                "green": primary,
                "yellow": "#b57614",
                "blue": secondary,
                "magenta": number,
                "cyan": "#427b58",
                "white": "#7c6f64",
                "brightBlack": comment,
                "brightRed": "#9d0006",
                "brightGreen": primary,
                "brightYellow": "#b57614",
                "brightBlue": secondary,
                "brightMagenta": number,
                "brightCyan": "#427b58",
                "brightWhite": text,
            }
        return {
            "background": base,
            "foreground": text,
            "cursor": accent,
            "cursorAccent": base,
            "selectionBackground": border,
            "black": base,
            "red": "#ea6962",
            "green": primary,
            "yellow": accent,
            "blue": secondary,
            "magenta": number,
            "cyan": "#8ec07c",
            "white": text,
            "brightBlack": comment,
            "brightRed": "#fb4934",
            "brightGreen": primary,
            "brightYellow": accent,
            "brightBlue": secondary,
            "brightMagenta": number,
            "brightCyan": "#8ec07c",
            "brightWhite": "#fbf1c7",
        }

    def apply_theme(self, theme_dict: dict):
        """Map Theta-IDE palette to xterm.js theme object and update web view."""
        xterm_theme = self.resolve_theme_palette(theme_dict)
        if not xterm_theme:
            return
        self._current_terminal_theme = xterm_theme
        if not self._is_ready:
            self._pending_theme = theme_dict
            return
        self.bridge.theme_received.emit(json.dumps(xterm_theme))

    @property
    def current_terminal_theme(self) -> dict:
        return getattr(self, "_current_terminal_theme", {})

    def closeEvent(self, event):
        self._closed = True
        self.pty.close()
        super().closeEvent(event)


class TerminalPanel(QWidget):
    """Terminal pane: a toolbar, the active shell, and a list of open shells to switch between.

    `terminal` is always the active TerminalWidget, so callers that drive one terminal
    (Settings' Restart terminal, the theme and font hooks, tests) keep working.
    """

    shells_changed = pyqtSignal()

    def __init__(self, cwd: Optional[str] = None, parent: Optional[QWidget] = None,
                 command: Optional[list[str]] = None, env: Optional[dict[str, str]] = None):
        super().__init__(parent)
        self.cwd = cwd or str(Path.cwd())
        self._command = command
        self._env = dict(env or {})
        self._theme: Optional[dict] = None
        self._font_size: Optional[int] = None
        self._colors: dict = dict(current_theme().get("colors", {}))
        self._next_number = 1
        self.terminals: list[TerminalWidget] = []

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Compact top control header
        self.toolbar = QFrame(self)
        self.toolbar.setObjectName("terminalToolbar")
        self.toolbar.setFixedHeight(28)
        tb_layout = QHBoxLayout(self.toolbar)
        tb_layout.setContentsMargins(10, 0, 10, 0)
        tb_layout.setSpacing(8)

        self.status_dot = QLabel("●", self.toolbar)
        self.status_dot.setObjectName("terminalStatusDot")
        tb_layout.addWidget(self.status_dot)

        self.title_label = QLabel("Terminal", self.toolbar)
        self.title_label.setObjectName("terminalTitle")
        tb_layout.addWidget(self.title_label)

        tb_layout.addStretch()

        # Split button: click opens the default shell, the arrow lists every installed shell
        self.new_btn = QToolButton(self.toolbar)
        self.new_btn.setObjectName("terminalNewShell")
        self.new_btn.setText("New shell")
        self.new_btn.setToolTip("Open another shell (arrow: choose which shell)")
        self.new_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.new_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.new_btn.clicked.connect(lambda: self.new_shell())
        self.shell_menu = QMenu(self.new_btn)
        self.shell_menu.aboutToShow.connect(lambda: self._fill_shell_menu(self.shell_menu))
        self.new_btn.setMenu(self.shell_menu)
        tb_layout.addWidget(self.new_btn)
        self.restart_btn = self.new_btn  # backwards-compatible alias

        self.close_btn = QPushButton("Close shell", self.toolbar)
        self.close_btn.setToolTip("End the selected shell")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.clicked.connect(lambda: self.close_shell())
        tb_layout.addWidget(self.close_btn)

        self.clear_btn = QPushButton("Clear", self.toolbar)
        self.clear_btn.setToolTip("Clear and reset terminal output")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_terminal)
        tb_layout.addWidget(self.clear_btn)

        root_layout.addWidget(self.toolbar)

        # Body: the active shell, with the list of open shells beside it
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self.stack = QStackedWidget(self)
        body.addWidget(self.stack, 1)

        self.shell_list = QListWidget(self)
        self.shell_list.setObjectName("terminalShellList")
        self.shell_list.setFixedWidth(170)
        self.shell_list.setToolTip("Open shells: click to switch, right-click for more")
        self.shell_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep keystrokes in the terminal
        self.shell_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.shell_list.customContextMenuRequested.connect(self._show_list_menu)
        self.shell_list.currentRowChanged.connect(self._on_row_changed)
        body.addWidget(self.shell_list)

        root_layout.addLayout(body, 1)

        self.new_shell()

        self.status_timer = QTimer(self)
        self.status_timer.setInterval(2000)
        # Polls for tmux starting or stopping; nobody sees the title while the panel is hidden.
        self.status_timer.timeout.connect(lambda: self.isVisible() and self._update_status())
        self.status_timer.start()

    def showEvent(self, event):
        super().showEvent(event)
        self._update_status()  # the poll skips while hidden, so catch up as soon as it's seen

    # ── Shells ───────────────────────────────────────────────────────────────

    @property
    def terminal(self) -> "TerminalWidget":
        """The active shell."""
        return self.stack.currentWidget()

    def shell_count(self) -> int:
        return len(self.terminals)

    def new_shell(self, command: Optional[list[str]] = None) -> "TerminalWidget":
        """Open another shell and switch to it: `command` if given (chosen from the menu),
        otherwise the default shell from Settings."""
        term = TerminalWidget(cwd=self.cwd, parent=self, command=command or self._command, env=self._env)
        term.chosen_shell = command is not None  # keeps its program when the Settings default changes
        term.shell_number = self._next_number
        self._next_number += 1
        if self._theme:
            term.apply_theme(self._theme)
        if self._font_size:
            term.set_font_size(self._font_size)
        term.session_started.connect(lambda t=term: self._on_session_changed(t))
        term.session_exited.connect(lambda code, t=term: self._on_session_changed(t, code))
        self.terminals.append(term)
        self.stack.addWidget(term)

        item = QListWidgetItem(self._shell_label(term))
        self.shell_list.addItem(item)
        self.set_current_shell(len(self.terminals) - 1)
        self.shells_changed.emit()
        return term

    def close_shell(self, index: Optional[int] = None):
        """End a shell (the active one by default). The pane always keeps at least one."""
        if index is None:
            index = self.terminals.index(self.terminal)
        if not 0 <= index < len(self.terminals):
            return
        term = self.terminals.pop(index)
        self.shell_list.blockSignals(True)
        self.shell_list.takeItem(index)
        self.shell_list.blockSignals(False)
        self.stack.removeWidget(term)
        term.close()
        term.deleteLater()
        if not self.terminals:
            self.new_shell()
        else:
            self.set_current_shell(min(index, len(self.terminals) - 1))
        self.shells_changed.emit()

    def set_current_shell(self, index: int):
        if not 0 <= index < len(self.terminals):
            return
        if self.shell_list.currentRow() != index:
            self.shell_list.setCurrentRow(index)  # re-enters through _on_row_changed
            return
        term = self.terminals[index]
        self.stack.setCurrentWidget(term)
        term.fit_terminal()
        term.focus_terminal()
        self._update_status()

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self.terminals):
            self.set_current_shell(row)

    def _fill_shell_menu(self, menu: QMenu):
        """Default shell first, then every shell installed on this machine."""
        menu.clear()
        default = self._command or get_default_shell()
        menu.addAction(f"Default ({Path(default[0]).stem})", lambda: self.new_shell())
        shells = available_shells()
        if shells:
            menu.addSeparator()
        for name, command_line in shells:
            argv = parse_shell_command(command_line)
            if argv:
                action = menu.addAction(name, lambda a=argv: self.new_shell(a))
                action.setToolTip(command_line)

    def _show_list_menu(self, pos):
        row = self.shell_list.indexAt(pos).row()
        menu = QMenu(self)
        menu.addAction("New shell", self.new_shell)
        choose = menu.addMenu("New shell with")
        self._fill_shell_menu(choose)
        if row >= 0:
            menu.addAction("Restart shell", lambda: self.terminals[row].restart_session())
            menu.addAction("Close shell", lambda: self.close_shell(row))
        menu.exec(self.shell_list.viewport().mapToGlobal(pos))

    # ── Status ───────────────────────────────────────────────────────────────

    def _shell_name(self, term: "TerminalWidget") -> str:
        argv = term.pty.command or get_default_shell()
        return Path(argv[0]).stem or "shell"

    def _shell_label(self, term: "TerminalWidget", exit_code: Optional[int] = None) -> str:
        label_text = f"{term.shell_number}  {self._shell_name(term)}"
        if exit_code is not None:
            label_text += f"  (exited {exit_code})"
        return label_text

    def _on_session_changed(self, term: "TerminalWidget", exit_code: Optional[int] = None):
        if term in self.terminals:
            row = self.terminals.index(term)
            item = self.shell_list.item(row)
            if item is not None:
                item.setText(self._shell_label(term, exit_code))
                item.setForeground(QColor(self._colors.get("muted" if exit_code is not None else "text", "#ebdbb2")))
        if term is self.terminal:
            self._update_status(exit_code)

    def _update_status(self, exit_code: Optional[int] = None):
        term = self.terminal
        if term is None:
            return
        name = self._shell_name(term)
        if exit_code is not None or not term.pty.is_alive():
            self.title_label.setText(f"Terminal · {name} · exited" if exit_code is None
                                     else f"Terminal · {name} · exited ({exit_code})")
            dot_role = "disabled"
        elif term.pty.is_tmux_active():
            self.title_label.setText("Terminal · tmux (Action: C-b)")
            dot_role = "secondary"
        else:
            self.title_label.setText(f"Terminal · {name}")
            dot_role = "primary"
        self.status_dot.setStyleSheet(f"color: {self._colors.get(dot_role, '#b8bb26')}; font-size: 11px;")

    # ── Commands and appearance ──────────────────────────────────────────────

    def configure(self, command: Optional[list[str]] = None, cwd: Optional[str] = None,
                  env: Optional[dict[str, str]] = None):
        """Shell settings for every shell: open ones use them from their next restart, new ones at once."""
        self._command = command
        if cwd:
            self.cwd = cwd
        self._env = dict(env or {})
        for term in self.terminals:
            own = term.pty.command if getattr(term, "chosen_shell", False) else command
            term.configure(command=own, cwd=cwd, env=env)

    def _restart_shell(self):
        self.terminal.restart_session()

    def _clear_terminal(self):
        self.terminal.clear()

    def set_font_size(self, size: int):
        """Update font size of every shell."""
        self._font_size = size
        for term in self.terminals:
            term.set_font_size(size)

    def apply_theme(self, theme_dict: dict):
        """Apply active theme palette to every shell, the toolbar and the shell list."""
        self._theme = theme_dict
        for term in self.terminals:
            term.apply_theme(theme_dict)
        colors = theme_dict.get("colors", {})
        if not colors:
            return
        self._colors = colors
        panel_bg = colors.get("surface", colors.get("panel", "#282828"))
        list_bg = colors.get("panel", "#282828")
        text_color = colors.get("text", "#ebdbb2")
        border_color = colors.get("border", "#504945")
        btn_bg = colors.get("raised", "#3c3836")
        accent = colors.get("accent", "#fabd2f")

        self.toolbar.setStyleSheet(f"""
            QFrame#terminalToolbar {{
                background-color: {panel_bg};
                border-bottom: 1px solid {border_color};
            }}
            QLabel {{
                background: transparent;
                color: {text_color};
            }}
            QLabel#terminalTitle {{
                font-weight: 600;
                font-size: 11px;
            }}
            QPushButton {{
                background-color: {btn_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                border-color: {accent};
            }}
            QToolButton#terminalNewShell {{
                background-color: {btn_bg};
                color: {text_color};
                border: 1px solid {border_color};
                border-radius: 3px;
                padding: 2px 18px 2px 8px;
                font-size: 11px;
            }}
            QToolButton#terminalNewShell:hover {{
                border-color: {accent};
            }}
            QToolButton#terminalNewShell::menu-button {{
                border-left: 1px solid {border_color};
                width: 14px;
            }}
        """)
        self.shell_list.setStyleSheet(f"""
            QListWidget#terminalShellList {{
                background-color: {list_bg};
                border: 0;
                border-left: 1px solid {border_color};
                outline: 0;
                padding: 4px;
                font-size: 11px;
            }}
            QListWidget#terminalShellList::item {{
                color: {text_color};
                padding: 5px 8px;
                border-radius: 4px;
            }}
            QListWidget#terminalShellList::item:hover {{
                background-color: {panel_bg};
            }}
            QListWidget#terminalShellList::item:selected {{
                background-color: {btn_bg};
                color: {accent};
            }}
        """)
        for row, term in enumerate(self.terminals):
            item = self.shell_list.item(row)
            if item is not None:
                item.setForeground(QColor(text_color if term.pty.is_alive() else colors.get("muted", text_color)))
        self._update_status()

    def closeEvent(self, event):
        for term in self.terminals:
            term.close()
        super().closeEvent(event)
