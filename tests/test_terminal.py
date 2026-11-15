"""Tests for embedded terminal emulator, PTY session lifecycle, and controls."""
import os
import sys
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"
if sys.platform == "darwin" and "QTWEBENGINE_CHROMIUM_FLAGS" not in os.environ:
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-gpu"

try:
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.terminal.pty_session import PtySession, get_default_shell, HAS_PTY
    from frontend.terminal.terminal_widget import TerminalWidget, TerminalPanel
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    HAS_PTY = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestPtySession(unittest.TestCase):
    def test_default_shell_resolution(self):
        cmd = get_default_shell()
        self.assertIsInstance(cmd, list)
        self.assertTrue(len(cmd) >= 1)
        self.assertTrue(os.path.isfile(cmd[0]))

    @unittest.skipIf(not HAS_PTY, "PTY not supported on current platform")
    def test_pty_lifecycle(self):
        pty = PtySession(command=[sys.executable, "-c", "import sys; print('READY'); sys.stdout.flush()"])
        received = []
        pty.data_ready.connect(lambda s: received.append(s))
        
        started = pty.start()
        self.assertTrue(started)
        self.assertTrue(pty.is_alive() or pty.process is not None)

        # Allow event loop to process output
        for _ in range(20):
            app.processEvents()
            time.sleep(0.05)
            if "READY" in "".join(received):
                break

        self.assertIn("READY", "".join(received))
        pty.close()
        self.assertFalse(pty.is_alive())

    @unittest.skipIf(not HAS_PTY or sys.platform == "win32", "/dev/tty is POSIX-only")
    def test_controlling_terminal_dev_tty(self):
        """Verify that TIOCSCTTY was set and child can access /dev/tty."""
        code = "import sys; f = open('/dev/tty', 'r'); print('TTY_SUCCESS'); sys.stdout.flush()"
        pty = PtySession(command=[sys.executable, "-c", code])
        received = []
        pty.data_ready.connect(lambda s: received.append(s))

        started = pty.start()
        self.assertTrue(started)

        for _ in range(30):
            app.processEvents()
            time.sleep(0.05)
            if "TTY_SUCCESS" in "".join(received):
                break

        self.assertIn("TTY_SUCCESS", "".join(received))
        pty.close()

    @unittest.skipIf(not HAS_PTY, "PTY not supported on current platform")
    def test_pty_resize(self):
        pty = PtySession()
        started = pty.start(cols=100, rows=30)
        self.assertTrue(started)
        pty.resize(120, 40)
        self.assertEqual(pty.initial_cols, 100)
        pty.close()


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTerminalPanelAndWidget(unittest.TestCase):
    def setUp(self):
        self.panel = TerminalPanel()

    def tearDown(self):
        self.panel.close()

    def test_panel_structure(self):
        self.assertIsNotNone(self.panel.toolbar)
        self.assertIsNotNone(self.panel.new_btn)
        self.assertIsNotNone(self.panel.clear_btn)
        self.assertIsNotNone(self.panel.status_dot)
        self.assertIsNotNone(self.panel.title_label)
        self.assertIsNotNone(self.panel.terminal)
        self.assertEqual(self.panel.new_btn.text(), "New shell")
        self.assertEqual(self.panel.close_btn.text(), "Close shell")
        self.assertEqual(self.panel.clear_btn.text(), "Clear")
        self.assertEqual(self.panel.shell_list.count(), 1)

    def test_theme_application(self):
        # Dark theme
        self.panel.apply_theme({
            "name": "Catppuccin",
            "colors": {
                "base": "#1e1e2e", "surface": "#181825", "panel": "#313244",
                "raised": "#45475a", "border": "#585b70", "text": "#cdd6f4",
                "accent": "#cba6f7", "primary": "#a6e3a1", "secondary": "#89b4fa",
                "number": "#f5c2e7", "comment": "#6c7086"
            }
        })
        # Light theme
        self.panel.apply_theme({
            "name": "Gruvbox Light",
            "colors": {
                "base": "#fbf1c7", "surface": "#f9f5d7", "panel": "#f2e5bc",
                "raised": "#ebdbb2", "border": "#d5c4a1", "text": "#3c3836",
                "accent": "#9d6000", "primary": "#79740e", "secondary": "#076678",
                "number": "#8f3f71", "comment": "#7c6f64"
            }
        })

    @unittest.skipIf(not HAS_PTY, "PTY not supported on current platform")
    def test_restart_on_enter_when_exited(self):
        term = self.panel.terminal
        # Simulate process exiting
        term.pty.close()
        self.assertFalse(term.pty.is_alive())

        # Sending Enter while dead should trigger restart_session
        term.bridge.send_input("\r")
        app.processEvents()
        self.assertTrue(term.pty.is_alive())

    def test_osc_52_set_clipboard(self):
        term = self.panel.terminal
        term.bridge.set_clipboard("tmux_copied_text")
        self.assertEqual(QApplication.clipboard().text(), "tmux_copied_text")

    @unittest.skipIf(not HAS_PTY, "PTY not supported on current platform")
    def test_tmux_environment_vars(self):
        code = "import os; print('TP=' + os.environ.get('TERM_PROGRAM', '')); print('CT=' + os.environ.get('COLORTERM', ''))"
        pty = PtySession(command=[sys.executable, "-c", code])
        received = []
        pty.data_ready.connect(lambda s: received.append(s))
        pty.start()
        for _ in range(20):
            app.processEvents()
            time.sleep(0.05)
            if "CT=" in "".join(received):  # the second line: TP= has arrived by then too
                break
        output = "".join(received)
        self.assertIn("TP=ghostty", output)
        self.assertIn("CT=truecolor", output)
        pty.close()

    def test_catppuccin_macchiato_theme(self):
        from frontend.terminal.terminal_widget import NATIVE_TERMINAL_PALETTES
        self.assertIn("catppuccin macchiato", NATIVE_TERMINAL_PALETTES)
        macchiato = NATIVE_TERMINAL_PALETTES["catppuccin macchiato"]
        self.assertEqual(macchiato["background"], "#24273a")
        self.assertEqual(macchiato["foreground"], "#cad3f5")

        # Test apply_theme with Catppuccin Macchiato
        self.panel.apply_theme({"name": "Catppuccin Macchiato", "colors": {}})
        self.assertEqual(self.panel.terminal.current_terminal_theme["background"], "#24273a")

    def test_terminal_html_rendering_assets(self):
        from frontend.terminal.terminal_widget import HTML_PATH
        content = HTML_PATH.read_text(encoding="utf-8")
        self.assertIn("Symbols Nerd Font Mono", content)
        self.assertIn("Symbols Nerd Font", content)
        self.assertIn("xterm-addon-canvas.js", content)
        self.assertIn("xterm-addon-unicode11.js", content)
        self.assertIn("lineHeight: 1.0", content)
        self.assertIn("customGlyphs: true", content)

        vendor_dir = HTML_PATH.parent / "vendor"
        self.assertTrue((vendor_dir / "SymbolsNerdFontMono-Regular.ttf").is_file())
        self.assertTrue((vendor_dir / "SymbolsNerdFont-Regular.ttf").is_file())
        self.assertTrue((vendor_dir / "xterm-addon-canvas.js").is_file())
        self.assertTrue((vendor_dir / "xterm-addon-unicode11.js").is_file())


if __name__ == "__main__":
    unittest.main()

