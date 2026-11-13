"""Configurable terminal shell: shell detection and parsing, venv activation, and the Settings page."""
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.terminal.pty_session import (
        HAS_PTY,
        available_shells,
        parse_shell_command,
        venv_environment,
    )
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    HAS_PTY = False


def quoted_executable() -> str:
    """The interpreter path as a single shell word.

    Quoting is not Windows-specific: on POSIX, shlex splits an unquoted path
    containing a space into two words, so a checkout under something like
    "Application Support" makes the parser report it cannot find the program.
    Double quotes are understood on both platforms.
    """
    return f'"{sys.executable}"'

@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestShellHelpers(unittest.TestCase):
    def test_empty_setting_means_auto_detect(self):
        self.assertIsNone(parse_shell_command(""))
        self.assertIsNone(parse_shell_command("   "))

    def test_unknown_program_falls_back(self):
        self.assertIsNone(parse_shell_command("no-such-shell-xyz --flag"))

    def test_absolute_path_with_arguments(self):
        self.assertEqual(parse_shell_command(f"{quoted_executable()} -i"), [sys.executable, "-i"])

    def test_detected_shells_round_trip_through_the_parser(self):
        shells = available_shells()
        self.assertTrue(shells, "no shell detected on this machine")
        for name, command in shells:
            argv = parse_shell_command(command)
            self.assertIsNotNone(argv, f"{name}: {command}")
            self.assertTrue(Path(argv[0]).is_file(), f"{name}: {argv[0]}")

    def test_venv_environment_puts_the_venv_first_on_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "venv"
            bin_dir = venv / ("Scripts" if sys.platform == "win32" else "bin")
            bin_dir.mkdir(parents=True)
            env = venv_environment(venv)
            self.assertEqual(env["VIRTUAL_ENV"], str(venv))
            self.assertTrue(env["PATH"].startswith(str(bin_dir) + os.pathsep))

    def test_missing_venv_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(venv_environment(Path(tmp) / "absent"), {})


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestTerminalSettings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.sm = self.window.settings_manager
        self.view = self.window.settings_view
        self.view.show_tab("terminal")

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_defaults_auto_detect_launch_folder_and_venv(self):
        command, cwd, env = self.window.terminal_config()
        self.assertIsNone(command)
        self.assertEqual(cwd, str(Path.cwd()))
        self.assertTrue(self.sm.terminal_activate_venv)

    def test_settings_reach_the_terminal_config(self):
        folder = Path(self.temp_dir.name)
        self.sm.set("terminal", "shell", quoted_executable())
        self.sm.set("terminal", "cwd", str(folder))
        self.sm.set("terminal", "activate_venv", False)
        command, cwd, env = self.window.terminal_config()
        self.assertEqual(command, [sys.executable])
        self.assertEqual(Path(cwd), folder)
        self.assertEqual(env, {})

    def test_missing_start_folder_falls_back_to_launch_folder(self):
        self.sm.set("terminal", "cwd", str(Path(self.temp_dir.name) / "gone"))
        self.assertEqual(self.window.terminal_config()[1], str(Path.cwd()))

    def test_terminal_page_is_registered(self):
        self.assertIn("terminal", self.view.tab_buttons)
        self.assertIn("terminal", self.view.page_widgets)

    def test_picking_a_detected_shell_saves_it(self):
        combo = self.view.terminal_shell_combo
        index = next(i for i in range(combo.count()) if combo.itemData(i))
        combo.setCurrentIndex(index)
        self.assertEqual(self.sm.terminal_shell, combo.itemData(index))
        self.assertFalse(self.view.terminal_shell_edit.isEnabled())

    def test_custom_command_is_saved_and_flagged_when_missing(self):
        combo = self.view.terminal_shell_combo
        combo.setCurrentIndex(combo.count() - 1)  # Custom command…
        self.assertTrue(self.view.terminal_shell_edit.isEnabled())
        self.view.terminal_shell_edit.setText("no-such-shell-xyz")
        self.view.terminal_shell_edit.editingFinished.emit()
        self.assertEqual(self.sm.terminal_shell, "no-such-shell-xyz")
        self.assertIn("not found", self.view.terminal_shell_status.text())

    def test_saved_custom_command_reloads_as_custom(self):
        self.view._load_terminal_shell("no-such-shell-xyz -x")
        combo = self.view.terminal_shell_combo
        self.assertEqual(combo.currentIndex(), combo.count() - 1)
        self.assertEqual(self.view.terminal_shell_edit.text(), "no-such-shell-xyz -x")

    @unittest.skipIf(not HAS_PTY, "PTY not supported on current platform")
    def test_restart_uses_the_configured_shell_and_venv(self):
        venv = Path(self.temp_dir.name) / "venv"
        (venv / ("Scripts" if sys.platform == "win32" else "bin")).mkdir(parents=True)
        code = "import os; print('VENV=' + os.environ.get('VIRTUAL_ENV', ''))"
        term = self.window.terminal_panel.terminal
        term.configure(command=[sys.executable, "-c", code], cwd=self.temp_dir.name,
                       env=venv_environment(venv))
        received = []
        term.pty.data_ready.connect(received.append)
        term.restart_session()
        deadline = time.time() + 10
        while time.time() < deadline and f"VENV={venv}" not in "".join(received):
            app.processEvents()
            time.sleep(0.05)
        self.assertIn(f"VENV={venv}", "".join(received))
        term.pty.close()


if __name__ == "__main__":
    unittest.main()
