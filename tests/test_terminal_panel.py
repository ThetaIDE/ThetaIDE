"""Terminal pane: several shells listed in a sidebar, and a toolbar without highlight boxes."""
import os
import sys
import time
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.terminal.pty_session import HAS_PTY, available_shells, parse_shell_command
    from frontend.terminal.terminal_widget import TerminalPanel
    from frontend.theme import BUILTINS
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False
    HAS_PTY = False


def pump(seconds=0.2):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.01)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestShellList(unittest.TestCase):
    def setUp(self):
        self.panel = TerminalPanel(command=[sys.executable, "-i"])
        self.panel.resize(900, 500)
        self.panel.show()
        pump()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        pump(0.05)

    def test_starts_with_one_listed_shell(self):
        self.assertEqual(self.panel.shell_count(), 1)
        self.assertEqual(self.panel.shell_list.count(), 2)
        self.assertIs(self.panel.terminal, self.panel.terminals[0])

    def test_new_shell_is_listed_and_becomes_active(self):
        new = self.panel.new_shell()
        self.assertEqual(self.panel.shell_list.count(), 3)
        self.assertIs(self.panel.terminal, new)
        self.assertEqual(self.panel.shell_list.currentRow(), 1)

    def test_clicking_a_listed_shell_switches_to_it(self):
        first = self.panel.terminal
        self.panel.new_shell()
        self.panel.shell_list.setCurrentRow(0)  # what a click does
        self.assertIs(self.panel.terminal, first)
        self.assertIs(self.panel.stack.currentWidget(), first)

    def test_closing_a_shell_removes_it_and_selects_a_neighbour(self):
        self.panel.new_shell()
        third = self.panel.new_shell()
        self.panel.close_shell(1)
        self.assertEqual(self.panel.shell_count(), 2)
        self.assertEqual(self.panel.shell_list.count(), 3)
        self.assertIs(self.panel.terminal, third)

    def test_closing_the_last_shell_opens_a_fresh_one(self):
        only = self.panel.terminal
        self.panel.close_shell()
        self.assertEqual(self.panel.shell_count(), 1)
        self.assertIsNot(self.panel.terminal, only)

    def test_shells_are_numbered_and_named_after_the_program(self):
        self.panel.new_shell()
        labels = [self.panel.shell_list.item(i).data(Qt.ItemDataRole.AccessibleTextRole) for i in range(self.panel.shell_count())]
        program = os.path.splitext(os.path.basename(sys.executable))[0]
        self.assertEqual(labels, [f"1  {program}", f"2  {program}"])

    def test_configure_reaches_open_and_new_shells(self):
        self.panel.configure(command=[sys.executable, "-q"], env={"THETA_TEST": "1"})
        self.assertEqual(self.panel.terminals[0].pty.command, [sys.executable, "-q"])
        new = self.panel.new_shell()
        self.assertEqual(new.pty.command, [sys.executable, "-q"])
        self.assertEqual(new.pty.extra_env, {"THETA_TEST": "1"})

    @unittest.skipIf(not HAS_PTY, "PTY not supported on current platform")
    def test_each_listed_shell_runs_its_own_process(self):
        second = self.panel.new_shell()
        for term in self.panel.terminals:
            if not term.pty.is_alive():
                term.pty.start()
        pump(0.5)
        first = self.panel.terminals[0]
        self.assertTrue(first.pty.is_alive() and second.pty.is_alive())
        self.assertIsNot(first.pty.process, second.pty.process)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestChoosingAShell(unittest.TestCase):
    """New shell's arrow menu opens a specific installed shell."""

    def setUp(self):
        self.panel = TerminalPanel(command=[sys.executable, "-i"])
        self.panel.show()
        pump()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        pump(0.05)

    def menu_actions(self):
        self.panel._fill_shell_menu(self.panel.shell_menu)
        return [a for a in self.panel.shell_menu.actions() if not a.isSeparator()]

    def test_new_shell_button_has_a_shell_menu(self):
        self.assertIs(self.panel.new_btn.menu(), self.panel.shell_menu)

    def test_menu_offers_the_default_and_every_installed_shell(self):
        names = [a.text() for a in self.menu_actions()]
        self.assertTrue(names[0].startswith("Default ("), names)
        self.assertEqual(names[1:], [name for name, _ in available_shells()])

    def test_choosing_a_shell_opens_it(self):
        actions = self.menu_actions()
        if len(actions) < 2:
            self.skipTest("no installed shell detected")
        actions[1].trigger()
        self.assertEqual(self.panel.shell_count(), 2)
        expected = parse_shell_command(available_shells()[0][1])
        self.assertEqual(self.panel.terminal.pty.command, expected)

    def test_new_shell_with_an_explicit_program(self):
        term = self.panel.new_shell([sys.executable, "-q"])
        self.assertEqual(term.pty.command, [sys.executable, "-q"])
        self.assertTrue(term.chosen_shell)

    def test_settings_change_keeps_chosen_shells_on_their_program(self):
        default_term = self.panel.terminal
        chosen = self.panel.new_shell([sys.executable, "-q"])
        self.panel.configure(command=[sys.executable, "-B"])
        self.assertEqual(default_term.pty.command, [sys.executable, "-B"])
        self.assertEqual(chosen.pty.command, [sys.executable, "-q"])


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestToolbarStyling(unittest.TestCase):
    def test_toolbar_labels_have_no_background_box(self):
        panel = TerminalPanel(command=[sys.executable, "-i"])
        self.addCleanup(panel.deleteLater)
        panel.apply_theme(BUILTINS["Gruvbox Light"])
        self.assertFalse(hasattr(panel, "toolbar"))
        self.assertIn("background: transparent", panel.shell_list.styleSheet())

    def test_status_dot_uses_the_theme(self):
        panel = TerminalPanel(command=[sys.executable, "-i"])
        self.addCleanup(panel.deleteLater)
        theme = BUILTINS["Nord"]
        panel.apply_theme(theme)
        self.assertFalse(hasattr(panel, "status_dot"))
        self.assertIn(theme["colors"]["text"], panel.shell_list.styleSheet())
        self.assertIs(panel.shell_list.itemWidget(panel.new_item), panel.new_btn)


if __name__ == "__main__":
    unittest.main()
