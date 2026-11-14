"""Unit tests for HotkeyManager and text-based hotkey settings."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.hotkeys import (
        HotkeyManager,
        parse_action_key,
        PANE_ALIASES,
        get_ctrl_key,
        get_ctrl_modifier,
        get_meta_key,
        get_meta_modifier,
    )
    from frontend.settings import SettingsManager
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHotkeySettings(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        # Canonical workspace file is data_dir.parent / "settings.toml"
        # Create an inner data dir so settings.toml stays isolated in self.temp_dir
        self.data_dir = Path(self.temp_dir.name) / "runs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.mgr = SettingsManager(self.data_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_hotkey_settings(self):
        self.assertTrue(self.mgr.hotkeys_enabled)
        self.assertEqual(self.mgr.action_key, "ctrl+b")
        self.assertAlmostEqual(self.mgr.hotkeys_leader_timeout, 1.5)
        
        # Verify requested hierarchy [0, 1, ...] -> [settings, components, experiments, ...]
        panes = self.mgr.hotkey_panes
        self.assertEqual(panes[0], "settings")
        self.assertEqual(panes[1], "components")
        self.assertIn(panes[2], ("config", "experiments"))
        self.assertIn("monitor", panes)
        self.assertIn("terminal", panes)
        self.assertIn("console", panes)

    def test_mutate_hotkey_settings(self):
        self.mgr.set("hotkeys", "action_key", "alt")
        self.assertEqual(self.mgr.action_key, "alt")

        self.mgr.set("hotkeys", "leader_timeout", 2.0)
        self.assertEqual(self.mgr.hotkeys_leader_timeout, 2.0)

        self.mgr.set("hotkeys", "enabled", False)
        self.assertFalse(self.mgr.hotkeys_enabled)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestKeyParsing(unittest.TestCase):
    def test_parse_caps_lock(self):
        key, mod = parse_action_key("caps_lock")
        self.assertEqual(key, Qt.Key.Key_CapsLock)
        self.assertIsNone(mod)

        key, mod = parse_action_key("CapsLock")
        self.assertEqual(key, Qt.Key.Key_CapsLock)

    def test_parse_modifiers(self):
        key, mod = parse_action_key("alt")
        self.assertEqual(key, Qt.Key.Key_Alt)
        self.assertEqual(mod, Qt.KeyboardModifier.AltModifier)

        key, mod = parse_action_key("ctrl")
        self.assertEqual(key, get_ctrl_key())
        self.assertEqual(mod, get_ctrl_modifier())

        key, mod = parse_action_key("cmd")
        self.assertEqual(key, get_meta_key())
        self.assertEqual(mod, get_meta_modifier())

        key, mod = parse_action_key("shift")
        self.assertEqual(key, Qt.Key.Key_Shift)
        self.assertEqual(mod, Qt.KeyboardModifier.ShiftModifier)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHotkeyManagerNavigation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name) / "workspace" / "runs"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.window = Window(self.data_dir)
        self.hm = self.window.hotkey_manager
        self.window.settings_manager.set("hotkeys", "action_key", "caps_lock")

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_pane_resolution_by_index(self):
        # 0 -> settings
        self.assertEqual(self.hm.get_pane_id_by_index(0), "settings")
        # 1 -> components
        self.assertEqual(self.hm.get_pane_id_by_index(1), "components")
        # 2 -> config / experiments
        self.assertEqual(self.hm.get_pane_id_by_index(2), "config")
        # 3 -> monitor
        self.assertEqual(self.hm.get_pane_id_by_index(3), "monitor")
        # 4 -> results
        self.assertEqual(self.hm.get_pane_id_by_index(4), "results")
        # 5 -> plots
        self.assertEqual(self.hm.get_pane_id_by_index(5), "plots")
        # 6 -> tensorboard
        self.assertEqual(self.hm.get_pane_id_by_index(6), "tensorboard")
        # The Job queue pane moved into the Monitor; 7 -> terminal
        self.assertEqual(self.hm.get_pane_id_by_index(7), "terminal")
        # 8 -> console
        self.assertEqual(self.hm.get_pane_id_by_index(8), "console")

    def test_switch_to_pane_by_index_direct(self):
        # Switch to 0 (Settings)
        res0 = self.hm.switch_to_pane_by_index(0)
        self.assertTrue(res0)
        self.assertIs(self.window.tabs.currentWidget(), self.window.settings_panel)

        # Switch to 1 (Components)
        res1 = self.hm.switch_to_pane_by_index(1)
        self.assertTrue(res1)
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)

        # Switch to 2 (Experiment builder / config)
        res2 = self.hm.switch_to_pane_by_index(2)
        self.assertTrue(res2)
        self.assertIs(self.window.tabs.currentWidget(), self.window.config_panel)

        # Switch to 7 (Terminal)
        res7 = self.hm.switch_to_pane_by_index(7)
        self.assertTrue(res7)
        self.assertIs(self.window.tabs.currentWidget(), self.window.terminal_panel)

        # Switch to 8 (Console)
        res8 = self.hm.switch_to_pane_by_index(8)
        self.assertTrue(res8)
        self.assertIs(self.window.tabs.currentWidget(), self.window.console_panel)

    def test_chorded_action_plus_number(self):
        """Simulate holding Caps Lock down while pressing a number key."""
        # Start on components
        self.window.tabs.setCurrentWidget(self.window.components_panel)

        # Press Caps Lock (hold)
        e_caps_down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        consumed = self.hm.eventFilter(self.window, e_caps_down)
        self.assertTrue(consumed)
        self.assertTrue(self.hm._action_key_held)

        # Press '2' while Caps Lock is held -> should switch to config (experiment)
        e_2 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_2, Qt.KeyboardModifier.NoModifier, "2")
        consumed_2 = self.hm.eventFilter(self.window, e_2)
        self.assertTrue(consumed_2)
        self.assertIs(self.window.tabs.currentWidget(), self.window.config_panel)

        # Release Caps Lock
        e_caps_up = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        consumed_up = self.hm.eventFilter(self.window, e_caps_up)
        self.assertTrue(consumed_up)
        self.assertFalse(self.hm._action_key_held)

    def test_sequential_leader_action_plus_number(self):
        """Simulate tapping Caps Lock (press & release), then pressing a number key."""
        # Start on config
        self.window.tabs.setCurrentWidget(self.window.config_panel)

        # Tap Caps Lock
        e_caps_down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps_down)
        e_caps_up = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps_up)

        self.assertTrue(self.hm._leader_active)

        # Press '0' -> should switch to Settings
        e_0 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.NoModifier, "0")
        consumed_0 = self.hm.eventFilter(self.window, e_0)
        self.assertTrue(consumed_0)
        self.assertIs(self.window.tabs.currentWidget(), self.window.settings_panel)
        self.assertFalse(self.hm._leader_active)

    def test_leader_escape_cancels(self):
        """Escape cancels active leader mode."""
        e_caps_down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps_down)
        e_caps_up = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps_up)
        self.assertTrue(self.hm._leader_active)

        e_esc = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        consumed = self.hm.eventFilter(self.window, e_esc)
        self.assertTrue(consumed)
        self.assertFalse(self.hm._leader_active)

    def test_leader_non_number_passes_through(self):
        """Typing a non-number during leader mode cancels leader and passes key through."""
        e_caps_down = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps_down)
        e_caps_up = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps_up)
        self.assertTrue(self.hm._leader_active)

        # User types 'w'
        e_w = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_W, Qt.KeyboardModifier.NoModifier, "w")
        consumed = self.hm.eventFilter(self.window, e_w)
        # Should NOT consume: let the letter pass through to text edit
        self.assertFalse(consumed)
        self.assertFalse(self.hm._leader_active)

    def test_unhide_hidden_pane_on_hotkey(self):
        """If a pane is hidden, switching to it via hotkey unhides and focuses it."""
        self.window.tabs.set_tab_visible("plots", False)
        self.assertFalse(self.window.tabs.is_tab_visible("plots"))

        # Action + 5 -> plots
        res = self.hm.switch_to_pane_by_index(5)
        self.assertTrue(res)
        self.assertTrue(self.window.tabs.is_tab_visible("plots"))
        self.assertIs(self.window.tabs.currentWidget(), self.window.plot_viewer)

    def test_window_switch_to_pane_method(self):
        self.window.switch_to_pane(1)
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)
        self.window.switch_to_pane("terminal")
        self.assertIs(self.window.tabs.currentWidget(), self.window.terminal_panel)
        self.window.switch_to_pane(0)
        self.assertIs(self.window.tabs.currentWidget(), self.window.settings_panel)

    def test_leader_double_tap_cancels(self):
        """Tapping the action key a second time cancels active leader mode."""
        e_caps = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps)
        self.assertTrue(self.hm._leader_active)

        # Tap again
        self.hm.eventFilter(self.window, e_caps)
        self.assertFalse(self.hm._leader_active)

    def test_shift_caps_lock_passes_through(self):
        """Shift + Caps Lock passes through without activating action key."""
        e_shift_caps = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_CapsLock,
            Qt.KeyboardModifier.ShiftModifier,
        )
        consumed = self.hm.eventFilter(self.window, e_shift_caps)
        self.assertFalse(consumed)
        self.assertFalse(self.hm._action_key_held)
        self.assertFalse(self.hm._leader_active)

    def test_hotkeys_disabled_setting(self):
        """When hotkeys.enabled is false, key events are not consumed."""
        self.window.settings_manager.set("hotkeys", "enabled", False)
        self.assertFalse(self.hm.enabled)

        e_caps = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        consumed = self.hm.eventFilter(self.window, e_caps)
        self.assertFalse(consumed)
        self.assertFalse(self.hm._leader_active)

    def test_custom_action_key_alt(self):
        """Switch action key to Alt; Alt + 1 switches to components."""
        self.window.settings_manager.set("hotkeys", "action_key", "alt")
        self.assertEqual(self.hm.action_key_name, "alt")
        self.assertEqual(self.hm.action_modifier, Qt.KeyboardModifier.AltModifier)

        # Start on config
        self.window.tabs.setCurrentWidget(self.window.config_panel)

        # Press Alt+1
        e_alt_1 = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_1,
            Qt.KeyboardModifier.AltModifier,
            "1",
        )
        consumed = self.hm.eventFilter(self.window, e_alt_1)
        self.assertTrue(consumed)
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)

    def test_custom_pane_order_in_settings(self):
        """Custom hotkeys.panes list in settings changes index-to-pane mapping."""
        self.window.settings_manager.set("hotkeys", "panes", ["settings", "terminal", "config"])
        self.assertEqual(self.hm.get_pane_id_by_index(0), "settings")
        self.assertEqual(self.hm.get_pane_id_by_index(1), "terminal")
        self.assertEqual(self.hm.get_pane_id_by_index(2), "config")

        # Action + 1 should now switch to terminal
        self.hm.switch_to_pane_by_index(1)
        self.assertIs(self.window.tabs.currentWidget(), self.window.terminal_panel)

    def test_terminal_precedence_in_terminal_pane(self):
        """When in terminal pane, keys pass through to terminal and tmux."""
        self.window.tabs.setCurrentWidget(self.window.terminal_panel)
        self.assertTrue(self.hm.is_in_terminal())

        # Test CapsLock key does not get intercepted in terminal
        e_caps = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        consumed = self.hm.eventFilter(self.window.terminal_panel, e_caps)
        self.assertFalse(consumed)

        # Test number key does not switch pane when in terminal
        e_0 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.NoModifier, 0, 0, 0x10000)
        consumed_0 = self.hm.eventFilter(self.window.terminal_panel, e_0)
        self.assertFalse(consumed_0)
        self.assertIs(self.window.tabs.currentWidget(), self.window.terminal_panel)

    def test_caps_plus_zero_moves_to_settings_when_out_of_terminal(self):
        """When out of terminal pane, caps + 0 moves to settings."""
        # Switch to components pane
        self.window.tabs.setCurrentWidget(self.window.components_panel)
        self.assertFalse(self.hm.is_in_terminal())

        # Press Caps Lock then 0
        e_caps = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_CapsLock, Qt.KeyboardModifier.NoModifier)
        self.hm.eventFilter(self.window, e_caps)
        e_0 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.NoModifier, "0")
        consumed = self.hm.eventFilter(self.window, e_0)
        self.assertTrue(consumed)
        self.assertIs(self.window.tabs.currentWidget(), self.window.settings_panel)

    def test_subsequent_number_without_action_does_not_switch(self):
        """Action + 1 switches to components, then pressing 3 alone does NOT switch."""
        self.window.settings_manager.set("hotkeys", "action_key", "ctrl+b")
        self.window.tabs.setCurrentWidget(self.window.config_panel)

        # Action + 1 (Ctrl+b + 1)
        e_ctrl_b = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, get_ctrl_modifier())
        self.hm.eventFilter(self.window, e_ctrl_b)
        e_1 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_1, Qt.KeyboardModifier.NoModifier, "1")
        consumed_1 = self.hm.eventFilter(self.window, e_1)
        self.assertTrue(consumed_1)
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)

        # Press 3 alone without action key
        e_3 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_3, Qt.KeyboardModifier.NoModifier, "3")
        consumed_3 = self.hm.eventFilter(self.window, e_3)
        self.assertFalse(consumed_3)
        # Should stay on components_panel, NOT switch to monitor
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)

    def test_shift_number_passes_through(self):
        """Shift + number (e.g. '!') is not swallowed as action hotkey."""
        self.window.tabs.setCurrentWidget(self.window.components_panel)
        e_shift_1 = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_1,
            Qt.KeyboardModifier.ShiftModifier,
            0,
            0,
            0x10000,
        )
        consumed = self.hm.eventFilter(self.window, e_shift_1)
        self.assertFalse(consumed)
        self.assertIs(self.window.tabs.currentWidget(), self.window.components_panel)

    def test_karabiner_ctrl_b_action_key_moves_to_settings(self):
        """When Caps Lock is remapped to Ctrl+b (Karabiner), Ctrl+b + 0 moves to settings out of terminal."""
        self.window.settings_manager.set("hotkeys", "action_key", "ctrl+b")
        self.window.tabs.setCurrentWidget(self.window.components_panel)
        self.assertFalse(self.hm.is_in_terminal())

        # Press Ctrl+b (emitted by Karabiner when Caps Lock is pressed)
        e_ctrl_b = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, get_ctrl_modifier())
        consumed_b = self.hm.eventFilter(self.window, e_ctrl_b)
        self.assertTrue(consumed_b)
        self.assertTrue(self.hm._leader_active)

        # Press 0 -> moves to settings
        e_0 = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_0, Qt.KeyboardModifier.NoModifier, "0")
        consumed_0 = self.hm.eventFilter(self.window, e_0)
        self.assertTrue(consumed_0)
        self.assertIs(self.window.tabs.currentWidget(), self.window.settings_panel)
        self.assertFalse(self.hm._leader_active)

    def test_karabiner_ctrl_b_in_terminal_passes_to_tmux(self):
        """When in terminal pane, Ctrl+b passes directly to tmux."""
        self.window.settings_manager.set("hotkeys", "action_key", "ctrl+b")
        self.window.tabs.setCurrentWidget(self.window.terminal_panel)
        self.assertTrue(self.hm.is_in_terminal())

        # Press Ctrl+b inside terminal
        e_ctrl_b = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, get_ctrl_modifier())
        consumed_b = self.hm.eventFilter(self.window.terminal_panel, e_ctrl_b)
        self.assertFalse(consumed_b)


if __name__ == "__main__":
    unittest.main()
