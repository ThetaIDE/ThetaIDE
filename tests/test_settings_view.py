"""SettingsView (the sidebar gear), and F1 opening the documentation."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.settings_view import SettingsDetailWindow, SettingsView
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSettingsViewStructureAndNavigation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.settings = self.window.settings_view
        self.settings.build_pages()  # These tests exercise controls from every category.
        self.window.show()
        self.window.tabs.setCurrentWidget(self.settings)

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_running_plugins_get_a_settings_button_at_startup(self):
        """The plugin lists are built before plugins start; they must be rebuilt once they run."""
        from PyQt6.QtWidgets import QToolButton

        def tips(layout):
            found = []
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.layout():
                    found += tips(item.layout())
                elif isinstance(item.widget(), QToolButton):
                    found.append(item.widget().toolTip())
            return found

        running = [self.window.plugin_manager.manifests[pid].name for pid in self.window.plugin_manager.instances]
        self.assertTrue(running)
        shown = tips(self.window.core_plugins_grid) + tips(self.window.plugins_grid)
        for name in running:
            self.assertIn(f"{name} Settings", shown)

    def test_settings_structure_and_window_bindings(self):
        """All expected tabs, attributes, and window bindings exist."""
        self.assertIsInstance(self.settings, SettingsView)
        self.assertIs(self.window.settings_panel, self.settings)

        # Tab buttons exist in sidebar
        expected_tabs = ["appearance", "core_plugins", "community_plugins", "hotkeys", "backend", "storage"]
        for tid in expected_tabs:
            self.assertIn(tid, self.settings.tab_buttons)
            self.assertIn(tid, self.settings.page_widgets)

        # Window attributes bound
        self.assertIsNotNone(self.window.settings_theme_select)
        self.assertIsNotNone(self.window.settings_action_key_combo)
        self.assertIsNotNone(self.window.settings_backend_status)  # Training pipeline page
        self.assertIsNotNone(self.window.settings_backend_status)
        self.assertIsNotNone(self.window.settings_runs_count_label)
        self.assertEqual(set(self.window.pane_sliders), set(self.window.tabs.tabs))  # built-in panes and plugin tabs

    def test_opens_on_the_first_category(self):
        """Settings shows a page straight away: Appearance, with no landing view in front."""
        app.processEvents()
        self.assertEqual(self.settings.active_tab_id, "appearance")
        self.assertTrue(self.settings.tab_buttons["appearance"].isChecked())
        self.assertTrue(self.settings.settings_window.isVisible())
        self.assertIs(self.settings.settings_window.pages_stack.currentWidget(), self.settings.page_widgets["appearance"])

    def test_click_tab_shows_its_page(self):
        self.settings.tab_buttons["hotkeys"].click()

        self.assertEqual(self.settings.active_tab_id, "hotkeys")
        self.assertTrue(self.settings.tab_buttons["hotkeys"].isChecked())
        self.assertFalse(self.settings.tab_buttons["appearance"].isChecked())
        self.assertIs(self.settings.settings_window.pages_stack.currentWidget(), self.settings.page_widgets["hotkeys"])
        self.assertIn("Hotkeys", self.settings.settings_window.title_label.text())

    def test_tab_switching_updates_settings_content(self):
        """Switching between tabs smoothly updates the title and page widget."""
        self.settings.tab_buttons["core_plugins"].click()
        self.assertEqual(self.settings.active_tab_id, "core_plugins")
        self.assertIn("Core Plugins", self.settings.settings_window.title_label.text())

        self.settings.tab_buttons["hotkeys"].click()
        self.assertEqual(self.settings.active_tab_id, "hotkeys")
        self.assertTrue(self.settings.tab_buttons["hotkeys"].isChecked())
        self.assertFalse(self.settings.tab_buttons["core_plugins"].isChecked())
        self.assertIn("Hotkeys", self.settings.settings_window.title_label.text())

        self.settings.tab_buttons["community_plugins"].click()
        self.assertEqual(self.settings.active_tab_id, "community_plugins")
        self.assertIn("Community Plugins", self.settings.settings_window.title_label.text())

    def test_clicking_the_active_tab_keeps_its_page(self):
        btn = self.settings.tab_buttons["hotkeys"]
        btn.click()
        btn.click()
        self.assertEqual(self.settings.active_tab_id, "hotkeys")
        self.assertTrue(btn.isChecked())
        self.assertTrue(self.settings.settings_window.isVisible())

    def test_page_fills_the_area_up_to_a_readable_width(self):
        self.window.resize(1800, 900)
        app.processEvents()
        card = self.settings.settings_window.card
        self.assertIsNone(card.graphicsEffect())
        self.assertEqual(card.width(), SettingsDetailWindow.MAX_PAGE_WIDTH)
        self.assertEqual(card.height(), self.settings.settings_window.height())

    def test_rebuilt_pane_switches_leave_no_stale_widgets_on_screen(self):
        """Plugins add tabs while the window is built, rebuilding the switches before the event loop runs;
        the old labels used to stay visible (stacked at the card's corner) until deleteLater fired."""
        from PyQt6.QtWidgets import QLabel

        self.settings.rebuild_pane_switches()  # a rebuild, with no event loop to run deleteLater
        card = self.settings.panes_grid.parentWidget()
        visible = [lbl.text() for lbl in card.findChildren(QLabel) if lbl.isVisible()]
        for _pid, entry in self.window.tabs.tabs.items():
            name = entry.get("text") or entry.get("short")
            self.assertLessEqual(visible.count(name), 1, name)

    def test_startup_zoom_posts_no_status_message_but_user_zoom_does(self):
        """A message posted before the window is visible doesn't hide the status bar's own widgets,
        so it was drawn over the theme label. Startup has nothing to confirm; user zooms still do."""
        with tempfile.TemporaryDirectory() as temp:
            fresh = Window(Path(temp))
            try:
                self.assertEqual(fresh.statusBar().currentMessage(), "")
            finally:
                fresh._quitting = True
                fresh.close()
        self.window.set_zoom(1.1, persist=False)
        self.assertEqual(self.window.statusBar().currentMessage(), "Zoom: 110%")
        self.window.set_zoom(1.0, persist=False)

    def test_no_animation_content_in_settings(self):
        self.assertNotIn("animation", self.settings.tab_buttons)
        self.assertNotIn("animation", self.settings.page_widgets)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestLogoAndF1OpenTheDocs(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.window.settings_view.build_pages()
        self.window.show()
        self.tabs = self.window.tabs

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def docs_open(self):
        entry = self.tabs.tabs.get("documentation")
        return entry is not None and entry["visible"] and self.tabs.currentWidget() is entry["widget"]

    def test_there_is_no_about_page(self):
        self.assertFalse(hasattr(self.window, "about_panel"))
        self.assertFalse(hasattr(self.window, "show_about"))
        help_menu = next(a.menu() for a in self.window.menuBar().actions() if a.text() == "Help")
        self.assertEqual([a.text() for a in help_menu.actions()], ["Documentation"])

    def test_f1_opens_the_docs_switching_the_plugin_on_if_needed(self):
        action = self.window.menu_actions["open_docs"]
        self.assertEqual(action.shortcut().toString(), "F1")
        if "documentation" in self.window.plugin_manager.instances:
            slider = self.window.plugin_sliders.get("documentation")
            if slider is not None:
                slider.setChecked(False)  # as the user would, in Settings > Core plugins
            else:
                self.window.on_plugin_slider_toggled("documentation", False)
        self.assertNotIn("documentation", self.tabs.tabs)
        self.tabs.setCurrentWidget(self.window.console_panel)
        action.trigger()
        self.assertTrue(self.docs_open())

    def test_f1_shows_a_hidden_docs_tab(self):
        self.window.show_docs()
        self.tabs.setCurrentWidget(self.window.console_panel)
        self.tabs.set_tab_visible("documentation", False)
        self.window.menu_actions["open_docs"].trigger()
        self.assertTrue(self.docs_open())

    def test_the_gear_is_the_only_button_below_the_tabs(self):
        self.assertFalse(hasattr(self.tabs, "logo_button"))
        layout = self.tabs.bar_layout
        self.assertIs(layout.itemAt(layout.count() - 1).widget(), self.tabs.settings_button)

    def test_default_menu_shortcuts_are_unique(self):
        """Qt fires neither action when two share a shortcut, so no two defaults may collide."""
        owners = {}
        for aid, action in self.window.menu_actions.items():
            for seq in action.shortcuts():
                self.assertNotIn(seq.toString(), owners, f"{aid} and {owners.get(seq.toString())} share it")
                owners[seq.toString()] = aid

    def test_menus_and_hotkeys_reach_settings(self):
        self.window.show_settings("hotkeys")
        self.assertIs(self.tabs.currentWidget(), self.window.settings_panel)
        self.assertEqual(self.window.settings_view.active_tab_id, "hotkeys")

        hotkeys = self.window.hotkey_manager
        self.tabs.setCurrentWidget(self.window.console_panel)
        self.assertTrue(hotkeys.switch_to_pane("settings"))
        self.assertIs(self.tabs.currentWidget(), self.window.settings_panel)
        self.assertFalse(hotkeys.switch_to_pane("about"))
        self.assertTrue(hotkeys.switch_to_pane("settings_about"))  # the combined pane's old name
        self.assertIs(self.tabs.currentWidget(), self.window.settings_panel)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSettingsViewPages(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.window = Window(Path(self.temp_dir.name))
        self.window.settings_view.build_pages()
        self.settings = self.window.settings_view
        self.settings.build_pages()  # These tests exercise controls from every category.
        self.window.show()
        self.window.tabs.setCurrentWidget(self.settings)

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_no_horizontal_scrollbars_in_settings(self):
        """Settings scroll area policy is ScrollBarAlwaysOff and no horizontal scroll range exists."""
        self.assertEqual(
            self.settings.settings_window.scroll_area.horizontalScrollBarPolicy(),
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )
        for tab_id in ["appearance", "core_plugins", "community_plugins", "backend", "storage"]:
            self.settings.tab_buttons[tab_id].click()
            QApplication.processEvents()
            h_bar = self.settings.settings_window.scroll_area.horizontalScrollBar()
            self.assertEqual(h_bar.maximum(), 0)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestHotkeysSettingsPage(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.settings = self.window.settings_view
        self.settings.build_pages()  # These tests exercise controls from every category.
        self.window.show()
        self.window.tabs.setCurrentWidget(self.settings)
        # Open hotkeys tab
        self.settings.tab_buttons["hotkeys"].click()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_hotkeys_page_components_and_minimal_card_1(self):
        """Card 1 has only the Action Key selector and Terminal Precedence slider."""
        # Kept in Card 1
        self.assertIsNotNone(self.window.settings_action_key_combo)
        self.assertIsNotNone(self.settings.slider_term)

        # Removed from Card 1 (no clutter)
        self.assertFalse(hasattr(self.settings, "hk_enabled_slider"))
        self.assertFalse(hasattr(self.settings, "custom_action_input"))
        self.assertFalse(hasattr(self.settings, "active_action_badge"))
        self.assertFalse(hasattr(self.settings, "leader_timeout_spin"))

        # Card 2 Hotkey Menu components
        self.assertIsNotNone(self.settings.conflict_banner)
        self.assertIsNotNone(self.settings.conflict_banner_label)
        self.assertFalse(self.settings.conflict_banner.isVisible())

        # All canonical panes tracked in Hotkey Menu
        expected_panes = ["components", "config", "monitor", "results", "plots", "tensorboard", "terminal", "console", "settings"]
        for pid in expected_panes:
            self.assertIn(pid, self.settings.pane_hotkey_edits)
            self.assertIn(pid, self.settings.pane_conflict_labels)

        # Card 3 Menu shortcuts
        from frontend.settings_view import DEFAULT_MENU_SHORTCUTS
        self.assertEqual(len(self.settings.menu_shortcut_edits), len(DEFAULT_MENU_SHORTCUTS))
        self.assertIn("open_launcher", self.settings.menu_shortcut_edits)  # the launcher is rebindable
        self.assertIn("zoom_in", self.settings.menu_shortcut_edits)
        self.assertIn("zoom_out", self.settings.menu_shortcut_edits)
        self.assertIn("reset_zoom", self.settings.menu_shortcut_edits)

    def test_customize_pane_hotkey_and_persistence(self):
        """User can change hotkey combo for any pane and it persists across sessions."""
        edit_components = self.settings.pane_hotkey_edits["components"]
        self.assertEqual(edit_components.get_canonical_combo(), "Action + 1")

        # Change components to Action + C
        edit_components.set_canonical_combo("Action + C")
        edit_components.comboChanged.emit("Action + C")

        # Verify persistence in settings manager
        self.assertEqual(self.window.settings_manager.hotkey_bindings.get("components"), "Action + C")

        # Clear shortcut
        edit_components.set_canonical_combo("")
        edit_components.comboChanged.emit("")
        self.assertNotIn("components", self.window.settings_manager.hotkey_bindings)

    def test_conflict_warning_detection_and_resolution(self):
        """Assigning duplicate combos triggers the conflict banner and row warnings."""
        edit_components = self.settings.pane_hotkey_edits["components"]
        edit_config = self.settings.pane_hotkey_edits["config"]

        # Initially no conflicts
        self.assertFalse(self.settings._validate_hotkey_conflicts())
        self.assertFalse(self.settings.conflict_banner.isVisible())

        # Cause conflict: assign Action + 1 to config as well
        edit_config.set_canonical_combo("Action + 1")
        edit_config.comboChanged.emit("Action + 1")

        # Conflict detected
        self.assertTrue(self.settings.conflict_banner.isVisible())
        self.assertIn("Action + 1", self.settings.conflict_banner_label.text())
        self.assertTrue(self.settings.pane_conflict_labels["components"].isVisible())
        self.assertTrue(self.settings.pane_conflict_labels["config"].isVisible())
        self.assertIn("Conflict", self.settings.pane_conflict_labels["components"].text())

        # Resolve conflict: assign Action + 2 back to config
        edit_config.set_canonical_combo("Action + 2")
        edit_config.comboChanged.emit("Action + 2")

        # Conflict banner and row labels should now be cleared
        self.assertFalse(self.settings.conflict_banner.isVisible())
        self.assertFalse(self.settings.pane_conflict_labels["components"].isVisible())
        self.assertFalse(self.settings.pane_conflict_labels["config"].isVisible())

    def test_reset_pane_mappings_to_default(self):
        """Resetting pane mappings restores default 0-9 bindings and clears conflicts."""
        # Change several hotkeys
        self.settings.pane_hotkey_edits["components"].set_canonical_combo("Action + X")
        self.settings.pane_hotkey_edits["components"].comboChanged.emit("Action + X")

        # Reset
        self.settings._reset_pane_mappings_to_default()

        self.assertEqual(self.settings.pane_hotkey_edits["components"].get_canonical_combo(), "Action + 1")
        self.assertEqual(self.settings.pane_hotkey_edits["settings"].get_canonical_combo(), "Action + 0")
        self.assertEqual(self.window.settings_manager.hotkey_bindings.get("components"), "Action + 1")
        self.assertFalse(self.settings.conflict_banner.isVisible())

    def test_action_key_preset_and_terminal_precedence(self):
        """Action key combo and terminal precedence slider persist changes."""
        # Select Caps Lock preset
        idx_caps = self.window.settings_action_key_combo.findData("caps_lock")
        self.assertGreaterEqual(idx_caps, 0)
        self.window.settings_action_key_combo.setCurrentIndex(idx_caps)
        self.assertEqual(self.window.settings_manager.action_key, "caps_lock")

        # Toggle terminal precedence
        init_term = self.window.settings_manager.hotkeys_terminal_precedence
        self.settings.slider_term.setChecked(not init_term)
        self.assertEqual(self.window.settings_manager.hotkeys_terminal_precedence, not init_term)

    def test_customize_menu_shortcuts(self):
        """User can customize menu action shortcuts and reset them."""
        from PyQt6.QtGui import QKeySequence

        # Change Launch Training from F5 to F6
        edit = self.settings.menu_shortcut_edits["launch_training"]
        edit.setKeySequence(QKeySequence("F6"))
        self.assertEqual(self.window.settings_manager.get("shortcuts", "launch_training"), "F6")
        if "launch_training" in self.window.menu_actions:
            self.assertEqual(self.window.menu_actions["launch_training"].shortcut().toString(), "F6")

        # Reset menu shortcuts
        self.settings._reset_menu_shortcuts_to_default()
        self.assertEqual(self.window.settings_manager.get("shortcuts", "launch_training"), "F5")
        if "launch_training" in self.window.menu_actions:
            self.assertEqual(self.window.menu_actions["launch_training"].shortcut().toString(), "F5")

    def test_customize_zoom_shortcuts(self):
        """User can bind custom hotkeys for zoom in and zoom out and reset them."""
        from PyQt6.QtGui import QKeySequence

        self.assertIn("zoom_in", self.settings.menu_shortcut_edits)
        self.assertIn("zoom_out", self.settings.menu_shortcut_edits)
        self.assertIn("reset_zoom", self.settings.menu_shortcut_edits)

        # Default shortcuts
        self.assertEqual(self.settings.menu_shortcut_edits["zoom_in"].keySequence().toString(), "Ctrl++")
        self.assertEqual(self.settings.menu_shortcut_edits["zoom_out"].keySequence().toString(), "Ctrl+-")
        self.assertEqual(self.settings.menu_shortcut_edits["reset_zoom"].keySequence().toString(), "Ctrl+0")

        # Customize zoom_in to Ctrl+= and zoom_out to Ctrl+Shift+-
        edit_in = self.settings.menu_shortcut_edits["zoom_in"]
        edit_in.setKeySequence(QKeySequence("Ctrl+="))
        self.assertEqual(self.window.settings_manager.get("shortcuts", "zoom_in"), "Ctrl+=")
        if "zoom_in" in self.window.menu_actions:
            self.assertIn("Ctrl+=", [s.toString() for s in self.window.menu_actions["zoom_in"].shortcuts()])

        # Reset menu shortcuts
        self.settings._reset_menu_shortcuts_to_default()
        self.assertEqual(self.window.settings_manager.get("shortcuts", "zoom_in"), "Ctrl++")
        self.assertEqual(self.settings.menu_shortcut_edits["zoom_in"].keySequence().toString(), "Ctrl++")

    def test_rebind_open_launcher_overwrites_combo_and_updates_action(self):
        """Clicking open_launcher edit listens and overwrites previous combo with single new combo."""
        from PyQt6.QtCore import Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        self.settings.show_tab("hotkeys")
        edit_launcher = self.settings.menu_shortcut_edits["open_launcher"]

        # Initial default is Ctrl+Space
        self.assertEqual(edit_launcher.keySequence().toString(), "Ctrl+Space")

        # Simulate clicking / focusing the edit -> starts recording
        edit_launcher._start_recording()
        self.assertTrue(edit_launcher._is_recording)
        self.assertEqual(edit_launcher.text(), "")

        # Simulate user pressing Ctrl+Shift+L
        ev = QKeyEvent(
            QEvent.Type.KeyPress,
            Qt.Key.Key_L,
            Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier,
        )
        edit_launcher.keyPressEvent(ev)

        # Combo overwrites previous value completely (not appended as a list of keys)
        self.assertEqual(edit_launcher.keySequence().toString(), "Ctrl+Shift+L")
        self.assertEqual(self.window.settings_manager.get("shortcuts", "open_launcher"), "Ctrl+Shift+L")
        if "open_launcher" in self.window.menu_actions:
            self.assertEqual(self.window.menu_actions["open_launcher"].shortcut().toString(), "Ctrl+Shift+L")

        # Reset back to default
        self.settings._reset_menu_shortcuts_to_default()
        self.assertEqual(edit_launcher.keySequence().toString(), "Ctrl+Space")
        self.assertEqual(self.window.settings_manager.get("shortcuts", "open_launcher"), "Ctrl+Space")

    def test_menu_shortcut_conflict_detection_and_resolution(self):
        """Assigning duplicate shortcut to open_launcher triggers conflict banners and row warnings."""
        from PyQt6.QtGui import QKeySequence

        self.settings.show_tab("hotkeys")
        edit_launcher = self.settings.menu_shortcut_edits["open_launcher"]
        edit_training = self.settings.menu_shortcut_edits["launch_training"]

        # Initially no conflicts
        self.assertFalse(self.settings._validate_hotkey_conflicts())
        self.assertFalse(self.settings.conflict_banner.isVisible())
        self.assertFalse(self.settings.menu_conflict_banner.isVisible())

        # Cause conflict: set open_launcher to F5 (which is assigned to launch_training)
        edit_launcher.setKeySequence(QKeySequence("F5"))

        # Conflict must be detected and alerted
        self.assertTrue(self.settings._validate_hotkey_conflicts())
        self.assertTrue(self.settings.conflict_banner.isVisible())
        self.assertTrue(self.settings.menu_conflict_banner.isVisible())
        self.assertIn("F5", self.settings.menu_conflict_banner_label.text())
        self.assertTrue(self.settings.menu_conflict_labels["open_launcher"].isVisible())
        self.assertTrue(self.settings.menu_conflict_labels["launch_training"].isVisible())
        self.assertIn("Conflict", self.settings.menu_conflict_labels["open_launcher"].text())
        self.assertTrue(edit_launcher._has_conflict)
        self.assertTrue(edit_training._has_conflict)

        # Resolve conflict: assign Alt+Space back to open_launcher
        edit_launcher.setKeySequence(QKeySequence("Alt+Space"))

        # Conflicts must now be cleared
        self.assertFalse(self.settings._validate_hotkey_conflicts())
        self.assertFalse(self.settings.conflict_banner.isVisible())
        self.assertFalse(self.settings.menu_conflict_banner.isVisible())
        self.assertFalse(self.settings.menu_conflict_labels["open_launcher"].isVisible())
        self.assertFalse(self.settings.menu_conflict_labels["launch_training"].isVisible())
        self.assertFalse(edit_launcher._has_conflict)
        self.assertFalse(edit_training._has_conflict)

        # Restore default
        self.settings._reset_menu_shortcuts_to_default()

    def test_menu_shortcut_clear_and_escape_cancel(self):
        """User can clear a menu shortcut via backspace or cancel recording via Escape."""
        from PyQt6.QtCore import Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        self.settings.show_tab("hotkeys")
        edit_launcher = self.settings.menu_shortcut_edits["open_launcher"]
        self.assertEqual(edit_launcher.keySequence().toString(), "Ctrl+Space")

        # Test Escape cancels recording without changing shortcut
        edit_launcher._start_recording()
        ev_esc = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        edit_launcher.keyPressEvent(ev_esc)
        self.assertEqual(edit_launcher.keySequence().toString(), "Ctrl+Space")

        # Test Backspace clears shortcut
        ev_bs = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier)
        edit_launcher.keyPressEvent(ev_bs)
        self.assertEqual(edit_launcher.keySequence().toString(), "")
        self.assertEqual(self.window.settings_manager.get("shortcuts", "open_launcher"), "")
        if "open_launcher" in self.window.menu_actions:
            self.assertTrue(self.window.menu_actions["open_launcher"].shortcut().isEmpty())

        # Restore default
        self.settings._reset_menu_shortcuts_to_default()
        self.assertEqual(edit_launcher.keySequence().toString(), "Ctrl+Space")

    def test_hotkey_manager_bypasses_when_shortcut_editor_focused(self):
        """HotkeyManager does not intercept key presses while a shortcut editor is focused."""
        from PyQt6.QtCore import Qt, QEvent
        from PyQt6.QtGui import QKeyEvent

        self.settings.show_tab("hotkeys")
        edit_launcher = self.settings.menu_shortcut_edits["open_launcher"]
        edit_launcher.setFocus()

        hm = self.window.hotkey_manager
        self.assertTrue(hm.is_in_hotkey_editor(edit_launcher))

        # Event filter should return False (let event pass to editor)
        ev_b = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
        filtered = hm.eventFilter(edit_launcher, ev_b)
        self.assertFalse(filtered)

    def test_zoom_actions_and_scaling(self):
        """Window zoom_in, zoom_out, and reset_zoom update zoom factor and persist."""
        self.assertEqual(self.window.zoom_factor, 1.0)
        self.assertEqual(self.window.settings_manager.zoom, 1.0)

        # Zoom in
        self.window.zoom_in()
        self.assertAlmostEqual(self.window.zoom_factor, 1.1)
        self.assertAlmostEqual(self.window.settings_manager.zoom, 1.1)
        self.assertAlmostEqual(self.window.theme_manager.zoom, 1.1)

        # Zoom out twice
        self.window.zoom_out()
        self.window.zoom_out()
        self.assertAlmostEqual(self.window.zoom_factor, 0.9)
        self.assertAlmostEqual(self.window.settings_manager.zoom, 0.9)

        # Reset zoom
        self.window.reset_zoom()
        self.assertAlmostEqual(self.window.zoom_factor, 1.0)
        self.assertAlmostEqual(self.window.settings_manager.zoom, 1.0)

    def test_display_scaling_appearance_card(self):
        """Appearance tab contains interactive zoom slider and controls."""
        self.settings.show_tab("appearance")
        self.assertIsNotNone(self.window.zoom_slider)
        self.assertIsNotNone(self.window.zoom_percent_label)
        self.assertEqual(self.window.zoom_percent_label.text(), "100%")

        # Changing zoom via slider
        self.window.zoom_slider.setValue(120)
        self.assertAlmostEqual(self.window.zoom_factor, 1.2)
        self.assertEqual(self.window.zoom_percent_label.text(), "120%")

    def test_show_resolved_defaults_appearance_setting(self):
        """Appearance tab contains Show Resolved Defaults toggle and syncs with settings_manager."""
        self.settings.show_tab("appearance")
        self.assertIsNotNone(self.window.toggle_resolved_defaults)
        self.assertFalse(self.window.settings_manager.show_resolved_defaults)

        # Toggle on
        self.window.toggle_resolved_defaults.setChecked(True)
        self.assertTrue(self.window.settings_manager.show_resolved_defaults)

        # Verify viewer view_mode updates
        viewer = self.window.config_tab_manager.current_viewer()
        self.assertEqual(viewer.view_mode, "resolved")

        # Toggle off
        self.window.toggle_resolved_defaults.setChecked(False)
        self.assertFalse(self.window.settings_manager.show_resolved_defaults)
        self.assertEqual(viewer.view_mode, "overrides")


