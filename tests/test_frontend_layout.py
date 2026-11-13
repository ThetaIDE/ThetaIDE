"""Tests for frontend layout enhancements: draggable/reorderable sidebar icons and pane visibility sliders."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Ensure QApplication exists in offscreen mode for tests
os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import QSize, Qt
    from PyQt6.QtGui import QPainter, QPixmap
    from PyQt6.QtWidgets import QApplication, QLabel, QWidget
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.sidetabs import SideTabs
    from frontend.widgets import ToggleSlider
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestToggleSlider(unittest.TestCase):
    def test_initial_state_and_toggle(self):
        slider = ToggleSlider(checked=True)
        self.assertTrue(slider.isChecked())

        slider.setChecked(False)
        self.assertFalse(slider.isChecked())

        # Test click toggle
        events = []
        slider.toggled.connect(lambda val: events.append(val))
        slider.click()
        self.assertTrue(slider.isChecked())
        self.assertEqual(events, [True])

        slider.click()
        self.assertFalse(slider.isChecked())
        self.assertEqual(events, [True, False])

    def test_size_and_paint(self):
        slider = ToggleSlider(checked=True)
        self.assertEqual(slider.sizeHint(), QSize(40, 22))

        pixmap = QPixmap(slider.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        slider.render(pixmap)
        self.assertFalse(pixmap.isNull())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestSideTabsDraggableAndVisibility(unittest.TestCase):
    def setUp(self):
        self.side_tabs = SideTabs()
        self.w1 = QLabel("Monitor Widget")
        self.w2 = QLabel("Results Widget")
        self.w3 = QLabel("Config Widget")
        self.w4 = QLabel("Plots Widget")

        self.side_tabs.addTab(self.w1, "Training monitor", "monitor", "Monitor", tab_id="monitor")
        self.side_tabs.addTab(self.w2, "Results browser", "results", "Results", tab_id="results")
        self.side_tabs.addTab(self.w3, "Experiment", "config", "Experiment", tab_id="config")
        self.side_tabs.addTab(self.w4, "Plot viewer", "plots", "Plots", tab_id="plots")

    def test_initial_tabs(self):
        self.assertEqual(self.side_tabs.tab_order, ["monitor", "results", "config", "plots"])
        self.assertTrue(self.side_tabs.is_tab_visible("monitor"))
        self.assertTrue(self.side_tabs.is_tab_visible("results"))
        self.assertEqual(self.side_tabs.currentWidget(), self.w1)

    def test_reorder_tabs(self):
        order_changes = []
        self.side_tabs.tabOrderChanged.connect(lambda order: order_changes.append(list(order)))

        # Move "monitor" from visible slot 0 to after "results" (slot 2)
        self.side_tabs.move_tab_to_visible_slot("monitor", 2)
        self.assertEqual(self.side_tabs.tab_order, ["results", "monitor", "config", "plots"])
        self.assertEqual(len(order_changes), 1)

        # Move "plots" to the very top (slot 0)
        self.side_tabs.move_tab_to_visible_slot("plots", 0)
        self.assertEqual(self.side_tabs.tab_order, ["plots", "results", "monitor", "config"])

    def test_set_tab_visible(self):
        vis_events = []
        self.side_tabs.tabVisibilityChanged.connect(lambda tid, vis: vis_events.append((tid, vis)))

        # Hide "results"
        self.side_tabs.set_tab_visible("results", False)
        self.assertFalse(self.side_tabs.is_tab_visible("results"))
        self.assertTrue(self.side_tabs.tabs["results"]["button"].isHidden())
        self.assertEqual(vis_events, [("results", False)])

        # Show "results"
        self.side_tabs.set_tab_visible("results", True)
        self.assertTrue(self.side_tabs.is_tab_visible("results"))
        self.assertFalse(self.side_tabs.tabs["results"]["button"].isHidden())
        self.assertEqual(vis_events, [("results", False), ("results", True)])

    def test_hide_active_tab_auto_switches(self):
        # Current widget is w1 (monitor)
        self.assertEqual(self.side_tabs.currentWidget(), self.w1)

        # Hide monitor -> should auto-switch to next visible tab (results, w2)
        self.side_tabs.set_tab_visible("monitor", False)
        self.assertEqual(self.side_tabs.currentWidget(), self.w2)

    def test_apply_tab_order(self):
        custom_order = ["plots", "config", "results", "monitor"]
        self.side_tabs.apply_tab_order(custom_order)
        self.assertEqual(self.side_tabs.tab_order, custom_order)


# The panes the window builds itself, in their default order. TensorBoard has a tab
# too, but it comes from the core plugin rather than the window, so it has no pane
# slider and is not part of the built-in set.
BUILTIN_PANES = [
    "components", "config", "workflows", "monitor", "results", "plots", "terminal", "console",
]
DEFAULT_TAB_ORDER = [
    "components", "config", "workflows", "monitor", "results", "plots", "tensorboard",
    "terminal", "console",
    # Registered after the panes settings.toml lists, so load_layout leaves it
    # at the end rather than in the position app.py adds it.
    "figures",
]


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestWindowLayoutAndSliders(unittest.TestCase):
    def setUp(self):
        from unittest.mock import patch

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

        for target, replacement in (
            ("frontend.app.TerminalPanel", HeadlessTerminalPanel),
            ("frontend.app.LocalBackend", InertBackend),
            ("frontend.app.Window.init_default_experiment", lambda _self: None),
        ):
            patcher = patch(target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.window.settings_view.show_tab("appearance")

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_default_tab_order(self):
        expected_order = DEFAULT_TAB_ORDER
        order = self.window.tabs.tab_order
        self.assertEqual(order[:len(expected_order)], expected_order)
        # Tabs from enabled plugins (e.g. arXiv Papers) follow the built-in panes
        self.assertTrue(set(order[len(expected_order):]) <= set(self.window.plugin_manager.instances))

    def test_pane_sliders_registered(self):
        expected_panes = BUILTIN_PANES
        for pane_id in expected_panes:
            self.assertIn(pane_id, self.window.pane_sliders)
            slider = self.window.pane_sliders[pane_id]
            self.assertIsInstance(slider, ToggleSlider)
            # Sliders start in step with the default sidebar visibility in settings.toml
            self.assertEqual(slider.isChecked(), pane_id in self.window.settings_manager.sidebar_visible)

    def test_every_tab_has_a_switch_including_plugin_tabs(self):
        # Plugins add their tabs after Settings is built; TensorBoard is hidden by default,
        # so without a switch Settings offered no way to show it.
        self.assertEqual(set(self.window.pane_sliders), set(self.window.tabs.tabs))
        self.assertIn("tensorboard", self.window.pane_sliders)
        self.assertFalse(self.window.pane_sliders["tensorboard"].isChecked())
        self.window.pane_sliders["tensorboard"].setChecked(True)
        self.assertTrue(self.window.tabs.is_tab_visible("tensorboard"))

    def test_a_plugin_switch_comes_and_goes_with_the_plugin(self):
        self.assertTrue(self.window.plugin_manager.disable_plugin("tensorboard"))
        self.assertNotIn("tensorboard", self.window.pane_sliders)
        self.assertTrue(self.window.plugin_manager.enable_plugin("tensorboard"))
        self.assertIn("tensorboard", self.window.pane_sliders)
        self.assertEqual(
            self.window.pane_sliders["tensorboard"].isChecked(), self.window.tabs.is_tab_visible("tensorboard")
        )

    def test_a_switch_follows_its_tab_when_shown_elsewhere(self):
        self.window.tabs.set_tab_visible("plots", True)
        self.assertTrue(self.window.pane_sliders["plots"].isChecked())
        self.window.tabs.set_tab_visible("plots", False)
        self.assertFalse(self.window.pane_sliders["plots"].isChecked())

    def test_toggle_slider_hides_pane(self):
        # Plots is hidden by default, so show it first
        self.window.pane_sliders["plots"].setChecked(True)
        self.assertTrue(self.window.tabs.is_tab_visible("plots"))

        # Toggle plots off
        self.window.pane_sliders["plots"].setChecked(False)
        self.assertFalse(self.window.tabs.is_tab_visible("plots"))

        # Toggle plots back on
        self.window.pane_sliders["plots"].setChecked(True)
        self.assertTrue(self.window.tabs.is_tab_visible("plots"))

    def test_cannot_hide_all_panes(self):
        # Turn off every pane except one, plugin panes (TensorBoard, arXiv Papers) included
        panes = list(self.window.pane_sliders)
        panes.remove("console")
        panes.append("console")
        self.window.tabs.setCurrentWidget(self.window.tabs.tabs["console"]["widget"])
        for p in panes[:-1]:
            self.window.pane_sliders[p].setChecked(False)

        # The last remaining pane should be disabled so it cannot be turned off
        last_slider = self.window.pane_sliders[panes[-1]]
        self.assertTrue(last_slider.isChecked())
        self.assertFalse(last_slider.isEnabled())

        # Attempt to toggle off the last pane
        self.window.on_pane_slider_toggled(panes[-1], False)
        self.assertTrue(last_slider.isChecked())

    def test_reset_sidebar_layout(self):
        # Alter order and hide some panes
        self.window.tabs.move_tab_to_visible_slot("plots", 0)
        self.window.pane_sliders["monitor"].setChecked(False)

        # Reset
        self.window.reset_sidebar_layout()
        self.assertEqual(self.window.tabs.tab_order[:len(DEFAULT_TAB_ORDER)], DEFAULT_TAB_ORDER)
        for p in BUILTIN_PANES:
            shown = p not in {"workflows", "monitor"}  # The monitor appears when jobs need it.
            self.assertEqual(self.window.tabs.is_tab_visible(p), shown)
            self.assertEqual(self.window.pane_sliders[p].isChecked(), shown)

    def test_workflows_is_hidden_by_default_but_can_be_shown(self):
        self.assertFalse(self.window.tabs.is_tab_visible("workflows"))
        self.window.pane_sliders["workflows"].setChecked(True)
        self.assertTrue(self.window.tabs.is_tab_visible("workflows"))

    def test_save_and_load_layout(self):
        # Reorder and hide a pane
        self.window.tabs.move_tab_to_visible_slot("terminal", 0)
        self.window.pane_sliders["console"].setChecked(False)
        self.window.save_layout()

        self.assertEqual(self.window.settings_manager.sidebar_order[0], "terminal")
        self.assertNotIn("console", self.window.settings_manager.sidebar_visible)

        # Create a new Window pointing to the same data directory
        window2 = Window(self.data_dir)
        window2.settings_view.show_tab("appearance")
        try:
            self.assertEqual(window2.tabs.tab_order[0], "terminal")
            self.assertFalse(window2.tabs.is_tab_visible("console"))
            self.assertFalse(window2.pane_sliders["console"].isChecked())
        finally:
            window2.close()

    def test_closing_does_not_save_the_layout_or_rebuild_the_experiment(self):
        # Shutdown removes each plugin's tab. Saving the layout after each removal rebuilt the open
        # experiment three times per tab, and on Windows its boxes flashed up as separate windows.
        from frontend.config_viewer import ConfigViewer

        self.assertTrue(self.window.plugin_manager.instances, "expected plugins with tabs to remove")
        writes, rebuilds = [], []
        original_set = self.window.settings_manager.set
        self.window.settings_manager.set = lambda *args, **kwargs: (writes.append(args[:2]), original_set(*args, **kwargs))
        for viewer in self.window.findChildren(ConfigViewer):
            viewer._render_boxes = lambda: rebuilds.append(1)
        self.window.close()
        self.assertFalse(self.window.plugin_manager.instances)
        self.assertNotIn("sidebar", [section for section, _key in writes])
        self.assertEqual(rebuilds, [])


if __name__ == "__main__":
    unittest.main()
