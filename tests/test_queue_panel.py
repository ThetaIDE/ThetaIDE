"""Unit tests for the frontend QueuePanel component."""
import os
import sys
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QWidget
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.queue_panel import QueuePanel, clock
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available in current Python environment")
class TestQueuePanel(unittest.TestCase):

    def setUp(self):
        self.moves = []
        self.removals = []
        self.opened = []
        self.toggled = []
        self.panel = QueuePanel(
            on_move=lambda jid, pos: self.moves.append((jid, pos)),
            on_remove=lambda jid: self.removals.append(jid),
            on_open=lambda jid: self.opened.append(jid),
            on_toggle=lambda run: self.toggled.append(run),
        )

    def test_clock_helper(self):
        self.assertEqual(clock(0), "0:00")
        self.assertEqual(clock(65), "1:05")
        self.assertEqual(clock(None), "0:00")
        self.assertEqual(clock(-10), "0:00")
        self.assertEqual(clock("invalid"), "0:00")

    def test_set_data_with_pending_job_null_started(self):
        """Active job with started=None (e.g. pending status) must not raise TypeError."""
        data = {
            "running": True,
            "active": [{
                "job_id": "job-pending-1",
                "experiment_id": "test_exp_pending",
                "status": "pending",
                "created": 1000.0,
                "started": None,
                "finished": None,
                "total_timesteps": 10000,
                "effective_timesteps": 10240,
            }],
            "queued": [],
            "finished": [],
        }
        self.panel.set_data(data)
        self.assertEqual(self.panel.table.rowCount(), 1)
        self.assertEqual(self.panel.table.item(0, 0).text(), "▶")
        self.assertEqual(self.panel.table.item(0, 1).text(), "test_exp_pending")
        self.assertEqual(self.panel.table.item(0, 2).text(), "10,240")
        self.assertEqual(self.panel.table.item(0, 3).text(), "pending")
        self.assertTrue(self.panel.table.item(0, 4).text().startswith("starting"))

    def test_set_data_with_null_and_missing_fields(self):
        """Jobs with null/missing experiment_id, timesteps, and timestamps should render cleanly."""
        data = {
            "running": False,
            "active": [{
                "job_id": "job-active-nulls",
                "status": "running",
            }],
            "queued": [{
                "job_id": "job-queued-nulls",
                "position": 0,
            }],
            "finished": [{
                "job_id": "job-finished-nulls",
                "status": "cancelled",
            }],
        }
        self.panel.set_data(data)
        self.assertEqual(self.panel.table.rowCount(), 3)
        # Check active row
        self.assertEqual(self.panel.table.item(0, 1).text(), "job-active-nulls")
        self.assertEqual(self.panel.table.item(0, 2).text(), "—")
        # Check queued row
        self.assertEqual(self.panel.table.item(1, 0).text(), "1")
        self.assertEqual(self.panel.table.item(1, 4).text(), "queued")
        # Check finished row
        self.assertEqual(self.panel.table.item(2, 4).text(), "never started")

    def test_button_actions_and_selection(self):
        data = {
            "running": False,
            "active": [],
            "queued": [
                {"job_id": "q1", "experiment_id": "exp1", "position": 0, "status": "queued"},
                {"job_id": "q2", "experiment_id": "exp2", "position": 1, "status": "queued"},
            ],
            "finished": [],
        }
        self.panel.set_data(data)
        self.panel.table.selectRow(0)
        self.panel.update_buttons()

        self.assertFalse(self.panel.up_button.isEnabled())
        self.assertTrue(self.panel.down_button.isEnabled())
        self.assertTrue(self.panel.remove_button.isEnabled())
        self.assertTrue(self.panel.open_button.isEnabled())

        self.panel.move_selected(1)
        self.assertEqual(self.moves, [("q1", 1)])

        self.panel.remove_selected()
        self.assertEqual(self.removals, ["q1"])

        self.panel.open_selected()
        self.assertEqual(self.opened, ["q1"])

    def test_set_offline(self):
        self.panel.set_offline("Connection refused")
        self.assertFalse(self.panel.run_button.isEnabled())
        self.assertEqual(self.panel.summary.toolTip(), "Connection refused")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available in current Python environment")
class TestQueuePanelRunControls(unittest.TestCase):
    """The run controls that used to be the menu bar's Run menu."""

    def test_buttons_call_back_and_hide_without_a_callback(self):
        calls = []
        panel = QueuePanel(lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: None,
                           on_launch=lambda: calls.append("launch"), on_add=lambda: calls.append("add"),
                           on_stop=lambda: calls.append("stop"), on_demo=lambda: calls.append("demo"))
        for button in (panel.launch_button, panel.add_button, panel.stop_button, panel.demo_button):
            button.click()
        self.assertEqual(calls, ["launch", "add", "stop", "demo"])
        bare = QueuePanel(lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: None)
        self.assertFalse(bare.launch_button.isVisibleTo(bare))

    def test_names_the_experiment_it_acts_on(self):
        panel = QueuePanel(lambda *a: None, lambda *a: None, lambda *a: None, lambda *a: None)
        panel.set_experiment("csc510/cartpole_demo")
        self.assertIn("csc510/cartpole_demo", panel.experiment.text())
        panel.set_experiment(None)
        self.assertEqual(panel.experiment.text(), "No experiment open")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available in current Python environment")
class TestRunControlsInTheWindow(unittest.TestCase):
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

    def setUp(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from frontend.app import Window

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

        self.temp = tempfile.TemporaryDirectory()
        with (
            patch("frontend.app.TerminalPanel", self.HeadlessTerminalPanel),
            patch("frontend.app.LocalBackend", InertBackend),
            patch.object(Window, "init_default_experiment", lambda _self: None),
        ):
            self.win = Window(Path(self.temp.name) / "runs")

    def tearDown(self):
        self.win._quitting = True
        self.win.close()
        self.temp.cleanup()

    def test_queue_sidebar_follows_active_jobs_and_can_be_opened_while_idle(self):
        state = {"running": False, "active": [], "queued": [], "finished": []}
        self.assertFalse(self.win.tabs.is_tab_visible("queue"))
        state["active"] = [{"job_id": "job-1", "status": "running"}]
        self.win.on_queue_changed(state)
        self.assertTrue(self.win.tabs.is_tab_visible("queue"))
        state["active"] = []
        self.win.on_queue_changed(state)
        self.assertFalse(self.win.tabs.is_tab_visible("queue"))
        self.win.on_pane_slider_toggled("queue", True)
        self.win.on_queue_changed(state)
        self.assertTrue(self.win.tabs.is_tab_visible("queue"))

    def test_the_run_menu_moved_to_the_queue_tab_and_its_shortcuts_still_work(self):
        menus = [a.text() for a in self.win.menuBar().actions()]
        self.assertNotIn("Run", menus)
        self.assertEqual(self.win.menu_actions["launch_training"].shortcut().toString(), "F5")
        self.assertIn(self.win.menu_actions["launch_training"], self.win.actions())
        panel = self.win.queue_panel
        self.assertIs(self.win.queue_button, panel.add_button)
        self.assertIs(self.win.stop_button, panel.stop_button)
        self.win.update_launch_state()
        self.assertFalse(panel.stop_button.isEnabled())  # nothing is running
        self.win.menu_actions["start_demo"].trigger()
        self.assertIsNotNone(self.win.active)
        self.assertTrue(panel.stop_button.isEnabled())
        self.assertFalse(panel.launch_button.isEnabled())
        panel.stop_button.click()
        self.assertIsNone(self.win.active)

    def test_the_runs_list_is_in_the_plots_tab_and_leaves_the_monitor_alone(self):
        from frontend.model import new_run

        sidebar = self.win.monitor.sidebar
        self.assertTrue(self.win.plot_viewer.isAncestorOf(sidebar))
        self.assertFalse(self.win.queue_panel.isAncestorOf(sidebar))
        self.assertFalse(self.win.monitor.isAncestorOf(sidebar))
        first, second = new_run(self.win.config()), new_run(self.win.config())
        self.win.runs[:0] = [first, second]
        self.win.refresh_runs()
        self.win.select_run(first)
        self.win.tabs.setCurrentWidget(self.win.plot_viewer)
        item = sidebar.items[second["id"]]
        sidebar.tree.itemPressed.emit(item, 0)  # a click plots it
        sidebar.tree.itemClicked.emit(item, 0)
        self.assertEqual(sidebar.compared, [second["id"]])
        self.assertIs(self.win.selected, first)  # the monitor is not driven from the runs list
        self.assertIs(self.win.tabs.currentWidget(), self.win.plot_viewer)

    def test_double_clicking_a_job_plots_its_run_in_the_plot_viewer(self):
        from frontend.model import new_run

        run = new_run(self.win.config(), simulated=False)
        run.update(status="completed", backend={"job_id": "job_1", "experiment_id": "exp_1"})
        self.win.runs.insert(0, run)
        self.win.refresh_runs()
        panel = self.win.queue_panel
        panel.set_data({"running": False, "active": [], "queued": [],
                        "finished": [{"job_id": "job_1", "experiment_id": "exp_1", "status": "completed"}]})
        self.assertEqual(panel.open_button.text(), "Open in Plot Viewer")
        panel.table.selectRow(0)  # a real double-click selects the row first
        panel.table.itemDoubleClicked.emit(panel.table.item(0, 1))
        self.assertIs(self.win.tabs.currentWidget(), self.win.plot_viewer)
        self.assertEqual(self.win.monitor.sidebar.compared, [run["id"]])
