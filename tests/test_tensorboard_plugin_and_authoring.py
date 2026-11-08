"""Tests for the modernized plugin authoring system and TensorBoard core plugin."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

from PyQt6.QtCore import QCoreApplication, QProcess
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from frontend.app import Window
from frontend.launcher import collect_items
from frontend.plugins.commands import Command, CommandRegistry
from frontend.plugins.context import PluginContext
from frontend.plugins.core.tensorboard import TensorBoardPlugin
from frontend.plugins.core.tensorboard.server import TensorBoardServer
from frontend.plugins.slots import ExtensionSlot, ExtensionSlotRegistry, ExtensionSlotWidget

app = QApplication.instance() or QApplication(sys.argv[:1])


class TestPluginAuthoringAndTensorboard(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.window = Window(data_dir=self.temp_dir, workspace=Path(self.temp_dir) / "workspace")

    def tearDown(self):
        if hasattr(self.window, "plugin_manager"):
            self.window.plugin_manager.shutdown()
        self.window.close()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_extension_slot_reactive_rebuild(self):
        """ExtensionSlot and ExtensionSlotWidget add, order, and remove actions dynamically."""
        registry = ExtensionSlotRegistry()
        slot = registry.get_slot("test.slot")
        self.assertIsNotNone(slot)

        widget = ExtensionSlotWidget(slot)
        # Initially empty slot makes container invisible
        self.assertFalse(widget.isVisible())
        self.assertEqual(len(widget._rendered_widgets), 0)

        # Add contribution
        slot.add_action("btn1", lambda: QPushButton("Button 1"), priority=50)
        self.assertTrue(widget.isVisible())
        self.assertIn("btn1", widget._rendered_widgets)

        # Add second contribution with higher priority number (lower display precedence)
        slot.add_action("lbl1", lambda: QLabel("Label 1"), priority=100)
        self.assertEqual(len(widget._rendered_widgets), 2)
        items = slot.get_items()
        self.assertEqual(items[0][0], "btn1")
        self.assertEqual(items[1][0], "lbl1")

        # Remove contribution
        slot.remove_action("btn1")
        self.assertNotIn("btn1", widget._rendered_widgets)
        self.assertEqual(len(widget._rendered_widgets), 1)

        # Remove last contribution -> widget hides
        slot.remove_action("lbl1")
        self.assertEqual(len(widget._rendered_widgets), 0)
        self.assertFalse(widget.isVisible())

    def test_command_registry_launcher_integration(self):
        """Commands registered via CommandRegistry surface in launcher collect_items."""
        cmd_registry = self.window.command_registry
        called = []

        cmd = Command(
            command_id="test.action",
            title="Execute Custom Action",
            callback=lambda: called.append(True),
            default_hotkey="Ctrl+Shift+X",
            keywords=("custom", "action", "run"),
            category="Plugin",
        )
        cmd_registry.register_command(cmd)

        items = collect_items(self.window)
        found = [it for it in items if it.key == "cmd:test.action"]
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].title, "Execute Custom Action")
        self.assertEqual(found[0].shortcut, "Ctrl+Shift+X")

        # Execute launcher action
        found[0].action()
        self.assertEqual(called, [True])

        # Unregister
        cmd_registry.unregister_command("test.action")
        items_after = collect_items(self.window)
        self.assertEqual([it for it in items_after if it.key == "cmd:test.action"], [])

    def test_context_process_supervision_and_cleanup(self):
        """Tracked child processes are automatically terminated on context cleanup."""
        context = PluginContext(
            plugin_id="proc_test",
            window=self.window,
            storage_file=Path(self.temp_dir) / "proc_test.json"
        )
        # Spawn a sleep process using the managed helper
        proc = context.spawn_process(sys.executable, ["-c", "import time; time.sleep(30)"])
        proc.start()
        self.assertTrue(proc.waitForStarted(2000))
        self.assertEqual(proc.state(), QProcess.ProcessState.Running)
        pid = int(proc.processId())
        self.assertGreater(pid, 0)

        # Context cleanup must kill the process tree cleanly.
        context.cleanup()
        deadline = time.time() + 3.0
        while time.time() < deadline and proc.state() != QProcess.ProcessState.NotRunning:
            proc.waitForFinished(100)
            QCoreApplication.processEvents()
            time.sleep(0.02)

        if proc.state() != QProcess.ProcessState.NotRunning:
            import psutil
            try:
                p = psutil.Process(pid)
                is_alive = p.is_running() and p.status() != psutil.STATUS_ZOMBIE
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                is_alive = False
            self.assertFalse(is_alive, f"PID {pid} should have been terminated")
        else:
            self.assertEqual(proc.state(), QProcess.ProcessState.NotRunning)

    def test_tensorboard_core_plugin_lifecycle(self):
        """TensorBoard plugin mounts tab, contributes footer button, registers command, and deactivates cleanly."""
        manager = self.window.plugin_manager

        # TensorBoard must be discovered as a core plugin
        self.assertIn("tensorboard", manager.manifests)
        manifest = manager.manifests["tensorboard"]
        self.assertTrue(manifest.is_core)
        self.assertTrue(manifest.default_enabled)

        # Since it is default_enabled, it is already active
        self.assertTrue(manager.is_plugin_enabled("tensorboard"))
        self.assertIn("tensorboard", self.window.tabs.tabs)

        # Verify extension slot contribution in monitor footer
        monitor_slot = self.window.extension_slots.get_slot("monitor.footer")
        self.assertIsNotNone(monitor_slot)
        items = dict(monitor_slot.get_items())
        self.assertIn("open_tensorboard", items)

        # Verify command registry has tensorboard.open
        cmd = self.window.command_registry.get_command("tensorboard.open")
        self.assertIsNotNone(cmd)
        self.assertEqual(cmd.title, "View: TensorBoard")
        self.assertEqual(cmd.default_hotkey, "Action + 6")

        # Now disable the plugin dynamically
        manager.disable_plugin("tensorboard")
        self.assertFalse(manager.is_plugin_enabled("tensorboard"))
        self.assertNotIn("tensorboard", self.window.tabs.tabs)

        # Button should be cleanly removed from monitor footer slot
        items_after = dict(monitor_slot.get_items())
        self.assertNotIn("open_tensorboard", items_after)

        # Command should be cleanly removed from command registry
        self.assertIsNone(self.window.command_registry.get_command("tensorboard.open"))

        # Re-enabling restores tab, footer slot, and command
        manager.enable_plugin("tensorboard")
        self.assertTrue(manager.is_plugin_enabled("tensorboard"))
        self.assertIn("tensorboard", self.window.tabs.tabs)
        self.assertIn("open_tensorboard", dict(monitor_slot.get_items()))
        self.assertIsNotNone(self.window.command_registry.get_command("tensorboard.open"))


if __name__ == "__main__":
    unittest.main()
