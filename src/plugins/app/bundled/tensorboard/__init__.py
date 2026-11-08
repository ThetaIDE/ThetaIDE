"""TensorBoard Core Plugin for Theta-IDE."""
from __future__ import annotations
from typing import Optional

from PyQt6.QtWidgets import QPushButton

from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

from .panel import TensorBoardPanel
from .server import TensorBoardServer


class TensorBoardPlugin(Plugin):
    """Core TensorBoard integration providing background server management and native UI controls."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: Optional[PluginContext] = None
        self.server: Optional[TensorBoardServer] = None
        self.panel: Optional[TensorBoardPanel] = None

    def activate(self, context: PluginContext) -> None:
        """Mount TensorBoard sidebar tab, extension slot button, and hotkeys."""
        self.context = context
        self.server = TensorBoardServer(parent=context.window)
        # Register tracker callback so context supervises and terminates server process on shutdown
        self.server.set_tracker_callback(context.track_process)

        self.panel = TensorBoardPanel(self.server, log=context.log)

        # 1. Mount sidebar tab
        context.add_sidebar_tab(
            tab_id="tensorboard",
            widget=self.panel,
            title="TensorBoard",
            icon_name="tensorboard",
            short_label="TensorBoard",
        )

        # 2. Register command for launcher and hotkeys
        context.register_command(
            command_id="tensorboard.open",
            title="View: TensorBoard",
            callback=lambda: context.select_sidebar_tab("tensorboard"),
            default_hotkey="Action + 6",
            keywords=("tensorboard", "tb", "tbd", "events", "scalars"),
            icon="tensorboard",
            detail="Open TensorBoard management panel",
        )

        # 3. Contribute 'Open TensorBoard' button to the Training Monitor footer slot
        def _make_monitor_button() -> QPushButton:
            btn = QPushButton("Open TensorBoard")
            btn.clicked.connect(lambda: context.select_sidebar_tab("tensorboard"))
            return btn

        context.contribute_to_slot(
            slot_id="monitor.footer",
            action_id="open_tensorboard",
            factory_or_widget=_make_monitor_button,
            priority=100,
        )

        context.log("TensorBoard plugin activated.")

    def deactivate(self) -> None:
        """Cleanly stop server and unmount all UI contributions."""
        if self.server:
            try:
                self.server.stop()
            except Exception as exc:
                if self.context:
                    self.context.log(f"Error stopping TensorBoard server: {exc}")
            self.server = None

        if self.panel:
            self.panel.deleteLater()
            self.panel = None

        if self.context:
            self.context.log("TensorBoard plugin deactivated.")
            self.context = None
