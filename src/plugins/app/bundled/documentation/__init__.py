"""Documentation Core Plugin for Theta-IDE."""
from __future__ import annotations

from typing import Optional

from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

from .panel import DocumentationPanel


class DocumentationPlugin(Plugin):
    """Core documentation plugin providing an interactive reference guide in Theta-IDE."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: PluginContext | None = None
        self.panel: DocumentationPanel | None = None

    def activate(self, context: PluginContext) -> None:
        """Mount the documentation tab to the IDE's primary sidebar."""
        self.context = context
        self.panel = DocumentationPanel(context=context)
        context.add_sidebar_tab(
            tab_id="documentation",
            widget=self.panel,
            title="Documentation",
            icon_name="documentation",
            short_label="Docs",
        )
        context.log("Documentation core plugin activated.")

    def deactivate(self) -> None:
        """Cleanly unmount and dispose of the documentation panel with zero system impact."""
        if self.panel:
            self.panel.cleanup()
            self.panel.deleteLater()
            self.panel = None

        if self.context:
            self.context.remove_sidebar_tab("documentation")
            self.context.log("Documentation core plugin deactivated.")
            self.context = None
