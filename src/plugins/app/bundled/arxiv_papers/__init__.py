"""arXiv Papers core plugin: search arXiv for the papers behind ThetaIDE's components, by tag."""
from __future__ import annotations

from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext

from .panel import ArxivPanel


class ArxivPapersPlugin(Plugin):
    """Adds the arXiv tab to the sidebar; turning the plugin off in Settings removes it."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: PluginContext | None = None
        self.panel: ArxivPanel | None = None

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = ArxivPanel(context=context)
        context.add_sidebar_tab(tab_id="arxiv_papers", widget=self.panel, title="arXiv Papers",
                                icon_name="arxiv", short_label="arXiv")

    def deactivate(self) -> None:
        if self.panel:
            self.panel.cleanup()
            self.panel.deleteLater()
            self.panel = None
        if self.context:
            self.context.remove_sidebar_tab("arxiv_papers")
            self.context = None
