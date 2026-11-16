"""One scrolling workspace for metric charts and optional monitor sections."""
from PyQt6.QtWidgets import QFrame, QScrollArea, QVBoxLayout, QWidget

from ..widgets import label


def embed_grid(view):
    """Let a shared chart grid participate in the workspace's single scroll area."""
    grid = view.grid
    content = grid.takeWidget()
    view.layout().replaceWidget(grid, content)
    grid.hide()
    grid.setParent(view)


class MonitorWorkspace(QScrollArea):
    def __init__(self, charts, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        self.sections = QVBoxLayout(content)
        self.sections.setContentsMargins(0, 0, 0, 0)
        self.sections.setSpacing(16)
        self.sections.addWidget(charts)
        self.setWidget(content)

    def add_section(self, title, view):
        section = QWidget()
        layout = QVBoxLayout(section)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(label(title, "heading"))
        layout.addWidget(view)
        self.sections.addWidget(section)
        section.hide()
        return section
