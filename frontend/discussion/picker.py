"""The figure picker: the compatible plot components of one experiment as clickable cards.

It occupies the Parameters column while open. Candidates with block findings are omitted;
candidates with warn findings show them inline, as the configuration pickers do. Choosing a
card emits ``chosen(experiment, selector)``; the panel then creates and opens the figure.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication

from ..config_panel.controls import ConfigBox, _make_subtle_tool_button, compatibility_status
from ..widgets import label
from . import model

NO_CANDIDATES = "No compatible plot components"


def parameter_summary(capability) -> str:
    """"6 parameters · 3 live", or "" when the schema cannot be read."""
    try:
        schema = model.capability_schema(capability)
    except model.FigureError:
        return ""
    count = len(schema.get("properties") or {})
    live = len(model.live_parameters(schema))
    text = f"{count} parameter{'' if count == 1 else 's'}"
    return f"{text} · {live} live" if live else text


class FigureCard(ConfigBox):
    """One candidate. Click, Enter or Space chooses it."""

    clicked = pyqtSignal()

    def __init__(self, candidate, parent=None):
        super().__init__(parent=parent, section="figureCard")
        self.candidate = candidate
        capability, package = candidate.capability, candidate.package
        self.setProperty("figureCard", True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        title = capability.metadata.get("label") or capability.id
        self.setAccessibleName(f"Add {title} figure")
        self.layout.setSpacing(6)

        heading = label(title, "cardTitle")
        self.layout.addWidget(heading)
        source = label(f"{package.id} {package.version}".strip(), "muted")
        source.setToolTip(capability.id)
        self.layout.addWidget(source)
        description = capability.metadata.get("description")
        if description:
            self.layout.addWidget(label(str(description), wrap=True))
        summary = parameter_summary(capability)
        if summary:
            self.layout.addWidget(label(summary, "muted"))
        for line in compatibility_status(candidate.warnings):
            self.layout.addWidget(line)

    @property
    def selector(self) -> str:
        return self.candidate.selector

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.clicked.emit()
            return
        super().keyPressEvent(event)


class FigurePicker(ConfigBox):
    """An "Add figure" box holding a card per compatible plot component, as Methods holds its
    method cards."""

    chosen = pyqtSignal(str, str)  # experiment, plotter selector
    cancelled = pyqtSignal()

    def __init__(self, experiment: str, candidates, parent=None):
        super().__init__(parent=parent, section="figurePicker")
        self.experiment = experiment
        self.cards: list[FigureCard] = []
        close = _make_subtle_tool_button("close", "Close without adding a figure", size=24, icon_size=14)
        close.clicked.connect(self.cancelled)
        self.set_header("Add figure").addWidget(close)
        self.add_widget(label(f"Plot components compatible with {experiment}", "muted", wrap=True))

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            offered = [c for c in candidates(experiment) if not c.blocked]
        except Exception as exc:  # the experiment does not compose
            self.add_widget(label(f"Cannot read {experiment}: {exc}", "muted", wrap=True))
            offered = None
        finally:
            QApplication.restoreOverrideCursor()
        if offered == []:
            self.add_widget(label(NO_CANDIDATES, "muted"))
        for candidate in offered or []:
            card = FigureCard(candidate)
            card.clicked.connect(lambda s=candidate.selector: self.chosen.emit(self.experiment, s))
            self.cards.append(card)
            self.add_widget(card)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()
            return
        super().keyPressEvent(event)
