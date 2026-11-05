"""The experiment pane's Recorded metrics box: pick the metrics a run records (telemetry.metrics).

A section's own checkbox records every metric of that section, including ones not seen yet ("losses/");
under it, one checkbox per metric this experiment's newest earlier run logged ("losses/total_loss").
What each choice skips is in src/app/telemetry/selection.py.
"""

from __future__ import annotations

import csv
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QVBoxLayout, QWidget

from src.app.telemetry.selection import ALWAYS_KEPT, SECTIONS

from .widgets import label

COLUMNS = 4  # sections per row
SECTION_HINTS = {
    "eval": "evaluation episodes run only if an eval/ metric is ticked",
    "system": "CPU and memory are sampled only if a system/ metric is ticked",
}


def logged_metrics(logs_dir: Path, group: str, experiment: str) -> list[str]:
    """Metric columns of the newest earlier run of `experiment` (results/logs/<group>/<experiment>[_<date>])."""
    folder = Path(logs_dir) / group
    if not group or not experiment or not folder.is_dir():
        return []
    runs = [p for p in folder.iterdir() if p.is_dir() and (p.name == experiment or p.name.startswith(experiment + "_"))]
    if not runs:
        return []
    newest = max(runs, key=lambda p: p.stat().st_mtime)
    names: set[str] = set()
    for metrics_csv in newest.glob("*/version_*/metrics.csv"):
        try:
            with metrics_csv.open(encoding="utf-8", newline="") as stream:
                names.update(next(csv.reader(stream), []))
        except OSError:
            continue
    return sorted(name for name in names if "/" in name and name not in ALWAYS_KEPT)


def section_of(name: str) -> str:
    return name.split("/", 1)[0]


class MetricPicker(QWidget):
    """Checkboxes for sections and metrics; `changed` carries the telemetry.metrics list."""

    changed = pyqtSignal(list)

    def __init__(self, selected: list[str], known: list[str], parent=None, *, row_factory=None, indent=18):
        super().__init__(parent)
        selected = list(selected or [])
        # Metrics without a section slash are valid names too. Keep them in a
        # separate group instead of treating the whole name as a section and
        # indexing a missing second part below (e.g. the malformed fixture's
        # integer value is displayed as "42" so validation can explain it).
        unsectioned = [name for name in known if "/" not in name]
        sections = list(SECTIONS) + sorted({section_of(n) for n in known + selected if "/" in n} - set(SECTIONS))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.section_boxes: dict[str, QCheckBox] = {}
        self.metric_boxes: dict[str, QCheckBox] = {}
        row_layout = None
        for index, section in enumerate(sections):
            if index % COLUMNS == 0:  # a new row of sections, each one a column of checkboxes
                row_layout = QHBoxLayout()
                row_layout.setSpacing(24)
                layout.addLayout(row_layout)
            column = QVBoxLayout()
            column.setSpacing(4)
            row_layout.addLayout(column, 1)
            whole = QCheckBox(f"{section}/  (all)")
            whole.setToolTip(f"Record every {section}/ metric, including ones not logged yet"
                             + (f": {SECTION_HINTS[section]}" if section in SECTION_HINTS else ""))
            whole.setChecked(f"{section}/" in selected)
            whole.toggled.connect(lambda _on, s=section: self._section_toggled(s))
            self.section_boxes[section] = whole
            column.addWidget(row_factory(section + "/", whole, True) if row_factory else whole)
            for name in (n for n in known if "/" in n and section_of(n) == section):
                box = QCheckBox(name.split("/", 1)[1])
                box.setToolTip(name)
                box.setChecked(name in selected or whole.isChecked())
                box.setEnabled(not whole.isChecked())
                box.setStyleSheet(f"margin-left: {indent}px;")
                box.toggled.connect(lambda _on: self._emit())
                self.metric_boxes[name] = box
                column.addWidget(row_factory(name, box, False) if row_factory else box)
            column.addStretch(1)
        if unsectioned:
            other = QVBoxLayout()
            other.setSpacing(4)
            other.addWidget(label("Other metrics", "heading"))
            for name in unsectioned:
                box = QCheckBox(name)
                box.setToolTip(name)
                box.setChecked(name in selected)
                box.toggled.connect(lambda _on: self._emit())
                self.metric_boxes[name] = box
                other.addWidget(row_factory(name, box, False) if row_factory else box)
            other.addStretch(1)
            layout.addLayout(other)
        if row_layout is not None:
            for _ in range(-len(sections) % COLUMNS):  # keep the last row's columns as wide as the others
                row_layout.addStretch(1)
        hint = label("Only ticked metrics are recorded. " + "; ".join(
            text[0].upper() + text[1:] for text in SECTION_HINTS.values()) + ". Step, epoch and transitions are "
            "always recorded." + ("" if known else " Metrics appear here once this experiment has run."), "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.empty_warning = label("Nothing is ticked: an empty list records every metric.", "configError")
        layout.addWidget(self.empty_warning)
        self.empty_warning.setVisible(not self.selection())

    def _section_toggled(self, section: str):
        """Ticked: the whole section (its metrics shown ticked, locked). Unticked: none of it, to pick from."""
        whole = self.section_boxes[section].isChecked()
        for name, box in self.metric_boxes.items():
            if section_of(name) == section:
                box.blockSignals(True)
                box.setChecked(whole)
                box.setEnabled(not whole)
                box.blockSignals(False)
        self._emit()

    def selection(self) -> list[str]:
        """telemetry.metrics: whole sections as "section/", other ticked metrics by name."""
        whole = [s for s, box in self.section_boxes.items() if box.isChecked()]
        names = [n for n, box in self.metric_boxes.items() if box.isChecked() and section_of(n) not in whole]
        return [f"{s}/" for s in whole] + names

    def _emit(self):
        selection = self.selection()
        self.empty_warning.setVisible(not selection)
        self.changed.emit(selection)
