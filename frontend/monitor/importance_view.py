"""Ranked parameter-importance bars shared by live and saved sweep dashboards."""
import math

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QGridLayout, QLabel, QProgressBar, QScrollArea, QVBoxLayout, QWidget

from ..widgets import label


class ImportanceView(QWidget):
    def __init__(self, *, compact=False):
        super().__init__()
        self.compact = compact
        layout = QVBoxLayout(self)
        layout.addWidget(label("Parameter importance", "cardTitle"))
        description = label("Which parameters explain the differences in objective scores? Higher bars indicate greater estimated importance.", "muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        description.setVisible(not compact)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        layout.addWidget(self.scroll, 1)
        self.note = label("", "muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.scores = []
        self._state = None
        self.set_study(None)

    def set_study(self, study):
        study = study or {}
        info = study.get("parameter_importance_info") or {}
        completed = info.get("completed_trials", sum(trial.get("state") == "complete" for trial in study.get("trials", [])))
        minimum = info.get("minimum_trials", 5)
        scores = study.get("parameter_importance") or {}
        scores = sorted(((str(name), float(value)) for name, value in scores.items()
                         if isinstance(value, (float, int)) and not isinstance(value, bool)
                         and math.isfinite(value) and 0 <= value <= 1), key=lambda item: (-item[1], item[0]))
        message = info.get("message")
        if scores:
            message = f"Based on {completed} completed trials · {info.get('evaluator', 'Estimated importance')}"
        elif not study:
            message = "Select a study to view parameter importance."
        elif info.get("status") == "waiting":
            message = f"Waiting for completed trials: {completed}/{minimum}."
        elif not message:
            message = (f"Waiting for completed trials: {completed}/{minimum}." if completed < minimum
                       else "No importance estimate was saved for this study.")
        state = (scores, message)
        if state == self._state:
            return
        self._state = state
        self.scores = scores
        self.status.setText(message)
        self.status.setVisible(not self.compact or not scores)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 12, 12, 12)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(18)
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 4)
        for row, (name, value) in enumerate(scores):
            title = QLabel(name)
            title.setWordWrap(True)
            title.setToolTip(name)
            bar = QProgressBar()
            bar.setRange(0, 10000)
            bar.setValue(round(value * 10000))
            bar.setTextVisible(False)
            bar.setFixedHeight(18)
            bar.setAccessibleName(f"{name}: {value:.1%} importance")
            bar.setToolTip(f"{name}: {value:.2%}")
            score = QLabel(f"{value:.1%}")
            score.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(title, row, 0)
            grid.addWidget(bar, row, 1)
            grid.addWidget(score, row, 2)
        grid.setRowStretch(len(scores), 1)
        old = self.scroll.takeWidget()
        if old is not None:
            old.deleteLater()
        self.scroll.setWidget(content)
        if self.compact:
            self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.scroll.setFixedHeight(max(60, grid.sizeHint().height()))
        self.scroll.setVisible(bool(scores))
        self.note.setText("Importance is relative and does not indicate whether increasing a parameter improves the score. "
                          "fANOVA compares parameters shared across completed trials.")
        self.note.setVisible(bool(scores) and not self.compact)
