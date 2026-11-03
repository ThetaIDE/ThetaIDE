"""Execution and replicate inspection for repeated local jobs."""
import json
from pathlib import Path

from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from ..charts import Chart
from ..widgets import label


class ExecutionsView(QWidget):
    def __init__(self, backend, open_run, parent=None):
        super().__init__(parent)
        self.backend, self.open_run = backend, open_run
        self.data = None
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.execution = QComboBox()
        self.execution.currentIndexChanged.connect(self.load)
        self.refresh_button = QPushButton("Refresh executions")
        self.refresh_button.clicked.connect(self.refresh)
        controls.addWidget(self.execution, 1)
        controls.addWidget(self.refresh_button)
        layout.addLayout(controls)
        self.summary = label("Refresh to list executions.", "muted")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.method = QComboBox()
        self.method.currentIndexChanged.connect(self.draw)
        layout.addWidget(self.method)
        self.chart = Chart("reward", "Evaluation reward — mean across training runs", band="spread", zero_based=False)
        self.chart.setMinimumHeight(200)
        layout.addWidget(self.chart, 1)
        self.note = label("", "muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        row = QHBoxLayout()
        self.replicate = QComboBox()
        self.replicate.currentIndexChanged.connect(self.show_replicate)
        self.open_button = QPushButton("Open individual run")
        self.open_button.clicked.connect(self.open_selected)
        row.addWidget(self.replicate, 1)
        row.addWidget(self.open_button)
        layout.addLayout(row)
        self.event = QComboBox()
        self.event.currentIndexChanged.connect(self.draw_episodes)
        layout.addWidget(self.event)
        self.episode_chart = Chart("reward", "Episode rewards for selected model and evaluation", zero_based=False)
        self.episode_chart.setMinimumHeight(160)
        layout.addWidget(self.episode_chart, 1)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select a replicate to inspect its status and evaluation episodes.")
        layout.addWidget(self.details, 1)

    def refresh(self):
        if not hasattr(self.backend, "list_executions"):
            self.summary.setText("Execution browsing is unavailable from this backend.")
            return
        self.backend.list_executions(self._listed)

    def _listed(self, paths, error):
        if error:
            self.summary.setText(f"Could not list executions: {error}")
            return
        selected = self.execution.currentData()
        self.execution.blockSignals(True)
        self.execution.clear()
        for path in paths:
            p = Path(path)
            self.execution.addItem(f"{p.parent.parent.parent.name} / {p.parent.parent.name} / {p.parent.name[:12]}", path)
        index = self.execution.findData(selected)
        self.execution.setCurrentIndex(max(0, index))
        self.execution.blockSignals(False)
        if self.execution.count():
            self.load()
        else:
            self.summary.setText("No repeated executions found.")

    def load(self, *_):
        path = self.execution.currentData()
        if path:
            self.backend.read_execution(path, self._loaded)

    def _loaded(self, data, error):
        if error:
            self.summary.setText(f"Could not read execution: {error}")
            return
        self.data = data
        runs = data["manifest"].get("runs", [])
        completed = sum(run.get("status") == "completed" for run in runs)
        self.summary.setText(f"{completed} of {len(runs)} trainings completed. " + "; ".join(data.get("errors", [])))
        self.method.blockSignals(True)
        self.method.clear()
        for name in data["curves"]:
            self.method.addItem(name)
        self.method.blockSignals(False)
        self.replicate.blockSignals(True)
        self.replicate.clear()
        for run in runs:
            self.replicate.addItem(f"{run['method']} · seed {run['training_seed']} · {run['status']}", run)
        self.replicate.blockSignals(False)
        self.draw()
        self.show_replicate()

    def draw(self, *_):
        if not self.data:
            return
        method = self.method.currentText()
        points = self.data["curves"].get(method, [])
        band = len(points) > 1 and all(p["std"] is not None for p in points)
        self.chart.band = "spread" if band else None
        self.chart.set_series([(method, [{"step": p["step"], "reward": p["mean"], "spread": p["std"], "n": p["n"]} for p in points], "#83a598")])
        counts = ", ".join(f"{p['step']:g}: {p['n']}" for p in points)
        self.note.setText("Mean of per-model episode means; " + ("band: sample standard deviation across trainings." if band else "training standard deviation band unavailable for this grid.") + f" Contributing runs by step: {counts or 'none'}. Grid: union of observed steps, interpolated only within each run's observed range.")

    def show_replicate(self, *_):
        run = self.replicate.currentData()
        if run and self.data:
            events = self.data["events"].get(run["run_id"], [])
            self.event.blockSignals(True)
            self.event.clear()
            for event in events:
                self.event.addItem(f"Evaluation at step {event.get('training_step')} · {event.get('episode_count')} episodes", event)
            self.event.blockSignals(False)
            self.details.setPlainText(json.dumps({"run": run, "evaluation_events": events}, indent=2))
        else:
            self.details.clear()
            self.event.clear()
        self.draw_episodes()

    def draw_episodes(self, *_):
        event = self.event.currentData() or {}
        outcomes = event.get("episode_outcomes", [])
        self.episode_chart.set_series([("Episode reward", [{"step": i + 1, "reward": value} for i, value in enumerate(outcomes)], "#b8bb26")])

    def open_selected(self):
        run = self.replicate.currentData()
        if run and self.data:
            self.open_run(self.data["manifest"]["group"], run["experiment_id"])
