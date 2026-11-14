"""Study-level tuning results from the portable tuner snapshots."""
from __future__ import annotations

import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget, QPushButton, QHeaderView,
                             QScrollArea, QFrame)

from ..plots import PlotCanvas
from .importance_view import ImportanceView
from .parallel_coordinates import ParallelCoordinates


# Compatibility import for existing desktop callers. Reading summaries is headless.
from src.app.pipeline.tuning_monitor import study_summaries


class SweepView(QWidget):
    trial_selected = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.studies = []
        self.lives = {}
        self._future_trials = 0
        layout = QVBoxLayout(self)
        # Retain the header state for existing integrations, but keep the
        # dashboard focused on its visualizations.
        header = QWidget(self)
        header.hide()
        header_layout = QVBoxLayout(header)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Study"))
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._show_study)
        bar.addWidget(self.selector, 1)
        self.chart_type = QComboBox()
        self.chart_type.currentIndexChanged.connect(self._draw_plot)
        self.summary = QLabel("No tuning studies available for this run.")
        self.summary.setWordWrap(True)
        self.summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_layout.addWidget(self.summary)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter trials by number, state, value, or parameter")
        self.filter.textChanged.connect(self._filter_trials)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.filter, 1)
        self.selection_count = QLabel("0 / 0 trials")
        filter_row.addWidget(self.selection_count)
        self.clear_filters = QPushButton("Clear filters")
        self.clear_filters.setEnabled(False)
        self.clear_filters.clicked.connect(self._clear_trial_filters)
        filter_row.addWidget(self.clear_filters)
        header_layout.addLayout(filter_row)
        self._filtered_trial_ids = set()
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        graphs = QVBoxLayout(content)
        graphs.setContentsMargins(0, 0, 0, 0)
        graphs.setSpacing(24)
        self.scroll.setWidget(content)
        layout.addWidget(self.scroll)
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        bar.addWidget(self.chart_type)
        overview_layout.addLayout(bar)
        trials_page = QWidget(header)
        trials_page.hide()
        trials_layout = QVBoxLayout(trials_page)
        parallel_page = QWidget()
        parallel_layout = QVBoxLayout(parallel_page)
        self.parallel_plot = ParallelCoordinates()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumHeight(380)
        scroll.setWidget(self.parallel_plot)
        parallel_layout.addWidget(scroll, 1)
        self.plot = PlotCanvas()
        self.plot.setMinimumHeight(360)
        overview_layout.addWidget(self.plot, 1)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Trial", "State", "Objective", "Duration (s)", "Parameters"])
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self._open_trial)
        self.table.itemSelectionChanged.connect(self._show_trial_detail)
        self.table.verticalHeader().hide()
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        for column in range(4):
            self.table.horizontalHeader().setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        trials_layout.addWidget(self.table, 1)
        graphs.addWidget(overview)
        graphs.addWidget(parallel_page)
        self.importance_view = ImportanceView(compact=True)
        graphs.addWidget(self.importance_view)
        graphs.addStretch()
        self.detail = QLabel("Select a trial to inspect its parameters.")
        self.detail.setWordWrap(True)
        self.detail.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        trials_layout.addWidget(self.detail)
        self.parameters = QTableWidget(0, 2)
        self.parameters.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.parameters.verticalHeader().hide()
        self.parameters.setShowGrid(False)
        self.parameters.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.parameters.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.parameters.setMaximumHeight(160)
        self.parameters.hide()
        trials_layout.addWidget(self.parameters)
        self.open_button = QPushButton("Open trial monitor")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(lambda: self._open_trial(self.table.currentRow(), 0))
        trials_layout.addWidget(self.open_button, 0, Qt.AlignmentFlag.AlignRight)
        self.parallel_plot.selection_changed.connect(self._filter_trials)

    def set_studies(self, studies):
        if studies == self.studies:
            return
        current = self.selector.currentData()
        self.studies = studies
        self.selector.blockSignals(True)
        self.selector.clear()
        for method, study in studies:
            self.selector.addItem(f"{method} · {study['study']}", (method, study['study']))
        index = self.selector.findData(current)
        self.selector.setCurrentIndex(index if index >= 0 else 0)
        self.selector.blockSignals(False)
        self._show_study()

    def _show_study(self, *_):
        index = self.selector.currentIndex()
        if not 0 <= index < len(self.studies):
            self._future_trials = 0
            self.parallel_plot.set_study(None)
            self.importance_view.set_study(None)
            self.summary.setText("No tuning studies available for this run.")
            self.table.setRowCount(0)
            self.chart_type.clear()
            self.open_button.setEnabled(False)
            self.parameters.hide()
            self.detail.setText("Select a trial to inspect its parameters.")
            self.plot.set_data([], "Trial", "Objective")
            self._clear_trial_filters()
            return
        method, study = self.studies[index]
        study_key = (method, study["study"])
        if study_key != self.parallel_plot.study_key:
            self.filter.blockSignals(True)
            self.filter.clear()
            self.filter.blockSignals(False)
        self.parallel_plot.set_study(study, study_key)
        self.importance_view.set_study(study)
        trials = sorted(study.get("trials", []), key=lambda t: t.get("id", 0))
        selected_chart = self.chart_type.currentText()
        self.chart_type.blockSignals(True)
        self.chart_type.clear()
        self.chart_type.addItems(["Optimization history", "Intermediate values"])
        parameters = sorted({key for trial in trials for key, value in trial.get("parameters", {}).items()
                             if isinstance(value, (int, float)) and not isinstance(value, bool)})
        self.chart_type.addItems([f"Parameter: {name}" for name in parameters])
        chart_index = self.chart_type.findText(selected_chart)
        self.chart_type.setCurrentIndex(max(0, chart_index))
        self.chart_type.blockSignals(False)
        counts = {state: sum(t.get("state") == state for t in trials) for state in ("complete", "running", "pending", "pruned", "fail")}
        requested = study.get('requested_trials')
        future = max(0, requested - max(0, len(trials) - (study.get('initial_trials') or 0))) if isinstance(requested, int) and study.get('stop_reason') == 'running' else 0
        self._future_trials = study.get('future_trials', future)
        pending = study.get('pending_trials', self._future_trials + counts['pending'])
        best = next((t for t in trials if t.get("id") == study.get("best_trial_id")), None)
        best_text = f"{best.get('value'):.5g} · Trial {best['id']}" if best and isinstance(best.get('value'), (int, float)) else "—"
        self.summary.setText(f"{study.get('objective_metric') or 'Objective'} · {study.get('direction', 'unknown').capitalize()}\n"
                             f"Best {best_text}    |    {counts['complete']} complete    ·    {counts['running']} running    ·    {pending} pending    ·    "
                             f"{counts['pruned']} pruned    ·    {counts['fail']} failed")
        self.summary.setToolTip("Estimated parameter importance: " + ", ".join(
            f"{name}: {value:.2f}" for name, value in study.get("parameter_importance", {}).items()))
        selected_id = self.table.currentItem().data(Qt.ItemDataRole.UserRole) if self.table.currentItem() else None
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(trials) + int(self._future_trials > 0))
        for row, trial in enumerate(trials):
            values = [trial.get("id"), trial.get("state"), trial.get("value"), trial.get("duration_seconds"),
                      ", ".join(f"{k}={v}" for k, v in trial.get("parameters", {}).items())]
            for col, value in enumerate(values):
                item = QTableWidgetItem("—" if value is None else str(value))
                item.setData(Qt.ItemDataRole.UserRole, trial.get("id"))
                self.table.setItem(row, col, item)
        if self._future_trials:
            for col, value in enumerate([f'{self._future_trials} future trials', 'pending', '—', '—',
                                         'Parameters will be sampled as capacity opens.']):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, None)
                self.table.setItem(len(trials), col, item)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)
        self.open_button.setEnabled(False)
        self.parameters.hide()
        self.detail.setText("Select a trial to inspect its parameters.")
        for row in range(self.table.rowCount()):
            if selected_id is not None and self.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == selected_id:
                self.table.selectRow(row)
                self._show_trial_detail()
                break
        self._filter_trials()

    def _filter_trials(self, *_):
        query = self.filter.text().casefold().strip()
        current = self.table.currentItem()
        current_id = current.data(Qt.ItemDataRole.UserRole) if current is not None else None
        matching = self.parallel_plot.matching_ids()
        self._filtered_trial_ids = set()
        for row in range(self.table.rowCount()):
            text = " ".join(self.table.item(row, col).text() for col in range(self.table.columnCount())
                            if self.table.item(row, col) is not None)
            trial_id = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            visible = (trial_id in matching if trial_id is not None else not self.parallel_plot.brushes) and (not query or query in text.casefold())
            self.table.setRowHidden(row, not visible)
            if visible and trial_id is not None:
                self._filtered_trial_ids.add(trial_id)
        self.parallel_plot.set_selected_ids(self._filtered_trial_ids)
        real_count = self.table.rowCount() - int(self._future_trials > 0)
        self.selection_count.setText(f"{len(self._filtered_trial_ids)} / {real_count} trials" +
                                    (f' · {self._future_trials} future pending' if self._future_trials else ''))
        self.clear_filters.setEnabled(bool(query or self.parallel_plot.brushes))
        if current_id not in self._filtered_trial_ids or not self.table.selectedItems():
            self.table.clearSelection()
            self.table.setCurrentItem(None)
            self.open_button.setEnabled(False)
            self.parameters.hide()
            self.detail.setText("Select a trial to inspect its parameters.")
        self._draw_plot()

    def _clear_trial_filters(self):
        self.filter.blockSignals(True)
        self.filter.clear()
        self.filter.blockSignals(False)
        self.parallel_plot.clear_brushes()

    def _draw_plot(self, *_):
        index = self.selector.currentIndex()
        if not 0 <= index < len(self.studies):
            return
        study = self.studies[index][1]
        trials = sorted((trial for trial in study.get("trials", []) if trial["id"] in self._filtered_trial_ids),
                        key=lambda trial: trial.get("id", 0))
        chart = self.chart_type.currentText()
        series = []
        xlabel = "Trial"
        if chart == "Intermediate values":
            xlabel = "Reported step"
            method = self.studies[index][0]
            for trial in trials:
                points = []
                label = f"{method} · {study['study']} · trial {trial['id']}"
                live = self.lives.get(label)
                metric = study.get("objective_metric")
                live_points = [(row.get("step"), row.get(metric)) for row in live.rows] if live and metric else []
                source = live_points if live_points else trial.get("intermediate_values", {}).items()
                for step, value in source:
                    try:
                        x, y = float(step), float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(x) and math.isfinite(y):
                        points.append((x, y))
                if points:
                    series.append((f"Trial {trial['id']} · {trial.get('state', '')}", sorted(points),
                                   "#fe8019" if trial.get("state") == "pruned" else "#83a598"))
        elif chart.startswith("Parameter: "):
            xlabel = chart.removeprefix("Parameter: ")
            for trial in trials:
                x, y = trial.get("parameters", {}).get(xlabel), trial.get("value")
                if trial.get("state") == "complete" and all(isinstance(v, (int, float)) and math.isfinite(v) for v in (x, y)):
                    series.append((f"Trial {trial['id']}", [(x, y)], "#83a598"))
        else:
            best = None
            points, best_points = [], []
            for trial in trials:
                value = trial.get("value")
                if trial.get("state") == "complete" and isinstance(value, (int, float)) and math.isfinite(value):
                    points.append((trial["id"], value))
                    best = value if best is None else (min if study.get("direction") == "minimize" else max)(best, value)
                    best_points.append((trial["id"], best))
            best_label = "Best in selection" if self.clear_filters.isEnabled() else "Best so far"
            series = [("Objective", points, "#83a598"), (best_label, best_points, "#b8bb26")]
        self.plot.set_data(series, xlabel, study.get("objective_metric") or "Objective")

    def set_lives(self, lives):
        self.lives = lives
        if self.chart_type.currentText() == "Intermediate values":
            self._draw_plot()

    def _show_trial_detail(self):
        index = self.selector.currentIndex()
        selected = self.table.selectedItems()
        if not selected or not 0 <= index < len(self.studies):
            return
        trial_id = selected[0].data(Qt.ItemDataRole.UserRole)
        if trial_id is None:
            self.detail.setText('Future trials are pending. Parameters are sampled when capacity opens.')
            self.parameters.hide()
            self.open_button.setEnabled(False)
            return
        trial = next((t for t in self.studies[index][1].get("trials", []) if t.get("id") == trial_id), None)
        if trial:
            self.detail.setText(f"Trial {trial_id} · {str(trial.get('state', '')).capitalize()}" +
                                (" · Checkpoint available" if trial.get("checkpoint") else ""))
            self.detail.setToolTip(str(trial.get("checkpoint") or ""))
            params = trial.get("parameters", {})
            self.parameters.setRowCount(len(params))
            for row, (name, value) in enumerate(params.items()):
                self.parameters.setItem(row, 0, QTableWidgetItem(str(name)))
                self.parameters.setItem(row, 1, QTableWidgetItem(str(value)))
            self.parameters.setVisible(bool(params))
            self.open_button.setEnabled(True)

    def _open_trial(self, row, _column):
        index = self.selector.currentIndex()
        item = self.table.item(row, 0)
        if item is not None and item.data(Qt.ItemDataRole.UserRole) is not None and 0 <= index < len(self.studies):
            method, study = self.studies[index]
            self.trial_selected.emit(f"{method} · {study['study']} · trial {item.data(Qt.ItemDataRole.UserRole)}")
