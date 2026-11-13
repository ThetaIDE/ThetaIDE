"""The Training monitor's live views: status strip, every chart, system use, logs, event feed and files.

Each view is given LiveRuns (frontend/live_run.py), one per agent of the selected run, and draws them;
none of them reads files or knows where the records came from.
"""

from __future__ import annotations

import bisect
from copy import deepcopy
import logging
import re
import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor, QDesktopServices, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..live_run import LiveRun, format_duration
from ..results.data import IncrementalXYThinner, chart_points, group_metrics, thin
from ..results.inspector import format_value
from ..results.metrics_catalog import describe
from ..results.plot_grid import PlotGrid, chart_title
from ..theme import theme_color
from ..widgets import label

PALETTE = ("#b8bb26", "#83a598", "#fabd2f", "#d3869b", "#8ec07c", "#fe8019")
COUNTERS = ("transitions", "step", "epoch")  # positions of a row, never charted as metrics
SYSTEM_LABELS = {
    "cpu_percent": "CPU  ·  % of one core, training process and its children",
    "rss_mb": "Memory  ·  MB, training process and its children",
    "host_cpu_percent": "Machine CPU  ·  %",
    "host_mem_percent": "Machine memory  ·  %",
    "host_load_1m": "Machine load  ·  1-minute average",
    "threads": "Threads",
    "children": "Child processes",
}
LEVEL_MARKS = {"error": "✖", "warning": "⚠", "info": "●"}
MAX_LOG_LINES = 20000
SPEED_EASING = 0.7  # share of the previous speed kept at each reading: the status strip's speed is eased


def latest_problem(lives: dict[str, LiveRun]) -> tuple[str, dict] | None:
    """(agent, event) of the newest warning or error any agent reported, or None."""
    newest = None
    newest_time = float("-inf")
    for agent, live in lives.items():
        for event in live.events:
            if event.get("level") not in ("warning", "error"):
                continue
            when = event.get("t") or 0
            # Match the last entry of a stable time sort, including tied timestamps.
            if when >= newest_time:
                newest, newest_time = (agent, event), when
    return newest


def _system_title(key: str) -> str:
    gpu = re.fullmatch(r"gpu(\d+)_mem_(allocated|reserved)_mb", key)
    if gpu:
        return f"GPU {gpu[1]} memory {gpu[2]}  ·  MB"
    return SYSTEM_LABELS.get(key, key)


def merged(lives: dict[str, LiveRun], attribute: str) -> list[tuple[str, dict]]:
    """(agent, item) for every item of `attribute` (events, artifacts) across agents, oldest first."""
    items = [(agent, item) for agent, live in lives.items() for item in getattr(live, attribute)]
    return sorted(items, key=lambda pair: pair[1].get("t") or 0)


def busiest(lives: dict[str, LiveRun]) -> LiveRun | None:
    """The agent that reported most recently: pipelines train their agents one after another."""
    reporting = [live for live in lives.values() if live.last_update is not None]
    return max(reporting, key=lambda live: live.last_update) if reporting else None


# ── Status strip ─────────────────────────────────────────────────────────────


class StatusStrip(QWidget):
    """Status · (the run's name) · elapsed · speed · time left, for the selected run.

    Speed and time left show while the run trains; "Updated" when it has gone quiet, or for a finished run.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(18)
        self._row = row
        self.badge = label("", "badge")
        row.addWidget(self.badge)
        row.addStretch()
        self.fields: dict[str, object] = {}
        self.captions: dict[str, object] = {}
        for key, caption in (("elapsed", "Elapsed"), ("speed", "Speed"), ("eta", "Time left"), ("updated", "Updated")):
            self.captions[key] = label(caption, "muted")
            row.addWidget(self.captions[key])
            self.fields[key] = label("—")
            row.addWidget(self.fields[key])
        self._speed: float | None = None  # smoothed speed of the run in `_speed_of`
        self._speed_of: LiveRun | None = None

    def _smoothed_rate(self, latest: LiveRun | None, live: bool) -> float | None:
        """The run's speed, eased toward each new reading so the number does not jitter every second."""
        rate = latest.rate() if latest is not None and live else None
        if rate and self._speed and latest is self._speed_of:
            rate = SPEED_EASING * self._speed + (1 - SPEED_EASING) * rate
        self._speed, self._speed_of = rate, latest
        return rate

    def add_title(self, *widgets: QWidget):
        """Put `widgets` (the run's name and description) after the badge, before the numbers."""
        for i, widget in enumerate(widgets):
            self._row.insertWidget(1 + i, widget)

    def add_trailing(self, widget: QWidget):
        """Put `widget` (e.g. buttons) at the end of the row."""
        self._row.addWidget(widget)

    def _show_fields(self, *keys: str):
        for key in self.fields:
            self.captions[key].setVisible(key in keys)
            self.fields[key].setVisible(key in keys)

    def show_run(self, status: str, lives: dict[str, LiveRun], started: float | None, live: bool, now=None):
        now = time.time() if now is None else now
        latest = busiest(lives)
        stalled = live and latest is not None and latest.stalled(now)
        if live and latest is not None and latest.stage in ("testing", "starting", "diagnostics"):
            status = latest.stage
        self.badge.setText(("STALLED" if stalled else status or "—").upper())
        tone = "#fb4934" if stalled or status in ("failed", "error") else theme_color("accent")
        self.badge.setStyleSheet(f"color: {tone};")
        if started is None and latest is not None and latest.start:
            started = (latest.start.get("run") or {}).get("started")
        end = latest.finish.get("t") if latest is not None and latest.finish else None
        self.fields["elapsed"].setText(format_duration((end or now) - started) if started and (live or end) else "—")
        rate = self._smoothed_rate(latest, live and status not in ("testing", "starting", "diagnostics"))
        self.fields["speed"].setText(f"{rate:,.0f} steps/s" if rate else "—")
        eta = None
        if latest is not None and live and rate and latest.budget and latest.progress() is not None:
            eta = max(0.0, (latest.budget - latest.progress()) / rate)
            if eta >= 90:  # over a minute and a half, tens of seconds are noise
                eta = round(eta / 10) * 10
        self.fields["eta"].setText(format_duration(eta))
        if latest is None or latest.last_update is None:
            self.fields["updated"].setText("—")
        elif live:
            ago = now - latest.last_update
            self.fields["updated"].setText(f"{format_duration(ago)} ago" + ("  ⚠ no new data" if stalled else ""))
        else:
            self.fields["updated"].setText(datetime.fromtimestamp(latest.last_update).strftime("%Y-%m-%d %H:%M"))
        if live:
            self._show_fields(*(("updated",) if stalled else ()))
        else:
            self._show_fields("updated")


# ── Charts: every metric each agent logged ───────────────────────────────────


Points = dict[str, dict[str, list[tuple[float, float]]]]  # series label -> metric -> [(x, value)], by x


def nearest(points: list[tuple[float, float]], x: float) -> tuple[float, float] | None:
    """The (x, value) in `points` (sorted by x) closest to `x`; the earlier one on a tie."""
    if not points:
        return None
    i = bisect.bisect_left(points, (x, float("-inf")))
    candidates = points[max(0, i - 1) : i + 1]
    return min(candidates, key=lambda p: abs(p[0] - x))


class PointInspector(QWidget):
    """Every metric of every series at the hovered step: one row per metric, one column per series.

    Runs log metrics on different rows (evaluations, losses, timings), so each cell is the series' value
    of that metric nearest the hovered step; one taken from another step is marked ≈ and says which.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)
        self.caption = label("Hover over any chart to see every metric of every series at that step.", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.table = QTableWidget(0, 0)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        layout.addWidget(self.table, 1)
        self.x: float | None = None
        self._fold()  # until a point is hovered: an empty table would only take the charts' room

    def _fold(self):
        self.table.hide()
        self.setMaximumHeight(self.caption.sizeHint().height() + 12)

    def _unfold(self):
        self.setMaximumHeight(16777215)  # QWIDGETSIZE_MAX
        self.table.show()

    def show_point(self, x: float, points: Points, colors: dict[str, str], axis_name: str):
        """Fill the table for step `x` from each series' points."""
        self.x = x
        if self.table.isHidden():
            self._unfold()
        labels = list(points)
        metrics = [
            m
            for names in group_metrics(sorted({m for series in points.values() for m in series})).values()
            for m in names
        ]
        self.table.clear()
        self.table.setRowCount(len(metrics))
        self.table.setColumnCount(len(labels))
        for col, name in enumerate(labels):
            header = QTableWidgetItem(name)
            if colors.get(name):
                header.setForeground(QColor(colors[name]))
            self.table.setHorizontalHeaderItem(col, header)
        self.table.setVerticalHeaderLabels(metrics)
        approximate = 0
        for row, metric in enumerate(metrics):
            self.table.verticalHeaderItem(row).setToolTip(describe(metric).label)
            for col, name in enumerate(labels):
                found = nearest(points[name].get(metric, []), x)
                if found is None:
                    item = QTableWidgetItem("—")
                else:
                    at, value = found
                    exact = at == x
                    approximate += not exact
                    item = QTableWidgetItem(("" if exact else "≈ ") + format_value(metric, value))
                    item.setToolTip(f"{metric} = {value!r} at {axis_name} {at:,.0f}")
                    if not exact:
                        item.setForeground(QColor(theme_color("muted")))
                self.table.setItem(row, col, item)
        note = (
            f"  ·  ≈ marks {approximate} value(s) from the nearest other step (hover a cell for which)"
            if approximate
            else ""
        )
        self.caption.setText(f"{axis_name} = {x:,.0f}  ·  {len(metrics)} metrics across {len(labels)} series{note}")


def budget_extent(lives: dict[str, LiveRun], axis: str) -> float | None:
    """The x-extent to fix the charts to: the largest training budget, while every run is within it.

    Fixing it keeps a live curve from being squeezed sideways each time it grows. Runs read from disk
    have no budget, and epochs-based ones are not measured in transitions: those charts fit the data.
    """
    if axis != "transitions":
        return None
    budgets = [live.budget for live in lives.values() if live.budget]
    if not budgets:
        return None
    reached = max((live.progress() or 0.0 for live in lives.values()), default=0.0)
    return float(max(budgets)) if reached <= max(budgets) else None


class ChartsView(QWidget):
    """One chart per logged metric, grouped by prefix in folding sections, one series per agent (or run).

    `header` (e.g. the Overview's cards) scrolls with the charts, above them. `lead` is (group, widget): a
    chart of the caller's drawn first in that group's section while a single run is shown; metrics in
    `exclude` are then left out, because it draws them. Comparing runs, every metric is charted here.
    `toolbar` sits on the first section's heading row and `group_columns` sets a section's columns.
    """

    def __init__(self, parent=None, header: QWidget | None = None, lead=None, toolbar: QWidget | None = None,
                 group_columns: dict[str, int] | None = None, show_caption: bool = True):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        top = QHBoxLayout()
        self.caption = label("", "muted")
        self.caption.setVisible(show_caption)
        top.addWidget(self.caption, 1)
        self.show_all = QPushButton("Show all plots")
        self.show_all.clicked.connect(self._show_all)
        self.show_all.hide()
        top.addWidget(self.show_all)
        layout.addLayout(top)
        self.grid = PlotGrid()
        self.grid.configure(collapsible=True, header=header, lead=lead, toolbar=toolbar, group_columns=group_columns,
                            compact_hide=True)
        self.grid.hide_requested.connect(self._hide)
        layout.addWidget(self.grid, 1)
        self.hidden: set[str] = set()
        self.exclude: set[str] = set()
        self.points: Points = {}
        self.axis_name = "step"
        self._lives: dict[str, LiveRun] = {}
        self._colors: dict[str, str] | None = None
        self._live = False
        self._comparing = False
        self._series_colors: dict[str, str] = {}
        self.grid.show_metrics([], "Metrics appear here as the run logs them.")  # the header, before any data

    def _hide(self, metric):
        self.hidden.add(metric)
        self.show_lives(self._lives, self._colors, self._live, comparing=self._comparing)

    def _show_all(self):
        self.hidden.clear()
        self.show_lives(self._lives, self._colors, self._live, comparing=self._comparing)

    def show_lives(self, lives: dict[str, LiveRun], colors: dict[str, str] | None = None, live: bool = False,
                   *, comparing: bool | None = None):
        """One series per entry of `lives` (label -> LiveRun), in `colors[label]` when given.

        `colors` is given when several runs are compared; `live` draws for a run still training: the
        x-axis fixed to its budget, a y-axis that only grows, and each curve's newest point marked.
        """
        self._lives, self._colors, self._live = lives, colors, live
        configs = [each.config for each in lives.values()]
        if configs != getattr(self, "_description_configs", None):
            from src.app.telemetry.monitors import available_monitors

            self._description_configs = deepcopy(configs)
            self._description_specs = []
            for config in configs:
                if not config:
                    continue
                try:
                    self._description_specs.extend(available_monitors(config, errors=[]))
                except (ValueError, KeyError, OSError) as exc:
                    logging.getLogger(__name__).warning("Cannot load metric descriptions: %s", exc)
        self._comparing = colors is not None if comparing is None else comparing
        rows = {name: live.rows for name, live in lives.items()}
        # Retain each source list: ordinary appends inspect only the new rows.
        sources = [(name, rs, len(rs)) for name, rs in rows.items()]
        previous = {name: (rs, size) for name, rs, size in getattr(self, "_point_sources", [])}
        rebuild = not hasattr(self, "_point_axis") or any(
            name not in previous or rs is not previous[name][0] or size < previous[name][1]
            for name, rs, size in sources
        )
        additions = {name: rs[0 if rebuild else previous[name][1]:] for name, rs, _ in sources}
        axis = getattr(self, "_point_axis", "step")
        if axis != "transitions" and any("transitions" in r for rs in additions.values() for r in rs):
            axis, rebuild = "transitions", True
        # Removing a source can also change the common x-axis.
        if set(previous) - set(rows):
            axis = "transitions" if any("transitions" in r for rs in rows.values() for r in rs) else "step"
            rebuild = True
        if rebuild:
            axis = "transitions" if any("transitions" in r for rs in rows.values() for r in rs) else "step"
            additions = rows
            self.points = {name: {} for name in rows}
            self._point_metrics = {}
            self._drawn_points = {}
            self._point_thinners = {}
        metrics = self._point_metrics
        for name, new_rows in additions.items():
            per_metric = self.points[name]
            changed_metrics = set()
            unordered = set()
            for r in new_rows:
                for key, value in r.items():
                    if key not in COUNTERS:
                        metrics.setdefault(key)
                        if axis in r:
                            points = per_metric.setdefault(key, [])
                            if points and r[axis] < points[-1][0]:
                                unordered.add(key)
                            points.append((r[axis], value))
                            changed_metrics.add(key)
            for metric in changed_metrics:
                if metric in unordered:
                    # Stable sorting preserves duplicate-step order, including late reports.
                    per_metric[metric].sort(key=lambda p: p[0])
                    self._point_thinners.pop((name, metric), None)
                self._drawn_points.pop((name, metric), None)
        self._point_sources = sources
        self._point_axis = axis
        axis, metrics = self._point_axis, self._point_metrics
        self.axis_name = "environment steps" if axis == "transitions" else "logged step"
        self._series_colors = {
            name: (colors or {}).get(name) or PALETTE[i % len(PALETTE)] for i, name in enumerate(rows)
        }
        shown = [m for m in metrics if m not in self.hidden]
        hidden = len(metrics) - len(shown)
        # Comparing runs, the lead chart (the selected run's alone) steps aside and every metric is charted.
        comparing = self._comparing
        self.grid.lead_hidden = comparing
        visible = shown if comparing or self.grid.lead is None else [m for m in shown if m not in self.exclude]
        self.show_all.setVisible(bool(hidden))
        self.show_all.setText(f"Show all plots ({hidden} hidden)")
        if not metrics:
            empty = "Metrics appear here as the run logs them."
        elif not shown:
            empty = "Every plot is hidden."
        else:
            empty = ""  # the rest are drawn by the lead chart
        descriptions = {}
        for metric in visible:
            matches = [spec for spec in self._description_specs if spec.description and
                       (spec.name == metric or (spec.name.endswith("/") and metric.startswith(spec.name)))]
            if matches:
                descriptions[metric] = max(matches, key=lambda spec: len(spec.name)).description
        self.grid.descriptions = descriptions
        self.grid.show_metrics(visible, empty)
        for metric in visible:
            self.grid.card(metric)  # refresh help even when only the run configuration changed
        names = ", ".join(lives) or "no agents yet"
        self.caption.setText(f"{len(metrics)} metrics  ·  x-axis: {self.axis_name}  ·  {names}")
        xmax = budget_extent(lives, axis)
        # Of a live run, the methods still training: a pipeline's finished methods stop growing.
        training = {name for name, each in lives.items() if live and not each.is_finished()}
        for metric, chart in self.grid.charts().items():
            series = []
            for name in rows:
                points = self.points[name].get(metric)
                if points:
                    key = (name, metric)
                    if key not in self._drawn_points:
                        thinner = self._point_thinners.setdefault(key, IncrementalXYThinner())
                        self._drawn_points[key] = chart_points(thinner.sample(points), metric)
                    series.append((name, self._drawn_points[key], self._series_colors[name]))
            chart.set_metric(metric, chart_title(metric))
            chart.hold_y = live
            chart.live_series = training
            chart.show_future = live and xmax is not None  # a stopped run will not reach the rest
            chart.set_series(series, xmax)


# ── System: resource use over time ───────────────────────────────────────────


class SystemView(QWidget):
    """CPU, memory and GPU memory of the training process over time, from its own samples."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        self.caption = label("", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self.grid = PlotGrid()
        self.grid.configure(compact_hide=True)
        layout.addWidget(self.grid, 1)

    def show_lives(self, lives: dict[str, LiveRun], colors: dict[str, str] | None = None):
        """Resource use per entry of `lives`, each against seconds since its own first sample."""
        samples = [(agent, t, data) for agent, live in lives.items() for t, data in live.system]
        keys: dict[str, None] = {}
        for _, _, data in samples:
            for key in data:
                keys.setdefault(key)
        ordered = [k for k in SYSTEM_LABELS if k in keys] + sorted(k for k in keys if k not in SYSTEM_LABELS)
        names = [f"system/{key}" for key in ordered]
        self.grid.show_metrics(
            names,
            "No resource samples. They come from the training process itself: this run predates the telemetry "
            "stream, or ran with telemetry.system_interval=0.",
        )
        starts = {agent: min(t for t, _ in run.system) for agent, run in lives.items() if run.system}
        longest = max((t - starts[a] for a, t, _ in samples), default=0.0)
        self.caption.setText(
            f"{len(samples)} samples over {format_duration(longest)}  ·  x-axis: seconds since each run's first sample"
            if samples
            else ""
        )
        for key, name in zip(ordered, names):
            chart = self.grid.cards[name].chart
            self.grid.cards[name].hide_button.hide()
            series = []
            for i, agent in enumerate(lives):
                points = [(t - starts[agent], data[key]) for t, data in lives[agent].system if key in data]
                if points:
                    color = (colors or {}).get(agent) or PALETTE[i % len(PALETTE)]
                    series.append((agent, thin(chart_points(points, name), name), color))
            chart.set_metric(name, _system_title(key))
            chart.set_series(series)


# ── Logs ─────────────────────────────────────────────────────────────────────


class _LogHighlighter(QSyntaxHighlighter):
    ERROR = re.compile(r"Traceback|Error|Exception|FAILED|✖", re.IGNORECASE)
    WARNING = re.compile(r"\bWARN(ING)?\b|⚠", re.IGNORECASE)

    def highlightBlock(self, text):
        color = None
        if self.ERROR.search(text):
            color = "#fb4934"
        elif self.WARNING.search(text):
            color = theme_color("accent")
        if color:
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(color))
            self.setFormat(0, len(text), fmt)


class LogView(QWidget):
    """The pipeline's output: search, errors-and-warnings only, and follow the end as lines arrive."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        bar = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find in log (Enter: next match)")
        self.search.returnPressed.connect(self.find_next)
        bar.addWidget(self.search, 1)
        self.problems_only = QCheckBox("Errors and warnings only")
        self.problems_only.toggled.connect(lambda _: self._redraw())
        bar.addWidget(self.problems_only)
        self.follow = QCheckBox("Follow")
        self.follow.setChecked(True)
        self.follow.setToolTip("Keep the newest output in view")
        bar.addWidget(self.follow)
        layout.addLayout(bar)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setMaximumBlockCount(MAX_LOG_LINES)
        self.text.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self._highlighter = _LogHighlighter(self.text.document())
        layout.addWidget(self.text, 1)
        self.empty = label("", "muted")
        layout.addWidget(self.empty)
        self.lines: list[str] = []

    @staticmethod
    def _is_problem(line: str) -> bool:
        return bool(_LogHighlighter.ERROR.search(line) or _LogHighlighter.WARNING.search(line))

    def set_lines(self, lines: list[str] | None, empty_text: str = ""):
        self.lines = list(lines or [])[-MAX_LOG_LINES:]
        self.empty.setText("" if self.lines else empty_text)
        self._redraw()

    def append(self, lines: list[str]):
        self.lines.extend(lines)
        if len(self.lines) > MAX_LOG_LINES:
            del self.lines[: len(self.lines) - MAX_LOG_LINES]
        self.empty.setText("")
        shown = [line for line in lines if self._is_problem(line)] if self.problems_only.isChecked() else lines
        if shown:
            self._write(shown)

    def _redraw(self):
        self.text.clear()
        lines = (
            [line for line in self.lines if self._is_problem(line)] if self.problems_only.isChecked() else self.lines
        )
        if lines:
            self._write(lines)

    def _write(self, lines):
        bar = self.text.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 2
        self.text.appendPlainText("\n".join(lines))
        if self.follow.isChecked() or at_end:
            bar.setValue(bar.maximum())

    def find_next(self):
        needle = self.search.text()
        if needle and not self.text.find(needle):
            self.text.moveCursor(self.text.textCursor().MoveOperation.Start)  # wrap around
            self.text.find(needle)


# ── Feed: what happened, newest first ────────────────────────────────────────


def _clock(t) -> str:
    return datetime.fromtimestamp(t).strftime("%H:%M:%S") if t else ""


class FeedView(QWidget):
    """Every event of the run, newest first: starts, checkpoints, new bests, warnings, errors, the finish."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Time", "Agent", "Event"])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 90)
        self.tree.setColumnWidth(1, 140)
        layout.addWidget(self.tree, 1)
        self.empty = label("", "muted")
        layout.addWidget(self.empty)

    @staticmethod
    def entries(lives: dict[str, LiveRun]) -> list[tuple[float, str, str, str]]:
        """(time, agent, level, text), newest first."""
        found = []
        for agent, live in lives.items():
            if live.start:
                info = live.start.get("run") or {}
                where = f" ({info['version']}, pid {info.get('pid')})" if info.get("version") else ""
                found.append((live.start.get("t") or 0, agent, "info", f"Run started{where}"))
            if live.finish:
                status = live.finish.get("status", "")
                level = "error" if status == "failed" else "info"
                found.append((live.finish.get("t") or 0, agent, level, f"Run finished: {status}"))
        for agent, event in merged(lives, "events"):
            found.append((event.get("t") or 0, agent, event.get("level") or "info", event.get("message") or ""))
        return sorted(found, key=lambda entry: entry[0], reverse=True)

    def show_lives(self, lives: dict[str, LiveRun]):
        entries = self.entries(lives)
        self.tree.clear()
        colors = {"error": "#fb4934", "warning": theme_color("accent")}
        for t, agent, level, text in entries:
            item = QTreeWidgetItem([_clock(t), agent, f"{LEVEL_MARKS.get(level, '●')}  {text}"])
            if level in colors:
                item.setForeground(2, QColor(colors[level]))
            self.tree.addTopLevelItem(item)
        self.empty.setText(
            ""
            if entries
            else "Events appear here as the run reports them: checkpoints, new bests, warnings and errors."
        )


# ── Files: what the run wrote ────────────────────────────────────────────────


class FilesView(QWidget):
    """Checkpoints and other files the run reported, openable; and its results folder."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        bar = QHBoxLayout()
        self.caption = label("", "muted")
        bar.addWidget(self.caption, 1)
        self.open_folder = QPushButton("Open run folder")
        self.open_folder.clicked.connect(self._open_folder)
        bar.addWidget(self.open_folder)
        layout.addLayout(bar)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File", "Kind", "Agent", "Written"])
        self.tree.setRootIsDecorated(False)
        self.tree.setColumnWidth(0, 420)
        self.tree.itemDoubleClicked.connect(self._open_item)
        layout.addWidget(self.tree, 1)
        self.folder: Path | None = None

    def show_lives(self, lives: dict[str, LiveRun], folder: Path | None):
        self.folder = folder
        self.open_folder.setEnabled(bool(folder and folder.exists()))
        latest: dict[str, tuple[str, dict]] = {}  # a checkpoint saved again replaces its earlier entry
        for agent, artifact in merged(lives, "artifacts"):
            latest[artifact.get("path", "")] = (agent, artifact)
        self.tree.clear()
        for path, (agent, artifact) in sorted(latest.items(), key=lambda kv: -(kv[1][1].get("t") or 0)):
            best = "  (best)" if (artifact.get("data") or {}).get("best") else ""
            item = QTreeWidgetItem([path, (artifact.get("kind") or "file") + best, agent, _clock(artifact.get("t"))])
            item.setData(0, Qt.ItemDataRole.UserRole, path)
            self.tree.addTopLevelItem(item)
        self.caption.setText(
            f"{len(latest)} file(s) reported  ·  double-click to open"
            if latest
            else "Files the run reports (checkpoints, figures) appear here."
        )

    def _open_item(self, item, _column):
        path = Path(item.data(0, Qt.ItemDataRole.UserRole) or "")
        target = path if path.exists() else path.parent
        if target.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))

    def _open_folder(self):
        if self.folder and self.folder.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.folder.resolve())))
