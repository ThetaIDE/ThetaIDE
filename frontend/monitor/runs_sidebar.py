"""Runs list (in the Plots tab): every run at a glance; click or tick runs to plot them together in
the Plot Viewer. The Training monitor is not driven from here: it shows the current or most recent run.

Live and queued runs come first, then the rest in the app's order (newest first). Each row has a status
mark, the run's name and seed, a small reward curve with the latest reward, and a checkbox. Ticked runs
are drawn in the Plot Viewer in their row's color.
"""

from __future__ import annotations

from PyQt6.QtCore import QPointF, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QHeaderView,
    QLineEdit,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..model import LIVE_STATUSES
from ..theme import theme_color
from ..widgets import label

PALETTE = ("#b8bb26", "#83a598", "#fabd2f", "#d3869b", "#8ec07c", "#fe8019", "#fb4934", "#689d6a")
RUN_ID = Qt.ItemDataRole.UserRole
TREND = Qt.ItemDataRole.UserRole + 1  # list of reward values, oldest first
COLOR = Qt.ItemDataRole.UserRole + 2
DETAIL = Qt.ItemDataRole.UserRole + 3  # the row's second line: how far along the run is
TREND_POINTS = 60  # at most this many points per sparkline
ROW_HEIGHT = 40  # two lines: the run, then its detail
STATUS_MARKS = {
    "completed": "✓",
    "failed": "✗",
    "interrupted": "✗",
    "stopped": "✗",
    "cancelled": "✗",
    "error": "✗",
    "queued": "◷",
}


def status_mark(status: str) -> str:
    return "●" if status in LIVE_STATUSES else STATUS_MARKS.get(status, "○")


def run_label(run) -> str:
    name = run["config"]["name"] if run.get("simulated") else (run.get("backend") or {}).get("experiment_id")
    return str(name or run["config"].get("name") or run["id"])


def reward_trend(run) -> list[float]:
    """The run's evaluation rewards, thinned to at most TREND_POINTS for drawing."""
    if run.get("_metrics_deferred"):
        return list(run["_metric_summary"].get("trend", []))
    values = [m["reward"] for m in run.get("metrics") or [] if isinstance(m.get("reward"), (int, float))]
    if len(values) > TREND_POINTS:
        stride = len(values) / TREND_POINTS
        values = [values[int(i * stride)] for i in range(TREND_POINTS - 1)] + [values[-1]]
    return values


def run_detail(run) -> str:
    """'training · 62%', 'done', 'queued'…: the second line of a run's row."""
    status = run.get("status", "")
    if status in LIVE_STATUSES:
        budget = (run.get("backend") or {}).get("effective_timesteps")
        step = (run.get("_metric_summary") or {}).get("step") if run.get("_metrics_deferred") else max((m.get("step") or 0 for m in run.get("metrics") or []), default=0)
        return f"training · {min(100, round(100 * step / budget))}%" if budget and step else "starting"
    return {"completed": "done"}.get(status, status)


class RunRowDelegate(QStyledItemDelegate):
    """The first column on two lines: the run (with its checkbox), then its detail in muted text."""

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop

    def paint(self, painter: QPainter, option, index):
        option.rect = option.rect.adjusted(0, 3, 0, 0)  # the first line a little off the top edge
        super().paint(painter, option, index)
        detail = index.data(DETAIL)
        if not detail:
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        widget = option.widget
        style = widget.style() if widget is not None else None
        text = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, widget) if style else opt.rect
        painter.save()
        painter.setPen(QColor(theme_color("muted")))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(QRectF(text.left() + 4, option.rect.top() + 19, text.width() - 4, 16),
                         Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, detail)
        painter.restore()

    def sizeHint(self, option, index):
        hint = super().sizeHint(option, index)
        return QSize(hint.width(), ROW_HEIGHT if index.data(RUN_ID) else hint.height())


class TrendDelegate(QStyledItemDelegate):
    """Draws a run's reward sparkline and latest value in the second column."""

    def paint(self, painter: QPainter, option, index):
        super().paint(painter, option, index)
        values = index.data(TREND) or []
        if not values:
            return
        middle = option.rect.center().y()  # a sparkline of one line's height, centred on the two-line row
        rect = QRectF(option.rect.left() + 4, middle - 8, option.rect.width() - 8, 16)
        text_width = 44
        area = QRectF(rect.left(), rect.top(), max(10.0, rect.width() - text_width - 6), rect.height())
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor(index.data(COLOR) or theme_color("primary"))
        if len(values) > 1:
            lo, hi = min(values), max(values)
            path = QPainterPath()
            for i, value in enumerate(values):
                x = area.left() + area.width() * i / (len(values) - 1)
                y = area.bottom() - area.height() * ((value - lo) / (hi - lo) if hi > lo else 0.5)
                if i == 0:
                    path.moveTo(QPointF(x, y))
                else:
                    path.lineTo(QPointF(x, y))
            painter.setPen(QPen(color, 1.4))
            painter.drawPath(path)
        painter.setPen(QColor(theme_color("muted")))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(
            QRectF(rect.right() - text_width, rect.top(), text_width, rect.height()),
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            f"{values[-1]:.1f}",
        )
        painter.restore()

    def sizeHint(self, option, index):
        return QSize(120, ROW_HEIGHT if index.data(RUN_ID) or index.data(TREND) else 26)


class RunsSidebar(QWidget):
    """Lists runs; `compare_changed` when the ticked set (what the Plot Viewer draws) changes.

    Clicking a row ticks or unticks it, like its checkbox. Which run the Training monitor shows is not
    chosen here: it follows the run training now, or the most recent one.
    """

    compare_changed = pyqtSignal(list)  # ids of the ticked runs, in the order they were ticked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("monitorRunsSidebar")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)
        layout.addWidget(label("RUNS", "eyebrow"))
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter runs…")
        self.filter.setClearButtonEnabled(True)
        self.filter.textChanged.connect(self._apply_filter)
        layout.addWidget(self.filter)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderHidden(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setIndentation(0)
        self.tree.setItemDelegateForColumn(0, RunRowDelegate(self.tree))
        self.tree.setItemDelegateForColumn(1, TrendDelegate(self.tree))
        self.tree.header().setStretchLastSection(False)
        self.tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 120)
        self.tree.itemChanged.connect(self._item_changed)
        self.tree.itemPressed.connect(self._pressed)
        self.tree.itemClicked.connect(self._clicked)
        layout.addWidget(self.tree, 1)
        self.clear_button = QPushButton("Clear plot")
        self.clear_button.setToolTip("Untick every run, taking them off the plot")
        self.clear_button.clicked.connect(self.clear_comparison)
        self.clear_button.hide()
        layout.addWidget(self.clear_button)
        self.items: dict[str, QTreeWidgetItem] = {}
        self.compared: list[str] = []  # ticked run ids, in tick order
        self._filling = False
        self._press = None  # (item, its check state) when the mouse went down on a row

    # ── Filling ──────────────────────────────────────────────────────────────

    def set_runs(self, runs):
        """Rebuild the list: live and queued runs first, then the rest. Ticks are kept."""
        self._filling = True
        self.tree.clear()
        self.items = {}
        ids = {run["id"] for run in runs}
        self.compared = [run_id for run_id in self.compared if run_id in ids]
        live = [run for run in runs if run.get("status") in (*LIVE_STATUSES, "queued")]
        rest = [run for run in runs if run not in live]
        for title, group in (("LIVE AND QUEUED", live), ("RECENT", rest)):
            if not group:
                continue
            heading = QTreeWidgetItem(self.tree, [title])
            heading.setFlags(Qt.ItemFlag.NoItemFlags)
            heading.setForeground(0, QColor(theme_color("secondary")))
            for run in group:
                item = QTreeWidgetItem(self.tree)
                item.setData(0, RUN_ID, run["id"])
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(0, Qt.CheckState.Checked if run["id"] in self.compared else Qt.CheckState.Unchecked)
                self.items[run["id"]] = item
                self._fill(item, run)
        self._paint_colors()
        self._apply_filter(self.filter.text())
        self._filling = False
        self.clear_button.setVisible(bool(self.compared))

    def _fill(self, item, run):
        seed = run["config"].get("seed")
        seed_text = f"  (seed {seed})" if seed is not None else ""
        item.setText(0, f"{status_mark(run.get('status', ''))}  {run_label(run)}{seed_text}")
        item.setToolTip(
            0, f"{run_label(run)} · {run.get('status', '')}\nClick or tick it to plot it in the Plot Viewer."
        )
        item.setData(0, DETAIL, run_detail(run))
        item.setData(1, TREND, reward_trend(run))

    def update_run(self, run):
        """Refresh one run's row (new metrics or status) without rebuilding the list."""
        item = self.items.get(run["id"])
        if item is not None:
            self._filling = True
            self._fill(item, run)
            self._filling = False

    def colors(self) -> dict[str, str]:
        """run id -> color of the ticked runs, in tick order: their curves in the Plots tab."""
        return {run_id: PALETTE[i % len(PALETTE)] for i, run_id in enumerate(self.compared)}

    def _paint_colors(self):
        colors = self.colors()
        filling, self._filling = self._filling, True  # recoloring fires itemChanged: not a tick
        for run_id, item in self.items.items():
            color = colors.get(run_id)  # resolved through the theme, as the Plot viewer draws its curve
            item.setData(1, COLOR, theme_color(color) if color else theme_color("comment"))
            item.setForeground(0, QColor(theme_color(color) if color else theme_color("text")))
        self._filling = filling

    def _apply_filter(self, text):
        needle = text.strip().lower()
        for item in self.items.values():
            item.setHidden(bool(needle) and needle not in item.text(0).lower())

    # ── Selection and comparison ─────────────────────────────────────────────

    def _pressed(self, item, _column):
        self._press = (item, item.checkState(0))

    def _clicked(self, item, _column):
        """A click on the row (not its checkbox, which Qt already toggled) ticks or unticks it."""
        press, self._press = self._press, None
        if press is None or press[0] is not item or not item.data(0, RUN_ID):
            return
        if item.checkState(0) == press[1]:
            checked = item.checkState(0) == Qt.CheckState.Checked
            item.setCheckState(0, Qt.CheckState.Unchecked if checked else Qt.CheckState.Checked)

    def _item_changed(self, item, column):
        if self._filling or column != 0:
            return
        run_id = item.data(0, RUN_ID)
        if not run_id:
            return
        ticked = item.checkState(0) == Qt.CheckState.Checked
        if ticked and run_id not in self.compared:
            self.compared.append(run_id)
        elif not ticked and run_id in self.compared:
            self.compared.remove(run_id)
        self._paint_colors()
        self.clear_button.setVisible(bool(self.compared))
        self.compare_changed.emit(list(self.compared))

    def tick(self, run_id):
        """Tick `run_id` (it is plotted), as a click on its checkbox would."""
        item = self.items.get(run_id)
        if item is not None and item.checkState(0) != Qt.CheckState.Checked:
            item.setCheckState(0, Qt.CheckState.Checked)

    def clear_comparison(self):
        self._filling = True
        for run_id in self.compared:
            item = self.items.get(run_id)
            if item is not None:
                item.setCheckState(0, Qt.CheckState.Unchecked)
        self._filling = False
        self.compared = []
        self._paint_colors()
        self.clear_button.hide()
        self.compare_changed.emit([])
