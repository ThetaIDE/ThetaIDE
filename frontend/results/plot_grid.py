"""Plot grid: one chart card per metric, grouped by prefix, each one hideable."""

from __future__ import annotations

import math

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QGridLayout, QHBoxLayout, QScrollArea, QToolButton, QVBoxLayout, QWidget

from ..charts import Chart
from ..widgets import label
from .data import group_metrics
from .metrics_catalog import describe, group_sort_key

COLUMNS = 2
HERO_HEIGHT = 340  # minimum height of a hero (Overview) or focused (Focus) chart


def chart_title(metric: str) -> str:
    """Catalog label plus the raw column name, so every chart says exactly which data it shows."""
    info = describe(metric)
    return metric if info.label == metric else f"{info.label}  ·  {metric}"


class ChartCard(QFrame):
    hide_requested = pyqtSignal(str)

    def __init__(self, metric: str, parent=None, compact_hide: bool = False):
        super().__init__(parent)
        self.metric = metric
        self.setObjectName("resultsChartCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.chart = Chart(metric, chart_title(metric), zero_based=False)
        self.chart.installEventFilter(self)
        self.chart.empty_text = "No data for the selected runs"
        self.chart.setMinimumHeight(220)
        layout.addWidget(self.chart)
        corner = QWidget()
        corner.setObjectName("resultsCardCorner")
        corner_layout = QHBoxLayout(corner)
        corner_layout.setContentsMargins(0, 0, 0, 0)
        corner_layout.setSpacing(6)
        self.value = label("", "resultsCardValue")
        self.value.setToolTip(
            "Latest value of the primary series (or the first series), as drawn: smoothed when Smoothing is on"
        )
        self.value.hide()
        corner_layout.addWidget(self.value)
        self.hide_button = QToolButton()
        self.hide_button.setText("−" if compact_hide else "Hide")
        self.hide_button.setObjectName("monitorPlotHide" if compact_hide else "")
        self.hide_button.setToolTip(f"Hide {metric} (bring it back from the Plots menu)")
        self.hide_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.hide_button.clicked.connect(lambda: self.hide_requested.emit(self.metric))
        corner_layout.addWidget(self.hide_button)
        self.chart.set_corner_widget(corner)  # top-right of the chart, like the Monitor's selector

    def eventFilter(self, watched, event):
        # Chart titles are painted rather than QLabel widgets. Limit their help
        # to the title strip so it does not obscure the plot's point inspector.
        if watched is self.chart and event.type() == QEvent.Type.ToolTip and event.pos().y() > 34:
            return True
        return super().eventFilter(watched, event)

    def set_value(self, text: str, color: str | None = None):
        """Show `text` (e.g. the latest value) beside the Hide button, or nothing when empty."""
        self.value.setText(text)
        self.value.setStyleSheet(f"color: {color};" if color else "")
        self.value.setVisible(bool(text))
        self.chart.position_corner()


class PlotGrid(QScrollArea):
    """Holds a ChartCard for every visible metric. Cards are reused across redraws.

    How they are laid out is set by `configure`: the number of columns, collapsible group headings,
    a `hero` metric drawn full width above the rest, or a single `focus` metric filling the grid.
    A `lead` (group, widget) puts a widget of the caller's first in that group's section (the Training
    monitor's headline chart); `toolbar` sits on the first section's heading row; `group_columns` gives
    a group its own number of columns (small multiples).
    """

    hide_requested = pyqtSignal(str)
    point_hovered = pyqtSignal(float)  # x of the point under the mouse, on any chart

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.body = QWidget()
        self.grid = QGridLayout(self.body)
        self.grid.setContentsMargins(0, 0, 8, 0)
        self.grid.setHorizontalSpacing(12)
        self.grid.setVerticalSpacing(10)
        self.setWidget(self.body)
        self.cards: dict[str, ChartCard] = {}
        self.empty = label("", "muted")
        self.empty.setWordWrap(True)
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.columns = COLUMNS
        self.collapsible = False
        self.collapsed: set[str] = set()  # groups whose cards are folded away under their heading
        self.hero: str | None = None
        self.focus: str | None = None
        self.header: QWidget | None = None
        self.flat = False
        self.lead: tuple[str, QWidget] | None = None
        self.lead_hidden = False  # leave the lead widget out (e.g. while runs are compared)
        self.toolbar: QWidget | None = None
        self.group_columns: dict[str, int] = {}
        self._metrics: list[str] = []
        self._empty_text = ""
        self._laid_lead: str | None = None  # the lead group of the current layout, if the lead is in it
        self._toolbar_row: QWidget | None = None
        self.compact_hide = False
        self.descriptions: dict[str, str] = {}

    def configure(self, columns=COLUMNS, collapsible=False, hero=None, focus=None, header=None, flat=False,
                  lead=None, toolbar=None, group_columns=None, compact_hide=False):
        """Change how cards are laid out; takes effect on the next `set_metrics`."""
        self.columns, self.collapsible, self.hero, self.focus = columns, collapsible, hero, focus
        self.header, self.flat = header, flat
        self.lead, self.toolbar, self.group_columns = lead, toolbar, dict(group_columns or {})
        self.compact_hide = compact_hide
        for card in self.cards.values():
            if card.property("flat") != flat:
                card.setProperty("flat", flat)
                card.style().unpolish(card)  # Qt only re-reads property selectors when re-polished
                card.style().polish(card)

    def relayout(self):
        self.set_metrics(self._metrics, self._empty_text)

    def toggle_group(self, group: str):
        (self.collapsed.discard if group in self.collapsed else self.collapsed.add)(group)
        self.relayout()

    def card(self, metric: str) -> ChartCard:
        card = self.cards.get(metric)
        if card is None:
            card = self.cards[metric] = ChartCard(metric, compact_hide=self.compact_hide)
            card.setProperty("flat", self.flat)
            card.hide_requested.connect(self.hide_requested)
            card.chart.point_hovered.connect(self.point_hovered)
        tooltip = metric + "\n\n" + (
            self.descriptions.get(metric) or describe(metric).description
            or "This component has not provided a description for this metric."
        )
        if card.chart.toolTip() != tooltip:
            card.chart.setToolTip(tooltip)
        return card

    def show_metrics(self, metrics: list[str], empty_text: str = ""):
        """set_metrics, skipped when nothing would change: live views redraw several times a second, and
        laying the cards out again each time makes them flicker and the scroll position jump."""
        unchanged = list(metrics) == self._metrics and empty_text == self._empty_text
        if self.grid.count() and unchanged and self._active_lead() == self._laid_lead:
            return
        self.set_metrics(metrics, empty_text)

    def _active_lead(self) -> str | None:
        return self.lead[0] if self.lead is not None and not self.lead_hidden else None

    def _clear(self):
        """Take every widget out of the grid; delete the ones made for the last layout (headings).

        Each widget is hidden before it is detached. Adding a widget to a layout queues a call that shows
        it; when the grid is laid out again before that call runs (a run's metrics arriving right after
        "Loading metrics…"), the call would show the detached, parentless widget as a window of its own.
        """
        keep = {id(card) for card in self.cards.values()}
        keep |= {id(w) for w in (self.header, self.empty, self.lead[1] if self.lead else None) if w is not None}
        if self.toolbar is not None:
            self.toolbar.hide()
            self.toolbar.setParent(None)  # it lives on in the next layout's heading row
        while self.grid.count():
            widget = self.grid.takeAt(0).widget()
            if widget is None:
                continue
            # Hide before detaching. Adding a widget to a layout queues a call that shows it; when the
            # grid is laid out again before that call runs (a run's metrics arriving right after
            # "Loading metrics…"), it would show the detached, parentless widget as its own window.
            widget.hide()
            widget.setParent(None)
            if id(widget) not in keep:
                widget.deleteLater()

    def _place(self, widget: QWidget, row: int, column: int, row_span: int, column_span: int):
        """Put `widget` in the grid and show it (_clear hid it, and a layout does not show a hidden widget)."""
        self.grid.addWidget(widget, row, column, row_span, column_span)
        widget.show()

    def _heading(self, group: str, count: int, folded: bool, first: bool) -> QWidget:
        depth = group.count("/")
        title = group.rsplit("/", 1)[-1].upper()
        if self.collapsible:
            heading = QToolButton()
            heading.setObjectName("resultsSection")
            heading.setText(f"{'    ' * depth}{'▸' if folded else '▾'}  {title}   {count}")
            heading.setToolTip(f"{'Show' if folded else 'Fold away'} the {group} plots")
            heading.setCursor(Qt.CursorShape.PointingHandCursor)
            heading.clicked.connect(lambda _=False, g=group: self.toggle_group(g))
        else:
            heading = label(title, "eyebrow")
        heading.setProperty("metricGroup", group)
        if not first or self.toolbar is None:
            return heading
        row = QWidget()  # the first heading carries the toolbar at its right
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(heading)
        layout.addStretch(1)  # all the spare width: the toolbar keeps its own size, at the right
        layout.addWidget(self.toolbar)
        self.toolbar.show()
        return row

    def set_metrics(self, metrics: list[str], empty_text: str = ""):
        """Lay out cards for `metrics` (already filtered to visible ones), grouped by prefix."""
        self._metrics, self._empty_text = list(metrics), empty_text
        lead = self._active_lead()
        self._laid_lead = lead
        self._clear()
        for r in range(self.grid.rowCount()):
            self.grid.setRowStretch(r, 0)
        for metric in [m for m in self.cards if m not in metrics]:
            self.cards.pop(metric).deleteLater()
        # Grid cells: enough that every group's column count divides them (2 and 4 columns: 4 cells).
        columns = math.lcm(self.columns, *self.group_columns.values()) if self.group_columns else self.columns
        for c in range(max(columns, self.grid.columnCount())):
            self.grid.setColumnStretch(c, 1 if c < columns else 0)
        row = 0
        if self.header is not None:
            self._place(self.header, row, 0, 1, columns)
            row += 1
        if not metrics and lead is None:
            self.empty.setText(empty_text)
            self._place(self.empty, row, 0, 1, columns)
            self.grid.setRowStretch(row, 1)
            return
        for metric in metrics:
            self.card(metric)  # every visible metric has a card, even when only one is on screen
        if self.focus is not None:
            focused = self.focus if self.focus in metrics else metrics[0]
            card = self.cards[focused]
            card.chart.setMinimumHeight(HERO_HEIGHT)
            self._place(card, row, 0, 1, columns)
            self.grid.setRowStretch(row, 1)
            return
        rest = list(metrics)
        if self.hero in metrics:
            card = self.cards[self.hero]
            card.chart.setMinimumHeight(HERO_HEIGHT)
            self._place(card, row, 0, 1, columns)
            row += 1
            rest.remove(self.hero)
        groups = group_metrics(rest)
        if lead is not None and lead not in groups:  # its section shows even with no other metric in it
            groups = {g: groups.get(g, []) for g in sorted([*groups, lead], key=group_sort_key)}
        # Each namespace owns its direct metrics and child namespaces. Store full
        # paths as collapse keys so equally named children remain independent.
        if self.collapsible:
            tree = {}
            for group in groups:
                tree[group] = []
            for metric in rest:
                path = metric.rsplit("/", 1)[0] if "/" in metric else "other"
                parts = path.split("/")
                for end in range(1, len(parts) + 1):
                    tree.setdefault("/".join(parts[:end]), [])
                tree[path].append(metric)
            groups = {}

            def visit(path):
                groups[path] = tree[path]
                for child in sorted(p for p in tree if p.rpartition("/")[0] == path):
                    visit(child)

            for root in sorted((p for p in tree if "/" not in p), key=group_sort_key):
                visit(root)
        for index, (group, names) in enumerate(groups.items()):
            ancestors = group.split("/")
            if any("/".join(ancestors[:i]) in self.collapsed for i in range(1, len(ancestors))):
                continue
            cells: list[QWidget] = [self.cards[m] for m in names]
            if group == lead:
                cells.insert(0, self.lead[1])
            folded = self.collapsible and group in self.collapsed
            count = sum(1 for m in rest if m.startswith(group + "/") or
                        (group == "other" and "/" not in m)) if self.collapsible else len(cells)
            if group == lead:
                count += 1
            self._place(self._heading(group, count, folded, index == 0), row, 0, 1, columns)
            row += 1
            if folded:
                continue
            per_row = self.group_columns.get(group, self.group_columns.get(group.split("/")[0], self.columns))
            span = columns // per_row
            alone = len(cells) == 1 and group == lead  # the lead chart alone in its section: full width
            for i, cell in enumerate(cells):
                if isinstance(cell, ChartCard):
                    cell.chart.setMinimumHeight(160 if per_row > 3 else 180 if per_row > 2 else 220)
                self._place(cell, row + i // per_row, (i % per_row) * span, 1, columns if alone else span)
            row += (len(cells) + per_row - 1) // per_row
        self.grid.setRowStretch(row, 1)

    def _place(self, widget, row, column, row_span=1, column_span=1):
        """Put `widget` in the grid and show it (set_metrics hid it, and a layout does not show a hidden widget)."""
        self.grid.addWidget(widget, row, column, row_span, column_span)
        widget.show()

    def charts(self) -> dict[str, Chart]:
        return {metric: card.chart for metric, card in self.cards.items()}
