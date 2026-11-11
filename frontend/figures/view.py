"""The figure workspace: pick panels, arrange them, style them, export.

Deliberately its own widget rather than an addition to frontend/plots.py.
That file is the run viewer and is being worked on separately; a figure
being assembled for a paper is a different job with different state, and
keeping them apart means neither has to wait for the other.

The preview is a real matplotlib canvas, so what is on screen is what gets
exported, including to PDF and SVG.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ..widgets import SpinBox as QSpinBox, DoubleSpinBox as QDoubleSpinBox
from ..theme import theme_color
from .model import FigureSpec, PanelRef, available_experiments, render, series_labels
from .sources import PanelSource

EXPORT_FILTER = "PNG image (*.png);;PDF document (*.pdf);;SVG image (*.svg)"

# The preview draws at screen resolution regardless of the export DPI, so a
# figure bound for 300 dpi print still reads correctly in the pane. The
# effective value is lowered further when the figure is wider than the panel,
# so the whole figure stays visible instead of being cropped.
PREVIEW_DPI = 100
MIN_PREVIEW_DPI = 24


class FigureWorkspace(QWidget):
    """Compose several plots into one figure and export it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.source = PanelSource()
        self.spec = FigureSpec(rows=1, cols=2)
        self._loading = False

        # Screen resolution for the preview; spec.dpi is applied on export.
        self.figure = Figure(figsize=(self.spec.width, self.spec.height), dpi=PREVIEW_DPI)
        self.canvas = FigureCanvasQTAgg(self.figure)

        self._build_ui()
        self._load_experiments()
        self._refresh_placed()
        self._redraw()

    # ── Layout ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)

        self.cmb_experiment = QComboBox()
        self.cmb_experiment.setMinimumWidth(320)
        self.cmb_experiment.currentIndexChanged.connect(self._on_experiment_changed)
        top = QHBoxLayout()
        top.addWidget(QLabel("Experiment:"))
        top.addWidget(self.cmb_experiment)
        btn_rescan = QPushButton("Rescan")
        btn_rescan.setToolTip("Look for runs that finished since this pane opened")
        btn_rescan.clicked.connect(lambda: self._rescan())
        top.addWidget(btn_rescan)
        top.addStretch()
        root.addLayout(top)

        # Left: what can be added.
        self.lst_catalog = QListWidget()
        self.lst_catalog.setToolTip("Double-click to add a panel to the figure")
        self.lst_catalog.itemDoubleClicked.connect(lambda _item: self._add_selected())
        btn_add = QPushButton("Add panel ->")
        btn_add.clicked.connect(lambda: self._add_selected())

        left = QWidget()
        left_box = QVBoxLayout(left)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_box.addWidget(QLabel("Available panels"))
        left_box.addWidget(self.lst_catalog, 1)
        left_box.addWidget(btn_add)
        self.lbl_catalog_note = QLabel("")
        self.lbl_catalog_note.setWordWrap(True)
        self.lbl_catalog_note.setStyleSheet(f"color: {theme_color('muted')};")
        left_box.addWidget(self.lbl_catalog_note)
        left.setMaximumWidth(260)

        # Middle: the figure itself.
        middle = QWidget()
        mid_box = QVBoxLayout(middle)
        mid_box.setContentsMargins(0, 0, 0, 0)
        mid_box.addWidget(NavigationToolbar2QT(self.canvas, middle))
        mid_box.addWidget(self.canvas, 1)

        # Right: the controls.
        right = QWidget()
        right_box = QVBoxLayout(right)
        right_box.setContentsMargins(0, 0, 0, 0)
        right_box.addWidget(QLabel("Figure"))
        right_box.addLayout(self._figure_form())
        right_box.addSpacing(8)
        right_box.addWidget(QLabel("Panels on the figure"))
        self.lst_placed = QListWidget()
        self.lst_placed.currentRowChanged.connect(lambda _row: self._show_panel_form())
        right_box.addWidget(self.lst_placed, 1)
        right_box.addLayout(self._placed_buttons())
        right_box.addWidget(QLabel("Selected panel"))
        self.panel_form_host = QWidget()
        self.panel_form = QFormLayout(self.panel_form_host)
        self.panel_form.setContentsMargins(0, 0, 0, 0)
        right_box.addWidget(self.panel_form_host)
        right.setMaximumWidth(300)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(middle)
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        root.addWidget(split, 1)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        bottom = QHBoxLayout()
        bottom.addStretch()
        btn_export = QPushButton("Export figure...")
        btn_export.setToolTip("Save exactly what is previewed, as PNG, PDF or SVG")
        btn_export.clicked.connect(lambda: self._export())
        bottom.addWidget(btn_export)
        root.addLayout(bottom)

    def _figure_form(self) -> QFormLayout:
        form = QFormLayout()
        self.spn_rows = QSpinBox()
        self.spn_rows.setRange(1, 6)
        self.spn_rows.setValue(self.spec.rows)
        self.spn_cols = QSpinBox()
        self.spn_cols.setRange(1, 6)
        self.spn_cols.setValue(self.spec.cols)
        for box in (self.spn_rows, self.spn_cols):
            box.valueChanged.connect(lambda _v: self._on_grid_changed())

        grid_row = QHBoxLayout()
        grid_row.addWidget(self.spn_rows)
        grid_row.addWidget(QLabel("x"))
        grid_row.addWidget(self.spn_cols)
        grid_row.addStretch()
        form.addRow("Grid", grid_row)

        self.spn_width = QDoubleSpinBox()
        self.spn_width.setRange(2.0, 40.0)
        self.spn_width.setValue(self.spec.width)
        self.spn_height = QDoubleSpinBox()
        self.spn_height.setRange(2.0, 40.0)
        self.spn_height.setValue(self.spec.height)
        for box in (self.spn_width, self.spn_height):
            box.setSuffix(" in")
            box.setSingleStep(0.5)
            box.valueChanged.connect(lambda _v: self._on_figure_changed())
        size_row = QHBoxLayout()
        size_row.addWidget(self.spn_width)
        size_row.addWidget(self.spn_height)
        form.addRow("Size", size_row)

        self.spn_dpi = QSpinBox()
        self.spn_dpi.setRange(72, 1200)
        self.spn_dpi.setValue(self.spec.dpi)
        self.spn_dpi.setToolTip("Most journals want 300 or more for raster figures")
        self.spn_dpi.valueChanged.connect(lambda _v: self._on_figure_changed())
        form.addRow("DPI", self.spn_dpi)

        self.txt_suptitle = QLineEdit(self.spec.suptitle)
        self.txt_suptitle.setPlaceholderText("Optional figure title")
        self.txt_suptitle.textChanged.connect(lambda _t: self._on_figure_changed())
        form.addRow("Title", self.txt_suptitle)

        self.chk_tight = QCheckBox("Tight layout")
        self.chk_tight.setChecked(self.spec.tight)
        self.chk_tight.toggled.connect(lambda _v: self._on_figure_changed())
        form.addRow(self.chk_tight)
        return form

    def _placed_buttons(self) -> QHBoxLayout:
        row = QHBoxLayout()
        for text, slot, tip in (
            ("Up", self._move_up, "Move this panel one slot earlier"),
            ("Down", self._move_down, "Move this panel one slot later"),
            ("Remove", self._remove_selected, "Take this panel off the figure"),
        ):
            btn = QPushButton(text)
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked=False, fn=slot: fn())
            row.addWidget(btn)
        return row

    # ── Experiments and catalogue ────────────────────────────────────────

    def _load_experiments(self) -> None:
        self._loading = True
        self.cmb_experiment.clear()
        for group, exp_id in available_experiments():
            self.cmb_experiment.addItem(f"{group} / {exp_id}", exp_id)
        self._loading = False
        if self.cmb_experiment.count():
            self._on_experiment_changed()
        else:
            self.lbl_catalog_note.setText("No runs with metrics found under results/logs.")

    def _rescan(self) -> None:
        self.source.invalidate()
        current = self.cmb_experiment.currentData()
        self._load_experiments()
        if current:
            index = self.cmb_experiment.findData(current)
            if index >= 0:
                self.cmb_experiment.setCurrentIndex(index)
        self._redraw()

    def current_experiment(self) -> str | None:
        return self.cmb_experiment.currentData()

    def _on_experiment_changed(self) -> None:
        if self._loading:
            return
        exp_id = self.current_experiment()
        self.lst_catalog.clear()
        if not exp_id:
            return

        refs = self.source.catalog(exp_id)
        for ref in refs:
            item = QListWidgetItem(ref.label)
            item.setData(Qt.ItemDataRole.UserRole, ref)
            self.lst_catalog.addItem(item)

        if refs:
            self.lbl_catalog_note.setText("")
        else:
            self.lbl_catalog_note.setText(
                "No composable panels for this run. Only plotters that draw into a "
                "supplied Axes can be arranged on a figure."
            )

    # ── Editing the figure ───────────────────────────────────────────────

    def _add_selected(self) -> None:
        item = self.lst_catalog.currentItem()
        if item is None:
            self._warn("Pick a panel from the list on the left first.")
            return
        try:
            self.spec.add(item.data(Qt.ItemDataRole.UserRole))
        except ValueError as exc:
            self._warn(str(exc))
            return
        self._sync_grid_boxes()
        self._refresh_placed()
        self.lst_placed.setCurrentRow(len(self.spec.used()) - 1)
        self._redraw()

    def _selected_slot(self) -> int | None:
        row = self.lst_placed.currentRow()
        if row < 0:
            return None
        item = self.lst_placed.item(row)
        return None if item is None else item.data(Qt.ItemDataRole.UserRole)

    def _remove_selected(self) -> None:
        slot = self._selected_slot()
        if slot is None:
            return
        self.spec.remove(slot)
        self.spec.compact()
        self._refresh_placed()
        self._redraw()

    def _move_up(self) -> None:
        self._move(-1)

    def _move_down(self) -> None:
        self._move(+1)

    def _move(self, delta: int) -> None:
        slot = self._selected_slot()
        if slot is None:
            return
        target = slot + delta
        if not 0 <= target < len(self.spec.panels):
            return
        self.spec.move(slot, target)
        self._refresh_placed()
        for row in range(self.lst_placed.count()):
            if self.lst_placed.item(row).data(Qt.ItemDataRole.UserRole) == target:
                self.lst_placed.setCurrentRow(row)
                break
        self._redraw()

    def _refresh_placed(self) -> None:
        self.lst_placed.blockSignals(True)
        self.lst_placed.clear()
        for slot, ref in enumerate(self.spec.panels):
            if ref is None:
                continue
            item = QListWidgetItem(f"{slot + 1}. {ref.title or ref.label}")
            item.setData(Qt.ItemDataRole.UserRole, slot)
            self.lst_placed.addItem(item)
        self.lst_placed.blockSignals(False)
        self._show_panel_form()

    def _on_grid_changed(self) -> None:
        if self._loading:
            return
        try:
            dropped = self.spec.set_grid(self.spn_rows.value(), self.spn_cols.value())
        except ValueError as exc:
            self._warn(str(exc))
            self._sync_grid_boxes()
            return
        self._refresh_placed()
        self._redraw()
        if dropped:
            # After the redraw, which writes its own status line and would
            # otherwise erase this. Shrinking the grid throws panels away, so
            # saying so is the whole point.
            names = ", ".join(d.title or d.label for d in dropped[:2])
            more = f" and {len(dropped) - 2} more" if len(dropped) > 2 else ""
            self._warn(f"{len(dropped)} panel(s) no longer fit and were removed: {names}{more}")

    def _sync_grid_boxes(self) -> None:
        self._loading = True
        self.spn_rows.setValue(self.spec.rows)
        self.spn_cols.setValue(self.spec.cols)
        self._loading = False

    def _on_figure_changed(self) -> None:
        if self._loading:
            return
        self.spec.width = self.spn_width.value()
        self.spec.height = self.spn_height.value()
        self.spec.dpi = self.spn_dpi.value()
        self.spec.suptitle = self.txt_suptitle.text()
        self.spec.tight = self.chk_tight.isChecked()
        self._redraw()

    # ── Per-panel controls ───────────────────────────────────────────────

    def _show_panel_form(self) -> None:
        while self.panel_form.rowCount():
            self.panel_form.removeRow(0)

        slot = self._selected_slot()
        ref = self.spec.panels[slot] if slot is not None else None
        if ref is None:
            note = QLabel("Select a panel to restyle it.")
            note.setStyleSheet(f"color: {theme_color('muted')};")
            self.panel_form.addRow(note)
            return

        txt_title = QLineEdit(ref.title or "")
        txt_title.setPlaceholderText(ref.label)
        txt_title.textChanged.connect(lambda text, s=slot: self._update(s, title=text or None))
        self.panel_form.addRow("Title", txt_title)

        # One swatch per series, so a panel comparing two methods can keep
        # them apart. The labels come from the drawn Axes, so they are only
        # known once the panel has been rendered at least once.
        for series in self._series_for(slot):
            btn = QPushButton(ref.colors.get(series) or "default")
            btn.setToolTip(f"Colour for {series}")
            btn.clicked.connect(lambda _checked=False, s=slot, name=series: self._pick_color(s, name))
            self.panel_form.addRow(series, btn)

        spn_width = QDoubleSpinBox()
        spn_width.setRange(0.5, 8.0)
        spn_width.setSingleStep(0.25)
        spn_width.setSpecialValueText("default")
        spn_width.setValue(ref.line_width if ref.line_width is not None else 0.5)
        spn_width.valueChanged.connect(lambda v, s=slot: self._update(s, line_width=None if v == 0.5 else v))
        self.panel_form.addRow("Line width", spn_width)

        chk_legend = QCheckBox("Legend")
        chk_legend.setChecked(ref.legend)
        chk_legend.toggled.connect(lambda v, s=slot: self._update(s, legend=v))
        self.panel_form.addRow(chk_legend)

        chk_grid = QCheckBox("Grid")
        chk_grid.setChecked(ref.grid)
        chk_grid.toggled.connect(lambda v, s=slot: self._update(s, grid=v))
        self.panel_form.addRow(chk_grid)

    def _axes_for(self, slot: int):
        """The Axes a slot was drawn into, or None when it has not been."""
        drawn = [i for i, p in enumerate(self.spec.panels) if p is not None]
        if slot not in drawn:
            return None
        index = drawn.index(slot)
        return self.figure.axes[index] if index < len(self.figure.axes) else None

    def _series_for(self, slot: int) -> list:
        ax = self._axes_for(slot)
        return series_labels(ax) if ax is not None else []

    def _pick_color(self, slot: int, series: str) -> None:
        ref = self.spec.panels[slot]
        current = (ref.colors.get(series) if ref else None) or "#1f77b4"
        chosen = QColorDialog.getColor(QColor(current), self, f"Colour for {series}")
        if not chosen.isValid():
            return
        colors = dict(ref.colors)
        colors[series] = chosen.name()
        self._update(slot, colors=colors)
        self._show_panel_form()

    def _update(self, slot: int, **changes) -> None:
        self.spec.update(slot, **changes)
        if "title" in changes:
            self._refresh_placed_labels()
        self._redraw()

    def _refresh_placed_labels(self) -> None:
        for row in range(self.lst_placed.count()):
            item = self.lst_placed.item(row)
            slot = item.data(Qt.ItemDataRole.UserRole)
            ref = self.spec.panels[slot]
            if ref is not None:
                item.setText(f"{slot + 1}. {ref.title or ref.label}")

    # ── Drawing and export ───────────────────────────────────────────────

    def _preview_dpi(self) -> int:
        """Dots per inch that fit the whole figure inside the canvas widget.

        The figure is a fixed size in inches, so at a fixed DPI a wide figure
        is simply larger than the widget and gets cropped: the right-hand
        column disappears and axis labels are cut mid-number. Lowering DPI
        shrinks the rendered pixels while leaving the layout in inches
        untouched, so the preview stays a faithful, smaller copy of what will
        be exported.
        """
        available = self.canvas.size()
        width, height = available.width(), available.height()
        if width <= 0 or height <= 0 or self.spec.width <= 0 or self.spec.height <= 0:
            return PREVIEW_DPI
        fit = min(width / self.spec.width, height / self.spec.height)
        return max(MIN_PREVIEW_DPI, min(PREVIEW_DPI, int(fit)))

    def resizeEvent(self, event):
        """Re-fit the preview when the pane changes size."""
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self) -> None:
        render(self.spec, self.source.resolve, figure=self.figure, dpi=self._preview_dpi())
        self.canvas.draw_idle()
        used = len(self.spec.used())
        if used:
            self.lbl_status.setText(
                f"{used} panel(s) on a {self.spec.rows}x{self.spec.cols} grid, "
                f"{self.spec.width:g}x{self.spec.height:g} in at {self.spec.dpi} dpi."
            )
            self.lbl_status.setStyleSheet(f"color: {theme_color('muted')};")
        else:
            self.lbl_status.setText("Add a panel from the left to start the figure.")
            self.lbl_status.setStyleSheet(f"color: {theme_color('muted')};")

    def _warn(self, message: str) -> None:
        self.lbl_status.setText(message)
        self.lbl_status.setStyleSheet("color: #fb4934;")

    def _export(self) -> None:
        if not self.spec.used():
            self._warn("There is nothing on the figure to export yet.")
            return

        suggested = str(Path.home() / "figure.png")
        path, chosen = QFileDialog.getSaveFileName(self, "Export figure", suggested, EXPORT_FILTER)
        if not path:
            return

        # Let the chosen filter decide when the typed name has no extension,
        # so "figure" under the PDF filter does not silently become a PNG.
        target = Path(path)
        if not target.suffix:
            ext = {"PDF document (*.pdf)": ".pdf", "SVG image (*.svg)": ".svg"}.get(chosen, ".png")
            target = target.with_suffix(ext)

        try:
            # Rendered again into a figure at the requested size and DPI
            # rather than scaling the preview, so the exported file is exactly
            # what the figure settings describe.
            out = Figure(figsize=(self.spec.width, self.spec.height), dpi=self.spec.dpi)
            render(self.spec, self.source.resolve, figure=out, dpi=self.spec.dpi)
            out.savefig(target, dpi=self.spec.dpi, bbox_inches="tight" if self.spec.tight else None)
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", f"Could not write {target}:\n{exc}")
            return

        self.lbl_status.setText(f"Exported to {target}")
        self.lbl_status.setStyleSheet(f"color: {theme_color('primary')};")
