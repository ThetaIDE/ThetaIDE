"""Interactive local plot viewer, using Qt without additional dependencies."""
import csv
import io
import math
from pathlib import Path

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QFileDialog,
    QMessageBox, QStackedWidget, QGraphicsView, QGraphicsScene, QSplitter,
)
from .widgets import label
from .theme import theme_color


def read_metrics_csv(path):
    """Read finite numeric columns of a CSV file; retain gaps and ignore text metadata columns."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return read_metrics_csv_stream(stream)


def read_metrics_csv_text(text):
    """read_metrics_csv() for CSV text already in memory (e.g. a file served by the backend)."""
    return read_metrics_csv_stream(io.StringIO(text.lstrip("\ufeff"), newline=""))


def read_metrics_csv_stream(stream):
    """Numeric columns of a CSV stream, by header; text columns are skipped."""
    reader = csv.DictReader(stream)
    headers = reader.fieldnames
    if not headers or len(set(headers)) != len(headers) or any(not h.strip() for h in headers):
        raise ValueError("CSV needs unique, nonempty column headers.")
    rows = []
    for row in reader:
        if len(rows) >= 50000:
            raise ValueError("This preview supports CSV files with up to 50,000 rows.")
        if None in row or None in row.values():
            raise ValueError("A CSV row has a different number of fields than its header.")
        rows.append(row)
    columns = {}
    for header in headers:
        values = []
        numeric = True
        for row in rows:
            text = row[header].strip()
            try:
                value = float(text) if text else None
            except ValueError:
                numeric = False
                break  # Text metadata is not an axis candidate.
            if value is not None and not math.isfinite(value):
                raise ValueError(f"Column '{header}' contains NaN or infinity.")
            values.append(value)
        if numeric and any(v is not None for v in values):
            columns[header] = values
    if len(columns) < 2:
        raise ValueError("Choose a CSV with at least two numeric columns and one data row.")
    return columns


class PlotCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.series = []
        self.xlabel, self.ylabel = "step", "reward"
        self.bounds = (0., 1., 0., 1.)
        self.auto_fit = True
        self.drag = None
        self.hover = None
        self.empty_text = "No plottable points yet.\nStart a demo run or open a metrics CSV."

    def area(self):
        return QRectF(76, 38, max(1, self.width() - 124), max(1, self.height() - 140))

    def set_data(self, series, xlabel, ylabel, reset=True):
        self.series, self.xlabel, self.ylabel = series, xlabel, ylabel
        self.hover = None
        if reset or self.auto_fit:
            self.reset_view()
        self.update()

    def reset_view(self):
        points = [p for _, data, _ in self.series for p in data]
        if points:
            xs, ys = zip(*points)
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
            dx = (xmax - xmin) * .05 or max(abs(xmin) * .05, .5)
            dy = (ymax - ymin) * .08 or max(abs(ymin) * .05, .5)
            self.bounds = xmin - dx, xmax + dx, ymin - dy, ymax + dy
        else:
            self.bounds = 0., 1., 0., 1.
        self.auto_fit = True
        self.update()

    def project(self, x, y):
        a = self.area()
        x0, x1, y0, y1 = self.bounds
        return QPointF(a.left() + (x - x0) / (x1 - x0) * a.width(),
                       a.bottom() - (y - y0) / (y1 - y0) * a.height())

    def zoom(self, factor, position=None):
        a = self.area()
        position = position or a.center()
        fx = (position.x() - a.left()) / a.width()
        fy = (a.bottom() - position.y()) / a.height()
        x0, x1, y0, y1 = self.bounds
        x, y = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        bounds = (x + (x0 - x) * factor, x + (x1 - x) * factor,
                  y + (y0 - y) * factor, y + (y1 - y) * factor)
        if all(math.isfinite(v) for v in bounds) and bounds[1] > bounds[0] and bounds[3] > bounds[2]:
            self.bounds = bounds
            self.auto_fit = False
            self.hover = None
            self.update()

    def wheelEvent(self, event):
        if self.area().contains(event.position()):
            self.zoom(.8 if event.angleDelta().y() > 0 else 1.25, event.position())
            event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.area().contains(event.position()):
            self.drag = (event.position(), self.bounds)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        if self.drag:
            origin, (x0, x1, y0, y1) = self.drag
            dx = (event.position().x() - origin.x()) / self.area().width() * (x1 - x0)
            dy = (event.position().y() - origin.y()) / self.area().height() * (y1 - y0)
            self.bounds = x0 - dx, x1 - dx, y0 + dy, y1 + dy
            self.auto_fit = False
            self.hover = None
        else:
            self.hover = event.position() if self.area().contains(event.position()) else None
        self.update()

    def mouseReleaseEvent(self, event):
        self.drag = None
        self.unsetCursor()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()

    def leaveEvent(self, event):
        self.hover = None
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(theme_color("base")))
        p.setFont(QFont("Segoe UI", 9))
        a = self.area()
        x0, x1, y0, y1 = self.bounds
        p.setPen(QColor(theme_color("text")))
        p.drawText(18, 23, f"{self.ylabel} / {self.xlabel}")
        for i in range(6):
            f = i / 5
            x, y = a.left() + a.width() * f, a.bottom() - a.height() * f
            p.setPen(QColor(theme_color("raised")))
            p.drawLine(QPointF(x, a.top()), QPointF(x, a.bottom()))
            p.drawLine(QPointF(a.left(), y), QPointF(a.right(), y))
            p.setPen(QColor(theme_color("muted")))
            p.drawText(QRectF(x - 35, a.bottom() + 7, 70, 20), Qt.AlignmentFlag.AlignCenter, f"{x0 + (x1 - x0) * f:.4g}")
            p.drawText(QRectF(2, y - 10, 66, 20), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{y0 + (y1 - y0) * f:.4g}")
        p.drawText(QRectF(a.left(), a.bottom() + 30, a.width(), 20), Qt.AlignmentFlag.AlignCenter, self.xlabel)
        p.save()
        p.setClipRect(a)
        nearest = None
        for name, points, color in self.series:
            path = QPainterPath()
            for i, (x, y) in enumerate(points):
                point = self.project(x, y)
                if i == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
                if self.hover and a.contains(point):
                    distance = (point.x() - self.hover.x()) ** 2 + (point.y() - self.hover.y()) ** 2
                    if nearest is None or distance < nearest[0]:
                        nearest = distance, point, x, y, color
            p.setPen(QPen(QColor(theme_color(color)), 2))
            p.drawPath(path)
            if len(points) == 1:
                p.drawEllipse(self.project(*points[0]), 3, 3)
        if nearest and nearest[0] < 1600:
            _, point, x, y, color = nearest
            p.setPen(QPen(QColor(theme_color(color)), 2))
            p.drawEllipse(point, 5, 5)
            self.setToolTip(f"{self.xlabel}: {x:.6g}\n{self.ylabel}: {y:.6g}")
        else:
            self.setToolTip("")
        p.restore()
        if not any(data for _, data, _ in self.series):
            p.setPen(QColor(theme_color("muted")))
            p.drawText(a, Qt.AlignmentFlag.AlignCenter, self.empty_text)
        self._draw_legend(p, a)

    def _draw_legend(self, p, a):
        """One entry per series, side by side, on up to two rows below the axis; '+N more' if they don't fit."""
        metrics = p.fontMetrics()
        x, row, rows, gap = a.left(), 0, 2, 24
        for i, (name, _, color) in enumerate(self.series):
            text = f"●  {name}"
            width = metrics.horizontalAdvance(text)
            if x > a.left() and x + width > a.right():
                x, row = a.left(), row + 1
            if row >= rows:
                p.setPen(QColor(theme_color("muted")))
                p.drawText(QRectF(a.left(), self.height() - 46 + (rows - 1) * 19, a.width(), 19),
                           Qt.AlignmentFlag.AlignRight, f"+{len(self.series) - i} more")
                return
            p.setPen(QColor(theme_color(color)))
            p.drawText(QRectF(x, self.height() - 46 + row * 19, min(width, a.width()) + 2, 19),
                       Qt.AlignmentFlag.AlignLeft, metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(a.width())))
            x += width + gap


class ImageCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(theme_color("base")))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def load(self, pixmap):
        self.scene().clear()
        self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(QRectF(pixmap.rect()))
        self.reset_view()

    def drawBackground(self, painter, rect):
        painter.fillRect(rect, QColor(theme_color("base")))

    def reset_view(self):
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        scale = self.transform().m11() / factor
        if .01 <= scale <= 100:
            self.scale(1 / factor, 1 / factor)

    def wheelEvent(self, event):
        self.zoom(.8 if event.angleDelta().y() > 0 else 1.25)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.reset_view()


class PlotViewer(QWidget):
    """Plots the runs ticked in the runs list (at its left) together, or a CSV / image opened from disk.

    There is no source picker: what is drawn is exactly what is ticked. An opened file is shown until the
    ticks change, and then the ticked runs are drawn again.
    """

    def __init__(self):
        super().__init__()
        self.runs = []
        self.compared: list[str] = []  # runs ticked in the runs list, in tick order
        self.compared_colors: dict[str, str] = {}
        self.imported = None  # (name, columns or QPixmap) of the file opened last, while it is shown
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(label("Plot Viewer", "heading"))
        # The runs list (add_runs_list) at the left, the plot and its controls beside it
        self.body = QSplitter(Qt.Orientation.Horizontal)
        self.body.setChildrenCollapsible(False)
        main = QWidget()
        outer, layout = layout, QVBoxLayout(main)
        layout.setContentsMargins(0, 0, 0, 0)
        self.body.addWidget(main)
        outer.addWidget(self.body, 1)
        tools = QHBoxLayout()
        for text, callback in (("Open CSV / image…", self.open_file), ("+", lambda: self.current_canvas().zoom(.8)),
                               ("−", lambda: self.current_canvas().zoom(1.25)),
                               ("Fit", lambda: self.current_canvas().reset_view()), ("Export PNG…", self.export_png)):
            button = QPushButton(text)
            if text in ("+", "−"):
                button.setToolTip("Zoom in" if text == "+" else "Zoom out")
                button.setAccessibleName(button.toolTip())
            button.clicked.connect(callback)
            tools.addWidget(button)
        tools.addStretch()
        layout.addLayout(tools)
        axes = QHBoxLayout()
        self.xaxis, self.yaxis = QComboBox(), QComboBox()
        for combo in (self.xaxis, self.yaxis):
            combo.setMaximumWidth(240)
        axes.addWidget(label("X", "muted"))
        axes.addWidget(self.xaxis, 1)
        axes.addWidget(label("Y", "muted"))
        axes.addWidget(self.yaxis, 1)
        axes.addStretch()
        layout.addLayout(axes)
        self.stack = QStackedWidget()
        self.canvas, self.image = PlotCanvas(), ImageCanvas()
        self.canvas.empty_text = "Tick runs in the list to plot them,\nor open a metrics CSV."
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.image)
        layout.addWidget(self.stack, 1)
        self.caption = label("", "muted")
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        layout.addWidget(label("Scroll to zoom · Drag to pan · Double-click to fit · Hover a curve to inspect", "muted"))
        self.xaxis.currentTextChanged.connect(self.redraw)
        self.yaxis.currentTextChanged.connect(self.redraw)
        self.refresh()

    def add_runs_list(self, runs_list, width=300):
        """Place the runs list at the left: its ticks choose what is plotted (and a click, the monitor's run)."""
        self.body.insertWidget(0, runs_list)
        self.body.setStretchFactor(1, 1)
        self.body.setSizes([width, 1000])

    def current_canvas(self):
        return self.stack.currentWidget()

    def update_runs(self, runs):
        """The app's runs changed (a new one, new metrics): redraw the ticked ones, keeping the view."""
        self.runs = runs
        if self.imported is None:
            self.refresh(reset=False)

    def set_compared(self, run_ids, colors):
        """The ticks changed: plot exactly the ticked runs, each in its row's color."""
        self.compared, self.compared_colors = list(run_ids), dict(colors)
        self.imported = None
        self.refresh()

    def data_for(self, run_id):
        for run in self.runs:
            if run["id"] == run_id:
                from .model import load_run_metrics
                try:
                    load_run_metrics(run)
                except (OSError, ValueError):
                    return None
                source = "simulated" if run.get("simulated", True) else "trained"
                return f"{run['config']['name']} [{run['id']}] · {source}", {
                    field: [m.get(field) for m in run["metrics"]]
                    for field in ["step"] + sorted({k for m in run["metrics"] for k in m} - {"step"})}
        return None

    def sources(self):
        """(name, columns, color) of everything to draw: the opened CSV, or the ticked runs."""
        if self.imported is not None:
            name, data = self.imported
            return [(name, data, "#b8bb26")]
        found = []
        for run_id in self.compared:
            entry = self.data_for(run_id)
            if entry is not None:
                found.append((*entry, self.compared_colors.get(run_id, "#b8bb26")))
        return found

    def refresh(self, reset=True):
        """Show what is chosen: an opened image, or curves; the axis lists offer every column of the sources."""
        if self.imported is not None and isinstance(self.imported[1], QPixmap):
            data = self.imported[1]
            self.stack.setCurrentIndex(1)
            self.xaxis.setEnabled(False)
            self.yaxis.setEnabled(False)
            self.image.load(data)
            self.caption.setText(f"Local image · {data.width()} × {data.height()} pixels · tick a run to go back")
            return
        self.stack.setCurrentIndex(0)
        self.xaxis.setEnabled(True)
        self.yaxis.setEnabled(True)
        columns = list(dict.fromkeys(column for _, data, _ in self.sources() for column in data))
        for combo, preferred in ((self.xaxis, "step"), (self.yaxis, "reward")):
            if [combo.itemText(i) for i in range(combo.count())] == columns:
                continue
            previous = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(columns)
            choice = previous if previous in columns else preferred
            index = combo.findText(choice)
            combo.setCurrentIndex(index if index >= 0 else min(1 if combo is self.yaxis else 0, len(columns) - 1))
            combo.blockSignals(False)
        self.redraw(reset=reset)

    def redraw(self, *_, reset=True):
        if self.imported is not None and isinstance(self.imported[1], QPixmap):
            return
        x, y = self.xaxis.currentText(), self.yaxis.currentText()
        series = []
        warnings = []
        for title, columns, color in self.sources():
            if x not in columns or y not in columns:
                warnings.append(f"{title}: chosen axes unavailable")
                continue
            pairs = sorted((a, b) for a, b in zip(columns[x], columns[y]) if a is not None and b is not None)
            series.append((title, pairs, color))
        self.canvas.set_data(series, x, y, reset=reset)
        count = sum(len(points) for _, points, _ in series)
        if self.imported is not None:
            provenance = "Imported CSV · session only · tick a run to go back"
        else:
            provenance = f"{len(series)} run{'s' * (len(series) != 1)}"
        self.caption.setText(f"{provenance} · {count:,} points" + (" · " + "; ".join(warnings) if warnings else ""))

    def open_path(self, path):
        path = Path(path)
        if path.suffix.lower() == ".csv":
            data = read_metrics_csv(path)
        else:
            data = QPixmap(str(path))
            if data.isNull():
                raise ValueError("Could not read that image. Choose a PNG, JPEG, or BMP file.")
        self.imported = path.name, data
        self.refresh()

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open plot or metrics", "",
                                             "Plots and metrics (*.csv *.png *.jpg *.jpeg *.bmp)")
        if path:
            try:
                self.open_path(path)
            except (OSError, ValueError, csv.Error) as exc:
                QMessageBox.warning(self, "Could not open plot", str(exc))

    def save_png(self, path):
        canvas = self.current_canvas()
        target = canvas.viewport() if isinstance(canvas, ImageCanvas) else canvas
        if not target.grab().save(str(path), "PNG"):
            raise OSError("The image could not be saved. Check the destination folder.")

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export current plot view", "thetaide-plot.png", "PNG (*.png)")
        if path:
            try:
                self.save_png(path)
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))
