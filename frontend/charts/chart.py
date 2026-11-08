"""Dependency-free metric chart shared by the Training monitor and the Results panel."""
import math

from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from ..theme import theme_color


class Chart(QWidget):
    """Small dependency-free plot; drawing only, no synthetic data hidden here.

    Axes fit the data with round tick values. `reference` (e.g. an environment's maximum
    reward) caps the y-axis and is drawn as a dashed line once the data approaches it.
    `band` names a per-point spread (e.g. "reward_std") drawn as a shaded ±1 band.
    """
    # x of the data point under the mouse (e.g. for an inspector showing every metric at that step)
    point_hovered = pyqtSignal(float)

    def __init__(self, metric="reward", title="Episode reward", band=None, zero_based=True):
        super().__init__()
        self.metric, self.title, self.band = metric, title, band
        self.zero_based = zero_based  # False fits the y-axis to the data (small changes stay visible)
        self.series = []
        self.xmax = None
        self.reference = None
        self.empty_text = "Launch training to see live metrics"
        self.corner = None
        self.setMinimumHeight(185)
        self.setMinimumWidth(240)
        self.setMouseTracking(True)
        self.hover_point = None
        self.zoom_start = None
        self.zoom_current = None
        self.custom_x_range = None
        self.baseline_series = []
        self.smoothing = 0.0
        self.primary = None  # a series name: hovering snaps only to its points, and it is drawn on top
        self.hover_all = False  # the hover card lists every series' value at the hovered step, not just one
        # Live drawing: hold_y lets the y-axis grow (with headroom) but never shrink, so it does not jump
        # with every new point; show_future shades the part of a fixed x-extent not trained yet; the
        # newest point of each series named in live_series (still training) is marked.
        self.hold_y = False
        self.show_future = False
        self.live_series: set[str] = set()
        self._extent = None  # (lo, hi, step) of the current series, worked out once per set_series
        self._held = None  # (lo, hi) the y-axis has grown to while hold_y is on
        self._held_key = None  # what the held range belongs to: the metric and series names

    def set_series(self, series, xmax=None, reference=None):
        """xmax fixes the x-axis extent (e.g. a run's training budget) instead of fitting the data."""
        self.xmax, self.reference = xmax, reference
        # Live runs report different metrics on different rows; skip points missing this chart's metric.
        self.series = [(name, [m for m in metrics if m.get(self.metric) is not None], color)
                       for name, metrics, color in series]
        key = (self.metric, tuple(name for name, _, _ in self.series))
        if key != self._held_key:
            self._held, self._held_key = None, key
        self._extent = None
        self.update()

    def _y_extent(self, points):
        """y_range() of the drawn points, held while hold_y is on; worked out once per set of series."""
        if self._extent is None:
            self._extent = self._hold(*self.y_range(points)) if points else (0.0, 1.0, 0.25)
        return self._extent

    def _hold(self, lo, hi, step):
        if not self.hold_y:
            self._held = None
            return lo, hi, step
        if self._held is not None:
            held_lo, held_hi = self._held
            if lo >= held_lo and hi <= held_hi:
                lo, hi = held_lo, held_hi
            else:  # grow past the new point by a share of the span, so the next few points fit as well
                span = max(hi, held_hi) - min(lo, held_lo)
                lo = min(held_lo, lo - 0.15 * span) if lo < held_lo else held_lo
                hi = max(held_hi, hi + 0.15 * span) if hi > held_hi else held_hi
                step = nice_step((hi - lo) / 4)
                lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
        self._held = (lo, hi)
        return lo, hi, nice_step((hi - lo) / 4)

    def set_baseline_series(self, series):
        """Set a pinned baseline series to render as a dashed reference line."""
        if not series:
            self.baseline_series = []
        else:
            self.baseline_series = [(name, [m for m in metrics if m.get(self.metric) is not None], color)
                                    for name, metrics, color in series]
        self.update()

    def set_smoothing(self, factor: float):
        """Exponential moving average smoothing factor (0.0 = raw, up to 0.99)."""
        self.smoothing = max(0.0, min(0.99, factor))
        self.update()

    def _smooth_points(self, points: list[dict], factor: float) -> list[dict]:
        if factor <= 0 or len(points) <= 1:
            return points
        smoothed = []
        last = None
        for p in points:
            val = p.get(self.metric)
            if val is None:
                continue
            if last is None:
                last = val
            else:
                last = last * factor + val * (1.0 - factor)
            smoothed.append({**p, self.metric: last})
        return smoothed

    def set_metric(self, metric, title):
        if metric != self.metric:  # redrawing the same metric keeps the hovered point
            self.hover_point = None
            self._extent = None
        self.metric, self.title = metric, title
        self.update()

    def set_corner_widget(self, widget):
        """Place a small control (e.g. a metric selector) in the chart's top-right corner."""
        widget.setParent(self)
        self.corner = widget
        self.position_corner()

    def position_corner(self):
        if self.corner is not None:
            self.corner.adjustSize()
            self.corner.move(self.width() - self.corner.width() - 12, 6)

    def resizeEvent(self, event):
        self.position_corner()
        super().resizeEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.zoom_start is not None:
            self.zoom_current = pos
            self.update()
            return

        area = QRectF(52, 45, self.width() - 74, self.height() - 80)
        if not area.contains(pos) or not self.series:
            if self.hover_point is not None:
                self.hover_point = None
                self.update()
            return

        points = [m for _, metrics, _ in self.series for m in metrics]
        if not points:
            if self.hover_point is not None:
                self.hover_point = None
                self.update()
            return

        xmax = self.xmax or max([m["step"] for m in points] or [10000])
        xlo, xhi = self.custom_x_range if self.custom_x_range else (0.0, float(xmax))
        lo, hi, _ = self._y_extent(points)

        def x_of(s):
            return area.left() + area.width() * (s - xlo) / max(1e-12, xhi - xlo)

        def y_of(v):
            return area.bottom() - area.height() * (v - lo) / max(1e-12, hi - lo)

        closest = None
        min_dist = float("inf")
        for name, metrics, color in self._snap_series():
            for m in metrics:
                s = m["step"]
                if s < xlo or s > xhi:
                    continue
                sx, sy = x_of(s), y_of(m[self.metric])
                dist = abs(pos.x() - sx)
                if dist < min_dist and dist < 35:
                    min_dist = dist
                    closest = (sx, sy, s, m[self.metric], name, color)

        if closest != self.hover_point:
            self.hover_point = closest
            self.update()
            if closest is not None:
                self.point_hovered.emit(float(closest[2]))

    def leaveEvent(self, event):
        if self.hover_point is not None or self.zoom_start is not None:
            self.hover_point = None
            self.zoom_start = None
            self.zoom_current = None
            self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            area = QRectF(52, 45, self.width() - 74, self.height() - 80)
            if area.contains(event.position()):
                self.zoom_start = event.position()
                self.zoom_current = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.zoom_start is not None:
            end_pos = event.position()
            area = QRectF(52, 45, self.width() - 74, self.height() - 80)
            points = [m for _, metrics, _ in self.series for m in metrics]
            xmax = self.xmax or max([m["step"] for m in points] or [10000])
            xlo, xhi = self.custom_x_range if self.custom_x_range else (0.0, float(xmax))

            def step_of(x):
                norm = (x - area.left()) / max(1e-12, area.width())
                return xlo + norm * (xhi - xlo)

            if abs(end_pos.x() - self.zoom_start.x()) > 15:
                s1 = step_of(min(self.zoom_start.x(), end_pos.x()))
                s2 = step_of(max(self.zoom_start.x(), end_pos.x()))
                s1 = max(0.0, s1)
                s2 = min(float(xmax), s2)
                if s2 - s1 > 10:
                    self.custom_x_range = (s1, s2)
            self.zoom_start = None
            self.zoom_current = None
            self.update()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.custom_x_range = None
            self.hover_point = None
            self.update()
        super().mouseDoubleClickEvent(event)

    def _snap_series(self):
        """The series hovering may snap to: only the primary one when it is plotted, else all."""
        primary = [entry for entry in self.series if entry[0] == self.primary]
        return primary or self.series

    def _draw_order(self):
        """Series in drawing order: the primary one last, so it stays on top."""
        return sorted(self.series, key=lambda entry: entry[0] == self.primary)

    def y_range(self, points):
        values = [m[self.metric] for m in points]
        if self.band:
            spreads = [(m[self.metric], m.get(self.band)) for m in points]
            values += [mean + s for mean, s in spreads if s is not None]
            # Rewards cannot go below zero; other spreads (e.g. a loss averaged over seeds) can.
            values += [max(0.0, mean - s) if self.zero_based else mean - s for mean, s in spreads if s is not None]
        anchor = [0.0] if self.zero_based else []
        lo, hi = min(values + anchor), max(values + anchor)
        if hi == lo:
            lo, hi = lo - 0.5 * (abs(lo) or 1), hi + 0.5 * (abs(hi) or 1)
        pad = (hi - lo) * 0.05
        lo, hi = (lo if self.zero_based and lo == 0 else lo - pad), hi + pad
        step = nice_step((hi - lo) / 4)
        lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
        if self.reference is not None and hi >= self.reference * 0.9:
            hi = self.reference  # close to the ceiling: show the full scale up to the maximum
            step = nice_step((hi - lo) / 4)
        return lo, hi, step

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme_color("panel")))
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(theme_color("text")))
        # Stop short of the corner control (e.g. a Hide button) so a long title never runs under it.
        corner = self.corner.width() + 8 if self.corner is not None and self.corner.isVisible() else 0
        reserved = 170 if self.reference is not None else 0
        room = self.width() - 32 - corner - reserved
        painter.drawText(16, 24, painter.fontMetrics().elidedText(self.title, Qt.TextElideMode.ElideMiddle, max(40, room)))
        area = QRectF(52, 45, self.width() - 74, self.height() - 80)
        points = [m for _, metrics, _ in self.series for m in metrics]
        xmax = self.xmax or max([m["step"] for m in points] or [10000])
        lo, hi, step = self._y_extent(points)
        xlo, xhi = self.custom_x_range if self.custom_x_range else (0.0, float(xmax))

        if self.custom_x_range:
            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(theme_color("comment")))
            painter.drawText(QRectF(16, 28, 220, 16), Qt.AlignmentFlag.AlignLeft, "Zoomed (double-click to reset)")

        def y_of(value):
            return area.bottom() - area.height() * (value - lo) / max(1e-12, hi - lo)

        def x_of(steps):
            return area.left() + area.width() * (steps - xlo) / max(1e-12, xhi - xlo)

        painter.setFont(QFont("Segoe UI", 8))
        value = lo
        while value <= hi + step / 2:
            y = y_of(value)
            painter.setPen(QColor(theme_color("raised")))
            painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))
            painter.setPen(QColor(theme_color("muted")))
            painter.drawText(QRectF(0, y - 8, 46, 16), Qt.AlignmentFlag.AlignRight, format_tick(value, step))
            value += step
        # Round x ticks (0, 25k, 50k…), not quarters of the extent: a 99,328-step budget would read 24.832k.
        x_step = nice_step((xhi - xlo) / 4)
        tick = math.ceil(xlo / x_step - 1e-9) * x_step
        while tick <= xhi + x_step * 1e-6:
            x = x_of(tick)
            painter.drawText(QRectF(x - 30, area.bottom() + 8, 60, 18), Qt.AlignmentFlag.AlignCenter,
                             format_tick(tick, x_step))
            tick += x_step
        if self.reference is not None and points:
            if hi >= self.reference:
                painter.setPen(QPen(QColor(theme_color("comment")), 1, Qt.PenStyle.DashLine))
                y = y_of(self.reference)
                painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))
                painter.drawText(QRectF(area.right() - 90, y + 2, 88, 14), Qt.AlignmentFlag.AlignRight,
                                 f"max {self.reference:g}")
            else:
                painter.setPen(QColor(theme_color("muted")))
                painter.drawText(QRectF(area.right() - 160, 12, 160, 16), Qt.AlignmentFlag.AlignRight,
                                 f"max possible {self.reference:g} ↑")

        # A held y-axis leaves headroom: a dashed line marks the highest value so far, below the top
        if self.hold_y and points:
            peak = max(m[self.metric] for m in points)
            if peak < hi - step * 0.25:
                painter.setPen(QPen(QColor(theme_color("border")), 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(area.left()), int(y_of(peak)), int(area.right()), int(y_of(peak)))

        painter.save()
        painter.setClipRect(area)

        # The part of a fixed x-extent (the training budget) the run has not reached yet
        if self.show_future and self.xmax and points:
            reached = x_of(max(m["step"] for m in points))
            if reached < area.right():
                future = QColor(theme_color("text"))
                future.setAlpha(8)
                painter.fillRect(QRectF(reached, area.top(), area.right() - reached, area.height()), future)

        # Render pinned baseline series (dashed reference line)
        for base_name, base_metrics, _ in self.baseline_series:
            if not base_metrics:
                continue
            b_path = QPainterPath()
            for i, m in enumerate(base_metrics):
                x, y = x_of(m["step"]), y_of(m[self.metric])
                if i == 0:
                    b_path.moveTo(x, y)
                else:
                    b_path.lineTo(x, y)
            painter.setPen(QPen(QColor(theme_color("comment")), 1.5, Qt.PenStyle.DashLine))
            painter.drawPath(b_path)

        for name, raw_metrics, color in self._draw_order():
            if not raw_metrics:
                continue
            metrics = self._smooth_points(raw_metrics, self.smoothing) if self.smoothing > 0 else raw_metrics

            if self.band and len(raw_metrics) > 1 and all(m.get(self.band) is not None for m in raw_metrics):
                band = QPainterPath()
                band.moveTo(x_of(raw_metrics[0]["step"]), y_of(raw_metrics[0][self.metric] + raw_metrics[0][self.band]))
                for m in raw_metrics[1:]:
                    band.lineTo(x_of(m["step"]), y_of(m[self.metric] + m[self.band]))
                for m in reversed(raw_metrics):
                    band.lineTo(x_of(m["step"]), y_of(max(lo, m[self.metric] - m[self.band])))
                band.closeSubpath()
                fill = QColor(theme_color(color))
                fill.setAlpha(35 if self.smoothing > 0 else 45)
                painter.fillPath(band, fill)

            # If smoothed, draw faint raw background line
            if self.smoothing > 0 and len(raw_metrics) > 1:
                raw_path = QPainterPath()
                for i, m in enumerate(raw_metrics):
                    x, y = x_of(m["step"]), y_of(m[self.metric])
                    if i == 0:
                        raw_path.moveTo(x, y)
                    else:
                        raw_path.lineTo(x, y)
                raw_pen_color = QColor(theme_color(color))
                raw_pen_color.setAlpha(60)
                painter.setPen(QPen(raw_pen_color, 1))
                painter.drawPath(raw_path)

            if len(metrics) == 1:
                # Single data point (e.g. fast smoke-test runs): draw glowing marker and dashed level line
                m0 = metrics[0]
                mx, my = x_of(m0["step"]), y_of(m0[self.metric])
                pen_color = QColor(theme_color(color))

                # Faint horizontal dashed line across the plot to make the value obvious
                painter.setPen(QPen(pen_color, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(area.left()), int(my), int(area.right()), int(my))

                # Glowing point marker
                glow_color = QColor(pen_color)
                glow_color.setAlpha(45)
                painter.setBrush(glow_color)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QRectF(mx - 8, my - 8, 16, 16))

                painter.setPen(QPen(pen_color, 2))
                painter.setBrush(QColor(theme_color("panel")))
                painter.drawEllipse(QRectF(mx - 5, my - 5, 10, 10))
                painter.setBrush(pen_color)
                painter.drawEllipse(QRectF(mx - 2.5, my - 2.5, 5, 5))
                painter.setBrush(Qt.BrushStyle.NoBrush)

                # Value label next to the marker
                val_text = f"{m0[self.metric]:.1f}" if abs(m0[self.metric]) >= 1 else f"{m0[self.metric]:.3f}"
                painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
                painter.setPen(pen_color)
                painter.drawText(QRectF(mx + 10, my - 10, 140, 20), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, f"{val_text} (step {int(m0['step'])})")
            else:
                path = QPainterPath()
                for i, m in enumerate(metrics):
                    x, y = x_of(m["step"]), y_of(m[self.metric])
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)

                # Subtle gradient fill underneath the curve (WandB / modern telemetry look)
                if not self.band and len(metrics) > 1:
                    area_path = QPainterPath(path)
                    area_path.lineTo(x_of(metrics[-1]["step"]), area.bottom())
                    area_path.lineTo(x_of(metrics[0]["step"]), area.bottom())
                    area_path.closeSubpath()
                    grad = QLinearGradient(0, area.top(), 0, area.bottom())
                    top_c = QColor(theme_color(color))
                    top_c.setAlpha(20 if self.smoothing > 0 else 28)
                    bot_c = QColor(theme_color(color))
                    bot_c.setAlpha(2)
                    grad.setColorAt(0.0, top_c)
                    grad.setColorAt(1.0, bot_c)
                    painter.fillPath(area_path, grad)

                painter.setPen(QPen(QColor(theme_color(color)), 3.5 if name == self.primary else 2))
                painter.drawPath(path)

                if name in self.live_series:  # the newest point, where the curve is growing
                    lx, ly = x_of(metrics[-1]["step"]), y_of(metrics[-1][self.metric])
                    halo = QColor(theme_color(color))
                    halo.setAlpha(60)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(halo)
                    painter.drawEllipse(QRectF(lx - 7, ly - 7, 14, 14))
                    painter.setBrush(QColor(theme_color(color)))
                    painter.drawEllipse(QRectF(lx - 3.5, ly - 3.5, 7, 7))
                    painter.setBrush(Qt.BrushStyle.NoBrush)

        painter.restore()

        # Selection rectangle during drag-to-zoom
        if self.zoom_start is not None and self.zoom_current is not None:
            zx1 = min(self.zoom_start.x(), self.zoom_current.x())
            zx2 = max(self.zoom_start.x(), self.zoom_current.x())
            z_rect = QRectF(zx1, area.top(), max(2.0, zx2 - zx1), area.height())
            z_fill = QColor(theme_color("primary"))
            z_fill.setAlpha(35)
            painter.fillRect(z_rect, z_fill)
            painter.setPen(QPen(QColor(theme_color("primary")), 1, Qt.PenStyle.DashLine))
            painter.drawRect(z_rect)

        # Hover point, crosshair guideline, and value tooltip card
        if self.hover_point is not None:
            hx, hy, h_step, h_val, h_name, h_color = self.hover_point
            # Vertical guideline
            painter.setPen(QPen(QColor(theme_color("raised")), 1, Qt.PenStyle.DashLine))
            painter.drawLine(int(hx), int(area.top()), int(hx), int(area.bottom()))

            # Outer glow and inner circle marker
            painter.setPen(QPen(QColor(theme_color(h_color)), 2))
            painter.setBrush(QColor(theme_color("panel")))
            painter.drawEllipse(QRectF(hx - 5, hy - 5, 10, 10))
            painter.setBrush(QColor(theme_color(h_color)))
            painter.drawEllipse(QRectF(hx - 2.5, hy - 2.5, 5, 5))

            # Floating tooltip card: the hovered value, or (hover_all) every series' nearest value
            rows = [(h_name, h_val, h_color)]
            if self.hover_all and len(self.series) > 1:
                # Only series that span the hovered step: a run that stopped earlier has no value there,
                # and its last point would read as if it did.
                rows = [(name, min(metrics, key=lambda m: abs(m["step"] - h_step))[self.metric], color)
                        for name, metrics, color in self.series
                        if metrics and min(m["step"] for m in metrics) <= h_step <= max(m["step"] for m in metrics)]
                rows = rows or [(h_name, h_val, h_color)]
            card_w = 136 if len(rows) == 1 else 220
            card_h = 46 if len(rows) == 1 else 26 + 17 * len(rows)
            cx = hx + 12 if hx + 12 + card_w <= area.right() else hx - 12 - card_w
            cy = max(area.top() + 4, min(hy - 23, area.bottom() - card_h - 4))
            card_rect = QRectF(cx, cy, card_w, card_h)

            painter.setBrush(QColor(theme_color("raised")))
            painter.setPen(QPen(QColor(theme_color("border")), 1))
            painter.drawRoundedRect(card_rect, 5, 5)

            painter.setFont(QFont("Segoe UI", 8))
            painter.setPen(QColor(theme_color("muted")))
            painter.drawText(QRectF(cx + 8, cy + 5, card_w - 16, 16), Qt.AlignmentFlag.AlignLeft, f"Step: {int(h_step):,}")
            if len(rows) == 1:
                painter.setPen(QColor(theme_color("text")))
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(QRectF(cx + 8, cy + 22, card_w - 16, 18), Qt.AlignmentFlag.AlignLeft, f"{self.title}: {h_val:.2f}")
            else:
                for i, (name, value, color) in enumerate(rows):
                    y = cy + 22 + 17 * i
                    bold = name == (self.primary or h_name)
                    painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold if bold else QFont.Weight.Normal))
                    painter.setPen(QColor(theme_color(color)))
                    painter.drawText(QRectF(cx + 8, y, card_w - 70, 16), Qt.AlignmentFlag.AlignLeft,
                                     painter.fontMetrics().elidedText(name, Qt.TextElideMode.ElideMiddle, int(card_w - 74)))
                    painter.setPen(QColor(theme_color("text")))
                    painter.drawText(QRectF(cx + card_w - 66, y, 58, 16), Qt.AlignmentFlag.AlignRight, f"{value:.4g}")

        if not points:
            painter.setPen(QColor(theme_color("muted")))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self.empty_text)


def nice_step(raw):
    """Round a raw tick interval up to 1, 2, 2.5 or 5 times a power of ten."""
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw * (1 - 1e-9))


def format_tick(value, step):
    """Tick label with just enough decimals for the tick interval (2.5-steps need one more)."""
    if abs(value) < step / 1e6:
        return "0"
    if abs(value) >= 1000 and step >= 100:
        return f"{value / 1000:g}k"
    exponent = math.floor(math.log10(step))
    mantissa = round(step / 10 ** exponent, 6)
    decimals = max(0, -exponent + (1 if mantissa == 2.5 else 0))
    return f"{value:.{decimals}f}"
