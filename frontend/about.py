"""About dialog with a shaded, software-rendered ASCII theta sculpture and animation settings."""
import math
import sys
import time

import numpy as np

from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
    QLabel, QSlider, QFrame, QComboBox
)

from .widgets import label, ToggleSlider
from .theme import theme_color


class AsciiTheta(QWidget):
    columns, rows = 76, 30
    ramp = ".,:;irsXA253hMHGS#9B&@"

    def __init__(self, parent=None, settings_manager=None):
        super().__init__(parent)
        # The frame is cached: projecting several thousand points costs ~35 ms, and widgets
        # drawn on top (the Settings card) force repaints of the area beneath them on every
        # scroll step. Those repaints now copy pixels; the projection runs only on update().
        self._frame_cache = None
        self._frame_cache_key = None
        self._held = False  # temporarily stopped by the host view; doesn't change the saved setting
        self.setMinimumSize(460, 280)
        self.setAccessibleName("Rotating three-dimensional ASCII theta logo")
        self.settings_manager = settings_manager
        if self.settings_manager is None and parent is not None:
            if hasattr(parent, "settings_manager"):
                self.settings_manager = parent.settings_manager
            elif hasattr(parent, "window") and hasattr(parent.window, "settings_manager"):
                self.settings_manager = parent.window.settings_manager

        # Initial parameter loading
        if self.settings_manager is not None:
            self.speed = getattr(self.settings_manager, "ascii_speed", 1.0)
            self.scale = getattr(self.settings_manager, "ascii_size", 1.0)
            self.thickness = getattr(self.settings_manager, "ascii_thickness", 1.0)
            self.tilt_factor = getattr(self.settings_manager, "ascii_tilt", 1.0)
            self.distance = getattr(self.settings_manager, "ascii_distance", 4.8)
            self.paused = not getattr(self.settings_manager, "ascii_animation", True)
        else:
            self.speed = 1.0
            self.scale = 1.0
            self.thickness = 1.0
            self.tilt_factor = 1.0
            self.distance = 4.8
            self.paused = False

        self.points = []
        self.rebuild_points()
        self.angle = .28
        self.last_time = time.monotonic()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.advance)

    def rebuild_points(self):
        # Elliptical ring, with a cylindrical crossbar to form a capital theta.
        r_ring = 0.18 * self.thickness
        r_bar = 0.13 * self.thickness
        points = []
        for i in range(160):
            u = math.tau * i / 160
            for j in range(28):
                v = math.tau * j / 28
                cu, su, cv, sv = math.cos(u), math.sin(u), math.cos(v), math.sin(v)
                points.append(((1 + r_ring * cv) * cu, 1.22 * (1 + r_ring * cv) * su,
                                r_ring * sv, cv * cu, cv * su / 1.22, sv))
        for i in range(100):
            for j in range(28):
                v = math.tau * j / 28
                points.append((-1 + 2 * i / 99, r_bar * math.cos(v), r_bar * math.sin(v),
                                0, math.cos(v), math.sin(v)))
        self.points = points
        self._points_array = np.array(points, dtype=float)
        self.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.last_time = time.monotonic()
        if not self.paused and not self._held:
            self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def set_paused(self, paused):
        self.paused = bool(paused)
        self.last_time = time.monotonic()
        if self.paused:
            self.timer.stop()
        elif self.isVisible() and not self._held:
            self.timer.start()
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_animation", not self.paused)
        self.update()

    def set_hold(self, held: bool):
        """Stop the animation while something covers it, without touching the user's
        animation setting; releasing the hold resumes it if the setting allows."""
        self._held = bool(held)
        if self._held:
            self.timer.stop()
        elif not self.paused and self.isVisible():
            self.last_time = time.monotonic()
            self.timer.start()

    def set_speed(self, speed: float):
        self.speed = max(0.0, float(speed))
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_speed", round(self.speed, 2))
        self.update()

    def set_scale(self, scale: float):
        self.scale = max(0.2, min(3.0, float(scale)))
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_size", round(self.scale, 2))
        self.update()

    def set_thickness(self, thickness: float):
        self.thickness = max(0.2, min(3.0, float(thickness)))
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_thickness", round(self.thickness, 2))
        self.rebuild_points()

    def set_tilt(self, tilt: float):
        self.tilt_factor = max(0.0, min(3.0, float(tilt)))
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_tilt", round(self.tilt_factor, 2))
        self.update()

    def set_distance(self, dist: float):
        self.distance = max(2.0, min(15.0, float(dist)))
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_distance", round(self.distance, 1))
        self.update()

    def reset_defaults(self):
        self.speed = 1.0
        self.scale = 1.0
        self.thickness = 1.0
        self.tilt_factor = 1.0
        self.distance = 4.8
        self.rebuild_points()
        if self.settings_manager is not None:
            self.settings_manager.set("appearance", "ascii_speed", 1.0)
            self.settings_manager.set("appearance", "ascii_size", 1.0)
            self.settings_manager.set("appearance", "ascii_thickness", 1.0)
            self.settings_manager.set("appearance", "ascii_tilt", 1.0)
            self.settings_manager.set("appearance", "ascii_distance", 4.8)
        self.update()

    def advance(self):
        now = time.monotonic()
        self.angle = (self.angle + min(now - self.last_time, .1) * .7 * self.speed) % math.tau
        self.last_time = now
        self.update()

    def frame(self):
        # Vectorized over all ~7k points: the per-point Python loop took ~22 ms a frame, which at
        # 30 fps kept most of a core busy whenever the Settings page was open.
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        tilt = (.20 + .12 * math.sin(self.angle)) * self.tilt_factor
        ct, st = math.cos(tilt), math.sin(tilt)
        dist = self.distance
        base_x = 19 * self.scale
        base_y = 8.8 * self.scale
        x, y, z, nx, ny, nz = self._points_array.T
        # Rotate about the vertical axis, then gently tilt toward the viewer.
        x, z = ca * x + sa * z, -sa * x + ca * z
        nx, nz = ca * nx + sa * nz, -sa * nx + ca * nz
        y, z = ct * y - st * z, st * y + ct * z
        ny, nz = ct * ny - st * nz, st * ny + ct * nz
        perspective = dist / np.maximum(0.1, dist - z)
        col = np.round(self.columns / 2 + x * perspective * base_x).astype(np.int64)  # both round half to even
        row = np.round(self.rows / 2 - y * perspective * base_y).astype(np.int64)
        visible = np.flatnonzero((col >= 0) & (col < self.columns) & (row >= 0) & (row < self.rows))
        if visible.size == 0:
            return {}
        # Per cell keep the nearest point (largest z); on a tie the earliest point wins, as before.
        key = row[visible] * self.columns + col[visible]
        order = np.lexsort((visible, -z[visible], key))
        _, first = np.unique(key[order], return_index=True)
        winners = visible[order[first]]
        normal_length = np.maximum(1e-6, np.sqrt(nx * nx + ny * ny + nz * nz))
        brightness = .20 + .80 * np.maximum(0, (-.35 * nx + .45 * ny + .82 * nz) / normal_length)
        top = len(self.ramp) - 1
        index = np.minimum(top, (brightness * top).astype(np.int64))
        ramp = self.ramp
        return {(int(col[i]), int(row[i])): (float(z[i]), ramp[index[i]], float(brightness[i]))
                for i in winners.tolist()}

    def update(self, *args):
        """Every state change goes through update(): drop the cached frame, then schedule a paint."""
        self._frame_cache = None
        super().update(*args)

    def paintEvent(self, event):
        ratio = self.devicePixelRatioF()
        key = (self.width(), self.height(), ratio, theme_color("base"), theme_color("text"))
        if self._frame_cache is None or self._frame_cache_key != key:
            self._frame_cache = QPixmap(max(1, round(self.width() * ratio)), max(1, round(self.height() * ratio)))
            self._frame_cache.setDevicePixelRatio(ratio)
            cache_painter = QPainter(self._frame_cache)
            self._render_frame(cache_painter)
            cache_painter.end()
            self._frame_cache_key = key
        painter = QPainter(self)
        target = QRectF(event.rect())
        painter.drawPixmap(target, self._frame_cache, self._source_rect(target, ratio))
        painter.end()

    @staticmethod
    def _source_rect(rect, ratio):
        return QRectF(rect.x() * ratio, rect.y() * ratio, rect.width() * ratio, rect.height() * ratio)

    def _render_frame(self, painter):
        painter.fillRect(self.rect(), QColor(theme_color("base")))
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        cell = min(self.width() / self.columns, self.height() / (self.rows * 1.8))
        height = cell * 1.8
        left = (self.width() - cell * self.columns) / 2
        top = (self.height() - height * self.rows) / 2
        mono_family = "Menlo" if sys.platform == "darwin" else "Consolas"
        font = QFont(mono_family)
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPixelSize(max(8, int(height)))
        painter.setFont(font)
        # Group cells by color so the pen changes four times per frame instead of once per
        # character, and draw at a computed baseline: the font is monospaced, so one offset
        # centers every glyph (much cheaper than a centered text box per character).
        metrics = QFontMetricsF(font)
        dx = (cell - metrics.horizontalAdvance("M")) / 2
        dy = (height + metrics.ascent() - metrics.descent()) / 2
        buckets = {"#ebdbb2": [], "#fabd2f": [], "#d79921": [], "#665c54": []}
        for (col, row), (_, char, brightness) in self.frame().items():
            color = "#ebdbb2" if brightness > .83 else "#fabd2f" if brightness > .48 else "#d79921" if brightness > .30 else "#665c54"
            buckets[color].append((left + col * cell + dx, top + row * height + dy, char))
        for color, glyphs in buckets.items():
            if not glyphs:
                continue
            painter.setPen(QColor(theme_color(color)))
            for x, y, char in glyphs:
                painter.drawText(QPointF(x, y), char)


class AnimationSettingsDialog(QDialog):
    """Interactive popup modal to configure ASCII Theta animation speed, scale, thickness, and 3D projection."""

    PRESETS = {
        "Default (Balanced)": {"speed": 100, "size": 100, "thickness": 100, "tilt": 100, "dist": 48},
        "Bold & Chunky": {"speed": 80, "size": 110, "thickness": 190, "tilt": 100, "dist": 48},
        "Delicate Wireframe": {"speed": 120, "size": 100, "thickness": 45, "tilt": 80, "dist": 48},
        "Dramatic 3D Tilt": {"speed": 90, "size": 120, "thickness": 120, "tilt": 190, "dist": 34},
        "Hyper Orbit": {"speed": 260, "size": 100, "thickness": 90, "tilt": 100, "dist": 48},
        "Slow Orbit": {"speed": 35, "size": 115, "thickness": 120, "tilt": 120, "dist": 48},
    }

    def __init__(self, target: AsciiTheta, settings_manager=None, parent=None):
        super().__init__(parent)
        self.target = target
        self.settings_manager = settings_manager or getattr(target, "settings_manager", None)
        self._updating_preset = False

        self.setWindowTitle("Theta Animation Settings")
        self.resize(460, 520)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(22, 20, 22, 20)
        main_layout.setSpacing(14)

        # Header title and description
        hdr = QVBoxLayout()
        hdr.setSpacing(4)
        title = label("Animation Settings", "heading")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        hdr.addWidget(title)
        subtitle = label("Fine-tune the rotating 3D ASCII sculpture in real time.", "muted")
        subtitle.setWordWrap(True)
        hdr.addWidget(subtitle)
        main_layout.addLayout(hdr)

        # Card 1: Presets & Animation Active Toggle
        ctrl_card = QFrame()
        ctrl_card.setObjectName("card")
        ctrl_layout = QVBoxLayout(ctrl_card)
        ctrl_layout.setContentsMargins(14, 12, 14, 12)
        ctrl_layout.setSpacing(10)

        row_active = QHBoxLayout()
        lbl_active_title = QLabel("Active Animation")
        lbl_active_title.setStyleSheet("font-weight: 600; font-size: 13px;")
        row_active.addWidget(lbl_active_title)
        self.status_lbl = label("Running" if not self.target.paused else "Paused", "muted")
        row_active.addWidget(self.status_lbl)
        row_active.addStretch()
        self.toggle_active = ToggleSlider(checked=not self.target.paused, parent=self)
        self.toggle_active.setToolTip("Toggle animation playback")
        self.toggle_active.toggled.connect(self._on_active_toggled)
        row_active.addWidget(self.toggle_active)
        ctrl_layout.addLayout(row_active)

        row_preset = QHBoxLayout()
        lbl_preset = QLabel("Preset Style")
        lbl_preset.setStyleSheet("font-size: 12px; color: #a89984;")
        row_preset.addWidget(lbl_preset)
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(["Custom"] + list(self.PRESETS.keys()))
        self.preset_combo.setCurrentText(self._match_initial_preset())
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        row_preset.addWidget(self.preset_combo, 1)
        ctrl_layout.addLayout(row_preset)

        main_layout.addWidget(ctrl_card)

        # Card 2: Sliders
        sliders_card = QFrame()
        sliders_card.setObjectName("card")
        sliders_layout = QVBoxLayout(sliders_card)
        sliders_layout.setContentsMargins(14, 12, 14, 12)
        sliders_layout.setSpacing(12)

        # 1. Speed Slider
        speed_init = int(round(self.target.speed * 100))
        speed_widget, self.slider_speed, self.val_speed = self._make_slider_row(
            "Rotation Speed", 0, 300, speed_init,
            lambda v: f"{v/100:.2f}x" if v > 0 else "0.00x (Frozen)",
            tooltip="Control the rotation speed multiplier"
        )
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        sliders_layout.addWidget(speed_widget)

        # 2. Size / Scale Slider
        size_init = int(round(self.target.scale * 100))
        size_widget, self.slider_size, self.val_size = self._make_slider_row(
            "Sculpture Size", 40, 180, size_init,
            lambda v: f"{v}%",
            tooltip="Control the overall scale of the 3D Theta"
        )
        self.slider_size.valueChanged.connect(self._on_size_changed)
        sliders_layout.addWidget(size_widget)

        # 3. Thickness Slider
        thick_init = int(round(self.target.thickness * 100))
        thick_widget, self.slider_thickness, self.val_thickness = self._make_slider_row(
            "Ring & Bar Thickness", 30, 250, thick_init,
            lambda v: f"{v}%",
            tooltip="Adjust the tube thickness of the outer ring and central crossbar"
        )
        self.slider_thickness.valueChanged.connect(self._on_thickness_changed)
        sliders_layout.addWidget(thick_widget)

        # 4. Tilt Angle Slider
        tilt_init = int(round(self.target.tilt_factor * 100))
        tilt_widget, self.slider_tilt, self.val_tilt = self._make_slider_row(
            "Forward Tilt", 0, 250, tilt_init,
            lambda v: f"{v/100:.2f}x",
            tooltip="Adjust the forward tilt angle toward the viewer"
        )
        self.slider_tilt.valueChanged.connect(self._on_tilt_changed)
        sliders_layout.addWidget(tilt_widget)

        # 5. Distance / Perspective Slider
        dist_init = int(round(self.target.distance * 10))
        dist_widget, self.slider_dist, self.val_dist = self._make_slider_row(
            "Perspective Depth", 25, 100, dist_init,
            lambda v: f"{v/10:.1f}",
            tooltip="Control 3D perspective projection focal depth"
        )
        self.slider_dist.valueChanged.connect(self._on_dist_changed)
        sliders_layout.addWidget(dist_widget)

        main_layout.addWidget(sliders_card)

        # Bottom Action Buttons
        btn_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Defaults")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setToolTip("Reset all animation parameters to default values")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()

        self.close_btn = QPushButton("Done")
        self.close_btn.setObjectName("primary")
        self.close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.close_btn.setDefault(True)
        self.close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(self.close_btn)

        main_layout.addLayout(btn_layout)

    def _make_slider_row(self, title: str, min_val: int, max_val: int, init_val: int, fmt_fn, tooltip: str = ""):
        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(0, 2, 0, 2)
        vbox.setSpacing(4)

        hdr = QHBoxLayout()
        name_lbl = QLabel(title)
        name_lbl.setStyleSheet("font-weight: 600; font-size: 12px;")
        val_lbl = QLabel(fmt_fn(init_val))
        val_lbl.setStyleSheet(f"font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 600; color: {theme_color('primary')};")
        hdr.addWidget(name_lbl)
        hdr.addStretch()
        hdr.addWidget(val_lbl)
        vbox.addLayout(hdr)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        slider.setCursor(Qt.CursorShape.PointingHandCursor)
        if tooltip:
            slider.setToolTip(tooltip)
        vbox.addWidget(slider)

        return container, slider, val_lbl

    def _match_initial_preset(self) -> str:
        s = int(round(self.target.speed * 100))
        sz = int(round(self.target.scale * 100))
        th = int(round(self.target.thickness * 100))
        ti = int(round(self.target.tilt_factor * 100))
        di = int(round(self.target.distance * 10))
        for name, cfg in self.PRESETS.items():
            if (cfg["speed"] == s and cfg["size"] == sz and cfg["thickness"] == th
                    and cfg["tilt"] == ti and cfg["dist"] == di):
                return name
        return "Custom"

    def _mark_custom(self):
        if not self._updating_preset:
            self.preset_combo.blockSignals(True)
            self.preset_combo.setCurrentText("Custom")
            self.preset_combo.blockSignals(False)

    def _on_active_toggled(self, active: bool):
        self.target.set_paused(not active)
        self.status_lbl.setText("Running" if active else "Paused")

    def _on_preset_changed(self, name: str):
        if name in self.PRESETS:
            cfg = self.PRESETS[name]
            self._updating_preset = True
            try:
                self.slider_speed.setValue(cfg["speed"])
                self.slider_size.setValue(cfg["size"])
                self.slider_thickness.setValue(cfg["thickness"])
                self.slider_tilt.setValue(cfg["tilt"])
                self.slider_dist.setValue(cfg["dist"])
            finally:
                self._updating_preset = False

    def _on_speed_changed(self, val: int):
        self._mark_custom()
        self.val_speed.setText(f"{val/100:.2f}x" if val > 0 else "0.00x (Frozen)")
        self.target.set_speed(val / 100.0)

    def _on_size_changed(self, val: int):
        self._mark_custom()
        self.val_size.setText(f"{val}%")
        self.target.set_scale(val / 100.0)

    def _on_thickness_changed(self, val: int):
        self._mark_custom()
        self.val_thickness.setText(f"{val}%")
        self.target.set_thickness(val / 100.0)

    def _on_tilt_changed(self, val: int):
        self._mark_custom()
        self.val_tilt.setText(f"{val/100:.2f}x")
        self.target.set_tilt(val / 100.0)

    def _on_dist_changed(self, val: int):
        self._mark_custom()
        self.val_dist.setText(f"{val/10:.1f}")
        self.target.set_distance(val / 10.0)

    def reset_to_defaults(self):
        self._updating_preset = True
        try:
            self.slider_speed.setValue(100)
            self.slider_size.setValue(100)
            self.slider_thickness.setValue(100)
            self.slider_tilt.setValue(100)
            self.slider_dist.setValue(48)
        finally:
            self._updating_preset = False
        self.preset_combo.setCurrentText("Default (Balanced)")
        self.toggle_active.setChecked(True)
        self.target.reset_defaults()


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About ThetaIDE")
        self.resize(620, 630)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = label("ThetaIDE", "brand")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        self.logo = AsciiTheta(self)
        layout.addWidget(self.logo, 1)
        buttons = QHBoxLayout()
        self.anim_btn = QPushButton("Animation Settings")
        self.anim_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.anim_btn.setToolTip("Configure ASCII Theta speed, size, thickness, and 3D effects")
        self.anim_btn.clicked.connect(self.open_animation_settings)
        buttons.addWidget(self.anim_btn)
        buttons.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        close.setDefault(True)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def open_animation_settings(self):
        dialog = AnimationSettingsDialog(target=self.logo, parent=self)
        dialog.exec()
