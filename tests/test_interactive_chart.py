import os
import unittest
import pytest

pytest.importorskip("PyQt6")

from PyQt6.QtCore import Qt, QPointF, QEvent
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

# Headless platform setup for offscreen testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication([])

from frontend.charts.chart import Chart


class TestInteractiveChart(unittest.TestCase):
    def setUp(self):
        self.chart = Chart("reward", "Episode reward")
        self.chart.resize(400, 300)
        self.series_data = [
            ("ppo", [{"step": 0, "reward": 10.0}, {"step": 500, "reward": 25.0}, {"step": 1000, "reward": 45.0}], "primary")
        ]
        self.chart.set_series(self.series_data, xmax=1000)

    def test_mouse_tracking_enabled(self):
        self.assertTrue(self.chart.hasMouseTracking())
        self.assertIsNone(self.chart.hover_point)
        self.assertIsNone(self.chart.custom_x_range)

    def test_box_zoom_and_double_click_reset(self):
        # Press at (100, 100)
        press_ev = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(100, 100),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.chart.mousePressEvent(press_ev)
        self.assertIsNotNone(self.chart.zoom_start)

        # Release at (260, 100)
        release_ev = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(260, 100),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.chart.mouseReleaseEvent(release_ev)
        self.assertIsNone(self.chart.zoom_start)
        self.assertIsNotNone(self.chart.custom_x_range)
        xlo, xhi = self.chart.custom_x_range
        self.assertTrue(0 <= xlo < xhi <= 1000)

        # Double click to reset
        dbl_ev = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            QPointF(200, 100),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.chart.mouseDoubleClickEvent(dbl_ev)
        self.assertIsNone(self.chart.custom_x_range)

    def test_single_point_render(self):
        """Ensure Chart renders cleanly with a single data point without crashing or errors."""
        from PyQt6.QtGui import QPixmap
        self.chart.set_series([("single", [{"step": 0, "reward": 31.5}], "primary")], xmax=640)
        pixmap = QPixmap(self.chart.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        self.chart.render(pixmap)
        self.assertFalse(pixmap.isNull())


if __name__ == "__main__":
    unittest.main()
