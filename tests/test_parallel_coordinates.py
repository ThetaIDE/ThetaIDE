"""Value-based brushing and linked sweep views, through actual mouse interactions."""
import copy
import os
import sys

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QTabWidget

from frontend.monitor.parallel_coordinates import ParallelCoordinates, category_key
from frontend.monitor.sweep_view import SweepView

app = QApplication.instance() or QApplication(sys.argv[:1])


def study():
    return {"study": "search", "objective_metric": "reward", "direction": "maximize", "trials": [
        {"id": 0, "state": "complete", "value": 10, "parameters": {"lr": .001, "batch": 32, "optimizer": "adam"}},
        {"id": 1, "state": "complete", "value": 90, "parameters": {"lr": .01, "batch": 64, "optimizer": "sgd"}},
        {"id": 2, "state": "complete", "value": 60, "parameters": {"lr": .02, "batch": 32, "optimizer": "adam"}},
        {"id": 3, "state": "complete", "value": 100, "parameters": {"lr": .03, "batch": 64, "optimizer": "sgd"}},
        {"id": 4, "state": "running", "value": None, "parameters": {"lr": .04, "optimizer": "adam"}},
        {"id": 5, "state": "complete", "value": 80, "parameters": {"lr": .005, "batch": 64, "optimizer": "adam"}},
    ]}


@pytest.fixture
def view():
    view = SweepView()
    view.set_studies([("method", study())])
    assert view.findChild(QTabWidget) is None
    view.resize(1100, 680)
    view.show()
    app.processEvents()
    view.scroll.ensureWidgetVisible(view.parallel_plot)
    app.processEvents()
    assert view.scroll.verticalScrollBar().maximum() > 0
    assert not view.table.isVisible()
    yield view
    view.close()
    view.deleteLater()


def point(chart, key, value):
    index = next(i for i, axis in enumerate(chart.axes) if axis.key == key)
    area = chart.plot_area()
    return QPoint(round(chart.axis_x(index)), round(area.bottom() - chart.axes[index].fraction(value) * area.height()))


def brush(chart, key, low, high):
    QTest.mousePress(chart, Qt.MouseButton.LeftButton, pos=point(chart, key, low))
    QTest.mouseMove(chart, point(chart, key, high))
    QTest.mouseRelease(chart, Qt.MouseButton.LeftButton, pos=point(chart, key, high))


def visible_ids(view):
    return {view.table.item(row, 0).data(Qt.ItemDataRole.UserRole) for row in range(view.table.rowCount())
            if not view.table.isRowHidden(row)}


def history_ids(view):
    return [trial for trial, value in view.plot.series[0][1]]


def test_multiple_axis_brushes_filter_history_and_table_together(view):
    row = next(row for row in range(view.table.rowCount())
               if view.table.item(row, 0).data(Qt.ItemDataRole.UserRole) == 0)
    view.table.selectRow(row)
    assert view.open_button.isEnabled()
    brush(view.parallel_plot, ("parameter", "lr"), .003, .025)
    assert visible_ids(view) == {1, 2, 5}
    assert history_ids(view) == [1, 2, 5]
    assert not view.open_button.isEnabled()  # hidden trials cannot be opened accidentally
    brush(view.parallel_plot, ("objective", "value"), 75, 100)
    assert visible_ids(view) == {1, 5}
    assert history_ids(view) == [1, 5]
    QTest.mouseClick(view.parallel_plot, Qt.MouseButton.LeftButton,
                     pos=point(view.parallel_plot, ("parameter", "optimizer"), "adam"))
    assert visible_ids(view) == {5}
    assert view.selection_count.text() == "1 / 6 trials"
    assert history_ids(view) == [5]
    assert view.plot.series[1][0] == "Best in selection"
    assert view.parallel_plot.selected_ids == {5}
    QTest.mouseClick(view.parallel_plot, Qt.MouseButton.RightButton,
                     pos=point(view.parallel_plot, ("parameter", "optimizer"), "adam"))
    assert visible_ids(view) == {1, 5}
    QTest.keyClick(view.parallel_plot, Qt.Key.Key_Escape)
    assert visible_ids(view) == set(range(6))
    assert history_ids(view) == [0, 1, 2, 3, 5]  # running trial has no objective yet


def test_text_filter_combines_with_brushes_and_clear_restores_every_view(view):
    brush(view.parallel_plot, ("parameter", "lr"), .003, .025)
    view.filter.setText("sgd")
    assert visible_ids(view) == {1}
    assert history_ids(view) == [1]
    view.filter.setText("no matching trial")
    assert visible_ids(view) == set()
    assert history_ids(view) == []
    assert view.parallel_plot.selected_ids == set()
    QTest.mouseClick(view.clear_filters, Qt.MouseButton.LeftButton)
    assert not view.filter.text()
    assert not view.parallel_plot.brushes
    assert visible_ids(view) == set(range(6))
    assert not view.clear_filters.isEnabled()


def test_live_updates_keep_actual_ranges_and_category_values_when_axes_expand(view):
    brush(view.parallel_plot, ("parameter", "lr"), .003, .025)
    QTest.mouseClick(view.parallel_plot, Qt.MouseButton.LeftButton,
                     pos=point(view.parallel_plot, ("parameter", "optimizer"), "adam"))
    ranges = copy.deepcopy(view.parallel_plot.brushes)
    updated = study()
    updated["trials"].extend([
        {"id": 6, "state": "complete", "value": 200, "parameters": {"lr": .5, "optimizer": "adadelta"}},
        {"id": 7, "state": "complete", "value": 95, "parameters": {"lr": .015, "optimizer": "adam"}},
    ])
    view.set_studies([("method", updated)])
    assert view.parallel_plot.brushes == ranges
    assert visible_ids(view) == {2, 5, 7}
    assert history_ids(view) == [2, 5, 7]
    assert view.selection_count.text() == "3 / 8 trials"


def test_filters_reset_between_studies_and_empty_selections_clear_the_chart(view):
    brush(view.parallel_plot, ("parameter", "lr"), .003, .025)
    view.filter.setText("adam")
    view.set_studies([("method", study()), ("other", dict(study(), study="other"))])
    view.selector.setCurrentIndex(1)
    assert not view.parallel_plot.brushes
    assert not view.filter.text()
    assert visible_ids(view) == set(range(6))
    view.set_studies([])
    assert not view.parallel_plot.axes
    assert not view.parallel_plot.trials
    assert visible_ids(view) == set()
    assert view.plot.series == []


def test_missing_values_and_typed_categories_are_not_confused_with_each_other():
    chart = ParallelCoordinates()
    data = {"trials": [
        {"id": 0, "parameters": {"choice": None, "flag": False, "value": 0}, "value": 1},
        {"id": 1, "parameters": {"choice": "None", "flag": True, "value": 1}, "value": 2},
        {"id": 2, "parameters": {"flag": False, "value": 2}, "value": None},
    ]}
    chart.set_study(data, "key")
    chart.brushes[("parameter", "choice")] = {category_key(None)}
    assert chart.matching_ids() == {0}
    assert ("parameter", "value") in {axis.key for axis in chart.axes}
    assert ("objective", "value") in {axis.key for axis in chart.axes}
    chart.clear_brushes()
    chart.brushes[("parameter", "flag")] = {category_key(False)}
    assert chart.matching_ids() == {0, 2}
    chart.brushes[("objective", "value")] = (0, 3)
    assert chart.matching_ids() == {0}
    chart.close()


def test_numeric_brush_keeps_its_meaning_when_mixed_categories_appear():
    chart = ParallelCoordinates()
    initial = {"trials": [{"id": 0, "parameters": {"choice": 1}, "value": 10}]}
    chart.set_study(initial, "same-study")
    chart.brushes[("parameter", "choice")] = (.5, 1.5)
    updated = copy.deepcopy(initial)
    updated["trials"].append({"id": 1, "parameters": {"choice": "1"}, "value": 20})
    chart.set_study(updated, "same-study")
    assert chart.axes[0].categories
    assert chart.matching_ids() == {0}  # the string category is not the numeric choice
    chart.resize(700, 400)
    chart.show()
    app.processEvents()
    assert not chart.grab().isNull()
    chart.close()
