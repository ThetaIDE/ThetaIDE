"""Refresh work scales with new telemetry, without losing chart state changes."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication([])

from frontend.live_run import LiveRun
from frontend.monitor.live_view import ChartsView, SystemView


def test_chart_history_reused_until_append_or_reset():
    run = LiveRun()
    run.rows = [{"step": 2, "loss": 3}, {"step": 1, "loss": 4}]
    view = ChartsView()
    view.show_lives({"a": run}, {"a": "#ffffff"}, True)
    points = view.points
    drawn = view._drawn_points[("a", "loss")]
    # A resource-only refresh must not rebuild metric history; colors still update.
    run.add([{"type": "system", "data": {"cpu_percent": 50}}])
    view.show_lives({"a": run}, {"a": "#000000"}, False)
    assert view.points is points
    assert view._drawn_points[("a", "loss")] is drawn
    assert view.grid.cards["loss"].chart.series[0][2] == "#000000"
    assert not view.grid.cards["loss"].chart.hold_y
    view._hide("loss")
    assert "loss" not in view.grid.charts()
    view._show_all()
    assert "loss" in view.grid.charts()
    run.rows.append({"step": 3, "loss": 2, "new_metric": 1})
    view.show_lives({"a": run})
    assert view.points["a"]["loss"] == [(1, 4), (2, 3), (3, 2)]
    assert "new_metric" in view.grid.charts()
    run.clear()
    run.rows = [{"transitions": 10, "loss": 9}]
    view.show_lives({"a": run})
    assert view.axis_name == "environment steps"
    assert view.points["a"]["loss"] == [(10, 9)]
    # Replacement with the same number of rows also invalidates the cache.
    run.rows = [{"step": 4, "loss": 8}]
    view.show_lives({"a": run})
    assert view.points["a"]["loss"] == [(4, 8)]
    view.show_lives({})
    assert view.points == {}
    view.close()


def test_system_refresh_does_not_relayout_unchanged_metrics(monkeypatch):
    view = SystemView()
    run = LiveRun()
    run.system.append((10, {"cpu_percent": 20}))
    view.show_lives({"a": run, "empty": LiveRun()})
    layouts = []
    original = view.grid.set_metrics
    monkeypatch.setattr(view.grid, "set_metrics", lambda *args: (layouts.append(args), original(*args)))
    run.system.append((11, {"cpu_percent": 40}))
    view.show_lives({"a": run})
    assert not layouts
    assert view.grid.cards["system/cpu_percent"].chart.series[0][1][-1]["system/cpu_percent"] == 40
    run.system.append((12, {"rss_mb": 200}))
    view.show_lives({"a": run})
    assert len(layouts) == 1
    view.close()


def test_latest_problem_preserves_timestamp_and_tie_order():
    from frontend.monitor.live_view import latest_problem
    first, second = LiveRun(), LiveRun()
    first.events = [{"level": "error", "t": 20}, {"level": "warning", "t": 1}]
    second.events = [{"level": "info", "t": 30}, {"level": "warning", "t": 20}]
    assert latest_problem({"a": first, "b": second}) == ("b", second.events[-1])
    second.events.pop()
    assert latest_problem({"a": first, "b": second}) == ("a", first.events[0])
    assert latest_problem({}) is None


def test_sparse_append_reuses_other_metric_and_handles_axis_changes():
    run = LiveRun()
    run.rows = [{'step': 2, 'loss': 3, 'accuracy': .5}]
    view = ChartsView()
    view.show_lives({'a': run})
    unchanged = view._drawn_points[('a', 'accuracy')]
    run.rows.append({'step': 1, 'loss': 4})
    run.rows.append({'step': 2, 'loss': 2})
    view.show_lives({'a': run})
    assert view._drawn_points[('a', 'accuracy')] is unchanged
    assert view.points['a']['loss'] == [(1, 4), (2, 3), (2, 2)]
    run.rows.append({'transitions': 10, 'loss': 1})
    view.show_lives({'a': run})
    assert view.axis_name == 'environment steps'
    assert view.points['a']['loss'] == [(10, 1)]
    other = LiveRun()
    other.rows = [{'step': 1, 'accuracy': .8}]
    view.show_lives({'a': run, 'b': other})
    view.show_lives({'b': other})
    assert view.axis_name == 'logged step'
    assert set(view.points) == {'b'}
    assert set(view.grid.charts()) == {'accuracy'}
    view.close()


def test_thinning_before_conversion_preserves_extrema_and_latest():
    from frontend.results.data import chart_points, thin, thin_xy
    points = [(i, float((i * 13) % 37)) for i in range(10001)]
    assert chart_points(thin_xy(points), 'loss') == thin(chart_points(points, 'loss'), 'loss')
    assert thin_xy(points)[-1] == points[-1]


def test_hidden_monitor_defers_scheduled_updates(monkeypatch):
    from types import SimpleNamespace
    from frontend.monitor.panel import MonitorPanel
    panel = MonitorPanel(SimpleNamespace(selected=None, runs=[]))
    calls = []
    monkeypatch.setattr(panel, 'refresh_live', lambda: calls.append(True))
    panel.live_changed()
    assert not panel._refresh_timer.isActive()
    panel.show()
    assert panel._refresh_timer.isActive()
    panel.hide()
    panel._refresh_if_visible()
    assert calls == []
    panel.show()
    panel._refresh_if_visible()
    assert calls == [True]
    panel.close()


def test_incremental_thinning_matches_full_rebuild_at_bucket_boundaries():
    from frontend.results.data import IncrementalXYThinner, thin_xy
    thinner = IncrementalXYThinner(limit=8)
    points = []
    for i in range(130):
        points.append((i, float((i * 13) % 37)))
        assert thinner.sample(points) == thin_xy(points, limit=8)
    del points[20:]
    assert thinner.sample(points) == thin_xy(points, limit=8)
    replacement = [(i, 2.) for i in range(100)]
    assert thinner.sample(replacement) == thin_xy(replacement, limit=8)


def test_late_metric_invalidates_incremental_thinning():
    from frontend.results.data import chart_points, thin_xy
    run = LiveRun()
    run.rows = [{'step': i, 'loss': float(i % 17)} for i in range(2200)]
    view = ChartsView()
    view.show_lives({'a': run})
    run.rows.append({'step': 3, 'loss': -100.})
    view.show_lives({'a': run})
    assert view._drawn_points[('a', 'loss')] == chart_points(thin_xy(view.points['a']['loss']), 'loss')
    view.close()
