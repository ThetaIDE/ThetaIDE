"""Training monitor live views: status strip, every chart, system use, logs, feed and files."""

import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.app import Window  # noqa: E402
from frontend.live_run import LiveRun, LiveRunCache, format_duration  # noqa: E402
from frontend.model import disk_run  # noqa: E402
from frontend.monitor.live_view import (  # noqa: E402
    ChartsView,
    FeedView,
    FilesView,
    LogView,
    StatusStrip,
    SystemView,
)
from src.app import telemetry  # noqa: E402
from src.app.telemetry.records import record  # noqa: E402

T0 = 1_760_000_000.0


def timed(rec, t):
    rec["t"] = t
    return rec


def sample_live(budget=10000):
    """A run 40 s in: four evaluations, losses, system samples, a checkpoint and a warning."""
    live = LiveRun(budget=budget)
    records = [timed(record("start", run={"version": "version_0", "pid": 42, "started": T0}, config={}), T0)]
    for i, reward in enumerate([10.0, 30.0, 25.0, 60.0]):
        data = {"eval/reward": reward, "losses/total_loss": 2.0 / (i + 1), "transitions": 1000.0 * (i + 1)}
        records.append(timed(record("metrics", step=i, data=data), T0 + 10 * (i + 1)))
        records.append(timed(record("system", data={"cpu_percent": 100.0 + i, "rss_mb": 500.0 + i}), T0 + 10 * i + 5))
    records.append(
        timed(record("artifact", path="ckpt/best_model.ckpt", kind="checkpoint", data={"best": True}), T0 + 41)
    )
    records.append(
        timed(record("artifact", path="ckpt/best_model.ckpt", kind="checkpoint", data={"best": True}), T0 + 42)
    )
    records.append(timed(record("event", kind="non_finite", level="warning", message="losses/x is nan"), T0 + 43))
    live.add(records, now=T0 + 43)
    return live


# ── The views on their own ───────────────────────────────────────────────────


def test_status_strip_shows_speed_time_left_and_a_stall():
    strip = StatusStrip()
    lives = {"ppo": sample_live()}
    strip.show_run("running", lives, None, live=True, now=T0 + 50)
    assert strip.badge.text() == "RUNNING"
    assert strip.fields["elapsed"].text() == "50s"
    assert strip.fields["speed"].text() == "100 steps/s"  # 3,000 transitions over the last 30 s
    assert strip.fields["eta"].text() == format_duration(60)  # 6,000 left at 100/s
    assert strip.fields["updated"].text() == "7s ago"
    strip.show_run("running", lives, None, live=True, now=T0 + 43 + 600)
    assert strip.badge.text() == "STALLED" and "no new data" in strip.fields["updated"].text()
    strip.show_run("completed", lives, None, live=False, now=T0 + 600)
    assert strip.badge.text() == "COMPLETED" and strip.fields["speed"].text() == "—"


def test_charts_view_plots_every_metric_per_agent_and_hides_on_request():
    view = ChartsView()
    second = sample_live()
    view.show_lives({"ppo": sample_live(), "dqn": second})
    assert set(view.grid.cards) == {"eval/reward", "losses/total_loss"}  # counters are axes, not charts
    chart = view.grid.cards["eval/reward"].chart
    assert [name for name, _, _ in chart.series] == ["ppo", "dqn"]
    assert [p["step"] for p in chart.series[0][1]] == [1000.0, 2000.0, 3000.0, 4000.0]
    assert "x-axis: environment steps" in view.caption.text()
    view.grid.cards["losses/total_loss"].hide_button.click()
    assert set(view.grid.cards) == {"eval/reward"} and view.show_all.isVisibleTo(view)
    view.show_all.click()
    assert set(view.grid.cards) == {"eval/reward", "losses/total_loss"}


def test_system_view_plots_samples_over_time_without_hide_buttons():
    view = SystemView()
    view.show_lives({"ppo": sample_live()})
    assert list(view.grid.cards) == ["system/cpu_percent", "system/rss_mb"]
    card = view.grid.cards["system/cpu_percent"]
    assert card.chart.title.startswith("CPU") and card.hide_button.isHidden()
    assert [p["step"] for p in card.chart.series[0][1]] == [0.0, 10.0, 20.0, 30.0]
    view.show_lives({})
    assert view.grid.cards == {} and "No resource samples" in view.grid.empty.text()


def test_feed_lists_events_newest_first_with_start_and_finish():
    live = sample_live()
    live.add([timed(record("finish", status="failed", summary={}), T0 + 60)], now=T0 + 60)
    entries = FeedView.entries({"ppo": live})
    assert entries[0][2:] == ("error", "Run finished: failed")
    assert entries[-1][3] == "Run started (version_0, pid 42)"
    assert [e[3] for e in entries if e[2] == "warning"] == ["losses/x is nan"]
    assert any(e[3].startswith("New best eval/reward 60") for e in entries)
    view = FeedView()
    view.show_lives({"ppo": live})
    assert view.tree.topLevelItemCount() == len(entries)


def test_files_view_lists_each_file_once():
    view = FilesView()
    view.show_lives({"ppo": sample_live()}, None)
    assert view.tree.topLevelItemCount() == 1
    item = view.tree.topLevelItem(0)
    assert (item.text(0), item.text(1)) == ("ckpt/best_model.ckpt", "checkpoint  (best)")
    assert not view.open_folder.isEnabled()


def test_log_view_filters_problems_and_keeps_new_lines():
    view = LogView()
    view.set_lines(["start", "WARNING: slow", "Traceback (most recent call last):", "done"])
    assert view.text.toPlainText().splitlines()[0] == "start"
    view.problems_only.setChecked(True)
    assert view.text.toPlainText().splitlines() == ["WARNING: slow", "Traceback (most recent call last):"]
    view.append(["fine", "RuntimeError: boom"])
    assert view.text.toPlainText().splitlines()[-1] == "RuntimeError: boom"
    view.problems_only.setChecked(False)
    assert view.text.toPlainText().splitlines()[-2:] == ["fine", "RuntimeError: boom"]
    view.set_lines([], "nothing captured")
    assert view.empty.text() == "nothing captured"


# ── In the window ────────────────────────────────────────────────────────────


@pytest.fixture
def window():
    with tempfile.TemporaryDirectory() as data_dir:
        win = Window(Path(data_dir))
        win.jobs.logs_dir = Path(data_dir) / "logs"
        try:
            yield win
        finally:
            win.close()


def open_view(window, title):
    window.monitor.show_view(title)
    window.monitor.refresh_live()


def test_a_followed_job_fills_the_live_views_as_records_arrive(window):
    run = disk_run({"group": "g", "experiment_id": "e", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    run["status"] = "running"
    run["backend"]["job_id"] = "job1"
    window.runs.insert(0, run)
    window.select_run(run)
    live = sample_live()
    window.live_runs[("job1", "ppo")] = live
    open_view(window, "Overview")
    charts = window.monitor.charts_view
    assert set(charts.points["ppo"]) == {"eval/reward", "losses/total_loss"}
    # The headline chart leads the eval section and draws reward, so the grid leaves reward out.
    assert charts.grid.lead == ("eval", window.monitor.reward_card_frame) and "eval/reward" in charts.exclude
    assert set(charts.grid.cards) == {"losses/total_loss"}
    window.on_job_records("job1", "ppo", [record("metrics", step=9, data={"val/f1": 0.5})], False)
    window.monitor.refresh_live()
    assert set(charts.grid.cards) == {"losses/total_loss", "val/f1"}
    assert [(s.agent, s.state) for s in window.monitor.pipeline_strip.segments] == [("ppo", "running")]
    assert not window.monitor.problem_bar.isHidden()
    assert "losses/x is nan" in window.monitor.problem_text.text()
    open_view(window, "Feed")
    assert window.monitor.feed_view.tree.topLevelItemCount() >= 3


def test_a_finished_run_is_read_from_disk_when_a_live_view_opens(window):
    version_dir = window.jobs.logs_dir / "g" / "old" / "ppo" / "version_0"
    run_log = telemetry.Run(version_dir, ["csv", "stream"], system_interval=0)
    for i in range(3):
        run_log.log({"eval/reward": 10.0 * i}, step=i, transitions=100 * i)
    run_log.system({"cpu_percent": 50.0, "rss_mb": 200.0})
    run_log.finish()
    run = disk_run({"group": "g", "experiment_id": "old", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    window.runs.insert(0, run)
    window.select_run(run)
    open_view(window, "System")
    assert list(window.monitor.system_view.grid.cards) == ["system/cpu_percent", "system/rss_mb"]
    open_view(window, "Overview")
    assert list(window.monitor.charts_view.points["ppo"]) == ["eval/reward"]
    assert [(s.agent, s.state) for s in window.monitor.pipeline_strip.segments] == [("ppo", "done")]
    assert window.monitor.open_log.isHidden()  # no warnings: the bar only says it has no recorded end
    assert window.monitor.status_strip.badge.text() == run["status"].upper()
    open_view(window, "Logs")
    assert "started outside ThetaIDE" in window.monitor.log_view.empty.text()
    open_view(window, "Files")
    assert window.monitor.files_view.open_folder.isEnabled()


def test_the_disk_cache_rereads_only_changed_runs(tmp_path):
    version_dir = tmp_path / "g" / "e" / "ppo" / "version_0"
    run_log = telemetry.Run(version_dir, ["stream"], system_interval=0)
    run_log.log({"eval/reward": 1.0}, step=1)
    cache = LiveRunCache()
    first = cache.load(tmp_path, "g", "e", ["ppo"])["ppo"]
    assert cache.load(tmp_path, "g", "e", ["ppo"])["ppo"] is first
    run_log.log({"eval/reward": 2.0}, step=2)
    run_log.finish()
    again = cache.load(tmp_path, "g", "e", ["ppo"])["ppo"]
    assert again is not first and again.latest("eval/reward") == 2.0 and again.is_finished()
    assert cache.cached(tmp_path, "g", "e", ["ppo", "missing"]) == {"ppo": again}


def test_output_missed_while_another_run_was_shown_is_read_again(window):
    first = disk_run({"group": "g", "experiment_id": "a", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    second = disk_run({"group": "g", "experiment_id": "b", "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    first["backend"]["job_id"] = "job_a"
    window.runs[:0] = [first, second]
    window.jobs._logs["job_a"] = ["line 1"]
    window.select_run(first)
    open_view(window, "Logs")
    assert window.monitor.log_view.text.toPlainText() == "line 1"
    window.select_run(second)
    window.jobs._logs["job_a"].append("line 2")  # arrives while the other run is shown
    window.select_run(first)
    window.monitor.refresh_live()
    assert window.monitor.log_view.text.toPlainText().splitlines() == ["line 1", "line 2"]


# ── Runs sidebar ─────────────────────────────────────────────────────────────


def make_run(name, status, rewards=(), job_id=None):
    run = disk_run({"group": "g", "experiment_id": name, "agents": [{"name": "ppo"}], "config": {}, "metadata": {}})
    run["status"] = status
    run["metrics"] = [{"step": i, "reward": r} for i, r in enumerate(rewards)]
    if job_id:
        run["backend"]["job_id"] = job_id
    return run


def sidebar_rows(sidebar):
    tree = sidebar.tree
    return [tree.topLevelItem(i).text(0) for i in range(tree.topLevelItemCount())]


def test_sidebar_lists_live_runs_first_with_reward_trends():
    from frontend.monitor.runs_sidebar import TREND, RunsSidebar

    sidebar = RunsSidebar()
    done = make_run("done", "completed", [1.0, 5.0, 9.0])
    live = make_run("live", "running", [2.0])
    queued = make_run("waiting", "queued")
    sidebar.set_runs([done, live, queued])
    rows = sidebar_rows(sidebar)
    assert rows[0] == "LIVE AND QUEUED" and rows[3] == "RECENT"
    assert [r.split("  ")[1] for r in rows[1:3]] == ["live", "waiting"]
    assert rows[4].startswith("✓  done")
    assert sidebar.items[done["id"]].data(1, TREND) == [1.0, 5.0, 9.0]
    done["metrics"].append({"step": 3, "reward": 12.0})
    sidebar.update_run(done)
    assert sidebar.items[done["id"]].data(1, TREND)[-1] == 12.0
    sidebar.filter.setText("wait")
    assert [run_id for run_id, item in sidebar.items.items() if not item.isHidden()] == [queued["id"]]


def test_ticking_runs_compares_them_and_survives_a_refresh():
    from frontend.monitor.runs_sidebar import PALETTE, RunsSidebar

    sidebar = RunsSidebar()
    a, b, c = (make_run(n, "completed") for n in "abc")
    sidebar.set_runs([a, b, c])
    signals = []
    sidebar.compare_changed.connect(signals.append)
    sidebar.items[c["id"]].setCheckState(0, Qt.CheckState.Checked)
    sidebar.items[b["id"]].setCheckState(0, Qt.CheckState.Checked)
    assert signals == [[c["id"]], [c["id"], b["id"]]]  # recoloring rows did not signal again
    assert sidebar.colors() == {c["id"]: PALETTE[0], b["id"]: PALETTE[1]}  # in tick order: their plot colors
    sidebar.set_runs([a, c])  # b is gone; c stays ticked
    assert sidebar.compared == [c["id"]]
    assert sidebar.items[c["id"]].checkState(0) == Qt.CheckState.Checked
    sidebar.clear_comparison()
    assert sidebar.compared == [] and signals[-1] == []


def test_ticking_a_run_plots_it_without_touching_the_monitor(window):
    shown = make_run("shown", "running", [1.0], job_id="job_s")
    other = make_run("other", "completed", [2.0], job_id="job_o")
    window.runs[:0] = [shown, other]
    window.refresh_runs()
    window.select_run(shown)
    window.live_runs[("job_s", "ppo")] = sample_live()
    window.live_runs[("job_o", "ppo")] = sample_live()
    open_view(window, "Overview")
    sidebar = window.monitor.sidebar
    sidebar.items[other["id"]].setCheckState(0, Qt.CheckState.Checked)
    window.monitor.refresh_live()
    assert "eval/reward" not in window.monitor.charts_view.grid.cards  # the monitor still shows one run
    loss = window.monitor.charts_view.grid.cards["losses/total_loss"].chart
    assert [name for name, _, _ in loss.series] == ["ppo"]
    open_view(window, "System")
    assert [name for name, _, _ in window.monitor.system_view.grid.cards["system/cpu_percent"].chart.series] == ["ppo"]
    plotted = window.plot_viewer.canvas.series  # the ticked run is drawn in the Plot viewer, in its row's color
    assert any(other["id"] in name and color == sidebar.colors()[other["id"]] for name, _, color in plotted)
    sidebar.clear_comparison()
    assert window.plot_viewer.canvas.series == []  # the plot draws exactly the ticked runs
    item = sidebar.items[other["id"]]
    sidebar.tree.itemPressed.emit(item, 0)  # a click on the row's text ticks it, as its checkbox would
    sidebar.tree.itemClicked.emit(item, 0)
    assert sidebar.compared == [other["id"]]
    assert window.selected is shown  # the monitor keeps showing the current run


# ── Point inspector ──────────────────────────────────────────────────────────


def test_nearest_picks_the_closest_logged_step():
    from frontend.monitor.live_view import nearest

    points = [(1000.0, 1.0), (2000.0, 2.0), (4000.0, 4.0)]
    assert nearest(points, 2000.0) == (2000.0, 2.0)
    assert nearest(points, 2900.0) == (2000.0, 2.0)
    assert nearest(points, 3100.0) == (4000.0, 4.0)
    assert nearest(points, -5.0) == (1000.0, 1.0) and nearest(points, 9e9) == (4000.0, 4.0)
    assert nearest([], 1.0) is None


def test_chart_view_has_no_point_inspector_table():
    view = ChartsView()
    assert not hasattr(view, "inspector")
    assert view.layout().indexOf(view.grid) >= 0


def test_the_overview_scrolls_rather_than_squashing_its_charts(window):
    from frontend.monitor.panel import OVERVIEW_REWARD_HEIGHT

    monitor = window.monitor
    assert monitor.views.widget(0) is monitor.charts_view
    assert monitor.overview_scroll is monitor.charts_view.grid  # the cards and every chart scroll together
    assert monitor.overview_scroll.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded
    run = make_run("busy", "running", [1.0], job_id="job_b")
    window.runs.insert(0, run)
    window.refresh_runs()
    window.select_run(run)
    live = sample_live()
    live.add([record("metrics", step=5, data={f"losses/l{i}": 1.0 for i in range(8)} | {"transitions": 5000.0})])
    window.live_runs[("job_b", "ppo")] = live
    window.resize(1200, 500)  # too short for the cards, the headline chart and two rows of losses
    window.tabs.setCurrentWidget(window.monitor_panel)
    window.show()
    monitor.refresh_live()
    deadline = time.monotonic() + 3  # the layout can take a few passes to settle when the machine is busy
    while time.monotonic() < deadline and not monitor.overview_scroll.verticalScrollBar().maximum():
        app.processEvents()
    assert monitor.reward_chart.height() >= OVERVIEW_REWARD_HEIGHT
    assert monitor.overview_scroll.verticalScrollBar().maximum() > 0


# ── Workspace: pipeline strip, live charts, steadier numbers ─────────────────


def test_the_pipeline_strip_shows_each_method_in_order_with_its_marks():
    from frontend.monitor.pipeline_strip import pipeline_segments

    done = LiveRun()
    done.add(
        [
            timed(record("start", run={"started": T0}, config={}), T0),
            timed(record("metrics", step=1, data={"eval/reward": 1.0, "transitions": 100.0}), T0 + 10),
            timed(record("artifact", path="a.ckpt", kind="checkpoint"), T0 + 30),
            timed(record("artifact", path="a.ckpt", kind="checkpoint"), T0 + 45),  # saved again: one mark
            timed(record("event", kind="new_best", level="info", message="best"), T0 + 20),
            timed(record("event", kind="new_best", level="info", message="best"), T0 + 40),  # only the latest
            timed(record("finish", status="completed"), T0 + 60),
        ],
        now=T0 + 60,
    )
    running = LiveRun(budget=10000)
    running.add(
        [
            timed(record("start", run={"started": T0 + 60}, config={}), T0 + 60),
            timed(record("event", kind="non_finite", level="warning", message="nan"), T0 + 70),
            timed(record("metrics", step=1, data={"eval/reward": 2.0, "transitions": 2500.0}), T0 + 80),
        ],
        now=T0 + 80,
    )
    segments = pipeline_segments(["a", "b", "c"], {"a": done, "b": running}, now=T0 + 100)
    assert [(s.agent, s.state) for s in segments] == [("a", "done"), ("b", "running"), ("c", "pending")]
    a, b, c = segments
    assert a.duration == 60 and a.marks == [(0.5, "checkpoint"), (40 / 60, "best")]
    assert b.fraction == 0.25 and b.weight == 160  # 40 s for a quarter of the budget
    assert b.marks == [(10 / 160, "warning")]
    assert c.weight == (60 + 160) / 2  # nothing known yet: as long as the others on average


def test_clicking_a_method_shows_only_its_curves_until_another_run_is_shown(window):
    shown = make_run("shown", "running", job_id="job_s")
    shown["backend"]["agents"] = ["a", "b"]
    other = make_run("other", "completed", job_id="job_o")
    window.runs[:0] = [shown, other]
    window.refresh_runs()
    window.select_run(shown)
    window.live_runs[("job_s", "a")] = sample_live()
    window.live_runs[("job_s", "b")] = sample_live()
    window.live_runs[("job_o", "ppo")] = sample_live()
    monitor = window.monitor
    open_view(window, "Overview")
    assert list(monitor.charts_view.points) == ["a", "b"]
    monitor.pipeline_strip.segment_clicked.emit("b")
    assert list(monitor.charts_view.points) == ["b"]
    monitor.pipeline_strip.segment_clicked.emit("")
    assert list(monitor.charts_view.points) == ["a", "b"]
    monitor.focus_agent("a")
    window.select_run(other)
    monitor.refresh_live()
    assert monitor.focused_agent is None and list(monitor.charts_view.points) == ["ppo"]


def test_a_live_run_is_charted_across_its_whole_budget():
    from frontend.monitor.live_view import budget_extent

    live = sample_live(budget=10000)  # 4,000 of 10,000 transitions so far
    assert budget_extent({"ppo": live}, "transitions") == 10000
    assert budget_extent({"ppo": live}, "step") is None  # epochs-based: fit the data
    assert budget_extent({"ppo": sample_live(budget=None)}, "transitions") is None  # read from disk
    assert budget_extent({"ppo": sample_live(budget=3000)}, "transitions") is None  # ran past it
    view = ChartsView()
    view.show_lives({"ppo": live}, live=True)
    chart = view.grid.cards["eval/reward"].chart
    assert chart.xmax == 10000 and chart.show_future and chart.live_series == {"ppo"} and chart.hold_y
    done = sample_live(budget=10000)
    done.add([record("finish", status="completed")], now=T0 + 50)
    view.show_lives({"ppo": live, "dqn": done}, live=True)
    assert chart.live_series == {"ppo"}  # a pipeline's finished method stops growing: no marker
    view.show_lives({"ppo": live}, live=False)  # finished: same extent, nothing left to shade or mark
    assert chart.xmax == 10000 and not chart.show_future and not chart.live_series and not chart.hold_y


def test_a_live_charts_y_axis_grows_with_headroom_but_never_shrinks():
    from frontend.charts import Chart

    chart = Chart("v", "v", zero_based=False)
    chart.hold_y = True

    def extent(values, name="a"):
        chart.set_series([(name, [{"step": i, "v": v} for i, v in enumerate(values)], "#ffffff")])
        return chart._y_extent([m for _, metrics, _ in chart.series for m in metrics])

    first = extent([1.0, 2.0])
    assert extent([1.4, 1.6]) == first  # a narrower spread keeps the axis
    grown = extent([1.0, 2.0, 5.0])
    assert grown[1] >= 5.0 + 0.15 * (5.0 - first[0]) - 1e-9  # past the new point, so the next ones fit
    assert extent([1.0, 2.0, 5.0, 4.0]) == grown
    assert extent([1.4, 1.6], name="b") != grown  # other series: start over
    chart.hold_y = False
    assert extent([1.4, 1.6])[1] < grown[1]


def test_the_grid_is_laid_out_again_only_when_its_metrics_change():
    from frontend.results.plot_grid import PlotGrid

    grid = PlotGrid()
    layouts = []
    original = grid.set_metrics
    grid.set_metrics = lambda metrics, empty="": (layouts.append(list(metrics)), original(metrics, empty))
    grid.show_metrics(["eval/reward", "losses/a"])
    grid.show_metrics(["eval/reward", "losses/a"])
    grid.show_metrics(["eval/reward", "losses/a", "losses/b"])
    assert layouts == [["eval/reward", "losses/a"], ["eval/reward", "losses/a", "losses/b"]]


def test_the_speed_is_eased_between_readings():
    strip = StatusStrip()
    live = sample_live()
    strip.show_run("running", {"ppo": live}, None, live=True, now=T0 + 50)
    assert strip.fields["speed"].text() == "100 steps/s"
    live.add([timed(record("metrics", step=4, data={"transitions": 7000.0}), T0 + 50)], now=T0 + 50)
    assert live.rate() == 150  # 6,000 transitions over the last 40 s
    strip.show_run("running", {"ppo": live}, None, live=True, now=T0 + 51)
    assert strip.fields["speed"].text() == "115 steps/s"  # 0.7 × 100 + 0.3 × 150


# ── Layout of design E ───────────────────────────────────────────────────────


def test_the_headline_chart_leads_the_eval_section_and_ticks_leave_the_monitor_alone(window):
    from frontend.monitor.panel import SECTION_COLUMNS

    shown = make_run("shown", "running", [1.0], job_id="job_s")
    other = make_run("other", "completed", [2.0], job_id="job_o")
    window.runs[:0] = [shown, other]
    window.refresh_runs()
    window.select_run(shown)
    window.live_runs[("job_s", "ppo")] = sample_live()
    window.live_runs[("job_o", "ppo")] = sample_live()
    monitor = window.monitor
    open_view(window, "Overview")
    grid = monitor.charts_view.grid
    position = grid.grid.getItemPosition(grid.grid.indexOf(monitor.reward_card_frame))
    total = grid.grid.columnCount()
    assert position[1] == 0 and position[3] == total  # first of the eval section; alone in it, so full width
    losses = grid.cards["losses/total_loss"]
    assert grid.grid.getItemPosition(grid.grid.indexOf(losses))[3] == total // SECTION_COLUMNS["losses"]
    monitor.sidebar.items[other["id"]].setCheckState(0, Qt.CheckState.Checked)
    monitor.refresh_live()
    assert grid.grid.indexOf(monitor.reward_card_frame) != -1  # ticked runs are compared in the Plots tab
    assert "eval/reward" not in grid.cards


def test_the_eval_section_starts_open_and_can_be_folded(window):
    run = make_run("shown", "running", [1.0], job_id="job_s")
    window.runs[:0] = [run]
    window.refresh_runs()
    window.select_run(run)
    window.live_runs[("job_s", "ppo")] = sample_live()
    monitor = window.monitor
    open_view(window, "Overview")
    grid = monitor.charts_view.grid
    assert grid.grid.indexOf(monitor.reward_card_frame) != -1
    assert grid.grid.indexOf(grid.cards["losses/total_loss"]) != -1
    grid.toggle_group("eval")
    monitor.refresh_live()
    assert grid.grid.indexOf(monitor.reward_card_frame) == -1


def test_the_eval_section_is_open_even_when_it_is_the_only_section(window):
    from frontend.model import new_run, sample, Config

    run = new_run(window.config())  # simulated: its only chart is the headline one, so the grid has no cards
    run["metrics"] = [sample(Config(**run["config"]), k + 1) for k in range(5)]
    window.runs.insert(0, run)
    window.refresh_runs()
    window.select_run(run)
    open_view(window, "Overview")
    grid = window.monitor.charts_view.grid
    assert grid.grid.indexOf(window.monitor.reward_card_frame) != -1


def test_the_status_row_opens_the_other_views_in_place_of_the_charts(window):
    monitor = window.monitor
    assert monitor.views.currentWidget() is monitor.charts_view
    monitor.view_buttons["Logs"].click()
    assert monitor.views.currentWidget() is monitor.log_view and monitor.view_buttons["Logs"].isChecked()
    monitor.view_buttons["Feed"].click()
    assert monitor.views.currentWidget() is monitor.feed_view and not monitor.view_buttons["Logs"].isChecked()
    monitor.view_buttons["Feed"].click()  # again: back to the charts
    assert monitor.views.currentWidget() is monitor.charts_view
    assert not any(button.isChecked() for button in monitor.view_buttons.values())
    monitor.show_view("System")
    monitor.open_log.click()
    assert monitor.views.currentWidget() is monitor.log_view


def test_the_bottom_bar_says_what_a_run_waits_for_and_then_shows_warnings(window):
    run = make_run("waiting", "queued", job_id="job_q")
    window.runs.insert(0, run)
    window.refresh_runs()
    window.select_run(run)
    monitor = window.monitor
    monitor.refresh_live()
    assert not monitor.problem_bar.isHidden() and "job queue" in monitor.problem_text.text()
    assert monitor.open_log.isHidden()
    run["status"] = "running"
    run["metrics"] = [{"step": 1, "reward": 1.0}]
    window.live_runs[("job_q", "ppo")] = sample_live()  # it has warned
    window.render_run()
    monitor.refresh_live()
    assert "losses/x is nan" in monitor.problem_text.text() and not monitor.open_log.isHidden()


def test_sidebar_rows_say_how_far_along_each_run_is():
    from frontend.monitor.runs_sidebar import DETAIL, RunsSidebar, run_detail

    live = make_run("live", "running", [1.0, 2.0])
    live["metrics"][-1]["step"] = 6200
    live["backend"]["effective_timesteps"] = 10000
    assert run_detail(live) == "training · 62%"
    assert run_detail(make_run("done", "completed")) == "done"
    assert run_detail(make_run("q", "queued")) == "queued"
    sidebar = RunsSidebar()
    sidebar.set_runs([live])
    assert sidebar.items[live["id"]].data(0, DETAIL) == "training · 62%"


def test_the_status_row_shows_speed_while_training_and_the_last_write_after():
    strip = StatusStrip()
    lives = {"ppo": sample_live()}
    strip.show_run("running", lives, None, live=True, now=T0 + 50)
    shown = [key for key, field in strip.fields.items() if not field.isHidden()]
    assert shown == ["elapsed", "speed", "eta"]
    strip.show_run("running", lives, None, live=True, now=T0 + 43 + 600)  # stalled: say since when
    assert not strip.fields["updated"].isHidden()
    strip.show_run("completed", lives, None, live=False, now=T0 + 600)
    assert [key for key, field in strip.fields.items() if not field.isHidden()] == ["elapsed", "updated"]
