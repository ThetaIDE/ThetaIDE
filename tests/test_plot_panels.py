"""Tests for the axes-aware plotting the Plots pane composes figures from.

The refactor that split plot_metric_series into data, drawing and the
file-writing wrapper has to stay behaviour-preserving, so these pin the
drawing contract rather than the pixels: a plotter draws into an Axes the
caller owns and never creates, saves or closes a figure behind its back.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from src.app.core.plotting.base import BasePlotter, PanelSpec
from src.app.workspace import WorkspacePaths
from src.app.core.plotting.capture import capture_figures, figures_from


def frame(n=40, metric="eval/reward", seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"transitions": np.arange(n) * 10, "step": np.arange(n), metric: rng.normal(size=n).cumsum()})


class StubPlotter(BasePlotter):
    """A metric-series plotter over in-memory frames, so no logs are needed."""

    draws_metric_series = True

    def __init__(self, runs=None, metrics=("eval/reward",)):
        super().__init__("stub", paths=WorkspacePaths.for_process())
        self._runs = runs if runs is not None else {"a": {"version_0": frame()}}
        self._metrics = list(metrics)

    def load_metrics(self, group, exp_id):
        return self._runs

    def get_effective_config(self, exp_id, cli_overrides=None, exp_config_name=None):
        cfg = {"metrics": self._metrics, "smoothing_window": 3}
        cfg.update(cli_overrides or {})
        return cfg, "grp", None


# ── Drawing into an Axes the caller owns ──────────────────────────────────


def test_draw_metric_uses_the_axes_it_is_given():
    fig = Figure()
    ax = fig.add_subplot(111)
    assert StubPlotter().draw_metric(ax, "exp", "grp", "eval/reward", {"smoothing_window": 3})
    assert len(ax.lines) == 1
    assert ax.get_title()


def test_draw_metric_creates_no_figure_of_its_own():
    before = set(plt.get_fignums())
    fig = Figure()
    StubPlotter().draw_metric(fig.add_subplot(111), "exp", "grp", "eval/reward", {})
    assert set(plt.get_fignums()) == before


def test_draw_metric_reports_false_for_an_absent_metric():
    fig = Figure()
    ax = fig.add_subplot(111)
    assert not StubPlotter().draw_metric(ax, "exp", "grp", "nope/missing", {})
    assert not ax.lines


def test_two_panels_coexist_in_one_figure():
    # The whole point: several plots composed into a single paper figure.
    plotter = StubPlotter(metrics=["eval/reward", "train/loss"])
    runs = {"a": {"version_0": frame(metric="eval/reward")}, "b": {"version_0": frame(metric="train/loss")}}
    plotter._runs = runs
    fig = Figure(figsize=(8, 6))
    left, right = fig.add_subplot(121), fig.add_subplot(122)
    assert plotter.draw_metric(left, "exp", "grp", "eval/reward", {})
    assert plotter.draw_metric(right, "exp", "grp", "train/loss", {})
    assert len(fig.axes) == 2
    assert left.lines and right.lines


def test_style_config_reaches_the_axes():
    fig = Figure()
    ax = fig.add_subplot(111)
    StubPlotter().draw_metric(
        ax, "exp", "grp", "eval/reward", {"style": {"line_width": 5.0}, "title": "Custom", "xlabel": "T"}
    )
    assert ax.lines[0].get_linewidth() == 5.0
    assert ax.get_title() == "Custom"
    assert ax.get_xlabel() == "T"


def test_multi_version_runs_draw_a_sem_band():
    runs = {"a": {"version_0": frame(seed=1), "version_1": frame(seed=2)}}
    fig = Figure()
    ax = fig.add_subplot(111)
    StubPlotter(runs=runs).draw_metric(ax, "exp", "grp", "eval/reward", {})
    assert ax.collections, "mean+-SEM should add a filled band"


# ── Panels ────────────────────────────────────────────────────────────────


def test_panels_are_offered_for_metric_series_plotters():
    panels = StubPlotter(metrics=["eval/reward"]).panels("exp")
    assert len(panels) == 1
    assert isinstance(panels[0], PanelSpec)
    assert panels[0].key == "stub/eval/reward"


def test_panels_draw_when_invoked():
    panel = StubPlotter().panels("exp")[0]
    fig = Figure()
    ax = fig.add_subplot(111)
    assert panel.draw(ax)
    assert ax.lines


def test_panels_bind_their_own_metric():
    # A late-binding closure bug would make every panel draw the last metric.
    runs = {"a": {"version_0": frame(metric="eval/reward").join(frame(metric="train/loss")["train/loss"])}}
    plotter = StubPlotter(runs=runs, metrics=["eval/reward", "train/loss"])
    panels = plotter.panels("exp")
    assert [p.key for p in panels] == ["stub/eval/reward", "stub/train/loss"]
    titles = []
    for panel in panels:
        fig = Figure()
        ax = fig.add_subplot(111)
        panel.draw(ax)
        titles.append(ax.get_ylabel())
    assert titles[0] != titles[1]


def test_metrics_with_no_data_are_not_offered_as_panels():
    assert StubPlotter(metrics=["eval/reward", "never/logged"]).panels("exp").__len__() == 1


def test_plotters_that_build_their_own_figures_offer_no_panels():
    class Opaque(BasePlotter):
        def __init__(self):
            super().__init__("opaque", paths=WorkspacePaths.for_process())

    assert Opaque().panels("exp") == []


# ── Capture fallback ──────────────────────────────────────────────────────


def test_capture_keeps_figures_a_plotter_would_discard():
    with capture_figures() as figures:
        for i in range(3):
            plt.figure()
            plt.plot([1, 2, 3], [i, i + 1, i + 2])
            plt.savefig("/nowhere/ignored.png")
            plt.close()
    assert len(figures) == 3
    assert all(fig.axes for fig in figures)


def test_capture_writes_nothing(tmp_path):
    target = tmp_path / "should_not_exist.png"
    with capture_figures():
        plt.figure()
        plt.plot([1, 2], [3, 4])
        plt.savefig(target)
        plt.close()
    assert not target.exists()


def test_captured_figures_are_still_editable():
    with capture_figures() as figures:
        plt.figure()
        plt.plot([1, 2, 3], [1, 4, 9])
        plt.title("before")
        plt.savefig("/nowhere/x.png")
        plt.close()
    ax = figures[0].axes[0]
    ax.set_title("after")
    ax.lines[0].set_color("#b8bb26")
    assert ax.get_title() == "after"


def test_capture_restores_pyplot_afterwards():
    original_savefig, original_close = plt.savefig, plt.close
    with capture_figures():
        pass
    assert plt.savefig is original_savefig
    assert plt.close is original_close


def test_capture_restores_pyplot_even_when_the_block_raises():
    original_savefig = plt.savefig
    with pytest.raises(RuntimeError):
        with capture_figures():
            raise RuntimeError("boom")
    assert plt.savefig is original_savefig


def test_figures_from_swallows_the_untrue_saved_messages(capsys):
    class Noisy(BasePlotter):
        def __init__(self):
            super().__init__("noisy", paths=WorkspacePaths.for_process())

        def run(self, exp_id, cli_overrides=None):
            plt.figure()
            plt.plot([1, 2], [1, 2])
            print("  Saved: somewhere/that/never/happened.png")
            plt.savefig("/nowhere/y.png")
            plt.close()

    figures = figures_from(Noisy(), "exp")
    assert len(figures) == 1
    assert "Saved:" not in capsys.readouterr().out
