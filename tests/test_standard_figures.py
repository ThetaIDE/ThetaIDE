"""The standard_figures Discussion figures over a seeded saved run, and the shared figure helpers.

Each test writes its own workspace (tests/discussion_fixtures.py). With Node installed, every
figure is also compiled and run in the vendored Vega and Vega-Lite builds.
"""

import json
import math

import numpy as np
import pytest

from discussion_fixtures import build_discussion_workspace, write_seeded_runs
from frontend.discussion import model
from src.app.core.plotting import figure_data
from src.app.core.plotting.figures import build_figure_spec, spec_error
from src.app.core.plotting.vega_specs import auto_text, legend_opacity, vega_lite_figure
from src.app.workspace import WorkspacePaths
from vega_support import requires_node, run_specs

FIGURES = ("convergence", "final_performance", "performance_profile", "run_variability", "losses",
           "sweep_scatter", "sweep_parallel", "metric_table")


@pytest.fixture
def paths(tmp_path, monkeypatch):
    root = build_discussion_workspace(tmp_path / "workspace")
    monkeypatch.setenv("THETA_WORKSPACE", str(root))
    return WorkspacePaths(root)


@pytest.fixture
def target(paths):
    return write_seeded_runs(paths)


def figure(paths, target, plotter, **params):
    doc = model.FigureDoc.load(model.create_figure(paths, target, plotter), paths=paths)
    for key, value in params.items():
        doc.set_param(key, value)
    spec = build_figure_spec(*doc.capability(), target, doc.values(), doc.live_params(), paths=paths)
    json.dumps(spec, allow_nan=False)  # the view receives strict JSON
    assert spec_error(spec, doc.live_params()) is None
    return doc, spec


def rows(spec):
    return spec["data"]["values"]


def params_of(spec):
    return {param["name"]: param.get("value") for param in spec["params"] if "value" in param}


# ── Every figure type ────────────────────────────────────────────────────────


def test_every_standard_figure_is_offered_with_a_label_description_and_grouped_parameters(paths, target):
    candidates = {c.selector: c for c in model.figure_candidates(paths, target)}
    for plotter in FIGURES:
        candidate = candidates[plotter]
        assert not candidate.blocked, (plotter, candidate.findings)
        assert candidate.package.id == "standard_figures" and candidate.package.version == "1.2.0"
        metadata = candidate.capability.metadata
        assert metadata["label"] and metadata["description"]
        schema = model.capability_schema(candidate.capability)
        groups = {prop.get("x-theta-group") for prop in schema["properties"].values()}
        assert groups <= {"Data", "Axes", "Style"} and None not in groups, plotter
        assert "title" in model.live_parameters(schema), plotter


@pytest.mark.parametrize("plotter", FIGURES)
def test_every_figure_builds_with_its_defaults_and_declares_its_live_params(paths, target, plotter):
    doc, spec = figure(paths, target, plotter)
    assert spec["$schema"].endswith("/vega-lite/v6.json")
    assert set(params_of(spec)) == set(doc.live_params())
    assert rows(spec)
    assert spec["title"]["text"]["expr"].startswith('title || "')  # an empty title shows the default


def test_convergence_aggregates_seeds_into_a_mean_and_band_and_can_show_runs(paths, target):
    _, spec = figure(paths, target, "convergence", metric="train/reward", smoothing="none")
    means = [row for row in rows(spec) if row["kind"] == "mean"]
    assert {row["method"] for row in means} == {"a2c", "ppo"}
    first = next(row for row in means if row["method"] == "ppo")
    # Three seeds offset by -1, 0 and +1: the mean is the middle seed, the SEM is 1/sqrt(3).
    assert first["y"] == pytest.approx(1.0)
    assert first["hi"] - first["y"] == pytest.approx(1 / math.sqrt(3))
    assert not [row for row in rows(spec) if row["kind"] == "run"]
    _, spec = figure(paths, target, "convergence", show_runs=True, methods=["ppo"])
    runs = [row for row in rows(spec) if row["kind"] == "run"]
    assert {row["method"] for row in rows(spec)} == {"ppo"} and len({row["run"] for row in runs}) == 3


@pytest.mark.parametrize("plotter", [p for p in FIGURES if not p.startswith("sweep_")])
def test_standard_metric_figures_combine_selected_sources(paths, target, plotter):
    other = write_seeded_runs(paths, "other/solo/20261117-100000")
    sources = [{"target": target, "method": "ppo", "version": "version_0"},
               {"target": other, "method": "ppo", "version": "version_2"}]
    _, spec = figure(paths, target, plotter, sources=sources, methods=["a2c"])
    assert {row["method"] for row in rows(spec)} == {f"ppo · {target}", f"ppo · {other}"}


def test_convergence_uses_only_selected_seeds_logging_the_metric(paths, target):
    other = write_seeded_runs(paths, "other/solo/20261117-100000")
    csv = paths.results / "logs" / other / "ppo" / "version_2" / "metrics.csv"
    csv.write_text("transitions,eval/other\n8,21\n16,22\n")
    sources = [{"target": target, "method": "ppo", "version": "version_0"},
               {"target": other, "method": "ppo", "version": "version_2"}]
    _, spec = figure(paths, target, "convergence", sources=sources, show_runs=True, smoothing="none",
                     metric="train/reward")
    assert {row["method"] for row in rows(spec)} == {f"ppo · {target}"}
    traces = [row for row in rows(spec) if row["kind"] == "run"]
    assert {row["run"].split(" · ")[-1] for row in traces} == {"version_0"}
    assert traces[0]["y"] == 0.0


def test_final_performance_scores_the_last_points_of_each_run(paths, target):
    _, spec = figure(paths, target, "final_performance", metric="train/reward", final_fraction=0.025)
    points = sorted(row["value"] for row in rows(spec) if row.get("kind") == "run" and row["method"] == "ppo")
    assert points == pytest.approx([39.0, 40.0, 41.0])  # the last point of each seed


def test_performance_profile_curves_fall_from_every_run_to_none(paths, target):
    _, spec = figure(paths, target, "performance_profile")
    for method in ("a2c", "ppo"):
        fractions = [row["fraction"] for row in rows(spec) if row["method"] == method]
        assert fractions[0] == 1 and fractions[-1] == 0 and fractions == sorted(fractions, reverse=True)


def test_losses_finds_every_logged_loss(paths, target):
    _, spec = figure(paths, target, "losses")
    assert {row["loss"] for row in rows(spec)} == {"losses/policy_loss", "losses/value_loss"}
    assert spec["resolve"] == {"scale": {"y": "independent"}}


def test_sweep_scatter_plots_finished_trials_and_optionally_pruned_ones(paths, target):
    _, spec = figure(paths, target, "sweep_scatter")
    assert len(rows(spec)) == 5 and {row["state"] for row in rows(spec)} == {"complete"}
    assert spec["encoding"]["x"]["scale"]["type"] == "log"  # learning rates span 100x
    assert "higher is better" in spec["layer"][0]["encoding"]["y"]["axis"]["title"]["expr"]
    _, spec = figure(paths, target, "sweep_scatter", parameter="gamma", include_pruned=True, color_by="learning_rate")
    pruned = [row for row in rows(spec) if row["state"] == "pruned"]
    assert len(pruned) == 1 and pruned[0]["value"] == 15.5  # its last intermediate value
    assert spec["encoding"]["x"]["scale"]["type"] == "linear"
    assert spec["encoding"]["color"]["field"] == "c"
    with pytest.raises(ValueError, match="swept: learning_rate, gamma"):
        figure(paths, target, "sweep_scatter", parameter="batch_size")


def test_sweep_parallel_normalises_each_axis(paths, target):
    _, spec = figure(paths, target, "sweep_parallel")
    values = [row for row in rows(spec) if row["kind"] == "value"]
    assert {row["key"] for row in values} == {"learning_rate", "gamma", "eval/reward"}
    assert all(0 <= row["y"] <= 1 for row in values)
    lr = {json.loads(row["raw"]): row["y"] for row in values if row["key"] == "learning_rate"}
    assert lr[1e-4] == 0 and lr[1e-2] == 1 and lr[1e-3] == pytest.approx(0.5)  # log-spaced


def test_sweep_figures_need_a_sweep(paths):
    candidates = {c.selector: c for c in model.figure_candidates(paths, "figs/online")}
    assert candidates["sweep_scatter"].blocked and candidates["sweep_parallel"].blocked


def test_metric_table_formats_each_cell_and_marks_the_best_method(paths, target):
    _, spec = figure(paths, target, "metric_table", final_fraction=0.025, decimals=1)
    cells = {(row["method"], row["metric"]): row for row in rows(spec)}
    assert cells[("ppo", "train/reward")]["text"] == "40.0 ± 1.0"  # standard deviation of 39, 40, 41
    assert cells[("ppo", "train/reward")]["best"] and not cells[("a2c", "train/reward")]["best"]
    _, spec = figure(paths, target, "metric_table", final_fraction=0.025, decimals=2, spread="sem")
    assert {row["text"] for row in rows(spec) if row["method"] == "ppo" and row["metric"] == "train/reward"} == {
        "40.00 ± 0.58"}
    _, spec = figure(paths, target, "metric_table", spread="none", highlight_best=False, metrics=["eval/reward"])
    assert {row["metric"] for row in rows(spec)} == {"eval/reward"}
    assert not any(row["best"] for row in rows(spec)) and "±" not in rows(spec)[0]["text"]


@requires_node
def test_every_figure_compiles_and_runs_in_the_vendored_vega(paths, target):
    items = []
    for plotter in FIGURES:
        doc, spec = figure(paths, target, plotter)
        values = doc.values()
        signals = {name: (values[name] + 1 if isinstance(values[name], (int, float)) and "opacity" not in name
                          else 0.5 if "opacity" in name else "Changed" if isinstance(values[name], str) else 1)
                   for name in doc.live_params()}
        items.append((spec, signals))
    for plotter, result in zip(FIGURES, run_specs(items)):
        assert result["ok"], (plotter, result.get("error"))
        problems = [w for w in result["warnings"] if "event source: window" not in w]
        assert not problems, (plotter, problems)
        assert "<svg" in result["svg"]


@requires_node
def test_live_title_and_axis_labels_change_the_drawn_text():
    spec = vega_lite_figure({"title": "", "x_label": ""}, ["title", "x_label"], [{"x": 1, "y": 2}],
                            title=auto_text("title", "Default title"), mark="point",
                            encoding={"x": {"field": "x", "type": "quantitative",
                                            "axis": {"title": auto_text("x_label", "Default x")}},
                                      "y": {"field": "y", "type": "quantitative"}})
    plain, changed = run_specs([(spec, {}), (spec, {"title": "Mine", "x_label": "My x"})])
    assert "Default title" in plain["svg"] and "Default x" in plain["svg"]
    assert "Mine" in changed["svg"] and "My x" in changed["svg"] and "Default" not in changed["svg"]


# ── Shared helpers ───────────────────────────────────────────────────────────


def test_aggregate_interpolates_onto_the_overlap_and_bands_widen_as_documented():
    runs = [(np.array([0, 1, 2, 3.0]), np.array([0, 1, 2, 3.0])),
            (np.array([1, 2, 3, 4.0]), np.array([2, 3, 4, 5.0]))]
    x, mean, lo, hi = figure_data.aggregate(runs, "std")
    assert x.tolist() == [1, 2, 3] and mean.tolist() == [1.5, 2.5, 3.5]
    assert (hi - mean).tolist() == pytest.approx([math.sqrt(0.5)] * 3)
    _, _, lo95, hi95 = figure_data.aggregate(runs, "ci95")
    assert (hi95 - mean) == pytest.approx(12.706 * (hi - mean) / math.sqrt(2))
    assert figure_data.aggregate(runs, "none")[2] is None and figure_data.aggregate(runs[:1])[2] is None
    with pytest.raises(ValueError):
        figure_data.aggregate(runs, "iqr")


def test_smoothing_and_statistics():
    y = np.array([0, 10, 0, 10.0])
    assert figure_data.smooth(y, "none", 5) is not None and figure_data.smooth(y, "none", 5).tolist() == y.tolist()
    assert figure_data.smooth(y, "ema", 3).tolist() == pytest.approx([0, 5, 2.5, 6.25])
    assert figure_data.iqm([1, 2, 3, 100]) == 2.5
    assert figure_data.STATISTICS["median"](np.array([1, 2, 30])) == 2
    assert figure_data.bootstrap_interval([1, 2, 3, 4]) == figure_data.bootstrap_interval([1, 2, 3, 4])  # seeded
    lo, hi = figure_data.bootstrap_interval([1, 2, 3, 4])
    assert 1 <= lo < 2.5 < hi <= 4
    assert figure_data.bootstrap_interval([7.0]) == (7.0, 7.0)


def test_metric_runs_fall_back_to_steps_when_the_quantity_is_missing():
    import pandas as pd
    data = {"ppo": {"0": pd.DataFrame({"step": [1, 2, 3], "r": [1.0, None, 3.0]})}}
    [run], label = figure_data.metric_runs(data, "r", "transitions")
    assert label == "Training steps" and run.x.tolist() == [1, 3] and run.y.tolist() == [1, 3]


def test_legend_opacity_dims_unselected_series():
    encoding = legend_opacity({"expr": "opacity"})
    assert encoding["condition"]["value"] == {"expr": "opacity"} and encoding["value"] == 0.15
