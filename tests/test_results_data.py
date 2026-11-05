"""Results data layer (frontend/results/data.py): pure logic, no Qt."""

import pytest

from frontend.results.data import (
    Manifest,
    MetricTable,
    Source,
    aggregate,
    chart_points,
    config_differences,
    decode_text,
    flatten,
    group_metrics,
    read_csv_rows,
    smooth,
    summarize,
    summarize_metric,
    to_float,
)
from frontend.results.metrics_catalog import describe

# Lightning PPO: evaluation rows carry eval/* and transitions; each epoch also gets a row
# with losses/* and an epoch-only duplicate without transitions.
LIGHTNING_ROWS = [
    {"epoch": "", "eval/reward": "20.0", "eval/reward_std": "2.0", "losses/total_loss": "", "transitions": "1000.0"},
    {"epoch": "0", "eval/reward": "", "eval/reward_std": "", "losses/total_loss": "0.5", "transitions": "1000.0"},
    {"epoch": "0", "eval/reward": "1020.0", "eval/reward_std": "", "losses/total_loss": "", "transitions": ""},
    {"epoch": "", "eval/reward": "40.0", "eval/reward_std": "3.0", "losses/total_loss": "", "transitions": "2000.0"},
    {"epoch": "1", "eval/reward": "", "eval/reward_std": "", "losses/total_loss": "0.25", "transitions": "2000.0"},
]

# Plugin runner metrics are retained as generic columns under losses/.
SB3_ROWS = [
    {
        "step": "1000",
        "transitions": "1000",
        "eval/reward": "15",
        "train/reward": "15",
        "losses/train/value_loss": "3.5",
        "losses/train/policy_gradient_loss": "-0.01",
    },
    {
        "step": "2000",
        "transitions": "2000",
        "eval/reward": "35",
        "train/reward": "35",
        "losses/train/value_loss": "2.0",
        "losses/train/policy_gradient_loss": "-0.02",
    },
]


def test_every_numeric_column_is_a_metric_including_engine_specific_names():
    table = MetricTable.from_rows(SB3_ROWS)
    # Nothing is whitelisted: SB3's own loss names come through untouched.
    assert set(table.metrics()) == {
        "eval/reward",
        "train/reward",
        "losses/train/value_loss",
        "losses/train/policy_gradient_loss",
    }
    assert table.axes() == ["transitions", "step"]


def test_metrics_are_ordered_by_group_then_name():
    table = MetricTable.from_rows(SB3_ROWS + [{"time/total": "1", "zeta": "2", "val/loss": "3"}])
    assert table.metrics() == [
        "eval/reward",
        "train/reward",
        "losses/train/policy_gradient_loss",
        "losses/train/value_loss",
        "val/loss",
        "time/total",
        "zeta",
    ]
    assert list(group_metrics(table.metrics())) == ["eval", "train", "losses", "val", "time", "other"]


def test_series_keeps_only_rows_with_both_values():
    table = MetricTable.from_rows(LIGHTNING_ROWS)
    # The epoch-only duplicate (reward 1020, no transitions) is not plotted.
    assert table.series("eval/reward") == [(1000.0, 20.0), (2000.0, 40.0)]
    assert table.series("losses/total_loss") == [(1000.0, 0.5), (2000.0, 0.25)]
    assert table.series("losses/total_loss", x="epoch") == [(0.0, 0.5), (1.0, 0.25)]
    assert table.series("missing") == []


def test_series_without_any_axis_uses_position():
    table = MetricTable.from_rows([{"score": "1"}, {"score": ""}, {"score": "3"}])
    assert table.series("score") == [(0.0, 1.0), (1.0, 3.0)]


def test_text_and_empty_columns_are_not_metrics():
    table = MetricTable.from_rows([{"name": "ppo", "blank": "", "nan": "nan", "r": "1"}])
    assert list(table.columns) == ["r"]
    assert [to_float(v) for v in ("inf", "x", None, True, "2.5")] == [None, None, None, None, 2.5]


def test_value_at_merges_every_row_for_a_point():
    table = MetricTable.from_rows(LIGHTNING_ROWS)
    # Reward and loss for transitions=2000 sit on different rows; the inspector sees both.
    assert table.value_at("transitions", 2000.0) == {
        "eval/reward": 40.0,
        "eval/reward_std": 3.0,
        "transitions": 2000.0,
        "epoch": 1.0,
        "losses/total_loss": 0.25,
    }


def test_extend_appends_new_rows_and_new_columns():
    table = MetricTable.from_rows(SB3_ROWS[:1])
    table.extend([{"transitions": "2000", "eval/reward": "35", "new/metric": "9"}])
    assert table.row_count == 2
    assert table.series("eval/reward") == [(1000.0, 15.0), (2000.0, 35.0)]
    assert table.series("new/metric") == [(2000.0, 9.0)]
    assert table.columns["losses/train/value_loss"] == [3.5, None]


def test_smooth_matches_the_monitor_ema():
    points = [(0, 0.0), (1, 10.0), (2, 10.0)]
    assert smooth(points, 0) == points
    assert smooth(points, 0.5) == [(0, 0.0), (1, 5.0), (2, 7.5)]


def test_aggregate_across_seeds():
    seeds = [[(1, 10.0), (2, 20.0)], [(1, 14.0), (2, 20.0)], [(1, 12.0)]]
    first, second = aggregate(seeds)
    assert (first.x, first.mean, first.n) == (1, 12.0, 3)
    assert first.std == pytest.approx(2.0) and first.sem == pytest.approx(2.0 / 3**0.5)
    assert (second.mean, second.std, second.n) == (20.0, 0.0, 2)


def test_summarize_respects_metric_direction():
    reward = summarize([(0, 1.0), (10, 5.0), (20, 3.0)], higher_is_better=True)
    assert (reward.final, reward.best, reward.best_x, reward.auc) == (3.0, 5.0, 10, 70.0)
    loss = summarize([(0, 1.0), (10, 0.2), (20, 0.4)], higher_is_better=False)
    assert (loss.best, loss.best_x) == (0.2, 10)
    assert summarize([]) is None
    table = MetricTable.from_rows(SB3_ROWS)
    assert summarize_metric(table, "losses/train/value_loss").best == 2.0  # a loss: lower wins


def test_catalog_describes_known_and_unknown_metrics():
    assert describe("eval/reward").label == "Episode reward" and describe("eval/reward").higher_is_better
    assert describe("losses/train/value_loss").higher_is_better is False
    unknown = describe("custom/episode_success_rate")
    assert unknown.label == "Episode success rate" and unknown.group == "custom"
    assert describe("val/huber_loss").higher_is_better is False
    assert describe("val/auroc").higher_is_better is True
    assert describe("x/temperature").higher_is_better is None


def test_manifest_groups_artifacts():
    manifest = Manifest.from_api(
        {
            "group": "g",
            "experiment_id": "e",
            "agents": {"ppo": ["version_0", "version_1"]},
            "artifacts": [
                {"path": "logs/g/e/ppo/version_0/metrics.csv", "kind": "metrics"},
                {"path": "plots/g/e/a.png", "kind": "figure"},
                {"path": "plots/g/e/b.png", "kind": "figure"},
            ],
        }
    )
    assert manifest.run_key == "g/e"
    assert manifest.kinds() == {"metrics": 1, "figure": 2}
    assert [a["path"] for a in manifest.by_kind("figure")] == ["plots/g/e/a.png", "plots/g/e/b.png"]
    assert manifest.metric_sources() == [("ppo", "version_0"), ("ppo", "version_1")]


def test_source_labels_and_chart_points():
    assert Source("g", "exp", "ppo").label == "exp / ppo"
    assert Source("g", "exp", "ppo", "version_2").label == "exp / ppo (version_2)"
    assert Source("g", "exp", "ppo").run_key == "g/exp"
    assert chart_points([(1.0, 2.0)], "eval/reward") == [{"step": 1.0, "eval/reward": 2.0}]


def test_nearest_x_picks_the_closest_logged_point():
    table = MetricTable.from_rows(LIGHTNING_ROWS)
    assert table.nearest_x("transitions", 1400) == 1000.0
    assert table.nearest_x("transitions", 1600) == 2000.0
    assert table.nearest_x("missing", 5) is None


def test_read_csv_rows_keeps_text_pads_short_rows_and_caps_rows():
    text = "﻿Method,Time,Note\nppo,14.8,fast\niql,9.5\n\nsac,3,x\n"
    preview = read_csv_rows(text, limit=2)
    assert preview.headers == ["Method", "Time", "Note"]  # byte-order mark removed
    assert preview.rows == [["ppo", "14.8", "fast"], ["iql", "9.5", ""]]
    assert (preview.total_rows, preview.truncated) == (3, True)
    assert decode_text(b"\xef\xbb\xbfa,b") == "a,b"


def test_flatten_and_config_differences():
    assert flatten({"env": {"name": "cp", "kw": {}}, "seeds": [1, 2]}) == {"env.name": "cp", "env.kw": {},
                                                                          "seeds": [1, 2]}
    a = {"seed": 1, "env": {"name": "cartpole"}, "methods": {"ppo": {"lr": 0.001}}}
    b = {"seed": 2, "env": {"name": "cartpole"}, "methods": {"iql": {"lr": 0.01}}}
    # Identical settings are hidden; a setting only one run has shows None for the other.
    assert config_differences({"a": a, "b": b}) == [("methods.iql.lr", [None, 0.01]),
                                                    ("methods.ppo.lr", [0.001, None]),
                                                    ("seed", [1, 2])]
    # One run: every setting, so the diff view doubles as a flat config view.
    assert [key for key, _ in config_differences({"a": a})] == ["env.name", "methods.ppo.lr", "seed"]


def test_series_groups_combine_an_agents_versions():
    from frontend.results.data import series_groups

    a0, a1 = Source("g", "e", "ppo", "version_0"), Source("g", "e", "ppo", "version_1")
    b = Source("g", "e", "iql", "version_0")
    assert series_groups([a0, a1, b], combine=False) == [(a0.label, [a0]), (a1.label, [a1]), (b.label, [b])]
    assert series_groups([a0, a1, b], combine=True) == [("e / ppo (mean of 2)", [a0, a1]), (b.label, [b])]


def test_seed_curves_from_different_trials_are_never_combined():
    from frontend.results.data import series_groups

    sources = [Source("g", "e", "ppo", f"version_{trial}_{seed}",
                      f"search · trial {trial} · seed {seed}", f"trial_{trial}")
               for trial in (35, 36) for seed in (11, 22)]
    groups = series_groups(sources, combine=True)
    assert len(groups) == 2
    assert groups[0] == ("e / ppo (search · trial 35 · mean of 2)", sources[:2])
    assert groups[1] == ("e / ppo (search · trial 36 · mean of 2)", sources[2:])


def test_combined_points_and_first_reaching():
    from frontend.results.data import SPREAD, combined_points, first_reaching

    points = combined_points([[(0, 10.0), (1, 20.0)], [(0, 14.0), (1, 20.0)]], "r", "std")
    assert [(p["step"], p["r"], p["n"]) for p in points] == [(0, 12.0, 2), (1, 20.0, 2)]
    assert points[0][SPREAD] == pytest.approx(2.828, abs=1e-3) and points[1][SPREAD] == 0.0
    curve = [(0, 1.0), (10, 5.0), (20, 9.0)]
    assert first_reaching(curve, 5.0) == 10 and first_reaching(curve, 10.0) is None
    assert first_reaching([(0, 3.0), (5, 0.5)], 1.0, higher_is_better=False) == 5  # a loss getting below 1


def test_summary_rows_and_csv():
    from frontend.results.data import SUMMARY_COLUMNS, summary_rows, to_csv

    rows = summary_rows([("run", "eval/reward", [(0, 1.0), (10, 3.0)]), ("run", "empty", [])], threshold=2.0)
    assert rows == [["run", "eval/reward", 3.0, 3.0, 10, 1.0, 3.0, 2.0, 20.0, 2, 10]]  # empty series skipped
    # No direction (e.g. time): no "best", min and max only.
    (timing,) = summary_rows([("run", "time/total", [(0, 1.0), (10, 3.0)])])
    assert timing[3:7] == [None, None, 1.0, 3.0]
    assert to_csv(list(SUMMARY_COLUMNS[:3]), [["a, b", "m", None]]) == 'Series,Metric,Final\n"a, b",m,\n'


def test_plot_viewer_reads_csv_text_like_files(tmp_path):
    from frontend.plots import read_metrics_csv, read_metrics_csv_text

    text = "step,reward,name\n0,1.5,ppo\n1,2.5,ppo\n"
    path = tmp_path / "m.csv"
    path.write_text(text, encoding="utf-8")
    assert read_metrics_csv_text(text) == read_metrics_csv(path) == {"step": [0.0, 1.0], "reward": [1.5, 2.5]}
    assert read_metrics_csv_text("﻿" + text) == read_metrics_csv(path)
