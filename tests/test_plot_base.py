"""Tests for BasePlotter load_metrics and method filtering."""
from pathlib import Path
import pandas as pd
import pytest

from plot.base import BasePlotter


class DummyPlotter(BasePlotter):
    name = "dummy"
    default_cfg = {}

    def plot(self, exp_id, cli_overrides=None, exp_config_name=None):
        pass


def test_load_metrics_filters_by_structured_methods(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.app.workspace import WorkspacePaths
    plotter = DummyPlotter("dummy", paths=WorkspacePaths(tmp_path))

    group = "test_grp"
    exp_id = "test_exp"

    # Create dummy log dirs for two methods: method_a and method_b
    log_dir_a = tmp_path / "results" / "logs" / group / exp_id / "method_a" / "version_0"
    log_dir_b = tmp_path / "results" / "logs" / group / exp_id / "method_b" / "version_0"
    log_dir_a.mkdir(parents=True)
    log_dir_b.mkdir(parents=True)

    df_a = pd.DataFrame({"step": [0, 1], "loss": [0.5, 0.4]})
    df_b = pd.DataFrame({"step": [0, 1], "loss": [0.8, 0.7]})
    df_a.to_csv(log_dir_a / "metrics.csv", index=False)
    df_b.to_csv(log_dir_b / "metrics.csv", index=False)

    # Mock get_experiment_config to return methods dict containing only method_a
    monkeypatch.setattr(
        plotter,
        "get_experiment_config",
        lambda exp_id, exp_config_name=None: {
            "methods": {
                "method_a": {"agent": "ppo"}
            }
        },
    )

    metrics = plotter.load_metrics(group, exp_id)
    assert "method_a" in metrics
    assert "method_b" not in metrics
    assert "version_0" in metrics["method_a"]


def test_get_active_aliases_and_is_method_active():
    # 1. Structured methods dict
    aliases, has_filter = BasePlotter.get_active_aliases({"methods": {"ppo": {}, "cql_tuned": {}}})
    assert has_filter is True
    assert "ppo" in aliases
    assert "cql_tuned" in aliases
    assert BasePlotter.is_method_active("ppo", aliases, has_filter)
    assert BasePlotter.is_method_active("ppo_0", aliases, has_filter)
    assert BasePlotter.is_method_active("ppo_12", aliases, has_filter)
    assert not BasePlotter.is_method_active("iql", aliases, has_filter)

    # 2. Legacy online_methods
    aliases_legacy, has_filter_legacy = BasePlotter.get_active_aliases({"online_methods": "ppo, blendrl"})
    assert has_filter_legacy is True
    assert "ppo" in aliases_legacy
    assert "blendrl" in aliases_legacy

    # 3. No filter
    aliases_none, has_filter_none = BasePlotter.get_active_aliases({})
    assert has_filter_none is False
    assert BasePlotter.is_method_active("any_method", aliases_none, has_filter_none)


def test_styling_helpers():
    from plot.base import clean_label, get_style, get_style_info

    # Default without overrides returns method name
    assert clean_label("my_algo") == "my_algo"

    # Style override for label
    assert clean_label("my_algo", style_override={"label": "Clean Label"}) == "Clean Label"

    # get_style incorporates overrides
    style = get_style("my_algo", style_override={"color": "#123456", "linestyle": "--"})
    assert style["label"] == "my_algo"
    assert style["color"] == "#123456"
    assert style["linestyle"] == "--"

    # get_style_info returns tuple
    c, ls, mk = get_style_info("my_algo", style_override={"color": "#123456", "marker": "s"})
    assert c == "#123456"
    assert ls == "-"
    assert mk == "s"


def test_distinct_colors_and_plot_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from src.app.workspace import WorkspacePaths
    plotter = DummyPlotter("convergence", paths=WorkspacePaths(tmp_path))

    group = "test_grp"
    exp_id = "test_exp"

    # Create dummy log dirs for two methods: method_one and method_two
    log_dir_1 = tmp_path / "results" / "logs" / group / exp_id / "method_one" / "version_0"
    log_dir_2 = tmp_path / "results" / "logs" / group / exp_id / "method_two" / "version_0"
    log_dir_1.mkdir(parents=True)
    log_dir_2.mkdir(parents=True)

    df_1 = pd.DataFrame({"transitions": [0, 10, 20], "eval/reward": [1.0, 2.0, 3.0]})
    df_2 = pd.DataFrame({"transitions": [0, 10, 20], "eval/reward": [0.5, 1.5, 2.5]})
    df_1.to_csv(log_dir_1 / "metrics.csv", index=False)
    df_2.to_csv(log_dir_2 / "metrics.csv", index=False)

    out_dir = tmp_path / "results" / "plots" / group / exp_id
    cfg = {
        "title": "Custom Benchmark Title",
        "xlabel": "Custom Env Steps",
        "ylabel": "Custom Return",
        "smoothing_window": 1,
        "x_axis": "transitions",
        "methods": {
            "method_one": {"label": "Method 1 Custom", "color": "#ff0000"},
            # method_two has zero style info: will use method name and auto-assigned distinct color
            "method_two": {},
        },
    }

    # Run plot_metric_series
    plotter.plot_metric_series(exp_id, group, out_dir, ["eval/reward"], cfg)

    # Verify plot was generated
    saved_plot = out_dir / "convergence" / "eval_reward.png"
    assert saved_plot.exists()
    assert saved_plot.stat().st_size > 0


def test_get_effective_config_general_and_plotter_overrides(tmp_path):
    from src.app.workspace import WorkspacePaths
    plotter = DummyPlotter("convergence", paths=WorkspacePaths(tmp_path))
    plotter.default_cfg = {"smoothing_window": 10, "xlabel": "Default Steps", "dpi": 300}

    # Mock experiment config with general plot options and plotter-specific options
    exp_cfg = {
        "group": "cartpole",
        "plots": {
            "title": "Global Experiment Title",
            "xlabel": "Global Steps",
            "convergence": {
                "smoothing_window": 3,
            },
        },
        "methods": {
            "ppo_test": {"agent": "ppo", "model": "dnn"},
        },
    }
    plotter.get_experiment_config = lambda exp_id, exp_config_name=None: exp_cfg

    effective_cfg, group, output_dir = plotter.get_effective_config("cartpole/test_exp")

    # Global plot parameters should be merged
    assert effective_cfg["title"] == "Global Experiment Title"
    assert effective_cfg["xlabel"] == "Global Steps"

    # Plotter-specific parameters should override defaults
    assert effective_cfg["smoothing_window"] == 3

    # Unmodified defaults should persist
    assert effective_cfg["dpi"] == 300

    # Methods should be passed down
    assert "methods" in effective_cfg
    assert "ppo_test" in effective_cfg["methods"]


