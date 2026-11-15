"""Tests for charting runs that are not reinforcement learning.

The Monitor keyed every point on `transitions`, the count of environment steps.
A supervised or unsupervised run has no environment, so it logs none, and all of
its rows were discarded: 61 rows of a real breast-cancer run became 0 points and
both charts showed "Launch training to see live metrics".

The column is now chosen once per run rather than per row, which is the whole
subtlety. Lightning writes an epoch-level duplicate of each row *without*
transitions, and those have to stay out of RL charts; a row-by-row fallback
would admit them and double every RL curve.
"""

from __future__ import annotations

import pytest

from frontend.model import COUNTER_COLUMNS, metric_points, position_column
from frontend.monitor import run_setup_label

# An RL run: evaluation rows, training rows, and Lightning's epoch duplicate.
RL_ROWS = [
    {"eval/reward": 20.0, "eval/reward_std": 8.5, "transitions": 0.0},
    {"losses/total_loss": 1.5, "losses/entropy": 0.69, "transitions": 512.0},
    {"eval/reward": 20.0, "step": 0.0},  # duplicate without transitions
    {"time/train": 0.3, "transitions": 600.0},
]

# A supervised run, as MLP component training module logs it.
SUPERVISED_ROWS = [
    {"epoch": 0.0, "step": 12.0, "train/loss": 0.60, "val/loss": 0.47},
    {"epoch": 1.0, "step": 24.0, "train/loss": 0.31, "val/loss": 0.22},
    {"epoch": 2.0, "step": 36.0, "train/loss": 0.12, "val/loss": 0.07},
]


# ---------------------------------------------------------------------------
# Choosing the column
# ---------------------------------------------------------------------------


def test_an_rl_run_is_keyed_on_transitions():
    assert position_column(RL_ROWS) == "transitions"


def test_a_supervised_run_falls_back_to_step():
    assert position_column(SUPERVISED_ROWS) == "step"


def test_epoch_is_used_when_there_is_nothing_else():
    assert position_column([{"epoch": 0.0, "val/loss": 0.5}]) == "epoch"


def test_transitions_wins_even_when_other_counters_are_present():
    """RL rows carry step and epoch too; preferring them would change the x-axis."""
    assert position_column([{"transitions": 100.0, "step": 1.0, "epoch": 0.0}]) == "transitions"


def test_the_preference_order_is_the_declared_one():
    assert COUNTER_COLUMNS == ("transitions", "step", "epoch")


def test_rows_with_no_counter_at_all_yield_nothing():
    assert position_column([{"val/loss": 0.5}]) is None
    assert metric_points([{"val/loss": 0.5}]) == []


# ---------------------------------------------------------------------------
# RL is untouched
# ---------------------------------------------------------------------------


def test_lightnings_duplicate_row_is_still_dropped():
    """The reason the choice is per run. A row-by-row fallback would keep this
    row via its `step`, double-counting every evaluation point."""
    points = metric_points(RL_ROWS)
    assert len(points) == 3
    assert [p["step"] for p in points] == [0, 512, 600]


def test_rl_points_keep_their_short_metric_keys():
    evaluation, training, _timing = metric_points(RL_ROWS)
    assert (evaluation["reward"], evaluation["reward_std"]) == (20.0, 8.5)
    assert (training["loss"], training["entropy"]) == (1.5, 0.69)


# ---------------------------------------------------------------------------
# Non-RL runs now chart
# ---------------------------------------------------------------------------


def test_every_supervised_row_becomes_a_point():
    assert len(metric_points(SUPERVISED_ROWS)) == len(SUPERVISED_ROWS)


def test_the_losses_are_carried_through_for_charting():
    points = metric_points(SUPERVISED_ROWS)
    assert [p["train/loss"] for p in points] == [0.60, 0.31, 0.12]
    assert [p["val/loss"] for p in points] == [0.47, 0.22, 0.07]


def test_points_are_placed_at_the_fallback_counter():
    assert [p["step"] for p in metric_points(SUPERVISED_ROWS)] == [12, 24, 36]


def test_the_counter_columns_are_not_offered_as_metrics():
    """epoch and step place a point; charting them against themselves is noise."""
    point = metric_points(SUPERVISED_ROWS)[0]
    assert "epoch" not in point
    assert set(COUNTER_COLUMNS) - {"step"} <= set(COUNTER_COLUMNS) - set(point)


def test_an_unsupervised_run_charts_its_reconstruction_loss():
    rows = [
        {"epoch": 0.0, "step": 10.0, "train/recon_loss": 1.04, "val/recon_loss": 0.91},
        {"epoch": 1.0, "step": 20.0, "train/recon_loss": 0.30, "val/recon_loss": 0.28},
    ]
    points = metric_points(rows)
    assert [p["val/recon_loss"] for p in points] == [0.91, 0.28]


def test_a_real_supervised_run_on_disk_charts():
    """The run that showed two empty charts: 61 rows, 0 points."""
    import csv
    from pathlib import Path

    found = sorted(Path("src/usr/results/logs").glob("*/*/*/version_*/metrics.csv"))
    non_rl = []
    for path in found:
        with open(path) as handle:
            header = handle.readline()
        if "transitions" not in header and "val/loss" in header:
            non_rl.append(path)
    if not non_rl:
        import pytest

        pytest.skip("no non-RL run on disk; run tabular_classification/quick_test to make one")
    with open(non_rl[0]) as handle:
        rows = [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in csv.DictReader(handle)]
    assert metric_points(rows), f"{non_rl[0]} still charts nothing"


# ---------------------------------------------------------------------------
# The run caption
# ---------------------------------------------------------------------------


def _run(simulated=False, env="breast_cancer", agents=("clf",)):
    return {"simulated": simulated, "backend": {"agents": list(agents)}}, {"env": env}


def test_a_supervised_run_is_labelled_with_its_own_environment_and_method():
    """It read "CartPole-v1 / PPO" for every app-launched run, so a breast-cancer
    classifier announced itself as CartPole with PPO."""
    run, config = _run()
    assert run_setup_label(run, config) == "breast_cancer  /  CLF"


def test_an_rl_run_on_another_environment_is_labelled_correctly_too():
    """The literal mislabelled these as well, not only the agent-less paradigms."""
    run, config = _run(env="seaquest", agents=("ppo",))
    assert run_setup_label(run, config) == "seaquest  /  PPO"


def test_several_methods_are_joined():
    run, config = _run(agents=("clf", "clf_deep"))
    assert run_setup_label(run, config) == "breast_cancer  /  CLF + CLF_DEEP"


def test_the_simulated_demo_run_has_a_generic_label():
    """A simulated run has no environment or method to display."""
    run, _ = _run(simulated=True)
    assert run_setup_label(run, {}) == "Simulated run"


def test_a_run_missing_its_details_degrades_rather_than_raising():
    assert run_setup_label({"simulated": False}, {}) == "?  /  ?"


# ---------------------------------------------------------------------------
# A run is described by the experiment, not by the builder's defaults
# ---------------------------------------------------------------------------


def test_builder_config_does_not_assume_an_installed_environment():
    """A fresh IDE workspace has no environment component to use as a default."""
    from frontend.model import Config

    assert (Config().env, Config().total_timesteps) == ("", 10000)


def test_composed_facts_replace_the_defaults():
    from dataclasses import replace

    from frontend.model import Config

    class _Window:
        composed_facts = {"env": "breast_cancer", "seed": 42, "total_timesteps": 10000, "paradigm": "supervised"}
        config_for_run = __import__("frontend.app", fromlist=["Window"]).Window.config_for_run

    out = _Window.config_for_run(_Window(), Config())
    assert out.env == "breast_cancer"
    assert out == replace(Config(), env="breast_cancer", seed=42, total_timesteps=10000)


def test_a_run_launched_before_composing_keeps_its_config():
    from frontend.model import Config

    class _Window:
        config_for_run = __import__("frontend.app", fromlist=["Window"]).Window.config_for_run

    assert _Window.config_for_run(_Window(), Config()) == Config()


def test_the_paradigm_is_not_written_into_the_config():
    """Config has no paradigm field; it rides on the run record instead."""
    from frontend.model import Config

    class _Window:
        composed_facts = {"env": "breast_cancer", "paradigm": "supervised", "seed": None, "total_timesteps": None}
        config_for_run = __import__("frontend.app", fromlist=["Window"]).Window.config_for_run

    out = _Window.config_for_run(_Window(), Config())
    assert out.env == "breast_cancer"
    assert not hasattr(out, "paradigm")
    assert out.seed == Config().seed, "a missing fact must not overwrite a real value"


# ---------------------------------------------------------------------------
# The step budget belongs to paradigms that step an environment
# ---------------------------------------------------------------------------


def test_rl_runs_are_measured_in_environment_steps():
    from frontend.monitor import measures_environment_steps

    for paradigm in ("online_rl", "offline_rl"):
        assert measures_environment_steps({"paradigm": paradigm}) is True


def test_agentless_paradigms_are_not():
    """They train by epochs, so a 10,000-step budget is a number they never reach."""
    from frontend.monitor import measures_environment_steps

    for paradigm in ("supervised", "unsupervised"):
        assert measures_environment_steps({"paradigm": paradigm}) is False


def test_a_run_without_a_recorded_paradigm_keeps_the_old_behaviour():
    """Records written before the paradigm was stored, and runs found on disk."""
    from frontend.monitor import measures_environment_steps

    assert measures_environment_steps({}) is True
    assert measures_environment_steps({"paradigm": "not_a_paradigm"}) is True


# ---------------------------------------------------------------------------
# The headline card follows the metric the run is judged by
# ---------------------------------------------------------------------------


def _headline(env):
    from frontend.monitor import headline_metric

    return headline_metric({}, {"env": env})


def test_rl_environments_keep_episode_reward():
    for env in ("cartpole", "seaquest"):
        key, info = _headline(env)
        assert (key, info.label) == ("reward", "Episode reward")


def test_offline_environments_headline_validation_loss():
    """breast_cancer and mimic declare monitor_metric: val/loss, which is already
    the number build_trainer selects best_model.ckpt on."""
    from frontend.config_panel.catalogs import config_tree
    tree = config_tree()
    for env in ("breast_cancer", "mimic"):
        if not tree or env not in tree.environments:
            pytest.skip(f"{env} environment uninstalled")
        key, info = _headline(env)
        assert (key, info.label) == ("val/loss", "Validation loss")


def test_the_declared_metric_is_mapped_to_its_run_record_key():
    """Environments declare the metrics.csv column; the points carry eval/reward
    under the short key `reward`, so the two have to be matched up."""
    from frontend.model import METRIC_COLUMNS

    assert METRIC_COLUMNS["reward"] == "eval/reward"
    assert _headline("cartpole")[0] == "reward"


def test_an_unknown_environment_falls_back_to_reward():
    """A run whose environment is gone looks exactly as it did before."""
    for env in ("not_an_environment", None, ""):
        assert _headline(env)[0] == "reward"


def test_every_environment_declares_what_it_is_judged_by():
    """The fallback should stay a safety net, not the usual path."""
    from frontend.config_model import ConfigTree

    tree = ConfigTree.discover()
    undeclared = [
        name
        for name, env in tree.environments.items()
        if "/" not in name and not env.raw.get("monitor_metric")
    ]
    assert not undeclared, f"environments with no monitor_metric: {sorted(undeclared)}"


def test_the_headline_value_comes_from_the_run():
    from frontend.model import latest, metric_points
    from frontend.monitor import headline_metric
    from frontend.config_panel.catalogs import config_tree

    tree = config_tree()
    if not tree or "breast_cancer" not in tree.environments:
        pytest.skip("breast_cancer environment uninstalled")

    run = {"metrics": metric_points([
        {"epoch": 0.0, "step": 13.0, "train/loss": 0.55, "val/loss": 0.41},
        {"epoch": 1.0, "step": 26.0, "train/loss": 0.04, "val/loss": 0.07},
    ])}
    key, info = headline_metric(run, {"env": "breast_cancer"})
    assert latest(run, key) == 0.07, "the card must not show train/loss"
    assert format(latest(run, key), info.fmt) == "0.0700"
