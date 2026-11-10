"""telemetry.metrics: an experiment records only the metrics it lists, and skips the work for the rest."""

import csv
import json

import pytest

from src.app.telemetry import settings_from_config
from src.app.telemetry.selection import MetricSelection, selection_errors, selection_from_config


def test_names_and_sections_are_kept_and_counters_always_are():
    selection = MetricSelection(["eval/reward", "losses/"])
    row = {"eval/reward": 1, "eval/reward_std": 2, "losses/total_loss": 3, "train/reward": 4,
           "step": 5, "epoch": 6, "transitions": 7}
    assert selection.filter(row) == {"eval/reward": 1, "losses/total_loss": 3, "step": 5, "epoch": 6,
                                     "transitions": 7}
    assert selection.keeps_section("eval") and selection.keeps_section("losses")
    assert not selection.keeps_section("system") and not selection.keeps_section("train")


def test_no_list_records_everything():
    for cfg in ({}, {"telemetry": {}}, {"telemetry": {"metrics": []}}):
        selection = selection_from_config(cfg)
        assert selection.everything and selection.keeps("anything/at_all") and selection.keeps_section("system")


def test_system_sampling_is_off_unless_a_system_metric_is_listed():
    assert settings_from_config({"telemetry": {"metrics": ["eval/reward"]}})[1] == 0.0
    assert settings_from_config({"telemetry": {"metrics": ["system/"], "system_interval": 5}})[1] == 5.0
    assert settings_from_config({})[1] > 0  # unchanged default when nothing is listed


@pytest.mark.parametrize("metrics, problem", [
    ("eval/reward", None),  # a single name is fine
    ([3], "must be a list"),
    (["  "], "must be a list"),
])
def test_the_list_is_checked(metrics, problem):
    errors = selection_errors({"telemetry": {"metrics": metrics}})
    assert (problem in errors[0]) if problem else errors == []


def test_a_sweep_must_keep_the_metric_it_optimises():
    cfg = {"telemetry": {"metrics": ["losses/"]}, "hydra": {"sweeper": {"direction": "maximize"}}}
    (error,) = selection_errors(cfg, "eval/reward")
    assert "eval/reward" in error
    assert selection_errors({**cfg, "telemetry": {"metrics": ["eval/"]}}, "eval/reward") == []


def test_the_run_records_only_listed_metrics(tmp_path):
    from src.app.telemetry.run import Run

    run = Run(tmp_path, ["csv", "stream"], metrics=MetricSelection(["losses/total_loss"]), system_interval=0)
    run.log({"losses/total_loss": 0.5, "losses/value_loss": 0.7, "eval/reward": 9}, step=1, transitions=128)
    run.finish()
    header = next(csv.reader(open(tmp_path / "metrics.csv", encoding="utf-8")))
    assert set(header) == {"step", "losses/total_loss", "transitions"}
    streamed = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    (row,) = [r for r in streamed if r["type"] == "metrics"]
    assert set(row["data"]) == {"losses/total_loss", "transitions"}


def test_lightning_loggers_record_only_listed_metrics(tmp_path):
    pytest.importorskip("lightning")
    from omegaconf import OmegaConf

    from src.app.core.lightning_builder import setup_loggers

    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "ppo"},
                            "telemetry": {"metrics": ["losses/total_loss"]}})
    csv_logger = setup_loggers(cfg, str(tmp_path))[0]
    csv_logger.log_metrics({"losses/total_loss": 1.0, "losses/value_loss": 2.0}, step=0)
    csv_logger.log_metrics({"losses/value_loss": 2.0}, step=1)  # nothing kept: no row at all
    csv_logger.save()
    rows = list(csv.DictReader(open(f"{csv_logger.log_dir}/metrics.csv", encoding="utf-8")))
    assert [set(r) for r in rows] == [{"losses/total_loss", "step"}] and len(rows) == 1


def test_no_eval_metric_skips_evaluation_episodes(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    from omegaconf import OmegaConf

    from src.app.core.callbacks import EnvironmentEvaluatorCallback
    from src.app.core.lightning_builder import build_trainer

    monkeypatch.chdir(tmp_path)
    base = {"group": "g", "experiment_id": "e", "seed": 1, "paradigm": "online_rl", "total_timesteps": 1000,
            "eval_episodes": 10, "agent": {"name": "ppo"}, "env": {"name": "cartpole", "num_envs": 1, "num_steps": 10}}
    skipped, _, _ = build_trainer(OmegaConf.create({**base, "telemetry": {"metrics": ["losses/"]}}))
    assert EnvironmentEvaluatorCallback not in [type(c) for c in skipped.callbacks]
    kept, _, _ = build_trainer(OmegaConf.create({**base, "telemetry": {"metrics": ["eval/reward"]}}))
    assert EnvironmentEvaluatorCallback in [type(c) for c in kept.callbacks]


def test_the_gui_can_import_the_selection_without_lightning():
    import subprocess
    import sys

    code = "import sys, src.app.telemetry.selection, src.app.pipeline.validation; print('lightning' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_parent_recording_selection_enables_nested_lazy_collector(tmp_path):
    from src.app.telemetry.run import Run
    calls = []
    with Run(tmp_path, [], config={"telemetry": {"metrics": ["interp/"]}}, system_interval=0) as run:
        assert run.monitor("interp/activations/", lambda: (calls.append(1) or {"interp/activations/norm": 2}))
        assert calls == [1]
        assert run.summary["interp/activations/norm"] == 2
