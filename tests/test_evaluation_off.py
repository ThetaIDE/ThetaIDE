"""An eval frequency of 0 turns evaluation off: no evaluation runs and no eval/ metric is recorded."""
import pytest
from omegaconf import OmegaConf

from src.app.telemetry.selection import evaluation_off, selection_errors, selection_from_config


@pytest.mark.parametrize("config,off", [
    ({"eval_freq_steps": 0}, True),
    ({"eval_freq": 0}, True),
    ({"eval_freq_steps": 0, "eval_freq": 500}, True),
    ({"eval_freq_steps": 2048}, False),
    ({}, False),
    ({"eval_freq_steps": False}, False),
])
def test_only_a_zero_eval_frequency_turns_evaluation_off(config, off):
    assert evaluation_off(OmegaConf.create(config)) is off


@pytest.mark.parametrize("metrics", [None, ["eval/reward", "losses/"], ["eval/"]])
def test_evaluation_off_drops_the_eval_section_from_every_selection(metrics):
    config = {"eval_freq_steps": 0, **({"telemetry": {"metrics": metrics}} if metrics else {})}
    selection = selection_from_config(OmegaConf.create(config))
    assert not selection.everything
    assert not selection.keeps_section("eval") and not selection.keeps("eval/reward")
    assert selection.keeps("transitions")  # plots still get their x-axis
    assert selection.filter({"eval/reward": 1.0, "eval/reward_std": 0.1, "transitions": 5.0}) == {"transitions": 5.0}
    if metrics is None or "losses/" in metrics:
        assert selection.keeps("losses/total_loss")


def test_evaluation_on_keeps_the_previous_selection():
    selection = selection_from_config(OmegaConf.create({"eval_freq_steps": 2048}))
    assert selection.everything and selection.keeps_section("eval")


def test_a_sweep_on_an_eval_metric_is_refused_when_evaluation_is_off():
    cfg = OmegaConf.create({"eval_freq_steps": 0})
    [problem] = selection_errors(cfg, "eval/reward")
    assert "evaluation is off" in problem
    assert selection_errors(cfg, "losses/total_loss") == []


def test_trainer_skips_the_evaluator_when_evaluation_is_off(tmp_path, monkeypatch):
    pytest.importorskip("lightning")
    from src.app.core.callbacks import EnvironmentEvaluatorCallback
    from src.app.core.lightning_builder import build_trainer

    monkeypatch.chdir(tmp_path)
    base = {"group": "g", "experiment_id": "e", "seed": 1, "paradigm": "online_rl", "total_timesteps": 1000,
            "eval_episodes": 10, "agent": {"name": "ppo"}, "env": {"name": "cartpole", "num_envs": 1, "num_steps": 10}}
    off, _, _ = build_trainer(OmegaConf.create({**base, "eval_freq_steps": 0}))
    assert EnvironmentEvaluatorCallback not in [type(c) for c in off.callbacks]
    on, _, _ = build_trainer(OmegaConf.create({**base, "eval_freq_steps": 200}))
    assert EnvironmentEvaluatorCallback in [type(c) for c in on.callbacks]


def test_eval_frequency_box_accepts_zero_and_shows_off(tmp_path):
    pytest.importorskip("PyQt6")
    import sys

    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    from frontend.config_viewer import ConfigViewer

    app = QApplication.instance() or QApplication(sys.argv[:1])
    path = tmp_path / "exp.yaml"
    path.write_text("paradigm: online_rl\ntotal_timesteps: 1000\neval_freq_steps: 2048\n", encoding="utf-8")
    viewer = ConfigViewer()
    viewer.load_file(path, "experiment/exp.yaml")
    viewer.show()
    app.processEvents()
    field = viewer.field_widgets["eval_freq_steps"]
    assert field.minimum() == 0
    field.setFocus()
    field.selectAll()
    QTest.keyClicks(field, "0")
    field.interpretText()
    assert viewer.raw_data["eval_freq_steps"] == 0
    assert field.text() == "Off"
    viewer.close()
