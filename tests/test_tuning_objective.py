import pytest

from src.app.pipeline.objective import objective_metric, objective_value
from src.app.telemetry.selection import selection_errors
from src.app.pipeline.tuning_config import tuning_disabled, tuning_enabled


def test_objective_uses_selected_metric_and_final_scalar():
    name = objective_metric({"tuning": {"metric": "val/accuracy"}}, "val/loss")
    assert name == "val/accuracy"
    assert objective_value({"val/loss": 0.1, "val/accuracy": 0.8}, name) == 0.8


@pytest.mark.parametrize("metrics", [{}, {"val/loss": float("nan")}, {"val/loss": float("inf")}, {"val/loss": [1, 2]}])
def test_invalid_objective_fails_instead_of_returning_a_score(metrics):
    with pytest.raises(ValueError, match="val/loss"):
        objective_value(metrics, "val/loss")


def test_metric_filter_checks_explicit_sweep_even_without_hydra_sweeper():
    cfg = {"telemetry": {"metrics": ["losses/"]}}
    assert selection_errors(cfg, "eval/reward", is_sweep=True)
    assert not selection_errors(cfg, "eval/reward", is_sweep=False)


def test_tuning_presence_is_authoritative_over_legacy_flags():
    assert not tuning_disabled({"tuning": {"enabled": False}})
    assert tuning_enabled({"tuning": {"enabled": False}, "sweep": False})
    assert tuning_enabled({"tuning": {"metric": "val/loss"}})
    assert tuning_enabled({"tuning": {}})
    assert tuning_disabled({"sweep": True})
    assert tuning_disabled({"tuning": None, "sweep": True})


def test_local_runner_uses_tuning_component_instead_of_context_flags(monkeypatch):
    from omegaconf import OmegaConf
    from src.app.pipeline import local_runner

    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "paradigm": "online_rl", "tuning": {}})
    methods = {
        "ordinary": {"agent": "ppo", "model": "dnn"},
        "tuned": {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": "interval(0.1, 0.2)"}},
    }
    plans = []
    monkeypatch.setattr(local_runner, "run_experiment", lambda overrides, **kwargs: plans.append(overrides))
    from src.app.pipeline.tuners_optuna import OptunaTuner
    monkeypatch.setattr(OptunaTuner, "prepare", lambda *args, **kwargs: None)
    monkeypatch.setattr(OptunaTuner, "finalize", lambda *args, **kwargs: None)
    context = {"methods": methods, "sanitized_extra_args": [], "storage_url": "sqlite:///test.db", "is_sweep": False}
    local_runner._run_methods_sequential(cfg, context)
    assert "--multirun" in plans[0]
    assert "--multirun" in plans[1]

    cfg.tuning = None
    plans.clear()
    from src.app.pipeline.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="tuned.*Add tuning"):
        local_runner._run_methods_sequential(cfg, {**context, "is_sweep": True})
    assert not plans


def test_method_preview_includes_the_search_space():
    from omegaconf import OmegaConf
    from src.app.pipeline.compose import ComposedExperiment, method_plans

    cfg = OmegaConf.create({"paradigm": "online_rl", "tuning": {}, "methods": {
        "method": {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": "interval(0.1, 0.2)"}},
    }})
    composed = ComposedExperiment("example", cfg, [], [], False)
    overrides = method_plans(composed)["method"]["train_overrides"]
    assert "--multirun" in overrides
    assert "agent.lr=interval(0.1, 0.2)" in overrides


def test_command_compiler_never_drops_a_method_search_space():
    from src.app.pipeline.commands import build_method_overrides

    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": "interval(0.1, 0.2)"}}
    overrides = build_method_overrides("method", method, cfg={"paradigm": "online_rl", "tuning": {}})
    assert "--multirun" in overrides
    assert "agent.lr=interval(0.1, 0.2)" in overrides
    from src.app.pipeline.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError, match="without a tuning component"):
        build_method_overrides("method", method, cfg={"paradigm": "online_rl"})
