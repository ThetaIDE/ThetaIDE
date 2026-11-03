import pytest

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.validation import _validate_tuning


def config():
    return {"tuning": {"metric": "eval/reward", "direction": "maximize"}, "methods": {
        "one": {"agent": "ppo", "model": "dnn", "tune": {
            "agent.lr": {"type": "float", "low": 0.01, "high": 0.1}}},
        "disabled": {"enabled": False, "agent": "ppo", "model": "dnn"},
    }}


def test_valid_sweep():
    _validate_tuning(config())


def test_missing_dimension_names_method():
    cfg = config()
    cfg["methods"]["two"] = {"agent": "ppo", "model": "dnn"}
    with pytest.raises(ConfigurationError, match="two"):
        _validate_tuning(cfg)


def test_invalid_bounds_fail_preflight():
    cfg = config()
    cfg["methods"]["one"]["tune"]["agent.lr"]["low"] = 1
    with pytest.raises(ConfigurationError, match="one.*bounds"):
        _validate_tuning(cfg)


@pytest.mark.parametrize("value", [float("nan"), [1, 2], {"width": 2}])
def test_invalid_choice_values_fail_preflight(value):
    cfg = config()
    cfg["methods"]["one"]["tune"]["agent.lr"] = {"type": "choice", "values": [0.1, value]}
    with pytest.raises(ConfigurationError, match="finite scalar"):
        _validate_tuning(cfg)


def test_invalid_sb3_parameter_is_reported_before_launch():
    cfg = config()
    cfg["methods"]["one"]["agent"] = "sb3/ppo"
    with pytest.raises(ConfigurationError, match="agent.learning_rate"):
        _validate_tuning(cfg)


def test_sb3_pruning_is_accepted_before_launch():
    cfg = config()
    cfg["methods"]["one"]["entrypoint"] = "src/usr/algorithms/sb3-suite/sb3_runner.py"
    cfg["tuning"]["options"] = {"optuna": {"pruner": "median"}}
    _validate_tuning(cfg)
