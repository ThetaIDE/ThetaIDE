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


def test_invalid_sb3_parameter_is_reported_before_launch():
    cfg = config()
    cfg["methods"]["one"]["agent"] = "sb3/ppo"
    with pytest.raises(ConfigurationError, match="agent.learning_rate"):
        _validate_tuning(cfg)
