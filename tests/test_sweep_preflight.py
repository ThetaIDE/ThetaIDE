import pytest

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.validation import _validate_tuning as _validate_in_workspace
from src.app.workspace import WorkspacePaths
from component_fixtures import install_fixture

_WORKSPACE = None


@pytest.fixture(scope="module", autouse=True)
def tuner_workspace(tmp_path_factory):
    """A workspace with the test tuner and an SB3 PPO agent profile."""
    global _WORKSPACE
    root = tmp_path_factory.mktemp("sweep-preflight")
    paths = WorkspacePaths(root / "workspace")
    install_fixture(paths, "search-tuner", root / "archives")
    agent = paths.root / "src/usr/config/agent/sb3/ppo.yaml"
    agent.parent.mkdir(parents=True)
    agent.write_text("learning_rate: 0.0003\n")
    _WORKSPACE = paths
    yield paths
    _WORKSPACE = None


def _validate_tuning(cfg, paths=None):
    return _validate_in_workspace(cfg, paths=paths or _WORKSPACE)


def config():
    return {"tuning": {"component": "search", "metric": "eval/reward", "direction": "maximize"}, "methods": {
        "one": {"agent": "ppo", "model": "dnn", "tune": {
            "agent.lr": {"type": "float", "low": 0.01, "high": 0.1}}},
        "disabled": {"enabled": False, "agent": "ppo", "model": "dnn"},
    }}


def test_valid_sweep():
    _validate_tuning(config())


@pytest.mark.parametrize("key", ["n_trials", "search_space", "monitor_metric", "enabled"])
def test_tuning_block_rejects_settings_outside_its_one_shape(key):
    cfg = config()
    cfg["tuning"][key] = 1
    with pytest.raises(ConfigurationError, match=f"Unknown tuning settings: {key}.*tuning.options.<component>"):
        _validate_tuning(cfg)


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
    cfg["tuning"]["options"] = {"search": {"pruner": "median"}}
    _validate_tuning(cfg)


def test_sweep_accepts_optional_agent_parameter_declared_by_method():
    cfg = config()
    cfg["methods"]["one"] = {
        "agent": {"name": "sb3/ppo", "learning_rate_final": 0.0001},
        "model": "dnn",
        "tune": {"agent.learning_rate_final": {"type": "log_float", "low": 1e-6, "high": 1e-4}},
    }
    _validate_tuning(cfg)
    # An undeclared path must still fail, even if another optional field exists.
    cfg["methods"]["one"]["tune"] = {"agent.typo": {"type": "float", "low": 0, "high": 1}}
    with pytest.raises(ConfigurationError, match="agent.typo.*absent"):
        _validate_tuning(cfg)


