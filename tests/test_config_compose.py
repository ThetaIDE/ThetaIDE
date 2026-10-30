"""GUI-to-CLI configuration equivalence (M0).

The ThetaIDE experiment builder expresses a config as Hydra overrides on a base recipe
(frontend.model.Config). These tests prove those overrides compose to exactly the same
config as hand-written recipes, and that the builder's exported recipe round-trips.
"""

import pytest

from frontend.model import BASE_EXPERIMENT, Config
from src.app.pipeline.compose import comparable_config, compose_experiment, validate_composed

REFERENCE = "thetaide/cartpole_ppo_reference"

TEST_COMPONENT_CONFIG = {"method_name": "ppo", "agent": "ppo", "model": "dnn"}
CASES = [
    Config(name="cartpole_ppo_baseline", **TEST_COMPONENT_CONFIG),
    Config(name="seed_study-7", seed=7, total_timesteps=25000, lr=1e-6, batch_size=256, gamma=0.9,
           **TEST_COMPONENT_CONFIG),
    Config(name="123", seed=0, total_timesteps=1000, lr=0.5, batch_size=32, gamma=1.0, tensorboard=False,
           **TEST_COMPONENT_CONFIG),
]


def gui_compose(config):
    return compose_experiment(BASE_EXPERIMENT, config.overrides())


@pytest.fixture(autouse=True)
def isolated_config(fixture_workspace, monkeypatch):
    import src.app.pipeline.compose as compose_module

    monkeypatch.chdir(fixture_workspace)
    monkeypatch.setattr(compose_module, "CONFIG_DIR", fixture_workspace / "in" / "config")


@pytest.fixture
def export_recipe(fixture_workspace):
    written = []

    def write(config):
        path = fixture_workspace / "in" / "config" / "experiment" / "thetaide" / f"_pytest_export_{len(written)}.yaml"
        path.write_text(config.recipe_yaml(), encoding="utf-8")
        written.append(path)
        return f"thetaide/{path.stem}"

    yield write
    for path in written:
        path.unlink(missing_ok=True)


def test_builder_command_is_a_valid_pipeline_invocation():
    config = CASES[0]
    command = config.command()
    assert command.startswith(f"python run_pipeline.py {BASE_EXPERIMENT} ")
    assert f"++methods.{config.method_name}.model={config.model}" in command
    # Validate through the same in-process pipeline service used by the IDE.
    validate_composed(gui_compose(config))


def test_default_builder_matches_hand_written_reference():
    # The clean IDE has no algorithm/model defaults. Supply test-only profiles
    # here so the integration test exercises composition without installed plugins.
    gui = gui_compose(CASES[0])
    reference = compose_experiment(REFERENCE)
    assert comparable_config(gui.cfg) == comparable_config(reference.cfg)
    validate_composed(gui)


def test_builder_defaults_do_not_select_an_installed_component():
    config = Config()
    assert config.env == config.agent == config.model == ""
    assert config.method_name == "method"


@pytest.mark.parametrize("config", CASES, ids=lambda c: c.name)
def test_exported_recipe_composes_like_builder_overrides(config, export_recipe):
    gui = gui_compose(config)
    exported = compose_experiment(export_recipe(config))
    assert comparable_config(gui.cfg) == comparable_config(exported.cfg)


@pytest.mark.parametrize("config", CASES, ids=lambda c: c.name)
def test_builder_values_reach_the_method_config(config):
    cfg = gui_compose(config).cfg
    assert cfg.experiment_id == config.name
    assert (cfg.seed, cfg.total_timesteps) == (config.seed, config.total_timesteps)
    method = cfg.methods.ppo
    assert (method.agent, method.model) == ("ppo", "dnn")
    assert (method.lr, method.batch_size, method.gamma) == (config.lr, config.batch_size, config.gamma)
    assert isinstance(method.lr, float) and isinstance(method.gamma, float)
    assert cfg.tensorboard is config.tensorboard
