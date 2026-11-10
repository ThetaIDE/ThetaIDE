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


def test_absent_tuning_does_not_compile_as_a_sweep_despite_legacy_flags():
    composed = compose_experiment(
        BASE_EXPERIMENT,
        ["++tuning=null", "sweep=true", "--multirun", "hydra/sweeper=optuna_online"],
    )
    assert not composed.is_sweep
    assert "--multirun" not in composed.launch_args
    assert not any(arg.startswith("hydra/sweeper=") for arg in composed.launch_args)


def test_composed_tuning_inherits_group_defaults_and_null_masks_them(fixture_workspace):
    group = fixture_workspace / "in/config/experiment/implicit_tuning"
    group.mkdir()
    (group / "_base.yaml").write_text("# @package _global_\ntuning: {component: optuna, enabled: false}\n")
    (group / "inherited.yaml").write_text(
        "# @package _global_\ndefaults: [/experiment/implicit_tuning/_base, _self_]\nsweep: false\n")
    composed = compose_experiment("implicit_tuning/inherited")
    assert composed.is_sweep
    assert composed.cfg.tuning.component == "optuna"
    assert not composed.cfg.sweep
    (group / "ordinary.yaml").write_text(
        "# @package _global_\ndefaults: [/experiment/implicit_tuning/_base, _self_]\ntuning: null\nsweep: true\n")
    composed = compose_experiment("implicit_tuning/ordinary")
    assert not composed.is_sweep
    assert composed.cfg.tuning is None


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


def test_sweep_preflight_checks_actual_trial_hydra_paths(monkeypatch):
    import src.app.pipeline.compose as pipeline
    from src.app.pipeline.exceptions import ConfigurationError

    composed = compose_experiment(REFERENCE, [
        "++tuning.component=optuna",
        "++methods.ppo.tune.model.missing.type=choice",
        "++methods.ppo.tune.model.missing.values=[1,2]",
    ])
    monkeypatch.setattr(pipeline, "method_plans", lambda _: {
        "ppo": {"settings": {"tune": {"model.missing": {"type": "choice", "values": [1, 2]}}},
                "train_overrides": ["model=dnn", "model.missing=choice(1,2)"]},
    })
    with pytest.raises(ConfigurationError, match="model.missing"):
        pipeline._validate_method_composition(composed)
    monkeypatch.setattr(pipeline, "method_plans", lambda _: {
        "ppo": {"settings": {"tune": {"model.hidden_sizes.0": {"type": "choice", "values": [32, 64]}}},
                "train_overrides": ["model=dnn", "model.hidden_sizes.0=choice(32,64)"]},
    })
    pipeline._validate_method_composition(composed)


def test_draft_preview_composes_the_yaml_without_writing_it(fixture_workspace, monkeypatch):
    from src.app.services import config_service
    monkeypatch.setattr(config_service, "CONFIG_DIR", fixture_workspace / "in" / "config")
    path = fixture_workspace / "in/config/experiment/thetaide/cartpole_ppo_reference.yaml"
    before = path.read_text()
    draft = before + "\nseed: 123\nmethods:\n  ppo:\n    agent: ppo\n    model: dnn\n    hidden_sizes: [17, 29]\n"
    report = config_service.compose_report.uncached(REFERENCE, [], draft)
    assert report["config"]["seed"] == 123
    assert report["methods"]["ppo"]["settings"]["hidden_sizes"] == [17, 29]
    assert path.read_text() == before


def test_component_schema_applies_to_the_actual_method_config(fixture_workspace):
    import json
    from src.app.pipeline.exceptions import ConfigurationError
    manifest_dir = fixture_workspace / "src/usr/models/dnn"
    manifest_dir.mkdir(parents=True)
    manifest_dir.joinpath(".theta_component.json").write_text(json.dumps({
        "config_path": "in/config/model/dnn.yaml",
        "parameter_schema": {"x-theta-validate": True, "type": "object", "properties": {
            "hidden_sizes": {"type": "array", "items": {"type": "integer", "minimum": 1}}
        }},
    }))
    base = ["++methods.ppo.agent=sb3/ppo", "++methods.ppo.hidden_sizes=[0,64]"]
    with pytest.raises(ConfigurationError, match="hidden_sizes"):
        validate_composed(compose_experiment(REFERENCE, base))
    base[-1] = "++methods.ppo.hidden_sizes=[16,64]"
    validate_composed(compose_experiment(REFERENCE, base))
