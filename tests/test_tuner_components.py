"""Exercise a second tuner through the real discovery, compiler and local runner."""

import pytest
from omegaconf import OmegaConf
import os
import subprocess
import sys
from pathlib import Path

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.tuners import get_tuner


def test_missing_component_fails_instead_of_falling_back_to_optuna(monkeypatch):
    monkeypatch.setattr("src.app.core.provider_loader.resolve_provider", lambda kind, name: None)
    with pytest.raises(ConfigurationError, match="not installed"):
        get_tuner({"tuning": {"component": "missing"}})


def test_incompatible_component_contract_is_rejected(monkeypatch):
    class OldTuner:
        api_version = 0
    monkeypatch.setattr("src.app.core.provider_loader.resolve_provider", lambda kind, name: OldTuner if (kind, name) == ("tuner", "old") else None)
    with pytest.raises(ConfigurationError, match="api_version = 1"):
        get_tuner(name="old")


def test_missing_optuna_dependencies_have_an_actionable_error(monkeypatch):
    from component_fixtures import optional_module
    _optuna_component = optional_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ConfigurationError, match="declared Python dependencies"):
        OptunaTuner().validate_config({"tuning": {"component": "optuna"}})


def test_optuna_component_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    from component_fixtures import optional_workspace
    workspace = optional_workspace("optuna")
    cfg = {"paradigm": "online_rl", "tuning": {"component": "optuna",
        "metric": "eval/reward", "direction": "maximize",
        "options": {"optuna": {"n_trials": 7, "sampler": "random", "seed": 23}},
    }}
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    overrides = build_method_overrides("method", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.n_trials=7" in overrides
    assert "++hydra.sweeper.sampler._target_=optuna.samplers.RandomSampler" in overrides
    assert "++hydra.sweeper.sampler.seed=23" in overrides


def test_qmc_and_tpe_specific_sampler_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    from component_fixtures import optional_workspace
    workspace = optional_workspace("optuna")
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    cfg = {"paradigm": "online_rl", "tuning": {"component": "optuna", "direction": "maximize", "options": {"optuna": {
        "sampler": "qmc", "seed": 8,
    }}}}
    overrides = build_method_overrides("m", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.sampler._target_=optuna.samplers.QMCSampler" in overrides
    assert "++hydra.sweeper.sampler.scramble=true" in overrides
    cfg["tuning"]["options"]["optuna"].update({"sampler": "tpe", "tpe_startup_trials": 3,
                                                "tpe_candidates": 12, "tpe_multivariate": True})
    overrides = build_method_overrides("m", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.sampler.n_startup_trials=3" in overrides
    assert "++hydra.sweeper.sampler.n_ei_candidates=12" in overrides
    assert "++hydra.sweeper.sampler.multivariate=true" in overrides


def test_pruning_time_and_failure_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    from component_fixtures import optional_workspace
    workspace = optional_workspace("optuna")
    cfg = {"paradigm": "supervised", "tuning": {"component": "optuna", "direction": "minimize", "options": {"optuna": {
        "pruner": "median", "pruner_startup_trials": 2, "pruner_warmup_steps": 3,
        "timeout_seconds": 60, "max_failed_trials": 1,
    }}}}
    method = {"model": "dnn", "tune": {"model.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    overrides = build_method_overrides("m", method, cfg=cfg, workspace=workspace)
    assert "++hydra.sweeper.pruner=median" in overrides
    assert "++hydra.sweeper.pruner_startup_trials=2" in overrides
    assert "++hydra.sweeper.pruner_warmup_steps=3" in overrides
    assert "++hydra.sweeper.timeout_seconds=60" in overrides
    assert "++hydra.sweeper.max_failed_trials=1" in overrides


def test_optuna_storage_override_is_identical_for_job_and_study():
    from src.app.pipeline.commands import build_method_overrides
    from component_fixtures import optional_workspace
    workspace = optional_workspace("optuna")
    from component_fixtures import optional_module
    _optuna_component = optional_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner

    cfg = OmegaConf.create({"experiment_id": "e", "paradigm": "online_rl",
                            "tuning": {"component": "optuna", "direction": "maximize", "options": {"optuna": {
                                "storage": "sqlite:///folder with spaces/new.db"}}}})
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "float", "low": 0.1, "high": 0.2}}}
    overrides = build_method_overrides("method", method, cfg=cfg, study_name="study one", workspace=workspace)
    assert OptunaTuner().storage(cfg) == "sqlite:///folder with spaces/new.db"
    assert '++hydra.sweeper.storage="sqlite:///folder with spaces/new.db"' in overrides
    assert '++hydra.sweeper.study_name="study one"' in overrides
    with pytest.raises(ValueError, match="compiled from the tuning block"):
        build_method_overrides("method", method, cfg=cfg,
                               extra_args=["hydra.sweeper.storage=sqlite:///different.db"], workspace=workspace)


def test_optuna_new_and_resume_studies_are_explicit(tmp_path):
    from component_fixtures import optional_module
    _optuna_component = optional_module('optuna', 'component.py')
    OptunaTuner = _optuna_component.OptunaTuner

    cfg = OmegaConf.create({
        "group": "g", "experiment_id": "e", "paradigm": "online_rl",
        "tuning": {"component": "optuna", "metric": "eval/reward", "direction": "maximize", "options": {"optuna": {
            "study_name": "study", "study_mode": "new"}}},
    })
    tuner = OptunaTuner()
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    tuner.prepare(cfg, storage, "study")
    with pytest.raises(ConfigurationError, match="already exists"):
        tuner.prepare(cfg, storage, "study")
    cfg.tuning.options.optuna.study_mode = "resume"
    tuner.prepare(cfg, storage, "study")
    with pytest.raises(ConfigurationError, match="does not exist"):
        tuner.prepare(cfg, storage, "missing")
    cfg.tuning.direction = "minimize"
    with pytest.raises(ConfigurationError, match="different objective direction"):
        tuner.prepare(cfg, storage, "study")


def test_optuna_sweeper_does_not_import_lightning_when_pruning_is_off(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "without_lightning.py"
    script.write_text(
        "import sys\n"
        "class BlockLightning:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'lightning' or fullname.startswith('lightning.'):\n"
        "            raise ModuleNotFoundError('Lightning intentionally unavailable')\n"
        "sys.meta_path.insert(0, BlockLightning())\n"
        "import optuna\n"
        "from hydra_plugins.theta_sweeper import ThetaSweeper\n"
        "ThetaSweeper('optuna', sampler=optuna.samplers.RandomSampler(), direction='minimize', storage=None,"
        " study_name='test', n_trials=1, n_jobs=1, search_space=None, pruner='none', pruner_startup_trials=5,"
        " pruner_warmup_steps=0, timeout_seconds=0, max_failed_trials=0)\n",
        encoding="utf-8",
    )
    from component_fixtures import optional_workspace
    workspace = optional_workspace("optuna")
    # The saved sweeper target resolves from the selected workspace's installed Optuna component.
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]),
           "THETA_WORKSPACE": str(workspace.root), "THETA_DATA_DIR": str(workspace.data_dir)}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


