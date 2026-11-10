"""Exercise a second tuner through the real discovery, compiler and local runner."""

import pytest
from omegaconf import OmegaConf
import os
import shutil
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
    from src.app.pipeline.tuners_optuna import OptunaTuner
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    with pytest.raises(ConfigurationError, match=r"\[optuna\]"):
        OptunaTuner().validate_config({"tuning": {}})


def test_optuna_component_settings_compile_with_legacy_fallback():
    from src.app.pipeline.commands import build_method_overrides
    cfg = {"paradigm": "online_rl", "tuning": {
        "metric": "eval/reward", "direction": "maximize", "n_trials": 10,
        "options": {"optuna": {"n_trials": 7, "sampler": "random", "seed": 23}},
    }}
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": "interval(0.1,0.2)"}}
    overrides = build_method_overrides("method", method, cfg=cfg, is_sweep=True)
    assert "++hydra.sweeper.n_trials=7" in overrides
    assert "++hydra.sweeper.sampler._target_=optuna.samplers.RandomSampler" in overrides
    assert "++hydra.sweeper.sampler.seed=23" in overrides


def test_qmc_and_tpe_specific_sampler_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": "interval(0.1,0.2)"}}
    cfg = {"paradigm": "online_rl", "tuning": {"direction": "maximize", "options": {"optuna": {
        "sampler": "qmc", "seed": 8,
    }}}}
    overrides = build_method_overrides("m", method, cfg=cfg, is_sweep=True)
    assert "++hydra.sweeper.sampler._target_=optuna.samplers.QMCSampler" in overrides
    assert "++hydra.sweeper.sampler.scramble=true" in overrides
    cfg["tuning"]["options"]["optuna"].update({"sampler": "tpe", "tpe_startup_trials": 3,
                                                "tpe_candidates": 12, "tpe_multivariate": True})
    overrides = build_method_overrides("m", method, cfg=cfg, is_sweep=True)
    assert "++hydra.sweeper.sampler.n_startup_trials=3" in overrides
    assert "++hydra.sweeper.sampler.n_ei_candidates=12" in overrides
    assert "++hydra.sweeper.sampler.multivariate=true" in overrides


def test_pruning_time_and_failure_settings_compile():
    from src.app.pipeline.commands import build_method_overrides
    cfg = {"paradigm": "supervised", "tuning": {"direction": "minimize", "options": {"optuna": {
        "pruner": "median", "pruner_startup_trials": 2, "pruner_warmup_steps": 3,
        "timeout_seconds": 60, "max_failed_trials": 1,
    }}}}
    method = {"model": "dnn", "tune": {"model.lr": "interval(0.1,0.2)"}}
    overrides = build_method_overrides("m", method, cfg=cfg, is_sweep=True)
    assert "++hydra.sweeper.pruner=median" in overrides
    assert "++hydra.sweeper.pruner_startup_trials=2" in overrides
    assert "++hydra.sweeper.pruner_warmup_steps=3" in overrides
    assert "++hydra.sweeper.timeout_seconds=60" in overrides
    assert "++hydra.sweeper.max_failed_trials=1" in overrides


def test_optuna_storage_override_is_identical_for_job_and_study():
    from src.app.pipeline.commands import build_method_overrides
    from src.app.tuners.optuna.component import OptunaTuner

    cfg = OmegaConf.create({"experiment_id": "e", "paradigm": "online_rl",
                            "hydra": {"sweeper": {"storage": "sqlite:///old.db"}},
                            "tuning": {"direction": "maximize", "options": {"optuna": {
                                "storage": "sqlite:///folder with spaces/new.db"}}}})
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": "interval(0.1,0.2)"}}
    overrides = build_method_overrides("method", method, cfg=cfg, is_sweep=True, study_name="study one")
    assert OptunaTuner().storage(cfg) == "sqlite:///folder with spaces/new.db"
    assert '++hydra.sweeper.storage="sqlite:///folder with spaces/new.db"' in overrides
    assert '++hydra.sweeper.study_name="study one"' in overrides
    with pytest.raises(ValueError, match="study preparation"):
        build_method_overrides("method", method, cfg=cfg, is_sweep=True,
                               extra_args=["hydra.sweeper.storage=sqlite:///different.db"])


def test_optuna_new_and_resume_studies_are_explicit(tmp_path):
    from src.app.pipeline.tuners_optuna import OptunaTuner

    cfg = OmegaConf.create({
        "group": "g", "experiment_id": "e", "paradigm": "online_rl",
        "tuning": {"metric": "eval/reward", "direction": "maximize", "options": {"optuna": {
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


def test_ordinary_run_never_resolves_a_tuner(monkeypatch):
    from src.app.pipeline import local_runner
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "paradigm": "online_rl"})
    monkeypatch.setattr(local_runner, "get_tuner", lambda cfg: pytest.fail("ordinary job resolved a tuner"))
    commands = []
    monkeypatch.setattr(local_runner, "run_experiment", lambda overrides, **kwargs: commands.append(overrides))
    local_runner._run_methods_sequential(cfg, {
        "methods": {"method": {"agent": "ppo", "model": "dnn"}},
        "sanitized_extra_args": [], "storage_url": None, "is_sweep": True,
    })
    assert len(commands) == 1 and "--multirun" not in commands[0]


def test_ordinary_run_works_when_optuna_imports_are_unavailable(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "without_optuna.py"
    script.write_text(
        "import sys\n"
        "class BlockOptuna:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'optuna' or fullname.startswith('optuna.') or fullname.startswith('hydra_plugins.hydra_optuna_sweeper'):\n"
        "            raise ModuleNotFoundError('Optuna intentionally unavailable')\n"
        "sys.meta_path.insert(0, BlockOptuna())\n"
        "from omegaconf import OmegaConf\n"
        "from src.app.pipeline import local_runner\n"
        "cfg=OmegaConf.create({'group':'g','experiment_id':'e','paradigm':'online_rl',"
        "'tuning':None})\n"
        "calls=[]\n"
        "local_runner.run_experiment=lambda overrides, **kw: calls.append(overrides)\n"
        "local_runner._run_methods_sequential(cfg, {'methods':{'m':{'agent':'ppo','model':'dnn'}},"
        "'sanitized_extra_args':[],'storage_url':None,'is_sweep':False})\n"
        "assert len(calls)==1 and '--multirun' not in calls[0]\n", encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


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
        "from hydra_plugins.theta_optuna_sweeper import ThetaOptunaSweeper\n"
        "ThetaOptunaSweeper(optuna.samplers.RandomSampler(), 'minimize', None, 'test', 1, 1, None)\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


def test_other_tuner_owns_compilation_and_lifecycle(monkeypatch):
    from src.app.pipeline import local_runner
    from src.app.pipeline.validation import _validate_tuning

    events = []

    class ExampleTuner:
        api_version = 1
        def settings(self):
            return []

        def validate_config(self, cfg):
            events.append("validate")

        def validate_dimension(self, target, dimension):
            assert dimension == {"type": "custom"}

        def compile(self, overrides, **context):
            return [*overrides, "++custom_tuner.active=true"]

        def job_entrypoint(self, cfg, method_cfg, script, python):
            return "custom_tuner_runner.py", python

        def study_name(self, cfg, method):
            return "custom-study"

        def storage(self, cfg):
            return "custom-storage"

        def prepare(self, cfg, storage, study):
            events.append(("prepare", storage, study))

        def finalize(self, group, experiment, method, storage, study):
            events.append(("finalize", group, experiment, method, storage, study))

    monkeypatch.setattr(
        "src.app.core.provider_loader.resolve_provider",
        lambda kind, name: ExampleTuner if (kind, name) == ("tuner", "example") else None,
    )
    cfg = OmegaConf.create({
        "group": "g", "experiment_id": "e", "paradigm": "online_rl",
        "tuning": {"component": "example", "metric": "eval/reward", "direction": "maximize"},
        "methods": {"method": {"agent": "ppo", "model": "dnn", "tune": {"model.width": {"type": "custom"}}}},
    })
    _validate_tuning(cfg)
    commands = []
    monkeypatch.setattr(local_runner, "run_experiment", lambda overrides, **kwargs: commands.append((overrides, kwargs)))
    local_runner._run_methods_sequential(cfg, {
        "methods": cfg.methods, "sanitized_extra_args": [], "storage_url": get_tuner(cfg).storage(cfg), "is_sweep": True,
    })
    assert "++custom_tuner.active=true" in commands[0][0]
    assert commands[0][1]["script_entrypoint"] == "custom_tuner_runner.py"
    assert not any("optuna" in item.lower() for item in commands[0][0])
    assert ("prepare", "custom-storage", "custom-study") in events
    assert ("finalize", "g", "e", "method", "custom-storage", "custom-study") in events


def test_slurm_uses_the_selected_tuners_runner_and_finalizer(tmp_path, monkeypatch):
    from src.app.pipeline import slurm_runner

    events = []
    class ExampleTuner:
        api_version = 1
        def settings(self): return []
        def validate_config(self, cfg): pass
        def validate_dimension(self, target, dimension): pass
        def compile(self, overrides, **context): return [*overrides, "custom_option=1"]
        def job_entrypoint(self, cfg, method_cfg, script, python): return "my_tuner/runner.py", None
        def study_name(self, cfg, method): return "my-study"
        def storage(self, cfg): return "my-storage"
        def prepare(self, cfg, storage, study): events.append((storage, study))
        def finalize(self, group, experiment, method, storage, study): pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.app.core.provider_loader.resolve_provider", lambda kind, name: ExampleTuner if (kind, name) == ("tuner", "example") else None)
    monkeypatch.setattr(slurm_runner, "generate_sbatch_header", lambda **kwargs: "#!/bin/bash\n")
    monkeypatch.setattr(slurm_runner, "get_shell_env_block", lambda site: "")
    monkeypatch.setattr(slurm_runner, "get_shell_python_cmd", lambda site: "python")
    monkeypatch.setattr(slurm_runner, "save_experiment_snapshot", lambda *args: None)
    monkeypatch.setattr(slurm_runner, "save_run_config", lambda *args: None)
    scripts = []
    monkeypatch.setattr(slurm_runner, "submit_sbatch", lambda content: scripts.append(content) or "42")
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "paradigm": "online_rl",
                            "site": {"consolidate": False}, "tuning": {"component": "example"}})
    method = {"agent": "ppo", "model": "dnn", "tune": {"agent.lr": {"type": "custom"}}}
    slurm_runner.run_slurm_training(cfg, {"methods": {"m": method}, "is_sweep": True,
                                         "storage_url": "my-storage", "sanitized_extra_args": []})
    assert events == [("my-storage", "my-study")]
    assert "my_tuner/runner.py" in scripts[0]
    assert "src.app.pipeline.tuner_finalize example g e m my-storage my-study" in scripts[0]
    assert "optuna" not in scripts[0].lower()


def test_independently_installed_tuner_runs_and_produces_plotman_artifacts(tmp_path):
    project = Path(__file__).resolve().parents[1]
    package = tmp_path / "package"
    shutil.copytree(project / "tests/fixtures/example_tuner", package)
    target = tmp_path / "installed"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--target", str(target), str(package)],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
    )
    assert install.returncode == 0, install.stdout + install.stderr
    script = tmp_path / "run_example.py"
    script.write_text(
        "import sys\n"
        "class BlockOptuna:\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'optuna' or fullname.startswith('optuna.') or fullname.startswith('hydra_plugins.hydra_optuna_sweeper'):\n"
        "            raise ModuleNotFoundError('Optuna intentionally unavailable')\n"
        "sys.meta_path.insert(0, BlockOptuna())\n"
        "from pathlib import Path\n"
        "from omegaconf import OmegaConf\n"
        "from src.app.pipeline.tuners import available_tuners, get_tuner\n"
        "from src.app.pipeline.validation import _validate_tuning\n"
        "from src.app.pipeline.local_runner import _run_methods_sequential\n"
        "from plot.manager import run_experiment_plots\n"
        "cfg = OmegaConf.create({'group':'g','experiment_id':'e','paradigm':'online_rl',"
        "'tuning':{'component':'example','metric':'score','direction':'minimize',"
        "'options':{'example':{'count':2}}},'methods':{'m':{'agent':'ppo','model':'dnn',"
        "'tune':{'agent.lr':{'type':'choice','values':[0.2,0.1]}}}}})\n"
        "assert 'example' in available_tuners()\n"
        "_validate_tuning(cfg)\n"
        "methods = OmegaConf.to_container(cfg.methods, resolve=True)\n"
        "methods['m']['python'] = sys.executable\n"
        "_run_methods_sequential(cfg, {'methods':methods,'sanitized_extra_args':[],"
        "'storage_url':get_tuner(cfg).storage(cfg),'is_sweep':True})\n"
        "p=Path('results/checkpoints/g/e/config.yaml'); p.parent.mkdir(parents=True,exist_ok=True)\n"
        "p.write_text('group: g\\nplots:\\n  tuning: {}\\n')\n"
        "run_experiment_plots('g/e')\n",
        encoding="utf-8",
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(target), str(project)]), "MPLBACKEND": "Agg",
           "THETA_WORKSPACE": str(tmp_path), "THETA_DATA_DIR": str(tmp_path / '.thetaide')}
    result = subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "results/checkpoints/g/e/m/best_params.yaml").read_text().strip() == "agent.lr: 0.1"
    assert (tmp_path / "results/plots/g/e/tuning/m/trials.csv").is_file()
    assert (tmp_path / "results/plots/g/e/tuning/m/objective.png").is_file()
