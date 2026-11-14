import json
import os
import subprocess
import sys
import shlex
import shutil
import importlib.util
import textwrap
from pathlib import Path

import optuna
import pytest

from component_fixtures import optional_module
_optuna_utils = optional_module('optuna', 'utils.py')
promote_best_trial_checkpoint = _optuna_utils.promote_best_trial_checkpoint
from src.app.pipeline.trial_artifacts import trial_checkpoint_dir


def test_study_names_cannot_escape_the_trial_directory(tmp_path):
    path = trial_checkpoint_dir(tmp_path, "../other", 0)
    assert path.is_relative_to(tmp_path / "sweeps")
    assert path.parent.name == "%2E%2E%2Fother"


def test_slurm_winner_command_quotes_study_arguments():
    from src.app.pipeline.slurm_runner import _winner_command

    command = _winner_command("python", "g", "exp", "method", "sqlite:///x.db", "study' $(touch nope)")
    assert shlex.split(command)[-1] == "study' $(touch nope)"


def test_plotman_reads_generic_tuning_summary_and_writes_table_and_figure(tmp_path, monkeypatch):
    from plot.standard_figures.tuning import TuningPlotter
    from src.app.core.plotter_registry import auto_discover_plotters, get_plotter_class

    monkeypatch.chdir(tmp_path)
    root = Path("results/checkpoints/g/e/m")
    root.mkdir(parents=True)
    (root / "tuning_summary.json").write_text(json.dumps({
        "schema_version": 1, "component": "another-tuner", "direction": "minimize",
        "trials": [
            {"id": 0, "state": "complete", "value": 4, "parameters": {"lr": 0.1},
             "intermediate_values": {"0": 5, "1": 4}},
            {"id": 1, "state": "fail", "value": None, "parameters": {"lr": 0.2}},
            {"id": 2, "state": "complete", "value": 2, "parameters": {"lr": 0.3}},
        ],
    }))
    study_root = root / "sweeps/older"
    study_root.mkdir(parents=True)
    (study_root / "tuning_summary.json").write_text((root / "tuning_summary.json").read_text())
    pruned_root = root.parent / "only_pruned"
    pruned_root.mkdir()
    (pruned_root / "tuning_summary.json").write_text(json.dumps({
        "schema_version": 1, "component": "example", "direction": "minimize",
        "trials": [{"id": 0, "state": "pruned", "value": None, "parameters": {},
                    "intermediate_values": {"0": 2}}],
    }))
    output = tmp_path / "plots"
    plotter = TuningPlotter()
    monkeypatch.setattr(plotter, "get_effective_config", lambda *args: ({}, "g", output))
    plotter.run("e")
    assert "param.lr" in (output / "tuning/m/trials.csv").read_text()
    assert (output / "tuning/m/objective.png").is_file()
    assert (output / "tuning/m/intermediate.png").is_file()
    auto_discover_plotters()
    assert get_plotter_class("tuning") is TuningPlotter
    from plot.manager import run_experiment_plots

    (root.parent / "config.yaml").write_text("group: g\nplots:\n  tuning: {}\n")
    from src.app.workspace import WorkspacePaths
    # Plotting resolves configs and outputs in the selected workspace, never the CWD.
    run_experiment_plots("g/e", paths=WorkspacePaths(tmp_path))
    assert Path("results/plots/g/e/tuning/m/trials.csv").is_file()
    assert Path("results/plots/g/e/tuning/m/objective.png").is_file()
    assert Path("results/plots/g/e/tuning/m/intermediate.png").is_file()
    assert Path("results/plots/g/e/tuning/m/studies/older/trials.csv").is_file()
    assert Path("results/plots/g/e/tuning/only_pruned/intermediate.png").is_file()


@pytest.mark.parametrize("extension", ["ckpt", "zip"])
def test_promotion_uses_study_and_trial_identity_not_hydra_job_index(tmp_path, monkeypatch, extension):
    monkeypatch.chdir(tmp_path)
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="study", storage=storage, direction="minimize")
    for value in [10.0, 1.0, 4.0]:
        study.tell(study.ask(), value)
    root = Path("results/checkpoints/g/e/m")
    winner = trial_checkpoint_dir(root, "study", 1)
    winner.mkdir(parents=True)
    (winner / f"best_model.{extension}").write_text("winner")
    wrong = root / "0" / "best_model.ckpt"
    wrong.parent.mkdir(parents=True)
    wrong.write_text("wrong")

    assert promote_best_trial_checkpoint("g", "e", "m", storage, "study") == "1"
    assert (root / f"best_model.{extension}").read_text() == "winner"
    assert json.loads((root / "best_trial_summary.json").read_text())["best_trial_id"] == 1
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["trials"][1]["checkpoint"] == str(winner / f"best_model.{extension}")
    assert summary["trials"][0]["checkpoint"] is None
    assert json.loads((root / "sweeps/study/tuning_summary.json").read_text())["study"] == "study"


def test_missing_winner_never_promotes_an_unrelated_checkpoint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="old", storage=storage)
    study.tell(study.ask(), 1.0)
    root = Path("results/checkpoints/g/e/m")
    root.mkdir(parents=True)
    (root / "best_model.ckpt").write_text("existing")
    with pytest.raises(FileNotFoundError, match="explicit trial-to-checkpoint mapping"):
        promote_best_trial_checkpoint("g", "e", "m", storage, "old")
    assert (root / "best_model.ckpt").read_text() == "existing"


def test_hydra_sweeps_pass_optuna_trial_numbers_after_resuming(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "objective.py"
    script.write_text(
        "from pathlib import Path\n"
        "import hydra\n"
        "from src.app.pipeline.trial_artifacts import trial_checkpoint_dir\n"
        f"@hydra.main(version_base=None, config_path={str(project / 'in/config')!r}, config_name='config')\n"
        "def main(cfg):\n"
        "    Path(f'trial_{cfg.theta_tuning.trial_number}.txt').write_text(cfg.theta_tuning.study_name)\n"
        "    path = trial_checkpoint_dir('results/checkpoints/g/e/m', cfg.theta_tuning.study_name, cfg.theta_tuning.trial_number)\n"
        "    path.mkdir(parents=True, exist_ok=True)\n"
        "    (path / 'best_model.ckpt').write_text(str(cfg.theta_tuning.trial_number))\n"
        "    return -float(cfg.theta_tuning.trial_number)\n"
        "if __name__ == '__main__': main()\n",
        encoding="utf-8",
    )
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    command = [
        sys.executable, str(script), "--multirun", "hydra/sweeper=optuna_offline",
        f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=resumed",
        "hydra.sweeper.n_trials=1", f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)",
        "++tuning.metric=score",
    ]
    for _ in range(2):
        subprocess.run(command, cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert (tmp_path / "trial_0.txt").read_text() == "resumed"
    assert (tmp_path / "trial_1.txt").read_text() == "resumed"
    study = optuna.load_study(study_name="resumed", storage=storage)
    assert [trial.number for trial in study.trials] == [0, 1]
    monkeypatch.chdir(tmp_path)
    promote_best_trial_checkpoint("g", "e", "m", storage, "resumed")
    assert Path("results/checkpoints/g/e/m/best_model.ckpt").read_text() == "1"
    assert json.loads(Path("results/checkpoints/g/e/m/tuning_summary.json").read_text())["objective_metric"] == "score"


def test_hydra_sweep_marks_invalid_objective_as_failed(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "bad_objective.py"
    script.write_text(
        "import hydra\n"
        f"@hydra.main(version_base=None, config_path={str(project / 'in/config')!r}, config_name='config')\n"
        "def main(cfg): return float('nan')\n"
        "if __name__ == '__main__': main()\n",
        encoding="utf-8",
    )
    storage = f"sqlite:///{tmp_path / 'failed.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", "hydra/sweeper=optuna_offline",
         f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=failed",
         "hydra.sweeper.n_trials=1", f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)",
         "++group=g", "++experiment_id=e", "++agent.name=m"],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    study = optuna.load_study(study_name="failed", storage=storage)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert study.user_attrs["theta_last_stop_reason"] == "failure_limit"
    summary = json.loads((tmp_path / "results/checkpoints/g/e/m/tuning_summary.json").read_text())
    assert summary["best_trial_id"] is None
    assert summary["trials"][0]["state"] == "fail"


def test_time_budget_waits_for_running_trial_but_stops_new_batches(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "slow_objective.py"
    script.write_text(
        "import hydra, time\n"
        f"@hydra.main(version_base=None, config_path={str(project / 'in/config')!r}, config_name='config')\n"
        "def main(cfg):\n"
        "    time.sleep(1.2)\n"
        "    return 1.0\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'timed.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", "hydra/sweeper=optuna_offline",
         f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=timed",
         "hydra.sweeper.n_trials=3", "hydra.sweeper.timeout_seconds=1",
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    study = optuna.load_study(study_name="timed", storage=storage)
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.user_attrs["theta_last_stop_reason"] == "time_budget"


def test_sweep_continues_after_allowed_trial_failure(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "sometimes_fails.py"
    script.write_text(
        "import hydra\n"
        f"@hydra.main(version_base=None, config_path={str(project / 'in/config')!r}, config_name='config')\n"
        "def main(cfg):\n"
        "    if cfg.theta_tuning.trial_number == 0: raise ValueError('broken candidate')\n"
        "    return 1.0\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'failures.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", "hydra/sweeper=optuna_offline",
         f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=failures",
         "hydra.sweeper.n_trials=2", "hydra.sweeper.max_failed_trials=1",
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=interval(1.0,2.0)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    study = optuna.load_study(study_name="failures", storage=storage)
    assert [trial.state for trial in study.trials] == [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.COMPLETE]
    assert study.user_attrs["theta_last_stop_reason"] == "trial_limit"


def test_qmc_sampler_runs_a_real_sweep(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "objective.py"
    script.write_text(
        "import hydra\n"
        f"@hydra.main(version_base=None, config_path={str(project / 'in/config')!r}, config_name='config')\n"
        "def main(cfg): return float(cfg.x)\n"
        "if __name__ == '__main__': main()\n", encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'qmc.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")])}
    result = subprocess.run(
        [sys.executable, str(script), "--multirun", "hydra/sweeper=optuna_offline",
         f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=qmc", "hydra.sweeper.n_trials=2",
         "++hydra.sweeper.sampler._target_=optuna.samplers.QMCSampler", "++hydra.sweeper.sampler.scramble=true",
         f"hydra.sweep.dir={tmp_path / 'hydra'}", "+x=range(0.1,0.5,0.2)"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    study = optuna.load_study(study_name="qmc", storage=storage)
    assert len(study.trials) == 2
    assert all(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    assert all(any(abs(next(iter(trial.params.values())) - expected) < 1e-9 for expected in (0.1, 0.3, 0.5))
               for trial in study.trials)


def test_interruption_marks_active_trial_and_exports_summary(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from component_fixtures import optional_module
    _optuna_sweeper = optional_module('optuna', 'sweeper.py')
    ThetaOptunaSweeper = _optuna_sweeper.ThetaOptunaSweeper

    class InterruptedLauncher:
        def launch(self, jobs, initial_job_idx):
            raise KeyboardInterrupt()

    monkeypatch.chdir(tmp_path)
    storage = f"sqlite:///{tmp_path / 'interrupted.db'}"
    sweeper = ThetaOptunaSweeper(optuna.samplers.RandomSampler(seed=1), "minimize", storage,
                                 "interrupted", 1, 1, None)
    sweeper.impl.config = OmegaConf.create({"group": "g", "experiment_id": "e", "agent": {"name": "m"}})
    sweeper.impl.launcher = InterruptedLauncher()
    with pytest.raises(KeyboardInterrupt):
        sweeper.sweep(["+x=interval(1.0,2.0)"])
    study = optuna.load_study(study_name="interrupted", storage=storage)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert study.user_attrs["theta_last_stop_reason"] == "interrupted"
    summary = json.loads(Path("results/checkpoints/g/e/m/tuning_summary.json").read_text())
    assert summary["stop_reason"] == "interrupted"


def test_resume_remains_compatible_with_legacy_optuna_distributions(tmp_path):
    import warnings
    from omegaconf import OmegaConf
    from component_fixtures import optional_module
    _optuna_sweeper = optional_module('optuna', 'sweeper.py')
    ThetaOptunaSweeper = _optuna_sweeper.ThetaOptunaSweeper

    class Result:
        return_value = 0.5

    class Launcher:
        def launch(self, jobs, initial_job_idx):
            return [Result() for _ in jobs]

    storage = f"sqlite:///{tmp_path / 'legacy.db'}"
    study = optuna.create_study(study_name="legacy", storage=storage, direction="minimize")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        trial = study.ask()
        trial._suggest("+x", optuna.distributions.UniformDistribution(1.0, 2.0))
        study.tell(trial, 1.0)
        sweeper = ThetaOptunaSweeper(optuna.samplers.RandomSampler(seed=1), "minimize", storage, "legacy", 1, 1, None)
        sweeper.impl.config = OmegaConf.create({"hydra": {"sweep": {"dir": str(tmp_path)}}})
        sweeper.impl.launcher = Launcher()
        sweeper.sweep(["+x=interval(1.0,2.0)"])
    assert len(optuna.load_study(study_name="legacy", storage=storage).trials) == 2


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 is optional")
@pytest.mark.parametrize('adaptive', [False, True])
def test_real_sb3_training_sweep_selects_its_checkpoint(tmp_path, monkeypatch, adaptive):
    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config)
    experiment = config / "experiment/smoke.yaml"
    experiment.write_text(
        "# @package _global_\n"
        "group: g\nexperiment_id: e\nparadigm: online_rl\ntotal_timesteps: 8\neval_episodes: 1\n"
        "env: {name: cartpole, env_id: CartPole-v1}\n"
        "agent: {name: m, algorithm: PPO, n_steps: 8, batch_size: 4, learning_rate: 0.001}\n"
        "model: {sb3_native_policy: true}\n"
        "tuning: {metric: eval/reward, direction: maximize}\n",
        encoding="utf-8",
    )
    runner = tmp_path / "src/usr/algorithms/sb3-suite/sb3_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/sb3_runner.py", runner)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/theta_sb3_policy.py", runner.parent / "theta_sb3_policy.py")
    storage = f"sqlite:///{tmp_path / 'sb3.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if adaptive:
        settings = tmp_path / 'settings.toml'
        settings.write_text('[training]\ncpu_percent=100\nmemory_percent=95\nmax_parallel=2\n')
        env.update(THETA_RESOURCE_POOL=str(tmp_path / 'resources.db'),
                   THETAIDE_SETTINGS_FILE=str(settings), THETA_MANAGED_SERIAL='1')
    command = [sys.executable, str(runner), "--multirun", "+experiment=smoke", "hydra/sweeper=optuna_online",
               f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=sb3-smoke",
               "hydra.sweeper.n_trials=1", "agent.learning_rate=choice(0.001,0.002)"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.chdir(tmp_path)
    promote_best_trial_checkpoint("g", "e", "m", storage, "sb3-smoke")
    root = Path("results/checkpoints/g/e/m")
    assert (root / "best_model.zip").is_file()
    study = optuna.load_study(study_name="sb3-smoke", storage=storage)
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["trials"][0]["value"] == study.best_value
    assert summary["objective_metric"] == "eval/reward"


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 is optional")
def test_real_sb3_sweep_records_a_pruned_trial(tmp_path):
    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config)
    (config / "experiment/smoke.yaml").write_text(
        "# @package _global_\ngroup: g\nexperiment_id: e\nparadigm: online_rl\n"
        "total_timesteps: 16\neval_freq_steps: 8\neval_episodes: 1\n"
        "env: {name: cartpole, env_id: CartPole-v1}\n"
        "agent: {name: m, algorithm: PPO, n_steps: 8, batch_size: 4, learning_rate: 0.001}\n"
        "model: {sb3_native_policy: true}\ntuning: {metric: eval/reward, direction: maximize}\n",
        encoding="utf-8",
    )
    runner = tmp_path / "src/usr/algorithms/sb3-suite/sb3_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/sb3_runner.py", runner)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/theta_sb3_policy.py", runner.parent / "theta_sb3_policy.py")
    storage = f"sqlite:///{tmp_path / 'sb3-pruning.db'}"
    study = optuna.create_study(study_name="sb3-pruning", storage=storage, direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=1000, intermediate_values={0: 1000}))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    command = [sys.executable, str(runner), "--multirun", "+experiment=smoke", "hydra/sweeper=optuna_online",
               f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=sb3-pruning",
               "hydra.sweeper.n_trials=1", "hydra.sweeper.pruner=median",
               "hydra.sweeper.pruner_startup_trials=1", "agent.learning_rate=choice(0.001,0.002)"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    trial = optuna.load_study(study_name="sb3-pruning", storage=storage).trials[-1]
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert 0 < trial.intermediate_values[0] < 1000
    assert not list((tmp_path / "results/checkpoints").rglob("best_model.zip"))
    assert "Error executing job with overrides" not in result.stderr


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 is optional")
def test_local_pipeline_sweep_trains_the_selected_blendrl(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from stable_baselines3 import PPO
    from src.app.pipeline.local_runner import _run_methods_sequential

    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config)
    for provider in ("blendrl", "dnn", "nsfr"):
        shutil.copytree(project / "src/usr/models" / provider, tmp_path / "src/usr/models" / provider)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    shutil.copytree(project / "src/usr/rules/cartpole", tmp_path / "src/usr/rules/cartpole")
    shutil.copytree(project / "in/envs/cartpole", tmp_path / "in/envs/cartpole")
    (config / "agent/ppo.yaml").write_text("algorithm: ppo\nn_steps: 8\nbatch_size: 4\nn_epochs: 1\nlearning_rate: 0.001\n")
    (config / "experiment/smoke.yaml").write_text(
        "# @package _global_\ngroup: g\nexperiment_id: e\nparadigm: online_rl\n"
        "total_timesteps: 16\neval_episodes: 1\n"
        "env: {name: cartpole, env_id: CartPole-v1, action_meanings: [left, right], "
        "symbolic_observation_builder: 'in.envs.cartpole.valuation:build_symbolic_observation'}\n"
        "tuning: {metric: eval/reward, direction: maximize, n_trials: 1}\n", encoding="utf-8")
    runner = tmp_path / "src/usr/algorithms/sb3-suite/sb3_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/sb3_runner.py", runner)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/theta_sb3_policy.py", runner.parent / "theta_sb3_policy.py")
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "experiment_name": "smoke",
                            "paradigm": "online_rl", "tuning": {"metric": "eval/reward", "direction": "maximize",
                            "n_trials": 1, "storage": storage}})
    method = {"agent": "ppo", "model": "blendrl", "entrypoint": str(runner), "python": sys.executable,
              "neural": {"type": "dnn", "hidden_sizes": [32, 32]},
              "symbolic": {"type": "nsfr", "rules": "variant2"},
              "blender": {"mode": "neural", "model": {"name": "basics/dnn", "hidden_sizes": [16, 16]}},
              "tune": {"agent.learning_rate": "choice(0.001,0.002)",
                       "model.neural.hidden_sizes.0": "choice(24,32)",
                       "model.symbolic.rules": "choice(variant1,variant2,default)"}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    _run_methods_sequential(cfg, {"methods": {"m": method}, "sanitized_extra_args": [],
                                  "storage_url": storage, "is_sweep": True})
    root = Path("results/checkpoints/g/e/m")
    saved = PPO.load(root / "best_model.zip")
    selected = saved.policy.theta_model_module
    from src.app.core.provider_loader import resolve_provider
    assert type(selected) is resolve_provider("model", "blendrl")
    assert selected.module_types == ["logic", "neural"]
    assert selected.policy_modules[1].network[0].out_features in (24, 32)
    assert selected.blender.network[0].out_features == 16
    assert json.loads((root / "tuning_summary.json").read_text())["trials"][0]["state"] == "complete"


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 is optional")
def test_cartpole_blendrl_experiment_sweep_compiles_and_runs(tmp_path, monkeypatch):
    import yaml
    from omegaconf import OmegaConf
    from stable_baselines3 import PPO
    from src.app.pipeline.local_runner import _run_methods_sequential
    from src.app.pipeline.validation import _validate_tuning

    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config)
    for provider in ("blendrl", "dnn", "nsfr"):
        shutil.copytree(project / "src/usr/models" / provider, tmp_path / "src/usr/models" / provider)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    shutil.copytree(project / "src/usr/rules/cartpole", tmp_path / "src/usr/rules/cartpole")
    shutil.copytree(project / "in/envs/cartpole", tmp_path / "in/envs/cartpole")
    experiment = config / "experiment/Belndrl Bench/CartpoleSweeping.yaml"
    # Keep this integration fixture independent of mutable user experiments.
    experiment.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "defaults": [{"override /env": "gym_classic_control/cartpole"}, "_self_"],
        "paradigm": "online_rl",
        "total_timesteps": 16, "eval_freq_steps": 8, "eval_episodes": 1,
        "methods": {"Ruleset2": {
            "agent": "sb3/ppo", "model": "blendrl", "actor_mode": "hybrid",
            "neural": {"type": "dnn", "hidden_sizes": [32, 32]},
            "symbolic": {"type": "nsfr", "rules": "variant2"},
            "blender": {"mode": "neural", "model": {"name": "basics/dnn", "hidden_sizes": [16, 16]}},
            "tune": {"agent.learning_rate": {"type": "log_float", "low": .0003, "high": .001},
                     "model.actor_mode": {"type": "choice", "values": ["hybrid", "neural"]},
                     "model.neural.hidden_sizes.0": {"type": "choice", "values": [24, 32]},
                     "model.blender.model.hidden_sizes.0": {"type": "choice", "values": [8, 16]}}}},
        "tuning": {"component": "optuna", "metric": "eval/reward", "direction": "maximize", "n_trials": 1},
    }
    dimensions = data["methods"]["Ruleset2"]["tune"]
    dimensions["agent.n_steps"] = {"type": "choice", "values": [8]}
    dimensions["agent.batch_size"] = {"type": "choice", "values": [4]}
    dimensions["agent.n_epochs"] = {"type": "int", "low": 1, "high": 1}
    experiment.write_text("# @package _global_\n" + yaml.safe_dump(data, sort_keys=False))
    runner = tmp_path / "src/usr/algorithms/sb3-suite/sb3_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/sb3_runner.py", runner)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/theta_sb3_policy.py", runner.parent / "theta_sb3_policy.py")
    storage = f"sqlite:///{tmp_path / 'cartpole.db'}"
    data.update({"group": "g", "experiment_id": "e", "experiment_name": "Belndrl Bench/CartpoleSweeping",
                 "paradigm": "online_rl"})
    data["tuning"]["storage"] = storage
    _validate_tuning(data)
    method = data["methods"]["Ruleset2"]
    method.update({"entrypoint": str(runner), "python": sys.executable})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    _run_methods_sequential(OmegaConf.create(data), {"methods": {"Ruleset2": method}, "sanitized_extra_args": [],
                                                   "storage_url": storage, "is_sweep": True})
    root = Path("results/checkpoints/g/e/Ruleset2")
    selected = PPO.load(root / "best_model.zip").policy.theta_model_module
    from src.app.core.provider_loader import resolve_provider
    assert type(selected) is resolve_provider("model", "blendrl")
    summary = json.loads((root / "tuning_summary.json").read_text())
    parameters = summary["trials"][0]["parameters"]
    assert set(parameters) == set(dimensions)
    assert summary["trials"][0]["intermediate_values"]
    assert selected.actor_mode == parameters["model.actor_mode"]
    assert selected.policy_modules[1].network[0].out_features == parameters["model.neural.hidden_sizes.0"]
    assert selected.blender.network[0].out_features == parameters["model.blender.model.hidden_sizes.0"]
    runtime_file = next(Path("results/logs/g/e/Ruleset2").glob("version_*/runtime.json"))
    runtime = json.loads(runtime_file.read_text())
    assert runtime["model_class"].endswith(".BlenderActorCritic")
    assert any(name.endswith(".NSFReasoner") for name in runtime["model_components"])


@pytest.mark.skipif(importlib.util.find_spec("stable_baselines3") is None, reason="SB3 is optional")
def test_local_pipeline_runs_a_real_sb3_sweep(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from src.app.pipeline.local_runner import _run_methods_sequential

    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config)
    (config / "agent/ppo.yaml").write_text("algorithm: ppo\nn_steps: 8\nbatch_size: 4\nlearning_rate: 0.001\n")
    (config / "model/dnn.yaml").write_text("architecture: dnn\nsb3_native_policy: true\n")
    (config / "experiment/smoke.yaml").write_text(
        "# @package _global_\ngroup: g\nexperiment_id: e\nparadigm: online_rl\n"
        "total_timesteps: 8\neval_episodes: 1\nenv: {name: cartpole, env_id: CartPole-v1}\n"
        "tuning: {metric: eval/reward, direction: maximize, n_trials: 1}\n", encoding="utf-8")
    runner = tmp_path / "src/usr/algorithms/sb3-suite/sb3_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/sb3_runner.py", runner)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/theta_sb3_policy.py", runner.parent / "theta_sb3_policy.py")
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "experiment_name": "smoke",
                            "paradigm": "online_rl", "tuning": {"metric": "eval/reward",
                            "direction": "maximize", "n_trials": 1}})
    method = {"agent": "ppo", "model": "dnn", "entrypoint": str(runner), "python": sys.executable,
              "tune": {"agent.learning_rate": "choice(0.001,0.002)"}}
    storage = f"sqlite:///{tmp_path / 'study folder' / 'pipeline.db'}"
    cfg.tuning.storage = storage
    cfg.tuning.options = {"optuna": {"sampler": "random", "seed": 23}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    _run_methods_sequential(cfg, {"methods": {"m": method}, "sanitized_extra_args": [],
                                  "storage_url": storage, "is_sweep": True})
    root = Path("results/checkpoints/g/e/m")
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert (root / "best_model.zip").is_file()
    assert summary["objective_metric"] == "eval/reward"
    assert summary["trials"][0]["parameters"]["agent.learning_rate"] in (0.001, 0.002)
    cfg.tuning.options.optuna.study_mode = "resume"
    cfg.tuning.options.optuna.study_name = summary["study"]
    _run_methods_sequential(cfg, {"methods": {"m": method}, "sanitized_extra_args": [],
                                  "storage_url": storage, "is_sweep": True})
    resumed = json.loads((root / "tuning_summary.json").read_text())
    assert [trial["id"] for trial in resumed["trials"]] == [0, 1]
    assert all(Path(trial["checkpoint"]).is_file() for trial in resumed["trials"])


@pytest.mark.skipif(importlib.util.find_spec("lightning") is None, reason="Lightning is not installed")
@pytest.mark.parametrize('adaptive', [False, True])
def test_real_lightning_trial_returns_the_selected_metric(tmp_path, monkeypatch, adaptive):
    project = Path(__file__).resolve().parents[1]
    script = tmp_path / "lightning_trial.py"
    script.write_text(textwrap.dedent(f"""
        from pathlib import Path
        import hydra
        import lightning as L
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from src.app.core.lightning_builder import finalize_training
        from src.app.pipeline.trial_artifacts import trial_checkpoint_dir

        class Tiny(L.LightningModule):
            def __init__(self, lr):
                super().__init__()
                self.layer = torch.nn.Linear(1, 1)
                self.lr = lr
            def training_step(self, batch, batch_idx):
                loss = self.layer(batch[0]).square().mean()
                self.log('train/loss', loss, on_step=False, on_epoch=True)
                return loss
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=self.lr)
            def train_dataloader(self):
                return DataLoader(TensorDataset(torch.ones(4, 1)), batch_size=2)

        @hydra.main(version_base=None, config_path={str(project / 'in/config')!r}, config_name='config')
        def main(cfg):
            model = Tiny(cfg.x)
            trainer = L.Trainer(max_epochs=1, logger=False, enable_checkpointing=False,
                                enable_progress_bar=False, enable_model_summary=False)
            trainer.fit(model)
            directory = trial_checkpoint_dir('results/checkpoints/g/e/m', cfg.theta_tuning.study_name,
                                             cfg.theta_tuning.trial_number)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / 'best_model.ckpt').write_text('initial checkpoint placeholder')
            return finalize_training(trainer, cfg, str(directory), 1.0, 0, 0)
        if __name__ == '__main__': main()
    """), encoding="utf-8")
    storage = f"sqlite:///{tmp_path / 'lightning.db'}"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    if adaptive:
        settings = tmp_path / 'settings.toml'
        settings.write_text('[training]\ncpu_percent=100\nmemory_percent=95\nmax_parallel=2\n')
        env.update(THETA_RESOURCE_POOL=str(tmp_path / 'resources.db'),
                   THETAIDE_SETTINGS_FILE=str(settings), THETA_MANAGED_SERIAL='1')
    command = [sys.executable, str(script), "--multirun", "hydra/sweeper=optuna_offline",
               f"hydra.sweeper.storage={storage}", "hydra.sweeper.study_name=lightning-smoke",
               "hydra.sweeper.n_trials=1", "++group=g", "++experiment_id=e", "++agent.name=m",
               "++tuning.metric=train/loss", "++tuning.direction=minimize", "+x=interval(0.01,0.02)"]
    result = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    monkeypatch.chdir(tmp_path)
    promote_best_trial_checkpoint("g", "e", "m", storage, "lightning-smoke")
    root = Path("results/checkpoints/g/e/m")
    assert (root / "best_model.ckpt").is_file()
    assert (root / "best_model.ckpt").read_bytes() != b"initial checkpoint placeholder"
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert summary["objective_metric"] == "train/loss"
    assert summary["trials"][0]["value"] > 0


@pytest.mark.skipif(importlib.util.find_spec("lightning") is None, reason="Lightning is not installed")
@pytest.mark.parametrize("pruning", [False, True])
def test_full_lightning_pipeline_sweep(tmp_path, monkeypatch, pruning, capfd):
    import numpy as np
    from omegaconf import OmegaConf
    from src.app.pipeline.local_runner import _run_methods_sequential

    project = Path(__file__).resolve().parents[1]
    config = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config)
    (config / "model/tiny.yaml").write_text("lightning_module: tiny_training\nlr: 0.01\n")
    (config / "experiment/smoke.yaml").write_text(
        "# @package _global_\ngroup: g\nexperiment_id: e\nparadigm: supervised\n"
        "epochs: 2\nbatch_size: 4\neval_episodes: 0\n"
        "tuning: {metric: val/loss, direction: minimize, n_trials: 1}\n", encoding="utf-8")
    data = tmp_path / "data.npz"
    np.savez(data, features=np.arange(24, dtype=np.float32).reshape(-1, 1) / 24,
             labels=(np.arange(24) % 2).astype(np.float32))
    plugin = tmp_path / "plugin"
    plugin.mkdir()
    (plugin / "tiny_training.py").write_text(textwrap.dedent("""
        import lightning as L
        import torch
        class TinyTraining(L.LightningModule):
            def __init__(self, architecture_name, input_dim, lr, **kwargs):
                super().__init__()
                self.layer = torch.nn.Linear(input_dim, 1)
                self.lr = lr
            def training_step(self, batch, batch_idx):
                x, y = batch
                loss = torch.nn.functional.mse_loss(self.layer(x).reshape(-1), y.float().reshape(-1))
                self.log('train/loss', loss, on_epoch=True)
                return loss
            def validation_step(self, batch, batch_idx):
                self.log('val/loss', torch.tensor(float(self.lr)), on_epoch=True)
            def configure_optimizers(self):
                return torch.optim.SGD(self.parameters(), lr=self.lr)
    """), encoding="utf-8")
    metadata = plugin / "tiny_training-0.0.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: tiny-training\nVersion: 0.0.1\n")
    (metadata / "entry_points.txt").write_text("[theta.components]\ntiny_training = tiny_training:TinyTraining\n")
    runner = tmp_path / "src/app/train.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/app/train.py", runner)
    storage = f"sqlite:///{tmp_path / 'lightning-pipeline.db'}"
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "experiment_name": "smoke",
                            "paradigm": "supervised", "dataset_path": str(data),
                            "tuning": {"metric": "val/loss", "direction": "minimize",
                                       "n_trials": 1, "storage": storage}})
    if pruning:
        study = optuna.create_study(study_name="pruning-study", storage=storage, direction="minimize")
        study.enqueue_trial({"model.lr": 0.01})
        study.enqueue_trial({"model.lr": 0.02})
        cfg.tuning.n_trials = 2
        cfg.tuning.options = {"optuna": {"study_mode": "resume", "study_name": "pruning-study",
                              "pruner": "median", "pruner_startup_trials": 1, "pruner_warmup_steps": 0}}
    method = {"model": "tiny", "entrypoint": str(runner), "python": sys.executable,
              "tune": {"model.lr": "choice(0.01,0.02)"}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(plugin), str(project), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    _run_methods_sequential(cfg, {"methods": {"m": method}, "sanitized_extra_args": [],
                                  "storage_url": storage, "is_sweep": True})
    root = Path("results/checkpoints/g/e/m")
    summary = json.loads((root / "tuning_summary.json").read_text())
    assert (root / "best_model.ckpt").is_file()
    assert summary["objective_metric"] == "val/loss"
    assert summary["trials"][0]["parameters"]["model.lr"] in (0.01, 0.02)
    if pruning:
        assert [trial["state"] for trial in summary["trials"]] == ["complete", "pruned"]
        assert summary["trials"][1]["checkpoint"] is None
        trials = optuna.load_study(study_name="pruning-study", storage=storage).trials
        assert trials[1].state == optuna.trial.TrialState.PRUNED
        assert set(trials[1].intermediate_values) == {0}  # stopped before the second validation epoch
        assert "0" in summary["trials"][1]["intermediate_values"]
        streams = list(Path("results/logs/g/e/m").glob("version_*/events.jsonl"))
        assert any(json.loads(line).get("status") == "pruned"
                   for stream in streams for line in stream.read_text().splitlines())
        _, stderr = capfd.readouterr()
        assert "Error executing job with overrides" not in stderr
        from plot.standard_figures.tuning import TuningPlotter

        TuningPlotter().run("g/e")
        assert Path("results/plots/g/e/tuning/m/intermediate.png").is_file()
