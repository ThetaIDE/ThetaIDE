from src.app.core.execution import RunContext
from src.app.workspace import WorkspacePaths

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "sb3_runner_telemetry", Path(__file__).resolve().parents[1] / "src/usr/algorithms/sb3-dev/sb3_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Capture:
    def __init__(self):
        self.rows = []

    def log(self, metrics, **position):
        self.rows.append((metrics, position))


def test_completed_episodes_are_logged_between_reporting_intervals():
    capture = Capture()
    episodes = runner.EpisodeMetrics(capture, window=2)
    episodes.record([{}, {"episode": {"r": 10, "l": 7}}], 17)
    episodes.record([{"episode": {"r": 30, "l": 9}}], 42)
    episodes.record([{"episode": {"r": 50, "l": 11}}], 81)
    assert [row[0]["train/episode_reward"] for row in capture.rows] == [10, 30, 50]
    assert capture.rows[-1] == ({"train/episode_reward": 50, "train/episode_length": 11,
                                "train/reward": 40, "train/length": 10},
                               {"step": 81, "transitions": 81})


def test_sb3_logger_dump_forwards_diagnostics_and_losses():
    logger_module = pytest.importorskip("stable_baselines3.common.logger")
    capture = Capture()
    callback = SimpleNamespace(num_timesteps=2048)

    class Writer(runner.ThetaSB3Writer, logger_module.KVWriter):
        pass

    logger = logger_module.Logger(folder=None, output_formats=[Writer(capture, callback)])
    logger.record("train/approx_kl", 0.02)
    logger.record("train/value_loss", 1.5)
    logger.record("time/fps", 120)
    logger.record("train/invalid", float("nan"))
    logger.dump(step=9)
    assert capture.rows == [({"diagnostics/approx_kl": 0.02, "losses/train/value_loss": 1.5,
                              "time/fps": 120.0}, {"step": 2048, "transitions": 2048})]


def test_component_qualified_algorithm_names_do_not_fall_back_to_ppo():
    algorithms = {"PPO": object(), "DQN": object()}
    assert runner.resolve_algorithm("sb3-dev/dqn", algorithms) is algorithms["DQN"]
    with pytest.raises(ValueError, match="Unsupported SB3 algorithm"):
        runner.resolve_algorithm("sb3-dev/typo", algorithms)


def test_sb3_options_are_forwarded_for_the_selected_algorithm():
    sb3 = pytest.importorskip("stable_baselines3")
    config = {"learning_rate": "3e-4", "n_epochs": 3, "gae_lambda": 0.8,
              "clip_range": 0.1, "batch_size": 32, "name": "example"}
    options = runner.algorithm_options(config, sb3.PPO)
    assert options == {"learning_rate": 0.0003, "n_epochs": 3, "gae_lambda": 0.8,
                       "clip_range": 0.1, "batch_size": 32}


def test_explicit_learning_rate_decay_is_validated_and_reaches_final_rate():
    sb3 = pytest.importorskip("stable_baselines3")
    options = runner.algorithm_options({"learning_rate": {"type": "linear", "initial": .002,
                                                        "final": .0001}}, sb3.PPO)
    schedule = options["learning_rate"]
    assert [schedule(p) for p in [1, .5, 0, -.5]] == pytest.approx([.002, .00105, .0001, .0001])
    scalar = runner.algorithm_options({"learning_rate": .002, "learning_rate_schedule": "linear",
                                      "learning_rate_final": .0001}, sb3.PPO)["learning_rate"]
    assert scalar(.5) == pytest.approx(.00105)
    for spec in [{"type": "typo", "initial": .001, "final": 0},
                 {"type": "linear", "initial": 0, "final": 0},
                 {"type": "linear", "initial": .001, "final": float("nan")}]:
        with pytest.raises(ValueError, match="learning_rate"):
            runner.algorithm_options({"learning_rate": spec}, sb3.PPO)


def test_qualified_dqn_job_saves_a_dqn_not_a_ppo(tmp_path, monkeypatch):
    import json
    sb3 = pytest.importorskip("stable_baselines3")
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "total_timesteps": 4,
                            "agent": {"name": "dqn", "algorithm": "sb3-dev/dqn", "buffer_size": 32,
                                      "learning_starts": 100, "batch_size": 4},
                            "seed": 73, "model": {"sb3_native_policy": True},
                            "env": {"env_id": "CartPole-v1", "num_envs": 2}})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(mode=SimpleNamespace(name="RUN"))))
    runner.execute(cfg, RunContext(
        'test', WorkspacePaths.for_process(),
        int(cfg.get('seed', 42)), tmp_path / 'results/checkpoints' / cfg.group / cfg.experiment_id /
        cfg.agent.name / str(cfg.get('variability_version') or '')))
    checkpoint = tmp_path / "results/checkpoints/g/e/dqn/best_model.zip"
    loaded = sb3.DQN.load(checkpoint)
    assert isinstance(loaded, sb3.DQN)
    assert loaded.buffer_size == 32
    assert loaded.n_envs == 2
    assert loaded.seed == 73
    runtime = json.loads((tmp_path / "results/logs/g/e/dqn/version_0/runtime.json").read_text())
    assert runtime["algorithm_class"].endswith(".DQN")
    assert runtime["policy_class"].endswith(".DQNPolicy")
    assert runtime["training_envs"] == 2


def test_final_evaluation_uses_last_update_and_best_checkpoint_survives(tmp_path, monkeypatch):
    import csv
    import json
    import torch
    sb3 = pytest.importorskip("stable_baselines3")
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    evaluations = []
    def evaluate(model, env, **kwargs):
        evaluations.append(model._n_updates)
        return ([500.] if model._n_updates == 0 else [100.]), [1]

    monkeypatch.setattr("stable_baselines3.common.evaluation.evaluate_policy", evaluate)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(mode=SimpleNamespace(name="RUN"))))
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "total_timesteps": 8,
                            "eval_freq_steps": 8, "eval_episodes": 1,
                            "variability_run_id": "test", "variability_method": "ppo", "variability_version": "version_0",
                            "agent": {"name": "ppo", "algorithm": "PPO", "n_steps": 8, "batch_size": 4,
                                      "n_epochs": 2, "learning_rate": .01},
                            "model": {"sb3_native_policy": True}, "env": {"env_id": "CartPole-v1"}})
    runner.execute(cfg, RunContext(
        'test', WorkspacePaths.for_process(),
        int(cfg.get('seed', 42)), tmp_path / 'results/checkpoints' / cfg.group / cfg.experiment_id /
        cfg.agent.name / str(cfg.get('variability_version') or '')))
    assert evaluations == [0, 0, 2, 2]
    root = tmp_path / "results/checkpoints/g/e/ppo/version_0"
    best, latest = sb3.PPO.load(root / "best_model.zip"), sb3.PPO.load(root / "latest_model.zip")
    assert best._n_updates == 0
    assert latest._n_updates == 2
    assert any(not torch.equal(v, latest.policy.state_dict()[k]) for k, v in best.policy.state_dict().items())
    events = [json.loads(line) for line in (tmp_path / "results/logs/g/e/ppo/version_0/evaluation_events.jsonl").read_text().splitlines()]
    assert [e["training_step"] for e in events if e["evaluation_mode"] == "greedy"] == [0, 8]
    assert events[-1]["episode_outcomes"] == [100.]
    with (tmp_path / "results/logs/g/e/ppo/version_0/metrics.csv").open() as stream:
        updates = [float(row["diagnostics/n_updates"]) for row in csv.DictReader(stream)
                   if row.get("diagnostics/n_updates")]
    assert updates[-1] == 2


def test_sb3_pruning_reports_evaluation_and_does_not_save_best(tmp_path, monkeypatch):
    optuna = pytest.importorskip("optuna")
    pytest.importorskip("stable_baselines3")
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import OmegaConf

    storage = f"sqlite:///{tmp_path / 'study.db'}"
    study = optuna.create_study(study_name="sb3", storage=storage, direction="maximize")
    study.add_trial(optuna.trial.create_trial(value=1000, intermediate_values={0: 1000}))
    trial = study.ask()
    cfg = OmegaConf.create({
        "group": "g", "experiment_id": "e", "total_timesteps": 16,
        "eval_freq_steps": 8, "eval_episodes": 1,
        "agent": {"name": "ppo", "algorithm": "PPO", "n_steps": 8, "batch_size": 8},
        "model": {"sb3_native_policy": True}, "env": {"env_id": "CartPole-v1"},
        "tuning": {"metric": "eval/reward"},
        "theta_tuning": {"component": "optuna", "pruner": "median", "pruner_startup_trials": 1,
                         "pruner_warmup_steps": 0, "study_name": "sb3", "storage": storage,
                         "optuna_trial_id": trial._trial_id, "trial_number": trial.number},
    })
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(mode=SimpleNamespace(name="RUN"))))
    result = runner.execute(cfg, RunContext(
        'test', WorkspacePaths.for_process(),
        int(cfg.get('seed', 42)), tmp_path / 'results/checkpoints' / cfg.group / cfg.experiment_id /
        cfg.agent.name / str(cfg.get('variability_version') or '')))
    assert result.status == "pruned"
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    frozen = study.trials[-1]
    assert frozen.state == optuna.trial.TrialState.PRUNED
    assert 0 < frozen.intermediate_values[0] < 1000
    assert not list((tmp_path / "results/checkpoints").rglob("best_model.zip"))


def test_episode_metrics_reach_live_monitor_points(tmp_path):
    import csv
    from frontend.model import metric_points_from_api
    from src.app.telemetry.run import Run
    from src.app.telemetry.sinks import CsvSink

    sink = CsvSink(tmp_path / "version_0")
    run = Run(tmp_path / "version_0", sinks=[sink], system_interval=0)
    runner.EpisodeMetrics(run).record([{"episode": {"r": 31, "l": 31}}], 31)
    run.finish()
    with (tmp_path / "version_0" / "metrics.csv").open() as f:
        points = metric_points_from_api(list(csv.DictReader(f)))
    assert points[0]["train/episode_reward"] == 31
    assert points[0]["train/reward"] == 31
