"""Paired evaluation protocol for the editable runner used by local experiments."""
import copy
import csv
import importlib.util
import json
import random
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


spec = importlib.util.spec_from_file_location(
    "sb3_dev_evaluation", Path(__file__).resolve().parents[1] / "src/usr/algorithms/sb3-dev/sb3_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_paired_rollouts_match_resets_reproduce_and_preserve_training_state(algorithm):
    resets = []

    class RecordReset(gym.Wrapper):
        def reset(self, **kwargs):
            obs, info = super().reset(**kwargs)
            resets.append(obs.copy())
            return obs, info

    env = DummyVecEnv([lambda: Monitor(RecordReset(gym.make("CartPole-v1", max_episode_steps=16)))])
    if algorithm == "ppo":
        model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, seed=73, device="cpu")
    else:
        model = DQN("MlpPolicy", env, buffer_size=32, learning_starts=100, seed=73, device="cpu")
        model.exploration_rate = .5
    model.policy.train()
    next(model.policy.children()).eval()  # preserve mixed module modes too
    modes = [module.training for module in model.policy.modules()]
    action_space = model.action_space
    action_rng = copy.deepcopy(action_space.np_random.bit_generator.state)
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    weights = {key: value.clone() for key, value in model.policy.state_dict().items()}
    try:
        results = runner.evaluate_policy_modes(model, env, episodes=3, seed=100)
        assert set(results) == {"greedy", "sampled"}
        assert all(len(values) == 3 for values in results.values())
        # DummyVecEnv auto-resets once after the last episode in each mode.
        np.testing.assert_array_equal(resets[:4], resets[4:])
        assert [module.training for module in model.policy.modules()] == modes
        assert model.action_space is action_space
        assert model.action_space.np_random.bit_generator.state == action_rng
        assert random.getstate() == python_state
        after_numpy = np.random.get_state()
        assert numpy_state[0] == after_numpy[0] and numpy_state[2:] == after_numpy[2:]
        np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
        assert torch.equal(torch.get_rng_state(), torch_state)
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, weights[key], rtol=0, atol=0)
        assert runner.evaluate_policy_modes(model, env, episodes=3, seed=100) == results
    finally:
        env.close()


def test_failed_evaluation_also_restores_rng_and_policy_mode(monkeypatch):
    env = DummyVecEnv([lambda: Monitor(gym.make("CartPole-v1"))])
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=4, seed=73, device="cpu")
    model.policy.train()
    state = torch.get_rng_state()

    def fail(*args, **kwargs):
        model.policy.eval()
        torch.rand(17)
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr("stable_baselines3.common.evaluation.evaluate_policy", fail)
    try:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            runner.evaluate_policy_modes(model, env, episodes=1, seed=100)
        assert model.policy.training
        assert torch.equal(torch.get_rng_state(), state)
    finally:
        env.close()


@pytest.mark.parametrize("repeated,evaluation_seed,experiment_id", [
    (False, None, "e"), (True, 1234, "e"), (True, 0, "path/dir/e"), (False, None, "path/dir/e"),
])
def test_runner_records_both_modes_and_keeps_greedy_checkpoint_selection(
    tmp_path, monkeypatch, repeated, evaluation_seed, experiment_id,
):
    calls = []

    def evaluate(model, env, **kwargs):
        greedy = kwargs["deterministic"]
        calls.append((model._n_updates, greedy, env._seeds[0], kwargs["n_eval_episodes"]))
        # Sampled score improves while greedy score falls: best must stay step zero.
        rewards = ([500., 500.] if model._n_updates == 0 else [100., 100.]) if greedy else (
            [10., 30.] if model._n_updates == 0 else [200., 300.])
        return rewards, [1, 1]

    monkeypatch.setattr("stable_baselines3.common.evaluation.evaluate_policy", evaluate)
    monkeypatch.setattr(HydraConfig, "get", staticmethod(lambda: SimpleNamespace(mode=SimpleNamespace(name="RUN"))))
    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({
        "group": "g", "experiment_id": experiment_id, "total_timesteps": 16, "seed": 73,
        "eval_freq_steps": 8, "eval_episodes": 2,
        "agent": {"name": "ppo", "algorithm": "PPO", "n_steps": 8, "batch_size": 4, "n_epochs": 1},
        "model": {"sb3_native_policy": True}, "env": {"env_id": "CartPole-v1"},
        "telemetry": {"system_interval": 0},
        "variability": {"evaluation_seed": None},
    })
    if repeated:
        cfg.update({"variability_run_id": "run-1", "variability_method": "ppo",
                    "variability_version": "version_0", "variability_evaluation_seed": evaluation_seed})
    runner.main.__wrapped__(cfg)
    base = evaluation_seed if repeated else 10000000
    assert calls == [(0, True, base, 2), (0, False, base, 2),
                     (0, True, base + 1, 2), (0, False, base + 1, 2),
                     (2, True, base + 2, 2), (2, False, base + 2, 2)]
    directory = tmp_path / "results/logs/g" / experiment_id / "ppo/version_0"
    rows = list(csv.DictReader((directory / "metrics.csv").open()))
    evaluations = [row for row in rows if row.get("eval/greedy_reward")]
    assert [float(row["step"]) for row in evaluations] == [0, 8, 16]
    assert [float(row["eval/sampled_reward"]) for row in evaluations] == [20, 20, 250]
    assert all(row["eval/reward"] == row["eval/greedy_reward"] for row in evaluations)
    assert float(evaluations[-1]["eval/sampled_reward_std"]) == 50
    events = [json.loads(line) for line in (directory / "evaluation_events.jsonl").read_text().splitlines()]
    assert [(event["training_step"], event["evaluation_mode"]) for event in events] == [
        (step, mode) for step in (0, 8, 16) for mode in ("greedy", "sampled")]
    assert events[-1]["episode_outcomes"] == [200, 300]
    assert events[-1]["deterministic_actions"] is False
    assert [event["metric"] for event in events[-2:]] == ["eval/reward", "eval/sampled_reward"]
    if repeated:
        assert all(event["run_id"] == "run-1" for event in events)
    root = tmp_path / "results/checkpoints/g" / experiment_id / "ppo"
    if repeated:
        root /= "version_0"
    assert PPO.load(root / "best_model.zip")._n_updates == 0
    assert PPO.load(root / "latest_model.zip")._n_updates == 2
    runtime = json.loads((directory / "runtime.json").read_text())
    assert runtime["evaluation_protocol"]["primary_mode"] == "greedy"


def test_metrics_reach_live_monitor_and_have_explicit_labels(tmp_path):
    from frontend.model import available_metrics, metric_points_from_api
    from frontend.monitor.panel import headline_metric
    from frontend.results.metrics_catalog import describe
    from src.app.telemetry.run import Run

    with Run(tmp_path, ["csv", "stream"], system_interval=0) as run:
        run.log({"eval/reward": 500., "eval/greedy_reward": 500., "eval/sampled_reward": 20.}, step=0)
    records = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    rows = [{**record["data"], "step": record["step"]} for record in records if record["type"] == "metrics"]
    current = {"metrics": metric_points_from_api(rows)}
    assert {"eval/greedy_reward", "eval/sampled_reward"} <= available_metrics(current)
    key, info = headline_metric(current, {})
    assert key == "reward" and info.label == "Greedy evaluation reward"
    assert describe("eval/sampled_reward").label == "Sampled evaluation reward"
