"""Validate that a PPO training budget clears Pendulum's no-learning controls."""
import argparse
import json
from pathlib import Path
import statistics

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

BUDGETS = (4096, 8192, 16384, 32768, 65536, 131072)
TRAIN_SEEDS = (947301, 947302, 947303)
EVAL_SEED = 1912744
EPISODES = 24
MARGIN = 200


def evaluate(model, seed, episodes=EPISODES):
    env = DummyVecEnv([lambda: Monitor(gym.make("Pendulum-v1"))])
    try:
        env.seed(seed)
        rewards, _ = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True,
                                     return_episode_rewards=True)
        return [float(reward) for reward in rewards]
    finally:
        env.close()


class RandomPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        return self.rng.uniform(-2.0, 2.0, size=(len(observation), 1)).astype(np.float32), state


def no_learning_controls(seeds, evaluation_seed, episodes, parameters):
    torch.set_num_threads(1)
    random = [evaluate(RandomPolicy(seed), evaluation_seed, episodes) for seed in seeds]
    untrained = []
    for seed in seeds:
        env = Monitor(gym.make("Pendulum-v1"))
        try:
            model = PPO("MlpPolicy", env, seed=seed, device="cpu", **parameters)
            untrained.append(evaluate(model, evaluation_seed, episodes))
        finally:
            env.close()
    return {"random_policy": random, "untrained_ppo": untrained}


class CalibrationPPO(PPO):
    def train(self):
        super().train()
        if self.num_timesteps in BUDGETS:
            self.record_point(self.num_timesteps, evaluate(self, EVAL_SEED))


def calibrate(output, profile):
    parameters = dict(n_steps=256, batch_size=64, n_epochs=3, learning_rate=0.0003, gamma=0.99)
    if profile == "short-horizon":
        parameters.update(n_steps=1024, n_epochs=10, learning_rate=0.001, gamma=0.95)
    torch.set_num_threads(1)
    baseline = {str(seed): evaluate(RandomPolicy(seed), EVAL_SEED) for seed in TRAIN_SEEDS}
    random_mean = statistics.mean(score for returns in baseline.values() for score in returns)
    runs = []
    result = {"env": "Pendulum-v1", "evaluation_seed": EVAL_SEED, "episodes": EPISODES,
              "margin": MARGIN, "profile": profile, "ppo_parameters": parameters,
              "random_returns": baseline, "random_mean": random_mean,
              "calibration_runs": runs, "qualified_budget": None}

    def save():
        pending = output.with_suffix(".tmp")
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(output)

    for seed in TRAIN_SEEDS:
        env = Monitor(gym.make("Pendulum-v1"))
        try:
            model = CalibrationPPO("MlpPolicy", env, seed=seed, device="cpu", **parameters)
            untrained = evaluate(model, EVAL_SEED)
            run = {"seed": seed, "untrained_returns": untrained, "points": []}
            runs.append(run)

            def record(step, rewards):
                mean = statistics.mean(rewards)
                run["points"].append({"step": step, "returns": rewards, "mean": mean})
                print(f"seed {seed}, {step} transitions: {mean:.1f}", flush=True)
                save()

            model.record_point = record
            print(f"seed {seed}, untrained: {statistics.mean(untrained):.1f}", flush=True)
            model.learn(total_timesteps=BUDGETS[-1])
        finally:
            env.close()
    for index, budget in enumerate(BUDGETS):
        if all(run["points"][index]["mean"] >=
               max(random_mean, statistics.mean(run["untrained_returns"])) + MARGIN for run in runs):
            result["qualified_budget"] = budget
            break
    save()
    print(f"Random mean {random_mean:.1f}; qualified budget {result['qualified_budget']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--profile", choices=("pilot", "short-horizon"), default="pilot")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    calibrate(args.output, args.profile)


if __name__ == "__main__":
    main()
