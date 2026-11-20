"""Measure raw rule controllers, without learned clause weights or a gate."""
import argparse
import importlib.util
import json
from pathlib import Path
import secrets

import gymnasium as gym
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--controller", choices=["full_controller", "attitude_descent"], default="full_controller")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    root = Path(__file__).resolve().parents[4]
    results = {"env_id": "LunarLander-v3", "seed": seed, "episodes": args.episodes,
               "policy": "raw deterministic controller; no learned clause weights", "controllers": {}}
    env = gym.make("LunarLander-v3")
    try:
        for name in (args.controller,):
            path = Path(__file__).resolve().parent / "resources/rules/lunarlander" / name / "valuation.py"
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            returns = []
            for episode in range(args.episodes):
                state, _ = env.reset(seed=seed + episode)
                total = 0.
                while True:
                    action = int(module.controller_action(torch.as_tensor(state)))
                    state, reward, terminated, truncated, _ = env.step(action)
                    total += reward
                    if terminated or truncated:
                        break
                returns.append(total)
            results["controllers"][name] = {"returns": returns, "mean": float(np.mean(returns)),
                                             "std": float(np.std(returns)),
                                             "fraction_return_ge_200": float(np.mean(np.array(returns) >= 200))}
    finally:
        env.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
