"""Universal Stable-Baselines3 Runner for Theta-IDE.

Executes SB3 algorithms (PPO, DQN, A2C, SAC) as a decoupled black-box engine,
logging training telemetry through src.app.telemetry (metrics.csv plus the events.jsonl live feed,
by default) and saving checkpoints adhering to the Theta-IDE execution contract.
"""

from __future__ import annotations

import os
import math
import sys
import time
from collections import deque
from numbers import Real
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.app import telemetry


class EpisodeMetrics:
    """Collect every completed training episode, including between logger dumps."""

    def __init__(self, run, window=100):
        self.run = run
        self.rewards = deque(maxlen=window)
        self.lengths = deque(maxlen=window)

    def record(self, infos, transitions):
        for info in infos:
            episode = info.get("episode")
            if not episode:
                continue
            reward, length = float(episode["r"]), float(episode["l"])
            self.rewards.append(reward)
            self.lengths.append(length)
            self.run.log({"train/episode_reward": reward, "train/episode_length": length,
                          "train/reward": sum(self.rewards) / len(self.rewards),
                          "train/length": sum(self.lengths) / len(self.lengths)},
                         step=transitions, transitions=transitions)


class ThetaSB3Writer:
    """Forward SB3's actual logger dumps, once per update, to Theta telemetry."""

    def __init__(self, run, callback):
        self.run, self.callback = run, callback

    def write(self, key_values, key_excluded, step=0):
        metrics = {}
        for key, value in key_values.items():
            if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
                continue
            if key.startswith("train/"):
                name = f"losses/{key}" if "loss" in key else f"diagnostics/{key[6:]}"
            elif key in ("time/fps", "time/time_elapsed"):
                name = key
            else:
                continue
            metrics[name] = float(value)
        if metrics:
            transitions = self.callback.num_timesteps
            self.run.log(metrics, step=transitions, transitions=transitions)

    def close(self):
        pass


@hydra.main(version_base=None, config_path="../../../../in/config", config_name="config")
def main(cfg: DictConfig):
    try:
        import stable_baselines3 as sb3
        from stable_baselines3 import A2C, DQN, PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.logger import KVWriter
    except ImportError:
        print(
            "\n[Error] Stable-Baselines3 is not installed in the active environment.\n"
            "Please run: pip install stable-baselines3 shimmy\n"
            "or install the SB3 plugin via the Theta Hub.\n"
        )
        sys.exit(1)

    try:
        import gymnasium as gym
    except ImportError:
        print("\n[Error] Gymnasium is required for SB3. Run: pip install gymnasium\n")
        sys.exit(1)

    group = str(cfg.get("group", "default_group"))
    exp_id = str(cfg.get("experiment_id", "default_exp"))
    agent_cfg = cfg.get("agent", {})
    agent_name = str(agent_cfg.get("name", "sb3_agent"))
    algo_name = str(agent_cfg.get("algorithm", "PPO")).upper()
    policy_type = str(agent_cfg.get("policy", "MlpPolicy"))
    total_timesteps = int(cfg.get("total_timesteps", agent_cfg.get("total_timesteps", 50000)))

    # Determine environment ID
    env_cfg = cfg.get("env", {})
    env_name = env_cfg.get("name")
    env_id = env_cfg.get("env_id") or env_name or "CartPole-v1"
    if env_id == "cartpole":
        env_id = "CartPole-v1"
    elif env_id == "mountaincar":
        env_id = "MountainCar-v0"

    print(f"\n=== [SB3 Engine] Initializing {algo_name} on {env_id} ===")
    print(f"Group: {group} | Experiment: {exp_id} | Total Timesteps: {total_timesteps}")

    # Telemetry: metrics.csv and the live feed by default; ++telemetry.sinks=[...] picks others
    sinks, system_interval = telemetry.settings_from_config(cfg)
    logger = telemetry.init(
        group,
        exp_id,
        agent_name,
        sinks=sinks,
        config=OmegaConf.to_container(cfg, resolve=True),
        system_interval=system_interval,
    )

    from hydra.core.hydra_config import HydraConfig

    hydra_cfg = HydraConfig.get()
    is_sweep = hydra_cfg.mode.name == "MULTIRUN"
    ckpt_dir = Path("results/checkpoints") / group / exp_id / agent_name.replace("/", "_")
    if is_sweep:
        ckpt_dir = ckpt_dir / "sweeps" / str(hydra_cfg.sweeper.study_name) / f"trial_{hydra_cfg.job.num}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Custom SB3 Callback to bridge telemetry into the Theta-IDE telemetry run
    class ThetaMetricsCallback(BaseCallback):
        def __init__(self, run: telemetry.Run):
            super().__init__()
            self.run = run
            self.episodes = EpisodeMetrics(run)

        def _on_training_start(self) -> None:
            self.model.logger.output_formats.append(SB3TelemetryWriter(self.run, self))

        def _on_step(self) -> bool:
            self.episodes.record(self.locals.get("infos", []), self.num_timesteps)
            return True

    # Build environment with Monitor wrapper for episode stats
    from stable_baselines3.common.monitor import Monitor

    is_atari = (env_name and "atari" in str(env_name).lower()) or (
        env_id and ("ale/" in str(env_id).lower() or "atari" in str(env_id).lower())
    )

    if is_atari:
        import ale_py
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

        try:
            gym.register_envs(ale_py)
        except Exception:
            pass
        game_stem = (str(env_name).split("/")[-1] if "/" in str(env_name) else str(env_id)).lower()
        if "ale/" not in game_stem:
            game_stem = f"ALE/{game_stem.capitalize()}-v5"
        else:
            game_stem = env_id

        atari_raw = gym.make(game_stem, frameskip=1, render_mode="rgb_array")
        atari_pre = AtariPreprocessing(atari_raw, screen_size=84, grayscale_obs=True, frame_skip=4, scale_obs=False)
        atari_stacked = FrameStackObservation(atari_pre, stack_size=4)
        env = Monitor(atari_stacked)
    else:
        env = Monitor(gym.make(env_id))

    # Resolve algorithm class
    algo_map = {"PPO": PPO, "DQN": DQN, "A2C": A2C, "SAC": SAC}
    AlgoClass = algo_map.get(algo_name, PPO)

    # Hyperparameters
    algo_kwargs = {}
    lr_val = agent_cfg.get("learning_rate", agent_cfg.get("lr"))
    if lr_val is not None:
        algo_kwargs["learning_rate"] = float(lr_val)
    if "gamma" in agent_cfg:
        algo_kwargs["gamma"] = float(agent_cfg["gamma"])
    if "batch_size" in agent_cfg:
        algo_kwargs["batch_size"] = int(agent_cfg["batch_size"])
    if "n_steps" in agent_cfg:
        algo_kwargs["n_steps"] = int(agent_cfg["n_steps"])

    model_arch = cfg.get("model", {}).get("architecture") if hasattr(cfg.get("model"), "get") else None
    if model_arch == "cnn":
        policy_type = "CnnPolicy"

    model = AlgoClass(policy_type, env, verbose=1, **algo_kwargs)

    class SB3TelemetryWriter(ThetaSB3Writer, KVWriter):
        pass

    cb = ThetaMetricsCallback(run=logger)

    start_time = time.time()
    status = "failed"
    objective = None
    logger.event("stage", "Training started", stage="training")
    try:
        model.learn(total_timesteps=total_timesteps, callback=cb)
        tuning = cfg.get("tuning")
        if is_sweep:
            metric = tuning.get("metric") if tuning is not None else "eval/reward"
            if metric != "eval/reward":
                raise ValueError(f"SB3 sweep objective {metric!r} is not supported; use 'eval/reward'.")
            from stable_baselines3.common.evaluation import evaluate_policy

            objective, _ = evaluate_policy(model, env, n_eval_episodes=max(1, int(cfg.get("eval_episodes") or 5)), deterministic=True)
            objective = float(objective)
            if not math.isfinite(objective):
                raise ValueError("SB3 evaluation did not produce a finite objective.")
            logger.log({metric: objective}, step=cb.n_calls, transitions=cb.num_timesteps)
        status = "completed"
    except BaseException as exc:
        logger.event("error", f"{type(exc).__name__}: {exc}", level="error")
        raise
    finally:
        end_time = time.time()
        # Save final models
        model.save(str(ckpt_dir / "best_model.zip"))
        model.save(str(ckpt_dir / "latest_model.zip"))
        logger.save_runtime_metadata(
            config=OmegaConf.to_container(cfg, resolve=True),
            training_time_seconds=end_time - start_time,
        )
        for name in ("best_model.zip", "latest_model.zip"):
            logger.artifact(ckpt_dir / name, kind="checkpoint")
        logger.finish(status=status)
        env.close()

    print(f"\n[SB3 Engine Complete] Total time: {end_time - start_time:.2f}s")
    print(f"Checkpoints saved to: {ckpt_dir}")
    print(f"Telemetry logged to: {logger.version_dir}")
    return objective


if __name__ == "__main__":
    main()
