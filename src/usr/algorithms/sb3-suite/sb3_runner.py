"""Universal Stable-Baselines3 Runner for Theta-IDE.

Executes SB3 algorithms (PPO, DQN, A2C, SAC) as a decoupled black-box engine,
logging training telemetry through src.app.telemetry (metrics.csv plus the events.jsonl live feed,
by default) and saving checkpoints adhering to the Theta-IDE execution contract.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.app import telemetry


@hydra.main(version_base=None, config_path="../../../../in/config", config_name="config")
def main(cfg: DictConfig):
    try:
        import stable_baselines3 as sb3
        from stable_baselines3 import A2C, DQN, PPO, SAC
        from stable_baselines3.common.callbacks import BaseCallback
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

    ckpt_dir = Path("results/checkpoints") / group / exp_id / agent_name.replace("/", "_")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Custom SB3 Callback to bridge telemetry into the Theta-IDE telemetry run
    class ThetaMetricsCallback(BaseCallback):
        def __init__(self, run: telemetry.Run, eval_freq: int = 1000):
            super().__init__()
            self.run = run
            self.eval_freq = eval_freq
            self.last_eval_step = 0

        def _on_step(self) -> bool:
            # Check episodic rewards from monitor or locals
            if self.n_calls % self.eval_freq == 0:
                infos = self.locals.get("infos", [])
                ep_rewards = []
                for info in infos:
                    if "episode" in info:
                        ep_rewards.append(info["episode"]["r"])

                metrics = {}
                if ep_rewards:
                    mean_r = float(sum(ep_rewards) / len(ep_rewards))
                    metrics["eval/reward"] = mean_r
                    metrics["train/reward"] = mean_r

                # Extract loss metrics from SB3 logger if available
                if hasattr(self.model, "logger") and hasattr(self.model.logger, "name_to_value"):
                    for k, v in self.model.logger.name_to_value.items():
                        if "loss" in k:
                            metrics[f"losses/{k}"] = float(v)

                self.run.log(
                    metrics,
                    step=self.n_calls,
                    transitions=self.num_timesteps,
                )
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

    cb = ThetaMetricsCallback(run=logger, eval_freq=1000)

    start_time = time.time()
    status = "failed"
    logger.event("stage", "Training started", stage="training")
    try:
        model.learn(total_timesteps=total_timesteps, callback=cb)
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


if __name__ == "__main__":
    main()
