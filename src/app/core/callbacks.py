import os
import json
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch

from src.app.core.checkpoints import CheckpointSchedule, checkpoint_name
from src.app.core.env_vectorized import VectorizedBaseEnv


class EnvironmentEvaluatorCallback(L.Callback):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.eval_env = None
        self.last_eval_step = -1
        self.train_start_time = None
        self.cumulative_eval_time = 0
        self.eval_event_index = 0

    def on_train_start(self, trainer, pl_module):
        self.train_start_time = time.time()
        if self.last_eval_step < 0:
            self.evaluate_and_log(trainer, pl_module, transitions=0)
            self.last_eval_step = 0

    def on_train_epoch_end(self, trainer, pl_module):
        total_timesteps = getattr(pl_module.cfg, "total_timesteps", 10000)
        eval_freq = (
            pl_module.cfg.get("eval_freq_steps", None)
            or pl_module.cfg.get("eval_freq", None)
            or (
                pl_module.cfg.agent.get("eval_freq_steps")
                if hasattr(pl_module.cfg, "agent") and hasattr(pl_module.cfg.agent, "get")
                else None
            )
            or max(1, total_timesteps // 4)
        )
        current_transitions = getattr(pl_module, "global_step_count", 0)

        should_eval = False
        target_transitions = current_transitions

        if current_transitions - self.last_eval_step >= eval_freq:
            should_eval = True
        elif current_transitions >= total_timesteps and self.last_eval_step < total_timesteps:
            should_eval = True
            target_transitions = total_timesteps

        if should_eval:
            self.last_eval_step = current_transitions
            self.evaluate_and_log(trainer, pl_module, transitions=target_transitions)

    def evaluate_and_log(self, trainer, pl_module, transitions):
        eval_start = time.time()
        avg_reward, std_reward = self.evaluate(trainer, pl_module)
        eval_end = time.time()

        eval_duration = eval_end - eval_start
        self.cumulative_eval_time += eval_duration
        transitions = int(round(transitions))

        if self.cfg is not None and self.cfg.get("variability_run_id"):
            event_path = (Path("results/logs") / str(self.cfg.group) / str(self.cfg.experiment_id)
                          / str(self.cfg.get("variability_method", "run")).replace("/", "_")
                          / str(self.cfg.get("variability_version", "version_0")) / "evaluation_events.jsonl")
            event_path.parent.mkdir(parents=True, exist_ok=True)
            event = {"schema_version": 1, "run_id": str(self.cfg.variability_run_id),
                     "method": str(self.cfg.variability_method), "event_index": self.eval_event_index,
                     "training_step": transitions, "checkpoint": "in-memory policy",
                     "metric": "eval/reward", "episode_count": len(self._episode_rewards),
                     "evaluation_seed": self._evaluation_seed,
                     "seed_policy": ("(base plus event index) modulo (2^32 minus worker count); vector workers derive their streams"
                                     if self._evaluation_seed is not None else "runner's existing evaluation seed policy"),
                     "deterministic_actions": None, "episode_outcomes": self._episode_rewards}
            with event_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")
            self.eval_event_index += 1

        metrics = {"eval/reward": avg_reward, "eval/reward_std": std_reward, "transitions": float(transitions)}

        if self.train_start_time is not None:
            current_total_time = eval_end - self.train_start_time
            pure_training_time = current_total_time - self.cumulative_eval_time
            metrics["time/eval"] = eval_duration
            metrics["time/train"] = pure_training_time
            metrics["time/total"] = current_total_time

        paradigm = (
            pl_module.cfg.get("paradigm", "online_rl")
            if hasattr(pl_module.cfg, "get")
            else getattr(pl_module.cfg, "paradigm", "online_rl")
        )
        if paradigm == "offline_rl":
            metrics["epoch"] = float(pl_module.current_epoch)

        log_step = self._lightning_log_step(trainer, transitions)
        # trainer.logger is only the first logger when several are configured (CSV, then TensorBoard)
        for logger in trainer.loggers:
            logger.log_metrics(metrics, step=log_step)

        pl_module.log("eval/reward", avg_reward, prog_bar=True, on_step=False, on_epoch=True)
        pl_module.log("transitions", float(transitions), logger=False, prog_bar=True)

        print(f"Evaluation at {transitions} transitions: Avg Reward = {avg_reward} (+/- {std_reward})")

    @staticmethod
    def _lightning_log_step(trainer, transitions):
        """The step Lightning itself uses for this epoch's logs, so evaluation lines up with losses.

        Lightning logs against fit_loop.epoch_loop._batches_that_stepped (one per training batch)
        and writes end-of-epoch metrics one below it; global_step may count optimizer steps instead.
        The attribute is private, hence the fallbacks.
        """
        epoch_loop = getattr(getattr(trainer, "fit_loop", None), "epoch_loop", None)
        batches = getattr(epoch_loop, "_batches_that_stepped", None)
        if isinstance(batches, int):
            return max(0, batches - 1)
        return trainer.global_step if hasattr(trainer, "global_step") else int(transitions)

    def evaluate(self, trainer, pl_module):
        cfg = self.cfg
        pl_module.eval()

        if self.eval_env is None:
            eval_n_envs = cfg.env.get("eval_n_envs", 20) if hasattr(cfg.env, "get") else 20
            target_n_envs = min(cfg.eval_episodes, eval_n_envs)
            self.eval_env = VectorizedBaseEnv.from_name(
                cfg.env.name,
                n_envs=target_n_envs,
                seed=cfg.get("variability_evaluation_seed", cfg.seed + 100),
            )

        eval_total_rewards = []
        n_eval_envs = self.eval_env.n_envs
        eval_cumulative_rewards = np.zeros(n_eval_envs)

        seed_base = cfg.get("variability_evaluation_seed")
        self._evaluation_seed = (int(seed_base) + self.eval_event_index) % (2**32 - n_eval_envs) if seed_base is not None else None
        obs = self.eval_env.reset(seed=self._evaluation_seed) if seed_base is not None else self.eval_env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = torch.as_tensor(obs, dtype=torch.float32, device=pl_module.device)

        while len(eval_total_rewards) < cfg.eval_episodes:
            with torch.no_grad():
                res = pl_module.get_action_and_value(obs)
                action = getattr(res, "action", res[0])

            step_res = self.eval_env.step(action.cpu().numpy())
            if len(step_res) == 5:
                next_obs, reward, terminations, truncations, infos = step_res
            else:
                next_obs, reward, terminations = step_res[:3]
                truncations = [False] * len(reward)

            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=pl_module.device)

            for k in range(n_eval_envs):
                eval_cumulative_rewards[k] += reward[k]
                if terminations[k] or truncations[k]:
                    eval_total_rewards.append(eval_cumulative_rewards[k])
                    eval_cumulative_rewards[k] = 0
                    if len(eval_total_rewards) >= cfg.eval_episodes:
                        break

        self._episode_rewards = [float(value) for value in eval_total_rewards]
        return np.mean(eval_total_rewards), np.std(eval_total_rewards)

    def on_fit_end(self, trainer, pl_module):
        if self.eval_env:
            self.eval_env.close()


class PeriodicCheckpointCallback(L.Callback):
    """Saves step_<transitions>.ckpt on the experiment's schedule and reports each one to telemetry."""

    def __init__(self, ckpt_dir: str, interval: int, total: int | None):
        super().__init__()
        self.ckpt_dir = ckpt_dir
        self.schedule = CheckpointSchedule(interval, total)

    def on_train_epoch_end(self, trainer, pl_module):
        transitions = int(getattr(pl_module, "global_step_count", 0))
        if not self.schedule.due(transitions):
            return
        os.makedirs(self.ckpt_dir, exist_ok=True)
        path = os.path.join(self.ckpt_dir, f"{checkpoint_name(transitions)}.ckpt")
        trainer.save_checkpoint(path)
        from src.app.telemetry.lightning import ThetaLogger

        for lg in trainer.loggers:
            if isinstance(lg, ThetaLogger):
                lg.run.artifact(path, kind="checkpoint", transitions=transitions)
                lg.run.event(
                    "checkpoint",
                    f"Saved {Path(path).name} at {transitions:,} transitions",
                    path=path,
                    transitions=transitions,
                )
