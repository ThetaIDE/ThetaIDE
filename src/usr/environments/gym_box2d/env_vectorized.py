"""Local Gymnasium Box2D provider; observations retain Gymnasium's native units."""
from __future__ import annotations

import gymnasium as gym
import torch

from src.app.api.environments import ActionCount, VectorizedBaseEnv


class VectorizedEnv(VectorizedBaseEnv):
    name = "gym_box2d"
    pred2action = {"noop": 0, "fire_left": 1, "fire_main": 2, "fire_right": 3}

    def __init__(self, name=None, env_id="LunarLander-v3", n_envs=1, seed=None,
                 mode="ppo", **kwargs):
        super().__init__()
        if env_id != "LunarLander-v3":
            raise ValueError("This Box2D provider currently supports discrete LunarLander-v3 only.")
        if isinstance(n_envs, bool) or not isinstance(n_envs, int) or n_envs < 1:
            raise ValueError("n_envs must be a positive integer.")
        self.env_id, self.n_envs, self.seed, self.mode = env_id, n_envs, seed, mode
        self.venv = gym.make_vec(env_id, num_envs=n_envs, vectorization_mode="sync")
        self.observation_space = self.venv.single_observation_space.shape
        self.n_features = self.observation_space[0]

    def reset(self, seed=None):
        obs, _ = self.venv.reset(seed=self.seed if seed is None else seed)
        return torch.as_tensor(obs, dtype=torch.float32)

    def step(self, actions, is_mapped=False):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs, rewards, terminated, truncated, infos = self.venv.step(actions)
        return torch.as_tensor(obs, dtype=torch.float32), rewards, terminated, truncated, infos

    @property
    def n_actions(self):
        return ActionCount(4)

    def close(self):
        self.venv.close()
