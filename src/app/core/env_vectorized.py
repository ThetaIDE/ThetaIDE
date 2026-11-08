"""Vectorized environment interface and Gymnasium adapter."""

from abc import ABC
from collections.abc import Sequence
from typing import Any

import torch

from src.app.core.utils import load_module


class ActionCount(int):
    """Integer representing action count that can also be called as a nullary function for backwards compatibility."""

    def __call__(self) -> int:
        return int(self)


class VectorizedBaseEnv(ABC):
    """Base abstract class for vectorized benchmark environments.

    Adheres to the standard Gymnasium vectorized environment protocol:
    - reset() -> obs tensor of shape (n_envs, *obs_shape)
    - step(actions) -> (next_obs, rewards, terminations, truncations, infos)
    """

    name: str
    pred2action: dict[str, int] = {}

    def reset(self, seed=None) -> torch.Tensor:
        """Reset all environments and return batched observations.

        Returns:
            Observation tensor of shape (n_envs, *obs_shape).
        """
        raise NotImplementedError

    def step(self, actions) -> tuple[torch.Tensor, Any, Any, Any, Any]:
        """Step all environments with batched actions.

        Returns:
            Tuple of (next_obs, rewards, terminations, truncations, infos).
        """
        raise NotImplementedError

    def get_action_meanings(self) -> Sequence[str]:
        return list(self.pred2action.keys())

    @property
    def n_actions(self) -> ActionCount:
        if getattr(self, "_n_actions_val", None) is not None:
            return ActionCount(self._n_actions_val)
        if hasattr(self, "pred2action") and self.pred2action:
            return ActionCount(len(list(set(self.pred2action.items()))))
        return ActionCount(getattr(self, "_n_actions", 2))

    @n_actions.setter
    def n_actions(self, val: int):
        self._n_actions_val = int(val)

    @staticmethod
    def from_name(name: str, **kwargs):
        """Instantiate a declared environment provider lazily."""
        import inspect
        from src.app.core.provider_loader import resolve_provider
        provider = resolve_provider("env", name)
        if provider is None: raise ValueError(f"No declared environment provider: {name!r}")
        target = provider.__init__ if isinstance(provider, type) else provider
        sig = inspect.signature(target)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        accepted = dict(kwargs) if accepts_kwargs else {k: v for k, v in kwargs.items() if k in sig.parameters}
        if "name" in sig.parameters or accepts_kwargs: accepted["name"] = name
        env = provider(**accepted)
        if env is None or not all(callable(getattr(env, method, None)) for method in ("reset", "step")):
            raise TypeError(f"Environment provider {name!r} must implement reset() and step()")
        return env

    def close(self):
        if hasattr(self, "env") and hasattr(self.env, "close"):
            self.env.close()
        elif hasattr(self, "envs"):
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()
        elif hasattr(self, "venv") and hasattr(self.venv, "close"):
            self.venv.close()


class StandardGymVectorEnv(VectorizedBaseEnv):
    """General-purpose vectorized wrapper for standard Gymnasium environments."""

    def __init__(
        self,
        env_id: str,
        n_envs: int = 1,
        seed: int | None = None,
        **kwargs,
    ):
        import gymnasium as gym

        self.name = env_id
        self.n_envs = max(1, int(n_envs))
        self.seed = seed
        self.venv = gym.make_vec(env_id, num_envs=self.n_envs, vectorization_mode="sync", **kwargs)

        obs_space = getattr(self.venv, "single_observation_space", None) or getattr(
            self.venv, "observation_space", None
        )
        self.observation_space = getattr(obs_space, "shape", (1,))

        act_space = getattr(self.venv, "single_action_space", None) or getattr(self.venv, "action_space", None)
        if hasattr(act_space, "n"):
            self._n_actions = int(act_space.n)
        elif hasattr(act_space, "shape"):
            self._n_actions = int(act_space.shape[0])
        else:
            self._n_actions = 2

        self.pred2action = {f"action_{i}": i for i in range(self._n_actions)}

    def reset(self, seed=None) -> torch.Tensor:
        seed_to_use = seed if seed is not None else self.seed
        obs, _ = self.venv.reset(seed=seed_to_use)
        return torch.as_tensor(obs, dtype=torch.float32)

    def step(self, actions, is_mapped: bool = False) -> tuple[torch.Tensor, Any, Any, Any, Any]:
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        obs, rewards, terminations, truncations, infos = self.venv.step(actions)
        return (
            torch.as_tensor(obs, dtype=torch.float32),
            rewards,
            terminations,
            truncations,
            infos,
        )

    @property
    def n_actions(self) -> ActionCount:
        return ActionCount(getattr(self, "_n_actions", 2))

    @n_actions.setter
    def n_actions(self, val: int):
        self._n_actions = int(val)
