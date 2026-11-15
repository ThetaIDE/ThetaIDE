"""Tests for the IDE's generic model and component extension contracts.

Plugin-specific architectures and algorithms are tested by their own packages.
These checks cover only the registry and protocols the IDE exposes to them.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.app.core.model_registry import MODEL_REGISTRY, build_model, register_model
from src.app.core.protocols import (
    DynamicTopologyProtocol,
    ExtraStateProtocol,
    HasModelCallbacks,
    walk_model_modules,
)


class DummyDynamicModel(nn.Module, DynamicTopologyProtocol, ExtraStateProtocol, HasModelCallbacks):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self._changed = False

    def forward(self, x):
        return self.linear(x)

    def has_topology_changed(self) -> bool:
        return self._changed

    def reset_topology_changed(self) -> None:
        self._changed = False

    def clone_topology_to(self, target: nn.Module) -> None:
        target.load_state_dict(self.state_dict())

    def get_callbacks(self) -> list:
        return []

    def extra_state(self) -> dict:
        return {"dummy_key": 42}

    def load_extra_state(self, state: dict) -> None:
        self.dummy_val = state.get("dummy_key")


class DummyComposite:
    def __init__(self, policy_modules):
        self.policy_modules = policy_modules


def test_register_and_build_custom_model():
    @register_model("test_dummy_plugin")
    def create_dummy(obs_dim=4, n_actions=2, **kwargs):
        return nn.Linear(obs_dim, n_actions)

    assert "test_dummy_plugin" in MODEL_REGISTRY
    model = build_model("test_dummy_plugin", obs_dim=8, n_actions=3)
    assert isinstance(model, nn.Linear)
    assert model.in_features == 8
    assert model.out_features == 3


def test_model_protocols_are_structural():
    model = DummyDynamicModel()
    assert isinstance(model, DynamicTopologyProtocol)
    assert isinstance(model, ExtraStateProtocol)
    assert isinstance(model, HasModelCallbacks)


def test_walk_model_modules_handles_standalone_and_composite_models():
    first = DummyDynamicModel()
    second = nn.Linear(4, 2)

    assert first in walk_model_modules(first)
    assert first in walk_model_modules(DummyComposite([first, second]))
    assert second in walk_model_modules(DummyComposite([first, second]))


def test_standard_gymnasium_vector_env_resolution():
    from src.app.core.env_vectorized import StandardGymVectorEnv, VectorizedBaseEnv

    env = VectorizedBaseEnv.from_name("CartPole-v1", n_envs=2)
    assert isinstance(env, StandardGymVectorEnv)
    assert env.n_actions == 2

    obs = env.reset()
    assert obs.shape == (2, 4)
    next_obs, rewards, term, trunc, infos = env.step(torch.tensor([0, 1]))
    assert next_obs.shape == (2, 4)
    assert len(rewards) == 2
    env.close()
