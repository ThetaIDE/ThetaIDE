"""Tests for the IDE's generic model and component extension contracts.

Plugin-specific architectures and algorithms are tested by their own packages.
These checks cover only the registry and protocols the IDE exposes to them.
"""

from __future__ import annotations

import torch.nn as nn

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


