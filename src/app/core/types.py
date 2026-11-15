"""Core data types and contracts for Theta-IDE."""

from typing import Any

import torch


class ActionResult:
    """Standard polymorphic return container for get_action_and_value().

    Supports:
      - Attribute access: res.action, res.logprob, res.entropy, res.value, res.aux
      - 4-element unpacking: action, logprob, entropy, value = res
      - Indexing: res[0]
      - Dynamic auxiliary metric access: res.get_aux("metric_name")
    """

    def __init__(
        self,
        action: torch.Tensor,
        logprob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
        aux: dict[str, Any] | None = None,
    ):
        self.action = action
        self.logprob = logprob
        self.entropy = entropy
        self.value = value
        self.aux: dict[str, Any] = aux if aux is not None else {}

    def __iter__(self):
        """Yield the 4 standard Actor-Critic elements for unpacking."""
        yield self.action
        yield self.logprob
        yield self.entropy
        yield self.value

    def __getitem__(self, index: int):
        return (self.action, self.logprob, self.entropy, self.value)[index]

    def __len__(self):
        return 4

    def get_aux(self, key: str, default: Any = None) -> Any:
        """Retrieve an auxiliary metric or tensor by key."""
        return self.aux.get(key, default)

    def __repr__(self):
        return (
            f"ActionResult(action={self.action.shape if hasattr(self.action, 'shape') else self.action}, "
            f"value={self.value.shape if hasattr(self.value, 'shape') else self.value}, "
            f"aux_keys={list(self.aux.keys())})"
        )
