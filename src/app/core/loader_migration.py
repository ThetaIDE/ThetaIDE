"""Explicit conversion of legacy environment hooks to experiment input bindings."""
from __future__ import annotations

from src.app.core.data_loaders import plain


def migrate_loader_config(config, *, state_provider=None):
    """Return a new configuration and changed paths. Never edits the original.

    A provider must be supplied for legacy state hooks: migration must not guess
    which implementation the experiment author intends to install or use.
    """
    result = plain(config)
    changed = []

    def migrate(node, prefix=""):
        env = node.get("env")
        if isinstance(env, dict) and "data_module" in env:
            node.setdefault("data_module", env.pop("data_module"))
            changed.append(prefix + "env.data_module")
        builder = env.pop("symbolic_observation_builder", None) if isinstance(env, dict) else None
        if builder is not None:
            changed.append(prefix + "env.symbolic_observation_builder")
        model = node.get("model")
        symbolic = model.get("symbolic") if isinstance(model, dict) else None
        if isinstance(symbolic, dict) and "observation_builder" in symbolic:
            builder = symbolic.pop("observation_builder")
            changed.append(prefix + "model.symbolic.observation_builder")
        if builder is not None and node.get("symbolic_state_loader") is None:
            if not state_provider:
                raise ValueError("Choose a symbolic state provider before migrating legacy observation builders.")
            node["symbolic_state_loader"] = {"name": state_provider, "params": {
                "builder": builder, "object_count": None,
            }}
        methods = node.get("methods") or {}
        for name, method in methods.items():
            if isinstance(method, dict) and name not in {"params", "shared_params", "common_params"}:
                migrate(method, prefix + "methods." + name + ".")

    migrate(result)
    return result, changed
