"""Universal plugin and entry-point discovery system for Theta-IDE.

Discovers, validates, and registers components distributed as Python packages via
standard pyproject.toml entry-point groups:
  - theta.models: Pluggable model architectures
  - theta.agents: Pluggable RL algorithms / methods
  - theta.environments: Pluggable domain environments and grounding hooks
  - theta.datamodules: Pluggable data modules and replay buffer samplers
  - theta.evaluators: Pluggable evaluation, safety, and verification hooks
  - theta.plugins: Pluggable Theta-IDE GUI plugins
"""

from __future__ import annotations

import importlib
import logging
from importlib.metadata import entry_points
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# Standard entry point groups recognized across Theta-IDE
THETA_ENTRY_POINTS = {
    "models": "theta.models",
    "agents": "theta.agents",
    "environments": "theta.environments",
    "datamodules": "theta.datamodules",
    "evaluators": "theta.evaluators",
    "plugins": "theta.plugins",
    "components": "theta.components",
    "tuners": "theta.tuners",
}

_DISCOVERY_CACHE: Dict[str, Dict[str, Any]] = {}
_DISCOVERY_PERFORMED: bool = False
ENVIRONMENT_REGISTRY: Dict[str, Any] = {}


def register_environment(*names: str):
    """Register environment classes or factories provided by installed packages."""

    def decorator(factory):
        for name in names:
            ENVIRONMENT_REGISTRY[name.lower()] = factory
        return factory

    return decorator


def discover_entry_points(group: str, force_reload: bool = False) -> Dict[str, Any]:
    """Discover and load all registered entry points for a given group.

    Args:
        group: Standard entry point group name (e.g. 'theta.models').
        force_reload: If True, bypass the internal cache.

    Returns:
        Dictionary mapping entry point name -> loaded object/callable.
    """
    if not force_reload and group in _DISCOVERY_CACHE:
        return _DISCOVERY_CACHE[group]

    discovered: Dict[str, Any] = {}
    try:
        eps = entry_points(group=group)
        for ep in eps:
            try:
                loaded = ep.load()
                discovered[ep.name] = loaded
                log.debug("Loaded entry point [%s] -> %s", group, ep.name)
            except Exception as e:
                log.warning("Failed to load entry point [%s:%s]: %s", group, ep.name, e)
    except Exception as e:
        log.debug("Entry point discovery error for group '%s': %s", group, e)

    _DISCOVERY_CACHE[group] = discovered
    return discovered


def load_all_plugins(force_reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """Scan and register all standard Theta entry point groups.

    Automatically populates:
      - Model registry (@register_model)
      - Agent registry (@register_agent)
      - Paradigm component registry (@register_component)
      - Environment registry (for VectorizedBaseEnv.from_name())
    """
    global _DISCOVERY_PERFORMED
    results: Dict[str, Dict[str, Any]] = {}

    for key, group in THETA_ENTRY_POINTS.items():
        found = discover_entry_points(group, force_reload=force_reload)
        results[key] = found

    # 1. Register discovered models
    from src.app.core.model_registry import register_model

    for name, model_cls in results.get("models", {}).items():
        if isinstance(model_cls, type):
            register_model(name)(model_cls)

    # 2. Register discovered agents / algorithms
    from src.app.core.agent_registry import register_agent

    for name, agent_cls in results.get("agents", {}).items():
        if isinstance(agent_cls, type):
            register_agent(name)(agent_cls)

    # 3. Register discovered components into paradigm loader
    from src.app.core.paradigm_loader import register_component

    for name, comp_cls in results.get("datamodules", {}).items():
        if isinstance(comp_cls, type):
            register_component(name)(comp_cls)
    for name, comp_cls in results.get("components", {}).items():
        if isinstance(comp_cls, type):
            register_component(name)(comp_cls)

    # 4. Register environment classes and factories for VectorizedBaseEnv.from_name().
    for name, env_factory in results.get("environments", {}).items():
        register_environment(name)(env_factory)

    _DISCOVERY_PERFORMED = True
    return results
