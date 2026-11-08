"""
Algorithm / Agent Registry — Single source of truth for algorithm class lookup.

Algorithms self-register using the @register_algorithm (or @register_agent) decorator:
    @register_algorithm("custom_algo")
    class CustomAlgorithm(OfflineAgentBase):
        ...

The pipeline resolves algorithms via get_algorithm_class() / get_agent_class():
    AgentClass = get_algorithm_class("custom_algo_tuned")
    # → Returns CustomAlgorithm (longest prefix match)
"""

import importlib
import os
import pkgutil
from typing import Any

AGENT_REGISTRY: dict[str, Any] = {}


def register_agent(*prefixes):
    """Decorator to register an algorithm class under one or more prefixes.

    Usage:
        @register_algorithm("custom_algo")
        class CustomAlgorithm(OfflineAgentBase):
            ...
    """

    def decorator(cls):
        for prefix in prefixes:
            AGENT_REGISTRY[prefix] = cls
        return cls

    return decorator


def get_agent_class(algo_name):
    """Load the selected declared provider; missing declarations never use registry fallbacks."""
    from src.app.core.provider_loader import resolve_provider
    selected = resolve_provider("algorithm", str(algo_name))
    if selected is None: raise ValueError(f"Unknown algorithm provider: {algo_name!r}; declare it in a native manifest")
    return selected


def auto_discover():
    """Import modules in src/usr/algorithms/ to trigger @register_agent decorators.

    Called once at startup (e.g., in train.py) to ensure all agents are registered
    before get_agent_class() is used.
    """
    from src.app.core.provider_loader import discover_kind
    discover_kind("algorithm")


def list_registered_agents():
    """Return sorted list of registered algorithm prefixes."""
    from src.app.core.provider_catalog import get_catalog
    agents = {key for p in get_catalog().packages for c in p.capabilities if c.kind == "algorithm" for key in (c.id, *c.aliases)}
    return sorted(agents)


# Semantic aliases aligning Algorithm terminology
register_algorithm = register_agent
get_algorithm_class = get_agent_class
list_registered_algorithms = list_registered_agents
ALGORITHM_REGISTRY = AGENT_REGISTRY
