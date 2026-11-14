"""Plotter Registry — Single source of truth for pluggable experiment plotters & visualization components.

Plotters compute convergence graphs, loss curves, domain verification plots, or executive reports.
They self-register using the @register_plotter decorator:
    @register_plotter("convergence")
    class ConvergencePlotter(BasePlotter):
        ...

The pipeline resolves plotters via get_plotter_class():
    PlotterClass = get_plotter_class("convergence")
"""

import importlib
import logging
import os
import pkgutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

log = logging.getLogger(__name__)

PLOTTER_REGISTRY: Dict[str, Type[Any]] = {}


def register_plotter(*names: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register a Plotter class under one or more names."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name in names:
            clean_name = str(name).strip()
            PLOTTER_REGISTRY[clean_name] = cls
            PLOTTER_REGISTRY[clean_name.lower()] = cls
            # Also register into paradigm_loader component registry for cross-compatibility
            try:
                from src.app.core.paradigm_loader import register_component

                register_component(clean_name)(cls)
            except Exception:
                pass
        return cls

    return decorator


def get_plotter_class(name, *, paths=None):
    """Load the selected declared provider; missing declarations never use registry fallbacks."""
    from src.app.core.provider_loader import resolve_provider
    selected = resolve_provider("plotter", str(name), paths)
    return selected


def list_registered_plotters(*, paths=None) -> List[str]:
    """List metadata selectors without loading optional plotting providers."""
    from src.app.core.provider_catalog import get_catalog
    return sorted({key for p in get_catalog(paths).packages for c in p.capabilities if c.kind == "plotter" for key in (c.id, *c.aliases)})


def auto_discover_plotters():
    from src.app.core.provider_loader import discover_kind
    return discover_kind("plotter")
