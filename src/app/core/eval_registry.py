"""Evaluator Registry — Single source of truth for pluggable model evaluators & verification protocols.

Evaluators compute verification metrics, rollouts, out-of-distribution tests, or domain scores.
They self-register using the @register_evaluator decorator:
    @register_evaluator("early_pred_auprc")
    class EarlyPredAUPRCEvaluator(BaseEvalProtocol):
        ...

The pipeline resolves evaluators via get_evaluator_class():
    EvalClass = get_evaluator_class("early_pred_auprc")
"""

import importlib
import os
import pkgutil
from typing import Any, Callable, Dict, List, Optional, Type

EVALUATOR_REGISTRY: Dict[str, Type[Any]] = {}


def register_evaluator(*names: str) -> Callable[[Type[Any]], Type[Any]]:
    """Decorator to register an Evaluator class under one or more names."""

    def decorator(cls: Type[Any]) -> Type[Any]:
        for name in names:
            clean_name = name.strip()
            EVALUATOR_REGISTRY[clean_name] = cls
            EVALUATOR_REGISTRY[clean_name.lower()] = cls
            # Also register into paradigm_loader component registry for cross-compatibility
            try:
                from src.app.core.paradigm_loader import register_component

                register_component(clean_name)(cls)
            except Exception:
                pass
        return cls

    return decorator


def get_evaluator_class(name):
    if not name: return None
    from src.app.core.provider_loader import resolve_provider
    selected = resolve_provider("evaluator", str(name))
    if selected is not None: return selected
    _register_builtins()
    value = EVALUATOR_REGISTRY.get(str(name)) or EVALUATOR_REGISTRY.get(str(name).lower())
    return value if value is not None and value.__module__.startswith('src.app.') else None


def auto_discover_evaluators() -> None:
    """Auto-discover evaluator modules in src/usr/eval/ and plugin entrypoints."""
    from src.app.core.provider_loader import discover_kind
    discover_kind("evaluator")
    _register_builtins()


def _register_builtins() -> None:
    """Ensure core built-in eval protocols are registered."""
    try:
        from src.app.core.paradigm_impls.base.online_rl import EpisodicRewardEvalProtocol

        register_evaluator("EpisodicRewardEvalProtocol")(EpisodicRewardEvalProtocol)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.offline_rl import OfflineRLEvalProtocol

        register_evaluator("OfflineRLEvalProtocol")(OfflineRLEvalProtocol)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.supervised import ClassificationEvalProtocol

        register_evaluator("ClassificationEvalProtocol")(ClassificationEvalProtocol)
    except Exception:
        pass

    try:
        from src.app.core.paradigm_impls.base.unsupervised import ReconstructionEvalProtocol

        register_evaluator("ReconstructionEvalProtocol")(ReconstructionEvalProtocol)
    except Exception:
        pass


def list_registered_evaluators() -> List[str]:
    """Return sorted list of registered evaluator names."""
    from src.app.core.provider_catalog import get_catalog
    names = {key for p in get_catalog().packages for c in p.capabilities if c.kind == "evaluator" for key in (c.id, *c.aliases)}
    from src.app.core.provider_metadata import BUILTIN_SELECTORS
    return sorted(names | set(BUILTIN_SELECTORS["evaluator"]))
