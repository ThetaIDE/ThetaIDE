"""Evaluator providers.

    ``register_evaluator``  decorator registering an evaluator class under one or more names.
"""
from ._exports import lazy_exports

_EXPORTS = {"register_evaluator": "src.app.core.eval_registry:register_evaluator"}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
