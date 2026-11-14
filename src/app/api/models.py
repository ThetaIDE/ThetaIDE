"""Model provider registration and construction.

Registration
    ``register_model``   decorator registering a model class or factory under one or more names.

Construction
    ``build_model``      instantiate the configured model for an environment and device.
    ``get_model_class``  the declared provider class for a model name.

Accessing these exports imports PyTorch.
"""
from ._exports import lazy_exports

_REGISTRY = "src.app.core.model_registry"
_EXPORTS = {name: f"{_REGISTRY}:{name}" for name in ("register_model", "build_model", "get_model_class")}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
