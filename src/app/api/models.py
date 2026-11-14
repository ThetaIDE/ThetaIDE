"""Model provider registration and construction.

Registration
    ``register_model``   decorator registering a model class or factory under one or more names.

Construction
    ``build_model``      instantiate the configured model for an environment and device.
    ``get_model_class``  the declared provider class for a model name.

Model protocols
    ``ActionResult``             what an actor returns for a batch of observations.
    ``walk_model_modules``       every nested model module, for hooks that visit the whole model.
    ``DynamicTopologyProtocol``, ``ExtraStateProtocol``, ``HasModelCallbacks``
                                 optional behaviors the host detects on a model.

Accessing these exports imports PyTorch.
"""
from ._exports import lazy_exports

_REGISTRY = "src.app.core.model_registry"
_EXPORTS = {
    **{name: f"{_REGISTRY}:{name}" for name in ("register_model", "build_model", "get_model_class")},
    "ActionResult": "src.app.core.types:ActionResult",
    **{name: f"src.app.core.protocols:{name}" for name in (
        "walk_model_modules", "DynamicTopologyProtocol", "ExtraStateProtocol", "HasModelCallbacks")},
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
