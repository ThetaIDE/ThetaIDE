"""Lightning agents of the base paradigms.

    ``OnlineAgentBase``, ``OfflineAgentBase``  Lightning base classes for algorithm capabilities.
    ``register_agent``                         decorator registering an agent class under one or more names.
    ``get_agent_class``                        the declared provider class for an algorithm name.

Accessing these exports imports PyTorch Lightning.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "OnlineAgentBase": "src.app.core.base_agent:OnlineAgentBase",
    "OfflineAgentBase": "src.app.core.base_agent:OfflineAgentBase",
    "register_agent": "src.app.core.agent_registry:register_agent",
    "get_agent_class": "src.app.core.agent_registry:get_agent_class",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
