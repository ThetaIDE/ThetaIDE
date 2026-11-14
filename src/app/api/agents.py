"""Lightning agents of the base paradigms.

    ``OnlineAgentBase``, ``OfflineAgentBase``  Lightning base classes for algorithm capabilities.
    ``register_agent``                         decorator registering an agent class under one or more names.
    ``get_agent_class``                        the declared provider class for an algorithm name.
    ``list_registered_agents``                 every declared algorithm name and alias.
    ``register_component``                     decorator registering a training component (a runner or
                                               Lightning data module) that paradigm configs name.

Accessing these exports imports PyTorch Lightning.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "OnlineAgentBase": "src.app.core.base_agent:OnlineAgentBase",
    "OfflineAgentBase": "src.app.core.base_agent:OfflineAgentBase",
    "register_agent": "src.app.core.agent_registry:register_agent",
    "get_agent_class": "src.app.core.agent_registry:get_agent_class",
    "list_registered_agents": "src.app.core.agent_registry:list_registered_agents",
    "register_component": "src.app.core.paradigm_loader:register_component",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
