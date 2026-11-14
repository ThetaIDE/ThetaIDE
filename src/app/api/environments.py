"""Environment provider base classes.

    ``VectorizedBaseEnv``  base class of vectorized environment capabilities.
    ``ActionCount``        integer action count that also describes the action space.

Accessing these exports imports PyTorch.
"""
from ._exports import lazy_exports

_EXPORTS = {name: f"src.app.core.env_vectorized:{name}" for name in ("VectorizedBaseEnv", "ActionCount")}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
