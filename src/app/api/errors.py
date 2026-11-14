"""Shared exception types.

``ConfigurationError``  an actionable recipe/configuration problem reported before
                        work starts. It is a ``ValueError``.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "ConfigurationError": "src.app.pipeline.exceptions:ConfigurationError",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
