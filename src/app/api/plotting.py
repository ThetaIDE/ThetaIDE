"""Plotter base class and shared figure styling.

Plotters
    ``BasePlotter``  base class of plotter capabilities. Constructors accept the
                     host's workspace as ``paths`` and pass it on:
                     ``def __init__(self, *, paths): super().__init__("name", paths=paths)``.
    ``PanelSpec``    one composable panel returned by ``BasePlotter.panels``.

Styling
    ``clean_label``, ``get_style``, ``get_style_info``, ``get_method_aliases``,
    ``get_canonical_method_name``, ``moving_average``, ``deep_update``, ``QUALITATIVE_PALETTE``
"""
from ._exports import lazy_exports

_BASE = "src.app.core.plotting.base"
_EXPORTS = {name: f"{_BASE}:{name}" for name in (
    "BasePlotter", "PanelSpec", "QUALITATIVE_PALETTE", "clean_label", "get_style", "get_style_info",
    "get_method_aliases", "get_canonical_method_name", "moving_average", "deep_update")}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
