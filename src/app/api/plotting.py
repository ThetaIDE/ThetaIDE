"""Plotter base class and shared figure styling.

Plotters
    ``BasePlotter``  base class of plotter capabilities. Constructors accept the
                     host's workspace as ``paths`` and pass it on:
                     ``def __init__(self, *, paths): super().__init__("name", paths=paths)``.
    ``PanelSpec``    one composable panel returned by ``BasePlotter.panels``.

Styling
    ``clean_label``, ``get_style``, ``get_style_info``, ``get_method_aliases``,
    ``get_canonical_method_name``, ``moving_average``, ``deep_update``, ``QUALITATIVE_PALETTE``

Discussion figures
    ``FIGURE_CONTRACT``  ``"figure/vega-lite-v1"``: a plotter capability that lists it in
                         ``provides`` implements ``figure_spec(exp_id, params)``.
    ``thin_xy``          thin long ``(x, y)`` series for drawing, keeping extrema and the last point.
"""
from ._exports import lazy_exports

_BASE = "src.app.core.plotting.base"
_EXPORTS = {name: f"{_BASE}:{name}" for name in (
    "BasePlotter", "PanelSpec", "QUALITATIVE_PALETTE", "clean_label", "get_style", "get_style_info",
    "get_method_aliases", "get_canonical_method_name", "moving_average", "deep_update")}
_EXPORTS["FIGURE_CONTRACT"] = "src.app.core.plotting.figures:FIGURE_CONTRACT"
_EXPORTS["thin_xy"] = "src.app.core.plotting.thinning:thin_xy"
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
