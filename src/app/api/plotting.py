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

Figure data (SDK 3.3; ``src.app.core.plotting.figure_data``)
    ``metric_runs``, ``Run``, ``by_method``, ``smooth``, ``aggregate``, ``final_scores``, ``iqm``,
    ``STATISTICS``, ``bootstrap_interval``, ``performance_profile``, ``study_summaries``,
    ``X_QUANTITIES``, ``X_LABELS``, ``SMOOTHING``, ``BANDS``

Figure specs (SDK 3.3; ``src.app.core.plotting.vega_specs``)
    ``vega_lite_figure``, ``live_params``, ``auto_text``, ``zoom_param``, ``legend_param``,
    ``legend_opacity``, ``hover_param``, ``hover_opacity``, ``color_encoding``,
    ``VEGA_LITE_SCHEMA``, ``LEGEND_POSITIONS``, ``NUMBER_FORMAT``
"""
from ._exports import lazy_exports

_BASE = "src.app.core.plotting.base"
_EXPORTS = {name: f"{_BASE}:{name}" for name in (
    "BasePlotter", "PanelSpec", "QUALITATIVE_PALETTE", "clean_label", "get_style", "get_style_info",
    "get_method_aliases", "get_canonical_method_name", "moving_average", "deep_update")}
_EXPORTS["FIGURE_CONTRACT"] = "src.app.core.plotting.figures:FIGURE_CONTRACT"
_EXPORTS["thin_xy"] = "src.app.core.plotting.thinning:thin_xy"
_EXPORTS.update({name: f"src.app.core.plotting.figure_data:{name}" for name in (
    "metric_runs", "Run", "by_method", "smooth", "aggregate", "final_scores", "iqm", "STATISTICS",
    "bootstrap_interval", "performance_profile", "study_summaries", "X_QUANTITIES", "X_LABELS",
    "SMOOTHING", "BANDS")})
_EXPORTS.update({name: f"src.app.core.plotting.vega_specs:{name}" for name in (
    "vega_lite_figure", "live_params", "auto_text", "zoom_param", "legend_param", "legend_opacity",
    "hover_param", "hover_opacity", "color_encoding", "VEGA_LITE_SCHEMA", "LEGEND_POSITIONS", "NUMBER_FORMAT")})
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
