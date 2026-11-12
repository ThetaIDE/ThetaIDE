"""Determine which configured plotters a run should execute."""

from collections.abc import Mapping, Sequence

GENERAL_PLOT_OPTIONS = {
    "output_dir", "title", "xlabel", "ylabel", "figsize", "dpi", "style",
    "smoothing_window", "x_axis", "metrics", "filename_prefix", "output_subdir", "use_cache",
}


def requested_plots(cfg):
    """Return explicitly selected plotters, including selections inherited through composition."""
    plots = cfg.get("plots")
    if isinstance(plots, Mapping):
        return {name: options for name, options in plots.items()
                if name not in GENERAL_PLOT_OPTIONS and options is not None}
    if isinstance(plots, Sequence) and not isinstance(plots, (str, bytes)):
        return {str(name): {} for name in plots}
    return {}


def should_run_plotting(cfg):
    return not cfg.get("no_plot", True) and bool(requested_plots(cfg))
