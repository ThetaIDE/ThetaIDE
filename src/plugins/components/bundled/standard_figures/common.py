"""What the standard figures share: loading a target's runs, method styles, labels and layers."""

import math

from src.app.api.plotting import (NUMBER_FORMAT, QUALITATIVE_PALETTE, auto_text, color_encoding, hover_opacity,
                                  hover_param, legend_opacity, legend_param, load_selected_metrics, zoom_param)


def load_runs(plotter, exp_id: str, params: dict) -> tuple[str, dict]:
    """Load the explicit results, or the home target with its existing method filters."""
    group = exp_id.split("/", 1)[0]
    if params.get("sources"):
        return group, load_selected_metrics(plotter.workspace, params["sources"])
    runs_data = plotter.load_metrics(group, exp_id)
    if params.get("methods"):
        runs_data = {name: versions for name, versions in runs_data.items() if name in params["methods"]}
    return group, runs_data


def data_title(exp_id: str, params: dict) -> str:
    return "Selected results" if params.get("sources") else exp_id


def method_styles(plotter, exp_id: str, group: str, methods: list, *, sources=()) -> dict:
    """``{method: {"label", "color", …}}`` from the experiment's method styles, as saved plots use."""
    if sources:
        styles = {}
        for index, key in enumerate(methods):
            target, _, method = key.rpartition("/")
            styles[key] = {"label": f"{method} · {target}",
                           "color": QUALITATIVE_PALETTE[index % len(QUALITATIVE_PALETTE)]}
        return styles
    experiment = plotter.get_experiment_config(exp_id)
    cfg = {**plotter.default_cfg, "methods": experiment.get("methods") or {}}
    return plotter.resolve_method_styles(list(methods), cfg, group)


def colors(styles: dict, methods: list, legend: str, *, field: str = "method", title: str = "Method") -> dict:
    labels = [styles[method]["label"] for method in methods]
    return color_encoding(labels, [styles[method]["color"] for method in methods], field=field, title=title,
                          legend=legend)


def metric_title(metric: str) -> str:
    return metric.replace("_", " ")


def finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def y_encoding(field: str, params: dict, fallback_title: str, **extra) -> dict:
    """A quantitative y with the shared axis parameters the figure declares: live ``y_label``,
    ``y_min`` and ``y_max``; ``y_scale`` regenerates."""
    scale = {"type": params.get("y_scale", "linear"), "nice": True}
    if scale["type"] != "log":
        scale["zero"] = False
    if "y_min" in params:
        scale.update({"domainMin": {"expr": "y_min"}, "domainMax": {"expr": "y_max"}})
    title = auto_text("y_label", fallback_title) if "y_label" in params else fallback_title
    return {"field": field, "type": "quantitative", "scale": scale, "axis": {"title": title}, **extra}


def x_axis(fallback_title: str) -> dict:
    return {"title": auto_text("x_label", fallback_title)}


def tooltip(*fields) -> list:
    """Tooltip entries: ``(field, title)`` pairs; numbers use the shared format."""
    entries = []
    for field, title, kind in fields:
        entry = {"field": field, "title": title, "type": kind}
        if kind == "quantitative":
            entry["format"] = NUMBER_FORMAT
        entries.append(entry)
    return entries


def interactive_line(filter_expr: str, y: dict, *, width="line_width", opacity="opacity", detail=None) -> dict:
    """The main series: zoom and pan, legend toggling, live line width and opacity."""
    layer = {"transform": [{"filter": filter_expr}], "params": [zoom_param(), legend_param()],
             "mark": {"type": "line", "strokeWidth": {"expr": width}},
             "encoding": {"y": y, "opacity": legend_opacity({"expr": opacity})}}
    if detail:
        layer["encoding"]["detail"] = {"field": detail}
    return layer


def hover_points(filter_expr: str, y: dict, tips: list, *, fields=("x", "method")) -> dict:
    """Invisible points that show the nearest value and its tooltip under the pointer."""
    return {"transform": [{"filter": filter_expr}], "params": [hover_param(fields)],
            "mark": {"type": "point", "filled": True, "size": 48},
            "encoding": {"y": y, "opacity": hover_opacity(), "tooltip": tips}}
