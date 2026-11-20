"""Run variability (figure/vega-lite-v1): one panel per method, every run and the mean."""

from src.app.api.plotting import (BasePlotter, aggregate, auto_text, by_method, hover_param, metric_runs, smooth,
                                  thin_xy, vega_lite_figure, zoom_param)

from .common import data_title, load_runs, method_styles, metric_title, tooltip, x_axis, y_encoding

LIVE = ["title", "line_width", "run_opacity", "x_label", "y_label", "y_min", "y_max"]


class RunVariabilityPlotter(BasePlotter):
    """Figure only: ``run`` writes nothing."""

    def __init__(self, *, paths):
        super().__init__("run_variability", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        pass

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        group, runs_data = load_runs(self, exp_id, params)
        metric = params["metric"]
        runs, x_title = metric_runs(runs_data, metric, params["x_quantity"], common_x=bool(params.get("sources")))
        if not runs:
            raise ValueError(f"No run of {data_title(exp_id, params)} logged {metric!r}.")
        grouped = by_method(runs)
        methods = sorted(grouped)
        styles = method_styles(self, exp_id, group, methods, sources=params.get("sources"))
        rows = []
        for method in methods:
            label = styles[method]["label"]
            smoothed = []
            for run in grouped[method]:
                y = smooth(run.y, params["smoothing"], params["smoothing_window"])
                smoothed.append((run.x, y))
                rows += [{"kind": "run", "method": label, "run": run.version, "x": float(px), "y": float(py)}
                         for px, py in thin_xy(list(zip(run.x.tolist(), y.tolist())))]
            x, mean, _, _ = aggregate(smoothed, "none")
            rows += [{"kind": "mean", "method": label, "run": "mean", "x": float(px), "y": float(py)}
                     for px, py in thin_xy(list(zip(x.tolist(), mean.tolist())))]

        labels = [styles[m]["label"] for m in methods]
        y_title = metric_title(metric)
        y = y_encoding("y", params, y_title)
        color = {"field": "method", "type": "nominal", "legend": None,
                 "scale": {"domain": labels, "range": [styles[m]["color"] for m in methods]}}
        tips = tooltip(("method", "Method", "nominal"), ("run", "Run", "nominal"), ("x", x_title, "quantitative"),
                       ("y", y_title, "quantitative"))
        panel = {
            "width": params["panel_width"], "height": params["panel_height"],
            "encoding": {"x": {"field": "x", "type": "quantitative", "axis": x_axis(x_title)}, "color": color},
            "layer": [
                {"transform": [{"filter": "datum.kind === 'run'"}], "params": [zoom_param()],
                 "mark": {"type": "line", "strokeWidth": 1, "opacity": {"expr": "run_opacity"}},
                 "encoding": {"y": y, "detail": {"field": "run"}}},
                {"transform": [{"filter": "datum.kind === 'mean'"}],
                 "mark": {"type": "line", "strokeWidth": {"expr": "line_width"}},
                 "encoding": {"y": y}},
                {"transform": [{"filter": "datum.kind === 'run'"}], "params": [hover_param(("x", "run"))],
                 "mark": {"type": "point", "filled": True, "size": 36},
                 "encoding": {"y": y, "opacity": {"condition": {"param": "hover", "empty": False, "value": 1},
                                                  "value": 0}, "tooltip": tips}},
            ],
        }
        return vega_lite_figure(
            params, LIVE, rows, title=auto_text("title", f"{data_title(exp_id, params)} · {metric} by run"), container=False,
            facet={"field": "method", "type": "nominal", "title": None, "sort": labels},
            columns=params["columns"], spec=panel, resolve={"scale": {"y": "shared"}})
