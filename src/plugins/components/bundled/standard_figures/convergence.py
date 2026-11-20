#!/usr/bin/env python3
import math

from src.app.api.plotting import (BasePlotter, aggregate, auto_text, by_method, metric_runs, smooth, thin_xy,
                                  vega_lite_figure)

from .common import (colors, data_title, hover_points, interactive_line, load_runs, method_styles, metric_title, tooltip,
                     x_axis, y_encoding)

LIVE = ["line_width", "opacity", "title", "x_label", "y_label", "y_min", "y_max"]


class ConvergencePlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("convergence", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        """One metric's smoothed mean per method with an uncertainty band (figure/vega-lite-v1).

        Each run is smoothed, then runs are averaged per method. ``show_runs`` adds every run
        faintly. Live parameters: line width, opacity, title, axis labels and y range.
        """
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
            smoothed = [(run.x, smooth(run.y, params["smoothing"], params["smoothing_window"]))
                        for run in grouped[method]]
            x, mean, lower, upper = aggregate(smoothed, params["band"])
            bands = zip(lower, upper) if lower is not None else ((None, None) for _ in x)
            points = [(float(px), float(py), lo, hi) for px, py, (lo, hi) in zip(x, mean, bands)]
            for px, py, lo, hi in thin_xy(points):
                row = {"kind": "mean", "method": label, "x": px, "y": py}
                if lo is not None and math.isfinite(lo) and math.isfinite(hi):
                    row["lo"], row["hi"] = float(lo), float(hi)
                rows.append(row)
            if params["show_runs"]:
                for run, (rx, ry) in zip(grouped[method], smoothed):
                    rows += [{"kind": "run", "method": label, "run": f"{label} · {run.version}", "x": float(px),
                              "y": float(py)} for px, py in thin_xy(list(zip(rx.tolist(), ry.tolist())))]

        y_title = metric_title(metric)
        y = y_encoding("y", params, y_title)
        layers = []
        if any("lo" in row for row in rows):
            layers.append({"transform": [{"filter": "datum.kind === 'mean' && isValid(datum.lo)"}],
                           "mark": {"type": "area"},
                           "encoding": {"y": y_encoding("lo", params, y_title), "y2": {"field": "hi"},
                                        "opacity": {"condition": {"param": "legend_pick", "empty": True,
                                                                  "value": 0.18}, "value": 0.04}}})
        if params["show_runs"]:
            layers.append({"transform": [{"filter": "datum.kind === 'run'"}],
                           "mark": {"type": "line", "strokeWidth": 1},
                           "encoding": {"y": y, "detail": {"field": "run"},
                                        "opacity": {"condition": {"param": "legend_pick", "empty": True,
                                                                  "value": 0.3}, "value": 0.05}}})
        layers.append(interactive_line("datum.kind === 'mean'", y))
        tips = tooltip(("method", "Method", "nominal"), ("x", x_title, "quantitative"),
                       ("y", y_title, "quantitative"), ("lo", "Lower", "quantitative"),
                       ("hi", "Upper", "quantitative"))
        layers.append(hover_points("datum.kind === 'mean'", y, tips))
        return vega_lite_figure(
            params, LIVE, rows, title=auto_text("title", f"{data_title(exp_id, params)} · {metric}"),
            encoding={"x": {"field": "x", "type": "quantitative", "axis": x_axis(x_title)},
                      "color": colors(styles, methods, params["legend"])},
            layer=layers)
