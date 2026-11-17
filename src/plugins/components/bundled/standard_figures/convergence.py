#!/usr/bin/env python3
import json
import math

from src.app.api.plotting import BasePlotter, thin_xy


class ConvergencePlotter(BasePlotter):
    def __init__(self, *, paths):
        super().__init__("convergence", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        cfg, group, output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics", ["eval/reward", "train/reward", "train/length"])
        self.plot_metric_series(exp_id, group, output_dir, metrics, cfg)

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        """One metric's smoothed mean per method, with a standard-error band (figure/vega-lite-v1).

        Live parameters: line_width, opacity and title. The rest change the data.
        """
        group = exp_id.split("/", 1)[0]
        metric = params["metric"]
        runs_data = self.load_metrics(group, exp_id)
        if params["methods"]:
            runs_data = {name: versions for name, versions in runs_data.items() if name in params["methods"]}
        active = self.active_methods_for(runs_data, metric)
        if not active:
            raise ValueError(f"No run of {exp_id} logged {metric!r}.")

        experiment = self.get_experiment_config(exp_id)
        cfg = {**self.default_cfg, "methods": experiment.get("methods") or {},
               "smoothing_window": params["smoothing_window"]}
        styles = self.resolve_method_styles(active, cfg, group)
        series, used_xlabel = self.collect_metric_series({name: runs_data[name] for name in active}, metric, cfg)
        xlabel, ylabel, automatic_title = self.resolve_labels(metric, exp_id, cfg, used_xlabel)

        rows, labels, colors = [], [], []
        for method, (x, y, sem) in series.items():
            label = styles[method]["label"]
            labels.append(label)
            colors.append(styles[method]["color"])
            bands = sem.tolist() if sem is not None else [None] * len(y)
            points = [(float(px), float(py), band) for px, py, band in zip(x.tolist(), y.tolist(), bands)
                      if math.isfinite(px) and math.isfinite(py)]
            for px, py, band in thin_xy(points):
                row = {"method": label, "x": px, "y": py}
                if band is not None and math.isfinite(band):
                    row["lo"], row["hi"] = py - band, py + band
                rows.append(row)

        return {
            "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
            "params": [
                {"name": "line_width", "value": params["line_width"]},
                {"name": "opacity", "value": params["opacity"]},
                {"name": "title", "value": params["title"]},
            ],
            "title": {"text": {"expr": f"title || {json.dumps(automatic_title)}"}},
            "width": "container",
            "height": "container",
            "autosize": {"type": "fit", "contains": "padding"},
            "data": {"values": rows},
            "encoding": {
                "x": {"field": "x", "type": "quantitative", "title": xlabel},
                "color": {"field": "method", "type": "nominal", "title": "Method",
                          "scale": {"domain": labels, "range": colors}},
            },
            "layer": [
                {"transform": [{"filter": "isValid(datum.lo)"}],
                 "mark": {"type": "area", "opacity": 0.15},
                 "encoding": {"y": {"field": "lo", "type": "quantitative"}, "y2": {"field": "hi"}}},
                {"mark": {"type": "line", "strokeWidth": {"expr": "line_width"}, "opacity": {"expr": "opacity"}},
                 "encoding": {"y": {"field": "y", "type": "quantitative", "title": ylabel}}},
            ],
        }
