"""Metric table (figure/vega-lite-v1): each method's final metrics with their spread across runs."""

import math

import numpy as np

from src.app.api.plotting import (STATISTICS, BasePlotter, auto_text, bootstrap_interval, final_scores, metric_runs,
                                  vega_lite_figure)

from .common import load_runs, method_styles, metric_title, tooltip

DEFAULT_METRICS = ("eval/reward", "train/reward")


class MetricTablePlotter(BasePlotter):
    """Figure only: ``run`` writes nothing."""

    def __init__(self, *, paths):
        super().__init__("metric_table", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        pass

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        group, runs_data = load_runs(self, exp_id, params)
        logged = {column for versions in runs_data.values() for df in versions.values() for column in df.columns}
        metrics = list(params["metrics"]) or [metric for metric in DEFAULT_METRICS if metric in logged]
        statistic, decimals = params["aggregate"], params["decimals"]
        cells = {}
        for metric in metrics:
            runs, _ = metric_runs(runs_data, metric)
            for method, values in final_scores(runs, fraction=params["final_fraction"], score=params["score"]).items():
                cells[(method, metric)] = values
        if not cells:
            raise ValueError(f"No run of {exp_id} logged {', '.join(metrics) or 'eval/reward or train/reward'}.")
        methods = sorted({method for method, _ in cells})
        styles = method_styles(self, exp_id, group, methods)

        def fmt(value):
            return f"{value:.{decimals}f}"

        rows = []
        for metric in metrics:
            summaries = {method: float(STATISTICS[statistic](np.asarray(cells[(method, metric)])))
                         for method in methods if (method, metric) in cells}
            if not summaries:
                continue
            pick = min if params["score"] == "lowest" else max
            best = pick(summaries, key=summaries.get) if len(summaries) > 1 else None
            for method, value in summaries.items():
                values = cells[(method, metric)]
                text = fmt(value)
                spread = params["spread"]
                if len(values) > 1 and spread in ("std", "sem"):
                    deviation = float(np.std(values, ddof=1))
                    text += f" ± {fmt(deviation / math.sqrt(len(values)) if spread == 'sem' else deviation)}"
                elif len(values) > 1 and spread == "bootstrap95":
                    lo, hi = bootstrap_interval(values, statistic)
                    text += f" [{fmt(lo)}, {fmt(hi)}]"
                rows.append({"method": styles[method]["label"], "metric": metric_title(metric), "text": text,
                             "value": value, "runs": len(values), "best": method == best and params["highlight_best"]})
        labels = [styles[method]["label"] for method in methods]
        tips = tooltip(("method", "Method", "nominal"), ("metric", "Metric", "nominal"),
                       ("value", statistic.upper(), "quantitative"), ("runs", "Runs", "quantitative"))
        return vega_lite_figure(
            params, ["title", "font_size"], rows, title=auto_text("title", f"{exp_id} · {statistic} of {params['score']} scores"),
            encoding={
                "x": {"field": "metric", "type": "nominal", "title": None, "sort": [metric_title(m) for m in metrics],
                      "axis": {"orient": "top", "labelAngle": 0, "domain": False, "ticks": False, "labelFontWeight": "bold"}},
                "y": {"field": "method", "type": "nominal", "title": None, "sort": labels,
                      "axis": {"domain": False, "ticks": False}},
                "text": {"field": "text"},
                "tooltip": tips,
            },
            layer=[{"transform": [{"filter": f"{'' if best else '!'}datum.best"}],
                    "mark": {"type": "text", "fontSize": {"expr": "font_size"},
                             "fontWeight": "bold" if best else "normal"}}
                   for best in (False, True)])
