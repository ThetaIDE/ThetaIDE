"""Final performance per method (figure/vega-lite-v1): bars, boxes or a strip of run scores."""

import numpy as np

from src.app.api.plotting import (STATISTICS, BasePlotter, auto_text, bootstrap_interval, final_scores, metric_runs,
                                  vega_lite_figure, zoom_param)

from .common import colors, data_title, finite, legend_opacity, legend_param, load_runs, method_styles, metric_title, tooltip, \
    y_encoding

LIVE = ["title", "opacity", "y_label", "y_min", "y_max"]
SCORE_TITLES = {"final": "Final", "best": "Best", "lowest": "Lowest"}


def method_scores(plotter, exp_id: str, params: dict):
    """``(group, scores, styles, methods)`` with ``scores = {method: [run score, …]}``."""
    group, runs_data = load_runs(plotter, exp_id, params)
    runs, _ = metric_runs(runs_data, params["metric"])
    scores = final_scores(runs, fraction=params.get("final_fraction", 0.1), score=params["score"])
    if not scores:
        raise ValueError(f"No run of {data_title(exp_id, params)} logged {params['metric']!r}.")
    methods = sorted(scores)
    return group, scores, method_styles(plotter, exp_id, group, methods, sources=params.get("sources")), methods


class FinalPerformancePlotter(BasePlotter):
    """Figure only: ``run`` writes nothing."""

    def __init__(self, *, paths):
        super().__init__("final_performance", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        pass

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        metric = params["metric"]
        _, scores, styles, methods = method_scores(self, exp_id, params)
        statistic = params["aggregate"]
        rows = []
        for method in methods:
            values = scores[method]
            label = styles[method]["label"]
            summary = {"kind": "summary", "method": label, "value": float(STATISTICS[statistic](np.asarray(values))),
                       "runs": len(values)}
            if params["interval"] == "bootstrap95" and len(values) > 1:
                summary["lo"], summary["hi"] = bootstrap_interval(values, statistic)
            rows.append(summary)
            rows += [{"kind": "run", "method": label, "value": value, "run": index + 1}
                     for index, value in enumerate(values) if finite(value)]

        y_title = f"{SCORE_TITLES[params['score']]} {metric_title(metric)}"
        y = y_encoding("value", params, y_title)
        color = colors(styles, methods, params["legend"])
        x = {"field": "method", "type": "nominal", "title": None, "sort": [styles[m]["label"] for m in methods],
             "axis": {"labelAngle": 0}}
        summary_tips = tooltip(("method", "Method", "nominal"), ("value", f"{statistic.upper()}", "quantitative"),
                               ("lo", "Lower 95%", "quantitative"), ("hi", "Upper 95%", "quantitative"),
                               ("runs", "Runs", "quantitative"))
        run_tips = tooltip(("method", "Method", "nominal"), ("run", "Run", "quantitative"),
                           ("value", y_title, "quantitative"))
        picked = legend_opacity({"expr": "opacity"})
        layers = []
        interactive = {"params": [zoom_param(encodings=("y",)), legend_param()]}
        chart = params["chart"]
        if chart == "bar":
            layers.append({"transform": [{"filter": "datum.kind === 'summary'"}], **interactive,
                           "mark": {"type": "bar", "width": {"band": 0.6}},
                           "encoding": {"y": y, "opacity": picked, "tooltip": summary_tips}})
        elif chart == "box":
            layers.append({"transform": [{"filter": "datum.kind === 'run'"}],
                           "mark": {"type": "boxplot", "extent": "min-max", "size": 28},
                           "encoding": {"y": y}})
        if any("lo" in row for row in rows):
            layers.append({"transform": [{"filter": "datum.kind === 'summary' && isValid(datum.lo)"}],
                           "mark": {"type": "rule", "strokeWidth": 1.5},  # mid grey reads on light and dark
                           "encoding": {"y": y_encoding("lo", params, y_title), "y2": {"field": "hi"},
                                        "color": {"value": "#8a8a8a"}}})
        if chart == "strip" or params["show_runs"]:
            point = {"transform": [{"filter": "datum.kind === 'run'"}],
                     "mark": {"type": "point", "filled": True, "size": 42},
                     "encoding": {"y": y, "opacity": picked, "tooltip": run_tips}}
            if chart != "bar":
                point.update(interactive)
            layers.append(point)
        if chart == "strip":
            layers.append({"transform": [{"filter": "datum.kind === 'summary'"}],
                           "mark": {"type": "tick", "thickness": 2, "size": 30},
                           "encoding": {"y": y, "tooltip": summary_tips}})
        return vega_lite_figure(params, LIVE, rows, title=auto_text("title", f"{data_title(exp_id, params)} · {metric}"),
                                encoding={"x": x, "color": color}, layer=layers)
