"""Performance profile (figure/vega-lite-v1): the fraction of runs above each score threshold."""

from src.app.api.plotting import BasePlotter, auto_text, performance_profile, vega_lite_figure

from .common import colors, hover_points, interactive_line, metric_title, tooltip, x_axis
from .final_performance import SCORE_TITLES, method_scores

LIVE = ["title", "line_width", "opacity", "x_label", "y_label"]


class PerformanceProfilePlotter(BasePlotter):
    """Figure only: ``run`` writes nothing."""

    def __init__(self, *, paths):
        super().__init__("performance_profile", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        pass

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        metric = params["metric"]
        _, scores, styles, methods = method_scores(self, exp_id, params)
        labelled = {styles[method]["label"]: scores[method] for method in methods}
        rows = [{**row, "x": row["threshold"], "y": row["fraction"]} for row in performance_profile(labelled)]
        x_title = f"{SCORE_TITLES[params['score']]} {metric_title(metric)} threshold"
        y = {"field": "y", "type": "quantitative", "scale": {"domain": [0, 1]},
             "axis": {"title": auto_text("y_label", "Fraction of runs above"), "format": "%"}}
        line = interactive_line("true", y)
        line["mark"]["interpolate"] = "step-after"
        tips = tooltip(("method", "Method", "nominal"), ("x", "Threshold", "quantitative"),
                       ("y", "Fraction above", "quantitative"))
        return vega_lite_figure(
            params, LIVE, rows, title=auto_text("title", f"{exp_id} · {metric} profile"),
            encoding={"x": {"field": "x", "type": "quantitative", "axis": x_axis(x_title), "scale": {"zero": False}},
                      "color": colors(styles, methods, params["legend"])},
            layer=[line, hover_points("true", y, tips)])
