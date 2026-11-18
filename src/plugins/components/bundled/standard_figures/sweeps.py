"""Sweep figures (figure/vega-lite-v1) from a target's tuning summaries: scatter and parallel coordinates."""

import json
import math

from src.app.api.plotting import (BasePlotter, NUMBER_FORMAT, auto_text, hover_opacity, hover_param, legend_opacity,
                                  legend_param, study_summaries, vega_lite_figure, zoom_param)

from .common import colors, finite, method_styles, tooltip, x_axis, y_encoding


def _last_intermediate(trial):
    points = []
    for step, value in (trial.get("intermediate_values") or {}).items():
        try:
            points.append((int(step), value))
        except (TypeError, ValueError):
            continue
    points = [(step, value) for step, value in sorted(points) if finite(value)]
    return points[-1][1] if points else None


def sweep_trials(plotter, exp_id: str, params: dict):
    """``(trials, names, objective, direction)``: finished trials with their parameters, the swept
    parameter names in first-seen order, the objective's name and the optimisation direction."""
    trials, names, objective, direction = [], {}, None, None
    for method, summary in study_summaries(plotter.workspace, exp_id):
        if params.get("methods") and method not in params["methods"]:
            continue
        objective = objective or summary.get("objective_metric")
        direction = direction or summary.get("direction")
        for trial in summary.get("trials") or []:
            value = trial.get("value")
            state = trial.get("state")
            if state == "pruned" and params.get("include_pruned"):
                value = _last_intermediate(trial)
            elif state != "complete":
                continue
            if not finite(value):
                continue
            parameters = trial.get("parameters") or {}
            names.update(dict.fromkeys(parameters))
            trials.append({"method": method, "study": summary.get("study"), "trial": trial.get("id"), "state": state,
                           "value": float(value), "parameters": parameters})
    if not trials:
        raise ValueError(f"{exp_id} has no finished sweep trials yet.")
    return trials, list(names), objective or "Objective", direction


def _numeric(values) -> bool:
    return all(finite(value) for value in values)


def _describe(parameters: dict) -> str:
    return ", ".join(f"{key}={value}" for key, value in parameters.items())


class SweepScatterPlotter(BasePlotter):
    """Figure only: ``run`` writes nothing."""

    def __init__(self, *, paths):
        super().__init__("sweep_scatter", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        pass

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        trials, names, objective, direction = sweep_trials(self, exp_id, params)
        parameter = params["parameter"] or names[0]
        if parameter not in names:
            raise ValueError(f"No trial of {exp_id} swept {parameter!r}; swept: {', '.join(names)}.")
        color_by = params["color_by"]
        if color_by and color_by not in names:
            raise ValueError(f"No trial of {exp_id} swept {color_by!r}; swept: {', '.join(names)}.")
        group = exp_id.split("/", 1)[0]
        methods = sorted({trial["method"] for trial in trials})
        styles = method_styles(self, exp_id, group, methods)
        rows = []
        for trial in trials:
            if parameter not in trial["parameters"]:
                continue
            row = {"method": styles[trial["method"]]["label"], "study": trial["study"], "trial": trial["trial"],
                   "state": trial["state"], "value": trial["value"], "x": trial["parameters"][parameter],
                   "parameters": _describe(trial["parameters"])}
            if color_by:
                row["c"] = trial["parameters"].get(color_by)
            rows.append(row)
        numeric_x = _numeric(row["x"] for row in rows)
        if not numeric_x:
            for row in rows:
                row["x"] = json.dumps(row["x"]) if not isinstance(row["x"], str) else row["x"]
        x_scale = params["x_scale"]
        if x_scale == "auto":
            present = [row["x"] for row in rows]
            x_scale = "log" if numeric_x and min(present) > 0 and max(present) >= 100 * min(present) else "linear"
        x = ({"field": "x", "type": "quantitative", "scale": {"type": x_scale, "nice": True, **({} if x_scale == "log" else {"zero": False})},
              "axis": x_axis(parameter)} if numeric_x else
             {"field": "x", "type": "nominal", "axis": x_axis(parameter)})
        if color_by:
            numeric_c = _numeric(row.get("c") for row in rows)
            color = {"field": "c", "type": "quantitative" if numeric_c else "nominal", "title": color_by,
                     "legend": None if params["legend"] == "none" else {"orient": params["legend"]}}
            if numeric_c:
                color["scale"] = {"scheme": "viridis"}
            opacity = {"expr": "opacity"}
            interaction = [zoom_param()]
        else:
            color = colors(styles, methods, params["legend"])
            opacity = legend_opacity({"expr": "opacity"})
            interaction = [zoom_param(), legend_param()]
        tips = tooltip(("method", "Method", "nominal"), ("study", "Study", "nominal"), ("trial", "Trial", "nominal"),
                       ("state", "State", "nominal"), ("value", objective, "quantitative"),
                       ("parameters", "Parameters", "nominal"))
        better = {"minimize": "lower is better", "maximize": "higher is better"}.get(direction)
        y_title = f"{objective} ({better})" if better else objective
        layers = [{"params": interaction,
                   "mark": {"type": "point", "filled": True, "size": {"expr": "point_size"}},
                   "encoding": {"y": y_encoding("value", params, y_title), "opacity": opacity,
                                "shape": {"field": "state", "type": "nominal", "title": "State",
                                          "legend": None if not params["include_pruned"] else {}},
                                "tooltip": tips}}]
        return vega_lite_figure(
            params, ["title", "point_size", "opacity", "x_label", "y_label", "y_min", "y_max"], rows,
            title=auto_text("title", f"{exp_id} · {objective} by {parameter}"),
            encoding={"x": x, "color": color}, layer=layers)


class SweepParallelPlotter(BasePlotter):
    """Figure only: ``run`` writes nothing."""

    def __init__(self, *, paths):
        super().__init__("sweep_parallel", paths=paths)

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        pass

    def figure_spec(self, exp_id: str, params: dict) -> dict:
        trials, names, objective, direction = sweep_trials(self, exp_id, params)
        keys = list(params["parameters"]) or names
        unknown = [key for key in keys if key not in names]
        if unknown:
            raise ValueError(f"No trial of {exp_id} swept {', '.join(unknown)}; swept: {', '.join(names)}.")
        columns = [*keys, objective]
        values = {key: [trial["parameters"].get(key) for trial in trials] for key in keys}
        values[objective] = [trial["value"] for trial in trials]

        def scaler(key):
            present = [value for value in values[key] if value is not None]
            if _numeric(present):
                low, high = min(present), max(present)
                if low > 0 and high >= 100 * low:  # log-spaced, such as learning rates
                    span = math.log(high / low)
                    return lambda v: math.log(v / low) / span, (low, high)
                return lambda v: 0.5 if high == low else (v - low) / (high - low), (low, high)
            ordered = sorted({json.dumps(value) for value in present})
            return (lambda v: ordered.index(json.dumps(v)) / max(1, len(ordered) - 1),
                    (json.loads(ordered[0]), json.loads(ordered[-1])) if ordered else (None, None))

        scalers = {key: scaler(key) for key in columns}
        rows = []
        for trial in trials:
            identity = f"{trial['method']} · {trial['study']} · {trial['trial']}"
            for key in columns:
                raw = trial["value"] if key == objective else trial["parameters"].get(key)
                if raw is None:
                    continue
                rows.append({"kind": "value", "line": identity, "key": key, "y": scalers[key][0](raw),
                             "raw": raw if isinstance(raw, str) else json.dumps(raw), "objective": trial["value"],
                             "method": trial["method"], "trial": trial["trial"]})
        for key in columns:
            low, high = scalers[key][1]
            for position, value in ((0, low), (1, high)):
                text = (format(value, NUMBER_FORMAT.replace("~", "")) if finite(value) and not isinstance(value, int)
                        else str(value))
                rows.append({"kind": "label", "key": key, "y": position, "top": 1, "text": text})

        x = {"field": "key", "type": "nominal", "sort": columns, "title": None,
             "scale": {"type": "point", "padding": 0.05},
             "axis": {"orient": "top", "labelAngle": 0, "domain": False, "ticks": False}}
        y = {"field": "y", "type": "quantitative", "axis": None, "scale": {"domain": [0, 1]}}
        legend = None if params["legend"] == "none" else {"orient": params["legend"]}
        color = {"field": "objective", "type": "quantitative", "title": objective, "legend": legend,
                 "scale": {"scheme": "viridis", "reverse": direction == "minimize"}}
        tips = tooltip(("method", "Method", "nominal"), ("trial", "Trial", "nominal"), ("key", "Axis", "nominal"),
                       ("raw", "Value", "nominal"), ("objective", objective, "quantitative"))
        layers = [
            {"transform": [{"filter": "datum.kind === 'label' && datum.y === 0"}], "mark": {"type": "rule",
             "color": "#8a8a8a", "strokeWidth": 1}, "encoding": {"x": x, "y": y, "y2": {"field": "top"}}},
            {"transform": [{"filter": "datum.kind === 'value'"}],
             "mark": {"type": "line", "strokeWidth": {"expr": "line_width"}, "opacity": {"expr": "opacity"}},
             "encoding": {"x": x, "y": y, "detail": {"field": "line"}, "color": color}},
            {"transform": [{"filter": "datum.kind === 'value'"}], "params": [hover_param(("line",))],
             "mark": {"type": "point", "filled": True, "size": 40},
             "encoding": {"x": x, "y": y, "color": color, "opacity": hover_opacity(), "tooltip": tips}},
            *({"transform": [{"filter": f"datum.kind === 'label' && datum.y === {position}"}],
               "mark": {"type": "text", "dx": 4, "align": "left", "baseline": baseline, "fontSize": 10,
                        "color": "#8a8a8a"},
               "encoding": {"x": x, "y": y, "text": {"field": "text"}}}
              for position, baseline in ((0, "bottom"), (1, "top"))),
        ]
        return vega_lite_figure(params, ["title", "line_width", "opacity"], rows,
                                title=auto_text("title", f"{exp_id} · sweep"), layer=layers,
                                config={"view": {"stroke": None}})
