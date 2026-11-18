"""Vega-Lite building blocks shared by Discussion figure providers (figure/vega-lite-v1).

They keep the standard figures consistent and within the contract:

- ``vega_lite_figure`` — the outer spec: schema, live params, sizing to the figure card, data.
- ``live_params`` — a top-level param per live parameter, with its current value.
- ``auto_text`` — an expression that shows a live text parameter, or a fallback when it is empty.
- ``zoom_param`` — zoom and pan by binding an interval selection to the scales; double-click resets.
- ``legend_param`` / ``legend_opacity`` — clicking a legend entry highlights that series.
- ``hover_param`` / ``hover_opacity`` — the nearest point under the pointer, for tooltips.
- ``color_encoding`` — method colours from ``resolve_method_styles``, with the legend position.
- ``number_format`` — the d3 format tooltips use for numbers.

Only ``"scales"`` and ``"legend"`` bindings are allowed by the contract; parameters bound to
input controls are rejected, because Theta's parameter box owns the controls.
"""

from __future__ import annotations

import json

VEGA_LITE_SCHEMA = "https://vega.github.io/schema/vega-lite/v6.json"
LEGEND_POSITIONS = ("right", "bottom", "top", "left", "none")
LEGEND_PARAM = "legend_pick"
HOVER_PARAM = "hover"
ZOOM_PARAM = "zoom"
DIMMED = 0.15
NUMBER_FORMAT = ".4~g"


def live_params(values: dict, names) -> list[dict]:
    """The top-level ``params`` entries for live parameters, in ``names`` order."""
    return [{"name": name, "value": values[name]} for name in names]


def auto_text(param: str, fallback: str) -> dict:
    """``{"expr": …}`` showing ``param`` when it is non-empty, else ``fallback``."""
    return {"expr": f"{param} || {json.dumps(str(fallback))}"}


def zoom_param(name: str = ZOOM_PARAM, encodings=("x", "y")) -> dict:
    return {"name": name, "select": {"type": "interval", "encodings": list(encodings)}, "bind": "scales"}


def legend_param(field: str = "method", name: str = LEGEND_PARAM) -> dict:
    return {"name": name, "select": {"type": "point", "fields": [field]}, "bind": "legend"}


def legend_opacity(on, name: str = LEGEND_PARAM, off: float = DIMMED) -> dict:
    """Full ``on`` opacity (a number or ``{"expr": …}``) for the picked legend entries, or for
    every series while nothing is picked; ``off`` for the others."""
    return {"condition": {"param": name, "empty": True, "value": on}, "value": off}


def hover_param(fields=("x", "method"), name: str = HOVER_PARAM) -> dict:
    return {"name": name, "select": {"type": "point", "fields": list(fields), "nearest": True,
                                     "on": "pointerover", "clear": "pointerout"}}


def hover_opacity(name: str = HOVER_PARAM) -> dict:
    return {"condition": {"param": name, "empty": False, "value": 1}, "value": 0}


def color_encoding(labels, colors, *, field: str = "method", title: str = "Method",
                   legend: str = "right") -> dict:
    """Nominal colour by ``field`` with explicit ``labels`` → ``colors``; ``legend="none"`` hides it."""
    if legend not in LEGEND_POSITIONS:
        raise ValueError(f"Unknown legend position {legend!r}; use one of {', '.join(LEGEND_POSITIONS)}")
    encoding = {"field": field, "type": "nominal", "title": title,
                "scale": {"domain": list(labels), "range": list(colors)}}
    # Symbols stay opaque: series layers dim themselves, not the legend.
    encoding["legend"] = None if legend == "none" else {"orient": legend, "symbolOpacity": 1}
    return encoding


def vega_lite_figure(values: dict, live, data: list, *, title: dict | str | None = None, container: bool = True,
                     **body) -> dict:
    """The outer spec: its live params, inline data and ``body``.

    ``container`` sizes a single view to the figure card. Faceted and concatenated specs
    cannot fill a container; pass ``container=False`` and give their panels a size instead.
    """
    spec = {"$schema": VEGA_LITE_SCHEMA, "params": live_params(values, live)}
    if title is not None:
        spec["title"] = title if isinstance(title, dict) and "text" in title else {"text": title}
    if container:
        spec.update({"width": "container", "height": "container", "autosize": {"type": "fit", "contains": "padding"}})
    spec["data"] = {"values": data}
    spec.update(body)
    return spec
