"""Shared contract for the scalar objective returned by a tuning trial."""

from __future__ import annotations

import math
from typing import Any, Mapping


def objective_metric(cfg: Any, default: str) -> str:
    component = cfg.get("objective") if hasattr(cfg, "get") else None
    if hasattr(component, "get") and component.get("component"):
        return "objective/" + str(component["component"])
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    metric = tuning.get("metric") if hasattr(tuning, "get") else None
    return str(metric or default)


def objective_value(metrics: Mapping[str, Any], name: str) -> float:
    """Return the final reported scalar; missing or invalid values fail the trial."""
    if name not in metrics:
        raise ValueError(f"Tuning objective metric {name!r} was not reported by the training engine.")
    try:
        value = float(metrics[name])
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"Tuning objective metric {name!r} must be a finite scalar.") from exc
    if not math.isfinite(value):
        raise ValueError(f"Tuning objective metric {name!r} must be a finite scalar.")
    return value
