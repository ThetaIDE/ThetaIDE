"""Data helpers shared by Discussion figure providers. Headless: numpy and pandas only.

A provider loads ``runs_data`` with ``BasePlotter.load_metrics`` (``{method: {version: DataFrame}}``),
then:

- ``metric_runs`` turns it into one ``Run`` (x, y) per logged version of a metric;
- ``smooth`` and ``aggregate`` give a per-method curve with an uncertainty band;
- ``final_scores`` reduces each run to one number, and ``bootstrap_interval`` / ``STATISTICS``
  summarise those numbers per method;
- ``performance_profile`` gives the fraction of runs above each score threshold;
- ``study_summaries`` reads a target's tuning summaries (``schema_version`` 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import moving_average

# Columns a provider may offer as the x quantity, in the order Results prefers them.
X_QUANTITIES = ("transitions", "step", "epoch", "time/total")
X_LABELS = {"transitions": "Transitions", "step": "Training steps", "epoch": "Epoch",
            "time/total": "Time (s)", "index": "Logged point"}
SMOOTHING = ("moving_average", "ema", "none")
BANDS = ("sem", "ci95", "std", "none")
# Two-sided 97.5% Student t quantiles for 1–30 degrees of freedom; larger samples use 1.96.
_T975 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
         2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
         2.052, 2.048, 2.045, 2.042)


@dataclass(frozen=True)
class Run:
    """One logged version of a metric, sorted by x, with finite values only."""

    method: str
    version: str
    x: np.ndarray
    y: np.ndarray


def _x_values(df: pd.DataFrame, rows: pd.Index, quantity: str) -> tuple[np.ndarray, str]:
    for column in dict.fromkeys((quantity, "step", "epoch")):
        if column in df.columns:
            full = pd.to_numeric(df[column], errors="coerce").interpolate(method="linear").ffill().bfill()
            values = full.loc[rows]
            if not values.empty and not values.isna().any() and values.nunique() > 1:
                return values.to_numpy(dtype=float), column
    return np.arange(len(rows), dtype=float), "index"


def metric_runs(runs_data: dict, metric: str, x_quantity: str = "transitions") -> tuple[list[Run], str]:
    """Every version's (x, y) for ``metric``, and the x label of the quantity actually used.

    The x quantity falls back to ``step``, ``epoch`` and finally the logged-point index for a
    run that did not log it, as the saved plots do.
    """
    runs, used = [], None
    for method, versions in sorted(runs_data.items()):
        for version, df in sorted(versions.items()):
            if metric not in df.columns:
                continue
            y = pd.to_numeric(df[metric], errors="coerce")
            rows = y.dropna().index
            if rows.empty:
                continue
            x, column = _x_values(df, rows, x_quantity)
            yv = y.loc[rows].to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(yv)
            order = np.argsort(x[keep], kind="stable")
            runs.append(Run(method, str(version), x[keep][order], yv[keep][order]))
            used = used or column
    return runs, X_LABELS.get(used or x_quantity, used or x_quantity)


def by_method(runs: list[Run]) -> dict[str, list[Run]]:
    grouped: dict[str, list[Run]] = {}
    for run in runs:
        grouped.setdefault(run.method, []).append(run)
    return grouped


def smooth(y: np.ndarray, method: str = "moving_average", window: int = 10) -> np.ndarray:
    """A trailing moving average or an exponential moving average (alpha = 2 / (window + 1))."""
    y = np.asarray(y, dtype=float)
    if method == "none" or window <= 1 or len(y) == 0:
        return y
    if method == "ema":
        return pd.Series(y).ewm(alpha=2 / (window + 1), adjust=False).mean().to_numpy()
    if method == "moving_average":
        return moving_average(y, window)
    raise ValueError(f"Unknown smoothing {method!r}; use one of {', '.join(SMOOTHING)}")


def t_quantile(degrees: int) -> float:
    return _T975[degrees - 1] if 1 <= degrees <= len(_T975) else 1.96


def _unique_increasing(x, y):
    x, index = np.unique(x, return_index=True)
    return x, np.asarray(y)[index]


def aggregate(runs: list[tuple[np.ndarray, np.ndarray]], band: str = "sem"):
    """The mean curve of several (x, y) runs and its band: ``(x, mean, lower, upper)``.

    Runs are interpolated onto the densest run's x, limited to the range every run covers.
    ``lower`` and ``upper`` are None with one run or ``band="none"``. ``sem`` is the standard
    error, ``ci95`` a Student t 95% interval of the mean and ``std`` one standard deviation.
    """
    if band not in BANDS:
        raise ValueError(f"Unknown band {band!r}; use one of {', '.join(BANDS)}")
    runs = [_unique_increasing(x, y) for x, y in runs if len(x)]
    if not runs:
        empty = np.array([])
        return empty, empty, None, None
    if len(runs) == 1:
        return runs[0][0], runs[0][1], None, None
    start, stop = max(x[0] for x, _ in runs), min(x[-1] for x, _ in runs)
    densest = max(runs, key=lambda run: len(run[0]))[0]
    grid = densest[(densest >= start) & (densest <= stop)]
    if len(grid) == 0:  # the runs do not overlap
        return runs[0][0], runs[0][1], None, None
    values = np.vstack([np.interp(grid, x, y) for x, y in runs])
    mean = values.mean(axis=0)
    if band == "none":
        return grid, mean, None, None
    spread = values.std(axis=0, ddof=1)
    n = len(runs)
    width = {"sem": spread / math.sqrt(n), "std": spread, "ci95": t_quantile(n - 1) * spread / math.sqrt(n)}[band]
    return grid, mean, mean - width, mean + width


def final_scores(runs: list[Run], *, fraction: float = 0.1, score: str = "final") -> dict[str, list[float]]:
    """One number per run: the mean of its last ``fraction`` of logged points (``final``), or
    its highest (``best``) or lowest (``lowest``) value."""
    scores: dict[str, list[float]] = {}
    for run in runs:
        if not len(run.y):
            continue
        if score == "final":
            count = max(1, math.ceil(len(run.y) * fraction))
            value = float(np.mean(run.y[-count:]))
        elif score == "best":
            value = float(np.max(run.y))
        elif score == "lowest":
            value = float(np.min(run.y))
        else:
            raise ValueError(f"Unknown score {score!r}; use final, best or lowest")
        scores.setdefault(run.method, []).append(value)
    return scores


def iqm(values, axis=-1):
    """Interquartile mean: the mean of the middle 50% (all values when there are fewer than four)."""
    values = np.sort(np.asarray(values, dtype=float), axis=axis)
    n = values.shape[axis]
    cut = n // 4
    middle = np.take(values, range(cut, n - cut), axis=axis) if n >= 4 else values
    return middle.mean(axis=axis)


STATISTICS = {
    "mean": lambda values, axis=-1: np.mean(values, axis=axis),
    "median": lambda values, axis=-1: np.median(values, axis=axis),
    "iqm": iqm,
}


def bootstrap_interval(values, statistic: str = "mean", *, level: float = 0.95, samples: int = 2000,
                       seed: int = 0) -> tuple[float, float]:
    """A percentile bootstrap interval of ``statistic`` over ``values`` (seeded, so figures are stable)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        point = float(STATISTICS[statistic](values)) if len(values) else float("nan")
        return point, point
    rng = np.random.default_rng(seed)
    resampled = values[rng.integers(0, len(values), size=(samples, len(values)))]
    stats = STATISTICS[statistic](resampled, axis=1)
    tail = (1 - level) / 2 * 100
    return float(np.percentile(stats, tail)), float(np.percentile(stats, 100 - tail))


def performance_profile(scores: dict[str, list[float]], *, points: int = 101) -> list[dict]:
    """Rows ``{method, threshold, fraction}``: the fraction of each method's runs scoring above
    each of ``points`` thresholds. The thresholds span every score with a 2.5% margin, so each
    curve starts at 1 and ends at 0."""
    every = [value for values in scores.values() for value in values]
    if not every:
        return []
    low, high = min(every), max(every)
    margin = (high - low) * 0.025 or abs(low) * 0.025 or 1.0
    thresholds = np.linspace(low - margin, high + margin, points)
    rows = []
    for method, values in scores.items():
        values = np.asarray(values, dtype=float)
        for threshold in thresholds:
            rows.append({"method": method, "threshold": float(threshold),
                         "fraction": float(np.mean(values > threshold))})
    return rows


def study_summaries(paths, exp_id: str) -> list[tuple[str, dict]]:
    """``(method, summary)`` for every valid tuning summary of a target, as the monitors read them."""
    from src.app.pipeline.tuning_monitor import local_study_summaries
    group, _, experiment = exp_id.partition("/")
    return local_study_summaries(paths.root, group, experiment)
