"""Results data layer: turn any engine's metrics.csv into chartable series. No Qt imports.

Every numeric column a run logged becomes a metric; nothing is filtered by name. Rows keep
their gaps: Lightning writes evaluation and training values on different rows, so a series
for one metric uses only the rows that report it.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass, field
from statistics import fmean, stdev

from .metrics_catalog import AXIS_COLUMNS, COUNTER_COLUMNS, describe, group_sort_key, metric_group

Point = tuple[float, float]


def to_float(value) -> float | None:
    """A finite float, or None for blanks, text, NaN and infinities."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass
class MetricTable:
    """One metrics.csv: columns of equal length, None where a row has no value."""

    columns: dict[str, list[float | None]] = field(default_factory=dict)
    row_count: int = 0

    @classmethod
    def from_rows(cls, rows: list[dict], columns: list[str] | None = None) -> MetricTable:
        """Build from csv.DictReader-style rows (values may be text, as the API sends them)."""
        names = list(columns or [])
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        table = cls()
        for name in names:
            values = [to_float(row.get(name)) for row in rows]
            if any(v is not None for v in values):  # drop text-only and empty columns
                table.columns[name] = values
        table.row_count = len(rows)
        return table

    def extend(self, rows: list[dict]) -> None:
        """Append rows (e.g. the new rows of a run that is still training)."""
        new = MetricTable.from_rows(rows)
        for name in set(self.columns) | set(new.columns):
            old_values = self.columns.get(name, [None] * self.row_count)
            self.columns[name] = old_values + new.columns.get(name, [None] * new.row_count)
        self.row_count += new.row_count

    def truncate(self, row_count: int) -> None:
        """Keep only the first `row_count` rows (e.g. to re-read a row that may have been half-written)."""
        row_count = max(0, min(row_count, self.row_count))
        for name in list(self.columns):
            self.columns[name] = self.columns[name][:row_count]
            if all(v is None for v in self.columns[name]):
                del self.columns[name]
        self.row_count = row_count

    def row(self, index: int) -> dict[str, float]:
        """The values of one row, by column (missing values left out)."""
        return {name: values[index] for name, values in self.columns.items() if values[index] is not None}

    def axes(self) -> list[str]:
        """Columns usable as an x-axis, most useful first."""
        return [name for name in AXIS_COLUMNS if name in self.columns]

    def metrics(self) -> list[str]:
        """Every measured column (everything except pure counters), grouped and ordered for display."""
        names = [name for name in self.columns if name not in COUNTER_COLUMNS]
        return sorted(names, key=lambda n: (group_sort_key(metric_group(n)), n))

    def series(self, metric: str, x: str | None = None) -> list[Point]:
        """(x, y) points for rows that have both values. Default x: the best available axis.

        With no axis column at all, x is the row's position among rows reporting the metric.
        """
        ys = self.columns.get(metric)
        if ys is None:
            return []
        x = x or next(iter(self.axes()), None)
        if x is None or x not in self.columns:
            return [(float(i), y) for i, y in enumerate(v for v in ys if v is not None)]
        xs = self.columns[x]
        return [(xv, yv) for xv, yv in zip(xs, ys) if xv is not None and yv is not None]

    def nearest_x(self, x_axis: str, x_value: float) -> float | None:
        """The logged x closest to `x_value` (runs rarely log at exactly the same steps)."""
        xs = [x for x in self.columns.get(x_axis, []) if x is not None]
        return min(xs, key=lambda x: abs(x - x_value)) if xs else None

    def value_at(self, x_axis: str, x_value: float) -> dict[str, float]:
        """Every metric's value at one x: the inspector's 'all data at this point'.

        Lightning splits one step across several rows, so values from every row with this x merge.
        """
        xs = self.columns.get(x_axis, [])
        found: dict[str, float] = {}
        for i, xv in enumerate(xs):
            if xv is None or xv != x_value:
                continue
            for name, values in self.columns.items():
                if values[i] is not None and name not in found:
                    found[name] = values[i]
        return found


def group_metrics(names: list[str]) -> dict[str, list[str]]:
    """Metric names by group (the prefix before '/'), in display order."""
    groups: dict[str, list[str]] = {}
    for name in sorted(names, key=lambda n: (group_sort_key(metric_group(n)), n)):
        groups.setdefault(metric_group(name), []).append(name)
    return groups


def smooth(points: list[Point], factor: float) -> list[Point]:
    """Exponential moving average, matching the Training monitor's chart (0 = raw)."""
    if factor <= 0 or len(points) <= 1:
        return list(points)
    out, last = [], None
    for x, y in points:
        last = y if last is None else last * factor + y * (1.0 - factor)
        out.append((x, last))
    return out


@dataclass(frozen=True)
class AggregatePoint:
    x: float
    mean: float
    std: float
    sem: float
    n: int


def aggregate(series: list[list[Point]]) -> list[AggregatePoint]:
    """Mean, std and standard error across runs (e.g. seeds) at each x they share.

    Runs are matched on exact x values, which holds for seeds of one experiment (they
    evaluate on the same schedule). An x reported by only some runs uses those runs.
    """
    by_x: dict[float, list[float]] = {}
    for points in series:
        for x, y in points:
            by_x.setdefault(x, []).append(y)
    result = []
    for x in sorted(by_x):
        ys = by_x[x]
        sd = stdev(ys) if len(ys) > 1 else 0.0
        result.append(AggregatePoint(x, fmean(ys), sd, sd / math.sqrt(len(ys)), len(ys)))
    return result


@dataclass(frozen=True)
class Summary:
    final: float
    best: float
    best_x: float
    minimum: float
    maximum: float
    mean: float
    auc: float
    count: int


def summarize(points: list[Point], higher_is_better: bool | None = True) -> Summary | None:
    """Headline numbers for one series. `best` is the max, or the min for lower-is-better metrics.

    `auc` is the trapezoidal area under the curve over x, useful for comparing learning speed.
    """
    if not points:
        return None
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    pick = min if higher_is_better is False else max
    best_x, best = pick(points, key=lambda p: p[1])
    auc = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(points) - 1))
    return Summary(ys[-1], best, best_x, min(ys), max(ys), fmean(ys), auc, len(points))


def summarize_metric(table: MetricTable, metric: str, x: str | None = None) -> Summary | None:
    return summarize(table.series(metric, x), describe(metric).higher_is_better)


# ── Run manifests ────────────────────────────────────────────────────────────

ARTIFACT_KINDS = (
    "metrics",
    "table",
    "figure",
    "report",
    "config",
    "metadata",
    "checkpoint",
    "cache",
    "tensorboard",
    "log",
    "other",
)


@dataclass
class Manifest:
    """Every artifact of one run, as listed by catalog.run_manifest(group, experiment_id)."""

    group: str
    experiment_id: str
    agents: dict[str, list[str]]
    artifacts: list[dict]
    truncated: bool = False
    version_details: dict[str, dict[str, dict]] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict) -> Manifest:
        return cls(
            data["group"],
            data["experiment_id"],
            dict(data.get("agents") or {}),
            list(data.get("artifacts") or []),
            bool(data.get("truncated")),
            dict(data.get("version_details") or {}),
        )

    @property
    def run_key(self) -> str:
        return f"{self.group}/{self.experiment_id}"

    def by_kind(self, kind: str) -> list[dict]:
        return [a for a in self.artifacts if a.get("kind") == kind]

    def kinds(self) -> dict[str, int]:
        """How many artifacts of each kind, in display order (only kinds present)."""
        counts = {kind: 0 for kind in ARTIFACT_KINDS}
        for a in self.artifacts:
            counts[a.get("kind", "other")] = counts.get(a.get("kind", "other"), 0) + 1
        return {kind: n for kind, n in counts.items() if n}

    def metric_sources(self) -> list[tuple[str, str]]:
        """(agent, version) for every metrics.csv, the series sources of this run."""
        return [(agent, version) for agent, versions in self.agents.items() for version in versions]


# ── Series sources and chart points ──────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    """One metrics.csv: an agent of a run, at one version (None = the newest)."""

    group: str
    experiment_id: str
    agent: str
    version: str | None = None
    display_version: str | None = field(default=None, compare=False)
    trial_version: str | None = field(default=None, compare=False)

    @property
    def run_key(self) -> str:
        return f"{self.group}/{self.experiment_id}"

    @property
    def label(self) -> str:
        """Legend text: experiment / agent, plus the version when one is pinned."""
        text = f"{self.experiment_id} / {self.agent}"
        return f"{text} ({self.display_version or self.version})" if self.version else text


MAX_DRAWN_POINTS = 2000


def thin(points: list[dict], metric: str, limit: int = MAX_DRAWN_POINTS) -> list[dict]:
    """At most about `limit` chart points, for drawing only: each bucket keeps its lowest and highest
    value (in x order), so spikes and dips survive. Shorter series come back unchanged."""
    return _thin(points, lambda point: point[metric], limit)


def thin_xy(points: list[Point], limit: int = MAX_DRAWN_POINTS) -> list[Point]:
    """Thin before constructing chart dictionaries, retaining the same extrema and latest point."""
    return _thin(points, lambda point: point[1], limit)


class IncrementalXYThinner:
    """The same extrema buckets as thin_xy, reused for an append-only sorted series.

    A bucket-width change rebuilds the sample; ordinary appends revisit only the
    last partial bucket and new points. Callers discard this cache after a reorder.
    """

    def __init__(self, limit: int = MAX_DRAWN_POINTS):
        self.limit = limit
        self._points = None
        self._size = self._count = 0
        self._buckets = []

    def sample(self, points: list[Point]) -> list[Point]:
        if len(points) <= self.limit:
            self._points = None
            self._buckets = []
            self._count = 0
            return points
        size = math.ceil(len(points) / (self.limit // 2))
        if points is not self._points or size != self._size or len(points) < self._count:
            self._buckets = []
            start = 0
        else:
            start = (self._count // size) * size
            del self._buckets[start // size:]
        for offset in range(start, len(points), size):
            bucket = points[offset:offset + size]
            low = min(range(len(bucket)), key=lambda i: bucket[i][1])
            high = max(range(len(bucket)), key=lambda i: bucket[i][1])
            self._buckets.append([bucket[i] for i in sorted({low, high})])
        self._points, self._size, self._count = points, size, len(points)
        kept = [point for bucket in self._buckets for point in bucket]
        if kept[-1] is not points[-1]:
            kept.append(points[-1])
        return kept


def _thin(points, value, limit):
    if len(points) <= limit:
        return points
    size = math.ceil(len(points) / (limit // 2))
    kept = []
    for start in range(0, len(points), size):
        bucket = points[start : start + size]
        low = min(range(len(bucket)), key=lambda i: value(bucket[i]))
        high = max(range(len(bucket)), key=lambda i: value(bucket[i]))
        kept.extend(bucket[i] for i in sorted({low, high}))
    if kept[-1] is not points[-1]:
        kept.append(points[-1])  # the latest value is always drawn
    return kept


def chart_points(points: list[Point], metric: str) -> list[dict]:
    """(x, y) points in the shape the shared Chart widget draws: {"step": x, metric: y}."""
    return [{"step": x, metric: y} for x, y in points]


# ── Artifact previews ────────────────────────────────────────────────────────


@dataclass
class CsvPreview:
    headers: list[str]
    rows: list[list[str]]
    total_rows: int

    @property
    def truncated(self) -> bool:
        return self.total_rows > len(self.rows)


def decode_text(data: bytes) -> str:
    """File bytes as text: UTF-8 (dropping a byte-order mark), unreadable bytes replaced."""
    return data.decode("utf-8-sig", errors="replace")


def read_csv_rows(text: str, limit: int = 5000) -> CsvPreview:
    """Every column of a CSV as text, for a table view: the first `limit` rows and the total count."""
    reader = csv.reader(io.StringIO(text.lstrip("\ufeff")))
    headers = next(reader, [])
    rows, total = [], 0
    for row in reader:
        if not row:
            continue
        total += 1
        if len(rows) < limit:
            rows.append(row + [""] * (len(headers) - len(row)))
    return CsvPreview(headers, rows, total)


def flatten(value, prefix: str = "") -> dict[str, object]:
    """Nested dicts as dotted keys ({"env": {"name": "x"}} -> {"env.name": "x"}); lists stay values."""
    if not isinstance(value, dict):
        return {prefix: value} if prefix else {}
    flat: dict[str, object] = {}
    for key, child in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict) and child:
            flat.update(flatten(child, name))
        else:
            flat[name] = child
    return flat


_MISSING = object()


def config_differences(configs: dict[str, dict]) -> list[tuple[str, list]]:
    """Settings whose values differ between runs: (dotted key, one value per run, None if absent).

    With a single run every setting is listed, so this doubles as a flat view of one config.
    """
    flat = {name: flatten(cfg or {}) for name, cfg in configs.items()}
    keys = sorted({key for f in flat.values() for key in f})
    rows = []
    for key in keys:
        values = [f.get(key, _MISSING) for f in flat.values()]
        if len(flat) > 1 and all(v == values[0] for v in values):
            continue
        rows.append((key, [None if v is _MISSING else v for v in values]))
    return rows


# ── Comparisons: combining versions, summaries, export ───────────────────────

# Chart-point key holding the spread (std or SEM) of a combined series; the chart shades ± this.
SPREAD = "__spread"
COMBINE_MODES = {None: "Separate", "std": "Mean ± std", "sem": "Mean ± SEM"}


def series_groups(sources: list[Source], combine: bool) -> list[tuple[str, list[Source]]]:
    """How ticked sources become plotted series: (legend label, the sources it covers).

    Combine repeated versions of each agent, keeping seeds of different tuning trials
    in separate series.
    """
    if not combine:
        return [(source.label, [source]) for source in sources]
    groups: dict[tuple, list[Source]] = {}
    for source in sources:
        groups.setdefault((source.group, source.experiment_id, source.agent, source.trial_version), []).append(source)
    result = []
    for (_, experiment_id, agent, trial), members in groups.items():
        trial_label = members[0].display_version.rsplit(" · seed ", 1)[0] if trial and members[0].display_version else None
        suffix = f"{trial_label} · mean of {len(members)}" if trial_label else f"mean of {len(members)}"
        label = members[0].label if len(members) == 1 else f"{experiment_id} / {agent} ({suffix})"
        result.append((label, members))
    return result


def combined_points(series: list[list[Point]], metric: str, spread: str) -> list[dict]:
    """Chart points for the mean of several series, with SPREAD holding the std or SEM at each x."""
    return [
        {"step": p.x, metric: p.mean, SPREAD: p.std if spread == "std" else p.sem, "n": p.n} for p in aggregate(series)
    ]


def first_reaching(points: list[Point], threshold: float, higher_is_better: bool | None = True) -> float | None:
    """x where a series first reaches `threshold` (at or above it, or at or below for a loss)."""
    for x, y in points:
        if (y <= threshold) if higher_is_better is False else (y >= threshold):
            return x
    return None


SUMMARY_COLUMNS = ("Series", "Metric", "Final", "Best", "Best at", "Min", "Max", "Mean", "Area under curve", "Points")


def summary_rows(entries: list[tuple[str, str, list[Point]]], threshold: float | None = None) -> list[list]:
    """One row per (series, metric, points): the SUMMARY_COLUMNS values, plus "Reaches threshold at"
    when a threshold is given. Values stay numbers (None when absent) for sorting and export."""
    rows = []
    for label, metric, points in entries:
        higher = describe(metric).higher_is_better
        summary = summarize(points, higher)
        if summary is None:
            continue
        # "Best" needs a direction: for metrics with none (time, entropy, KL, …) Min and Max say it all.
        directed = higher is not None
        row = [
            label,
            metric,
            summary.final,
            summary.best if directed else None,
            summary.best_x if directed else None,
            summary.minimum,
            summary.maximum,
            summary.mean,
            summary.auc,
            summary.count,
        ]
        if threshold is not None:
            row.append(first_reaching(points, threshold, higher))
        rows.append(row)
    return rows


def to_csv(headers: list[str], rows: list[list]) -> str:
    """CSV text; None becomes an empty cell. Line endings are left to the file the caller writes."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator=chr(10))
    writer.writerow(headers)
    writer.writerows([["" if v is None else v for v in row] for row in rows])
    return out.getvalue()
