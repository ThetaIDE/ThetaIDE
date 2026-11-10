"""LiveRun: everything one training run has reported so far, and what can be worked out from it.

Fed telemetry records (src/app/telemetry/records.py) by whichever source reads them, it keeps the metric
rows, a bounded history of system samples, events, artifacts, the config and the final summary. From
those it derives the live numbers a monitor shows: training speed, time remaining, whether the run has
gone quiet, and "new best" events for evaluation and validation metrics. No Qt: it is plain data.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any

from src.app.telemetry.records import is_finite_number, metrics_row

from .results.metrics_catalog import describe

SYSTEM_SAMPLES_KEPT = 1800  # an hour at the default 2 s sampling interval
RATE_WINDOW_SECONDS = 60.0  # training speed is measured over the last minute
STALL_SECONDS = 120.0  # no record for this long while running: the run has gone quiet
BEST_PREFIXES = ("eval/", "val/")  # metrics whose new bests are announced (losses improve too often)
PROGRESS_KEYS = ("transitions", "step")  # what "how far along" is measured in, in order of preference


class LiveRun:
    """The state of one run, built up from its telemetry records."""

    def __init__(self, budget: float | None = None, stall_seconds: float = STALL_SECONDS):
        self.budget = budget  # total transitions (or steps) the run will train for, if known
        self.stall_seconds = stall_seconds
        self.clear()

    def clear(self):
        """Forget everything read so far (the source started over)."""
        self.rows: list[dict[str, float]] = []
        self.system: deque[tuple[float, dict[str, float]]] = deque(maxlen=SYSTEM_SAMPLES_KEPT)
        self.events: list[dict[str, Any]] = []  # reported by the run, and derived here (source "derived")
        self.artifacts: list[dict[str, Any]] = []
        self.start: dict[str, Any] | None = None
        self.finish: dict[str, Any] | None = None
        self.config: dict[str, Any] = {}
        self.best: dict[str, tuple[float, float | None]] = {}  # metric -> (best value, step)
        self.last_update: float | None = None
        self.stage: str | None = None
        self._latest_progress: float | None = None
        self._progress: deque[tuple[float, float]] = deque()  # (time, progress) within the rate window

    # ── Feeding ──────────────────────────────────────────────────────────────

    def add(self, records: list[dict[str, Any]], reset: bool = False, now: float | None = None) -> list[dict]:
        """Take in new records. Returns the events this added (reported and derived), oldest first."""
        if reset:
            self.clear()
        now = time.time() if now is None else now
        added: list[dict[str, Any]] = []
        for rec in records:
            kind = rec.get("type")
            when = rec.get("t") or now  # CSV-derived records carry no time: use when they arrived
            if kind == "metrics":
                added.extend(self._add_metrics(rec, when))
            elif kind == "system":
                self.system.append((when, dict(rec.get("data") or {})))
            elif kind == "event":
                if rec.get("kind") == "stage" and (rec.get("data") or {}).get("stage"):
                    stage = rec["data"]["stage"]
                    if stage != self.stage:
                        self._progress.clear()
                    self.stage = stage
                self.events.append(rec)
                added.append(rec)
            elif kind == "artifact":
                self.artifacts.append(rec)
            elif kind == "start":
                self.start = rec
                self.config = dict(rec.get("config") or {})
            elif kind == "config":
                self.config.update(rec.get("config") or {})
            elif kind == "finish":
                self.finish = rec
        if records:
            self.last_update = now
        return added

    def _add_metrics(self, rec: dict[str, Any], when: float) -> list[dict]:
        row = metrics_row(rec)
        if not row:
            return []
        self.rows.append(row)
        progress = next((row[k] for k in PROGRESS_KEYS if k in row), None)
        if progress is not None:
            self._latest_progress = progress
            if self.stage in (None, "training"):
                self._progress.append((when, progress))
            while self._progress and when - self._progress[0][0] > RATE_WINDOW_SECONDS:
                self._progress.popleft()
        derived = []
        for key, value in row.items():
            higher = describe(key).higher_is_better
            if higher is None or not key.startswith(BEST_PREFIXES):
                continue
            previous = self.best.get(key)
            if previous is None or (value > previous[0] if higher else value < previous[0]):
                self.best[key] = (value, row.get("step"))
                if previous is not None:  # the first value is not news
                    event = {
                        "type": "event",
                        "t": when,
                        "kind": "new_best",
                        "level": "info",
                        "message": f"New best {key} {value:.4g} (was {previous[0]:.4g})",
                        "data": {"metric": key, "value": value, "previous": previous[0], "step": row.get("step")},
                        "source": "derived",
                    }
                    self.events.append(event)
                    derived.append(event)
        return derived

    # ── Derived numbers ──────────────────────────────────────────────────────

    def progress(self) -> float | None:
        """How far the run got: its latest transitions (or step) count."""
        return self._latest_progress

    def rate(self) -> float | None:
        """Transitions (or steps) per second over the last minute, once there are two points to compare."""
        if self.stage not in (None, "training") or len(self._progress) < 2:
            return None
        (t0, p0), (t1, p1) = self._progress[0], self._progress[-1]
        return (p1 - p0) / (t1 - t0) if t1 > t0 and p1 >= p0 else None

    def eta_seconds(self) -> float | None:
        """Seconds until the budget is reached at the current rate."""
        rate, progress = self.rate(), self.progress()
        if not self.budget or not rate or progress is None or rate <= 0:
            return None
        return max(0.0, (self.budget - progress) / rate)

    def latest_system(self) -> dict[str, float]:
        return self.system[-1][1] if self.system else {}

    def latest(self, metric: str) -> float | None:
        return next((row[metric] for row in reversed(self.rows) if is_finite_number(row.get(metric))), None)

    def is_finished(self) -> bool:
        return self.finish is not None

    def stalled(self, now: float | None = None) -> bool:
        """True when a run that has reported something has said nothing for `stall_seconds`."""
        if self.finish is not None or self.last_update is None:
            return False
        return (time.time() if now is None else now) - self.last_update > self.stall_seconds


def format_duration(seconds: float | None) -> str:
    """'1h 04m', '3m 12s' or '42s'; an em dash when unknown."""
    if seconds is None:
        return "—"
    seconds = max(0, int(seconds))
    if seconds >= 3600:
        return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m"
    if seconds >= 60:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds}s"


class LiveRunCache:
    """LiveRuns of finished (or other sessions') runs, read from disk and re-read only when files change.

    Each agent's newest version_N is read whole through AgentFeed: its events.jsonl when it has one,
    else its metrics.csv (rows only, without times, so no speed or system history).
    """

    def __init__(self):
        self._entries: dict[tuple, tuple[tuple, LiveRun]] = {}

    def load_version(self, agent_dir, version):
        """Read a specific trial's telemetry without switching to a newer trial."""
        from pathlib import Path
        from src.app.services.metrics import AgentFeed

        feed = AgentFeed(Path(agent_dir), version=version)
        version_dir = feed.newest_version()
        if version_dir is None:
            return None
        stamp = (str(version_dir), *(_file_stamp(version_dir / name) for name in ("events.jsonl", "metrics.csv")))
        key = ("version", str(version_dir))
        cached = self._entries.get(key)
        if cached is None or cached[0] != stamp:
            live = LiveRun()
            records, _ = feed.read()
            live.add(records, now=_file_time(version_dir))
            self._entries[key] = cached = (stamp, live)
        return cached[1]

    def cached(self, logs_dir, group: str, experiment_id: str, agents: list[str]) -> dict[str, LiveRun]:
        """What was read before, without touching the disk (possibly out of date)."""
        found = {}
        for agent in agents:
            entry = self._entries.get((str(logs_dir), group, experiment_id, agent))
            if entry is not None:
                found[agent] = entry[1]
        return found

    def load(self, logs_dir, group: str, experiment_id: str, agents: list[str]) -> dict[str, LiveRun]:
        from pathlib import Path

        from src.app.services.metrics import AgentFeed

        lives = {}
        for agent in agents:
            feed = AgentFeed(Path(logs_dir) / group / experiment_id / agent)
            version_dir = feed.newest_version()
            if version_dir is None:
                continue
            stamp = (str(version_dir), *(_file_stamp(version_dir / name) for name in ("events.jsonl", "metrics.csv")))
            key = (str(logs_dir), group, experiment_id, agent)
            cached = self._entries.get(key)
            if cached is None or cached[0] != stamp:
                live = LiveRun()
                records, _ = feed.read()
                live.add(records, now=_file_time(version_dir))
                self._entries[key] = cached = (stamp, live)
            lives[agent] = cached[1]
        return lives


def _file_stamp(path) -> tuple:
    try:
        stat = path.stat()
    except OSError:
        return (None,)
    return (stat.st_size, stat.st_mtime_ns)


def _file_time(version_dir) -> float:
    """When the run last wrote anything: the time given to records that carry none (CSV rows)."""
    times = []
    for name in ("events.jsonl", "metrics.csv"):
        try:
            times.append((version_dir / name).stat().st_mtime)
        except OSError:
            pass
    return max(times, default=time.time())
