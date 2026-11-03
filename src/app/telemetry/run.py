"""The run: what training code talks to. It turns calls into records and hands each to every sink.

    run = telemetry.init("group", "experiment", "method", config=cfg)
    run.log({"eval/reward": 187.3}, step=1200, transitions=24576)
    run.event("checkpoint", "saved best_model.ckpt")
    run.finish()

Which sinks a run writes to is a list of names (see sinks.py), so the destination can change without
changing any training code. Hydra runs read it from the `telemetry` config key:
    ++telemetry.sinks=[csv,stream]  ++telemetry.system_interval=2   (0 turns system sampling off)
and `telemetry.metrics` chooses which metrics are recorded (selection.py).
"""

from __future__ import annotations

import atexit
import math
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from .records import is_finite_number, record, scalar
from .selection import MetricSelection, selection_from_config
from .sinks import DEFAULT_SINKS, Sink, resolve_sink
from .system import DEFAULT_INTERVAL, SystemSampler


class Run:
    """One training run's telemetry, written to a list of sinks. Safe to call from several threads."""

    def __init__(
        self,
        version_dir: str | Path,
        sinks: Iterable[str | Sink] = DEFAULT_SINKS,
        config: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        system_interval: float = DEFAULT_INTERVAL,
        metrics: MetricSelection | None = None,
    ):
        self.version_dir = Path(version_dir)
        self.metrics = metrics or MetricSelection()  # which metrics log() records
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.sinks: list[Sink] = [resolve_sink(spec, self.version_dir) for spec in sinks]
        self.summary: dict[str, Any] = {}  # the latest value of every metric, like a WandB run summary
        self.finished = False
        self.exit_status = "ended"  # how the run is finished if the process exits first (see _at_exit)
        self._step = 0
        self._started = time.time()
        self._lock = threading.RLock()
        self._warned: set[str] = set()
        self._failed_sinks: set[int] = set()
        info = {
            "version_dir": str(self.version_dir),
            "version": self.version_dir.name,
            "pid": os.getpid(),
            "host": platform.node(),
            "python": sys.version.split()[0],
            "started": self._started,
            **(meta or {}),
        }
        self._emit("start", record("start", run=info, config=config or {}))
        self.sampler = SystemSampler(self.system, system_interval).start() if system_interval > 0 else None
        atexit.register(self._at_exit)

    # ── Logging ──────────────────────────────────────────────────────────────

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        transitions: int | float | None = None,
        epoch: int | None = None,
    ) -> None:
        """One row of metrics. `step` defaults to the previous step + 1, as in ContractLogger."""
        with self._lock:
            self._step = int(step) if step is not None else self._step + 1
            data = {str(k): scalar(v) for k, v in self.metrics.filter(metrics).items()}
            if transitions is not None:
                data["transitions"] = scalar(transitions)
            if epoch is not None:
                data["epoch"] = scalar(epoch)
            self._emit("write", record("metrics", step=self._step, data=data))
            for key, value in data.items():
                if is_finite_number(value):
                    self.summary[key] = value
                elif isinstance(value, float) and not math.isfinite(value) and key not in self._warned:
                    self._warned.add(key)  # once per metric: a diverging run would otherwise flood the feed
                    self.event("non_finite", f"{key} is {value} at step {self._step}", level="warning", metric=key)

    def event(self, kind: str, message: str, level: str = "info", **data: Any) -> None:
        """Something worth showing in the live feed: a checkpoint, a warning, an error."""
        self._emit("write", record("event", kind=kind, level=level, message=message, data=data))

    def artifact(self, path: str | Path, kind: str = "file", **data: Any) -> None:
        """A file the run wrote (checkpoint, figure, report, ...)."""
        self._emit("write", record("artifact", path=str(path), kind=kind, data=data))

    def system(self, sample: dict[str, float]) -> None:
        """One resource-use sample (the system sampler calls this on its own thread)."""
        self._emit("write", record("system", data=dict(sample)))

    def update_config(self, config: dict[str, Any]) -> None:
        """Hyperparameters that become known after the start (e.g. reported by Lightning)."""
        self._emit("write", record("config", config=config))

    def save_runtime_metadata(
        self,
        config: dict[str, Any] | None = None,
        training_time_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        """runtime.json in the version folder, exactly as ContractLogger writes it."""
        from src.app.core.contract_logger import write_runtime_metadata

        return write_runtime_metadata(self.version_dir, self._started, self._step, config, training_time_seconds, extra)

    def finish(self, status: str = "completed", summary: dict[str, Any] | None = None) -> None:
        """End the run: stop sampling, write the summary, close every sink. Later calls do nothing."""
        if self.sampler is not None:
            self.sampler.stop()
        with self._lock:
            if self.finished:
                return
            self.finished = True
            self._emit("finish", record("finish", status=status, summary={**self.summary, **(summary or {})}))
        atexit.unregister(self._at_exit)

    def _at_exit(self):
        self.finish(status=self.exit_status)  # the process exited without finishing the run

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc_type is not None and not self.finished:
            self.event("error", f"{exc_type.__name__}: {exc}", level="error")
        self.finish(status="failed" if exc_type is not None else "completed")

    # ── Delivery ─────────────────────────────────────────────────────────────

    def _emit(self, method: str, rec: dict[str, Any]) -> None:
        """Hand `rec` to every sink. A failing sink is reported once and skipped; training goes on."""
        with self._lock:
            for i, sink in enumerate(self.sinks):
                if i in self._failed_sinks:
                    continue
                try:
                    getattr(sink, method)(rec)
                except Exception as exc:
                    self._failed_sinks.add(i)
                    print(f"[telemetry] {type(sink).__name__} stopped after an error: {exc!r}", file=sys.stderr)


def version_dir_for(
    group: str, experiment_id: str, agent: str, base_dir: str | Path = "results/logs", version: str | None = None
) -> Path:
    """results/logs/<group>/<experiment_id>/<agent>/version_N: the next free N unless `version` is given."""
    from src.app.core.contract_logger import ContractLogger

    return ContractLogger._resolve_version_dir(
        Path(base_dir) / group / experiment_id / agent.replace("/", "_"), version
    )


def init(
    group: str | None = None,
    experiment_id: str | None = None,
    agent: str | None = None,
    *,
    version_dir: str | Path | None = None,
    version: str | None = None,
    base_dir: str | Path = "results/logs",
    sinks: Iterable[str | Sink] | None = None,
    config: dict[str, Any] | None = None,
    system_interval: float = DEFAULT_INTERVAL,
    metrics: MetricSelection | None = None,
) -> Run:
    """Start a run in an explicit directory or version under its agent folder."""
    if version_dir is None:
        if not (group and experiment_id and agent):
            raise ValueError("telemetry.init needs version_dir, or group, experiment_id and agent")
        version_dir = version_dir_for(group, experiment_id, agent, base_dir, version=version)
    elif version is not None:
        raise ValueError("Specify either version_dir or version, not both")
    meta = {k: v for k, v in (("group", group), ("experiment_id", experiment_id), ("agent", agent)) if v}
    return Run(version_dir, DEFAULT_SINKS if sinks is None else sinks, config, meta, system_interval, metrics)


def settings_from_config(cfg: Any) -> tuple[list[str], float]:
    """(sinks, system_interval) from a run config's `telemetry` key, with the defaults for anything unset.

    A `telemetry.metrics` list without a system/ entry turns system sampling off (nothing would record it).
    """
    section = cfg.get("telemetry") if hasattr(cfg, "get") else None
    section = section if hasattr(section, "get") else {}
    sinks = section.get("sinks")
    interval = section.get("system_interval")
    interval = float(interval) if interval is not None else DEFAULT_INTERVAL
    if not selection_from_config(cfg).keeps_section("system"):
        interval = 0.0
    return [str(s) for s in sinks] if sinks is not None else list(DEFAULT_SINKS), interval
