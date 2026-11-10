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
from contextlib import contextmanager
import math
import os
import platform
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from .records import is_finite_number, record, scalar
from .selection import MetricSelection, selection_from_config
from .monitors import available_monitors, monitors_from_config, system_monitors
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
        self.metrics = metrics if metrics is not None else selection_from_config(config)
        self.monitors = monitors_from_config(config)
        self._config = config or {}
        self._monitor_specs = None
        self._stage = "starting"
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
        from src.app.services.worker_priority import scheduling_metadata
        if scheduling := scheduling_metadata():
            info["scheduling"] = scheduling
        self._emit("start", record("start", run=info, config=config or {}))
        # CSV has no resource records. Avoid probing entirely when no destination
        # consumes system samples; custom archival sinks can still retain them.
        system_consumers = any(getattr(sink, "records_system", True)
                               and (not getattr(sink, "live", False)
                                    or any(self.monitors.keeps(spec.name) for spec in system_monitors(config)))
                               for sink in self.sinks)
        self.sampler = SystemSampler(self.system, system_interval).start() if (
            system_interval > 0 and self.metrics.keeps_section("system") and system_consumers) else None
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
        if kind == "stage" and data.get("stage"):
            self._stage = str(data["stage"])
        self._emit("write", record("event", kind=kind, level=level, message=message, data=data))

    def set_stage(self, stage: str, message: str | None = None, **data: Any) -> None:
        self.event("stage", message or stage.capitalize(), stage=stage, **data)

    @contextmanager
    def phase(self, stage: str, message: str | None = None, **data: Any):
        """Expose synchronous testing/diagnostics and resume the previous stage on success."""
        previous = self._stage
        self.set_stage(stage, message, **data)
        yield
        self.set_stage(previous)

    def artifact(self, path: str | Path, kind: str = "file", **data: Any) -> None:
        """A file the run wrote (checkpoint, figure, report, ...)."""
        output_root = os.environ.get("THETA_OUTPUT_ROOT")
        if output_root:
            try:
                path = Path(path).resolve().relative_to(Path(output_root).resolve()).as_posix()
            except ValueError:
                pass  # External artifact references remain explicit.
        self._emit("write", record("artifact", path=str(path), kind=kind, data=data))

    def system(self, sample: dict[str, float]) -> None:
        """One resource-use sample (the system sampler calls this on its own thread)."""
        self._emit("write", record("system", data=dict(sample)))

    def monitor(self, name: str, collect: Callable[[], dict[str, Any]], *, step: int | None = None) -> bool:
        """Collect an optional diagnostic lazily, only when its stream is enabled.

        ``collect()`` returns a mapping within the advertised name/prefix. Unlike
        normal training ``log()``, this is optional observation work: disabling
        it also skips collection and its recording. Never use for losses,
        checkpoint inputs or objective values.
        """
        if not self.wants_monitor(name) or not (
            self.metrics.keeps_section(name) if name.endswith("/") else self.metrics.keeps(name)
        ):
            return False
        values = collect()
        if any(not (key.startswith(name) if name.endswith("/") else key == name) for key in values):
            raise ValueError(f"Monitor {name!r} returned metrics outside its declared output")
        self.log(values, step=step)
        return True

    def wants_monitor(self, name: str) -> bool:
        """Whether any declared scalar remains selected before optional collection."""
        if not self.monitors.keeps(name):
            return False
        if not name.endswith("/") or not self.monitors.disabled:
            return True
        if self._monitor_specs is None:
            self._monitor_specs = available_monitors(self._config)
        outputs = [spec.name for spec in self._monitor_specs if spec.name.startswith(name)]
        return not outputs or any(self.monitors.keeps(key) and self.metrics.keeps(key) for key in outputs)

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
            filtering = bool(self.monitors.disabled)
            live_record = self.monitors.filter_record(rec) if filtering else rec
            for i, sink in enumerate(self.sinks):
                if i in self._failed_sinks:
                    continue
                try:
                    delivered = live_record if filtering and getattr(sink, "live", False) else rec
                    if delivered is not None:
                        getattr(sink, method)(delivered)
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
