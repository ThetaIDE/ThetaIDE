"""Sinks: where a run's telemetry records go. Any object with start/write/finish is one.

Built in:
    csv     metrics.csv through ContractLogger (metrics records only), the file the Results browser reads
    stream  events.jsonl: every record, one JSON object per line, appended as it happens (the live feed)
    memory  a list in memory, for tests and in-process consumers

Other destinations plug in without touching the training code: register_sink("name", factory), or name
a factory by import path in the run's sink list ("my_package.sinks:WandbSink").
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from .records import encode

STREAM_FILE = "events.jsonl"


@runtime_checkable
class Sink(Protocol):
    """A destination for telemetry records. `version_dir` is the run's results/logs/.../version_N folder."""

    def start(self, rec: dict[str, Any]) -> None: ...

    def write(self, rec: dict[str, Any]) -> None: ...

    def finish(self, rec: dict[str, Any]) -> None: ...


SinkFactory = Callable[[Path], Sink]


class StreamSink:
    """Appends every record to <version_dir>/events.jsonl, flushed line by line so readers see it at once."""

    live = True  # receives telemetry.monitors filtering; archival sinks do not

    def __init__(self, version_dir: Path):
        self.path = Path(version_dir) / STREAM_FILE
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8", newline="\n")

    def _append(self, rec):
        if self._file is None:
            return
        self._file.write(encode(rec) + "\n")
        self._file.flush()

    def start(self, rec):
        self._append(rec)

    def write(self, rec):
        self._append(rec)

    def finish(self, rec):
        self._append(rec)
        if self._file is not None:
            self._file.close()
            self._file = None


class CsvSink:
    """Writes metrics records to <version_dir>/metrics.csv exactly as ContractLogger always has."""

    records_system = False

    def __init__(self, version_dir: Path):
        from src.app.core.contract_logger import ContractLogger

        version_dir = Path(version_dir)
        self.logger = ContractLogger(log_dir=version_dir.parent, version=version_dir.name)

    def start(self, rec):
        pass

    def write(self, rec):
        if rec.get("type") != "metrics":
            return
        metrics = dict(rec.get("data") or {})
        transitions = metrics.pop("transitions", None)
        epoch = metrics.pop("epoch", None)
        self.logger.log(metrics, step=rec.get("step"), transitions=transitions, epoch=epoch)

    def finish(self, rec):
        self.logger.close()


class MemorySink:
    """Keeps every record in `records`."""

    def __init__(self, version_dir: Path | None = None):
        self.version_dir = version_dir
        self.records: list[dict[str, Any]] = []
        self.finished = False

    def start(self, rec):
        self.records.append(rec)

    def write(self, rec):
        self.records.append(rec)

    def finish(self, rec):
        self.records.append(rec)
        self.finished = True


SINKS: dict[str, SinkFactory] = {"csv": CsvSink, "stream": StreamSink, "memory": MemorySink}
DEFAULT_SINKS = ("csv", "stream")


def register_sink(name: str, factory: SinkFactory) -> None:
    """Make `name` usable in a run's sink list (e.g. telemetry.sinks=[csv,stream,name])."""
    SINKS[name] = factory


def resolve_sink(spec: str | Sink, version_dir: Path) -> Sink:
    """A sink from a registered name, a "module:attribute" import path, or a ready-made sink object."""
    if not isinstance(spec, str):
        return spec
    factory = SINKS.get(spec)
    if factory is None:
        module_name, _, attribute = spec.partition(":")
        if not attribute:
            raise ValueError(
                f"Unknown telemetry sink {spec!r}: use one of {', '.join(sorted(SINKS))} or 'package.module:Factory'"
            )
        factory = getattr(importlib.import_module(module_name), attribute)
    return factory(Path(version_dir))
