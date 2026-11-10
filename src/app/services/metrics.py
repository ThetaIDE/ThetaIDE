"""Reading a run's metrics.csv and events.jsonl: whole files, or only what was appended since last time."""

from __future__ import annotations

import csv
from pathlib import Path


def latest_version_file(agent_dir: Path, name: str) -> Path | None:
    """`name` inside the newest version_N folder of an agent's log directory (by N, so version_10 > version_9)."""

    def number(path: Path) -> int:
        suffix = path.parent.name.removeprefix("version_")
        return int(suffix) if suffix.isdigit() else -1

    files = sorted(agent_dir.glob(f"version_*/{name}"), key=number)
    return files[-1] if files else None


def latest_metrics_csv(agent_dir: Path) -> Path | None:
    return latest_version_file(agent_dir, "metrics.csv")


def _numeric(row: dict) -> dict[str, float]:
    values = {}
    for key, value in row.items():
        if key and value not in (None, ""):
            try:
                values[key] = float(value)
            except (TypeError, ValueError):
                pass
    return values


def read_metrics_rows(path: Path) -> list[dict[str, float]]:
    """Every complete row of a metrics CSV, numeric columns only."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    lines = text.split("\n")
    if lines and lines[-1] != "":
        lines = lines[:-1]  # the last line is still being written
    return [_numeric(row) for row in csv.DictReader(lines)]


def read_metrics_table(path: Path) -> list[dict[str, str]]:
    """Every row of a metrics CSV as strings (all columns), for the Results browser."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


class MetricsTail:
    """Reads only the rows appended to one metrics CSV since the previous read.

    Lightning rewrites the file from scratch when a new column appears, and the rewrite is
    usually *longer* than what it replaces: a supervised run logs `epoch,step,val/loss` during
    the validation sanity check and then `epoch,step,train/loss,training_time_seconds,val/loss`
    once training starts. Watching for the file to shrink therefore misses the common case, and
    a stale header zipped onto wider rows files every value under the wrong name - train/loss
    arriving as val/loss, and the columns past the end of the old header disappearing entirely.

    So the header itself is compared on every read. A changed header means the file was
    rewritten, whichever direction its size went, and `reset` tells the caller to drop what it
    had and take these rows as the whole run.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.header: list[str] = []

    def read(self) -> tuple[list[dict[str, float]], bool]:
        """(new rows, reset). Rows are only returned once their line is complete."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return [], False
        reset = False
        if size < self.offset:
            self.offset, self.header, reset = 0, [], True
        try:
            with open(self.path, "rb") as f:
                header_line = f.readline().decode("utf-8", errors="replace")
                if not header_line.endswith("\n"):
                    return [], reset  # header still being written
                header = next(csv.reader([header_line.rstrip("\r\n")]), [])
                if header != self.header:
                    # A rewrite. Only a reset if we had already read a different header:
                    # the first read of a file is not one.
                    reset = reset or bool(self.header)
                    self.header, self.offset = header, f.tell()
                elif not self.offset:
                    self.offset = f.tell()
                f.seek(self.offset)
                raw = f.read()
        except OSError:
            return [], reset
        last_newline = raw.rfind(b"\n")
        if last_newline == -1:
            return [], reset
        complete = raw[: last_newline + 1]
        self.offset += len(complete)
        rows = []
        for line in complete.decode("utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            values = next(csv.reader([line]), [])
            numeric = _numeric(dict(zip(self.header, values)))
            if numeric:
                rows.append(numeric)
        return rows, reset


class StreamTail:
    """Reads only the records appended to one events.jsonl since the previous read.

    The file is only ever appended to; if it shrinks anyway (deleted and started again) the tail starts
    over and reports `reset`. A last line without its newline is still being written and waits.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0

    def read(self) -> tuple[list[dict], bool]:
        from src.app.telemetry.records import decode

        try:
            size = self.path.stat().st_size
        except OSError:
            return [], False
        reset = size < self.offset
        if reset:
            self.offset = 0
        try:
            with open(self.path, "rb") as f:
                f.seek(self.offset)
                raw = f.read()
        except OSError:
            return [], reset
        last_newline = raw.rfind(b"\n")
        if last_newline == -1:
            return [], reset
        self.offset += last_newline + 1
        lines = raw[: last_newline + 1].decode("utf-8", errors="replace").splitlines()
        return [rec for line in lines if (rec := decode(line)) is not None], reset


def _version_number(path: Path) -> int:
    suffix = path.name.removeprefix("version_")
    return int(suffix) if suffix.isdigit() else -1


class AgentFeed:
    """Every telemetry record of one agent's newest run, read incrementally.

    Reads <agent_dir>/version_N/events.jsonl when the run writes the live stream, else its metrics.csv
    (older runs, engines that only write the CSV), turning each row into a metrics record. Reports
    `reset` whenever what was read before no longer applies: a newer version_N started, the stream
    appeared next to a CSV already being read, or a file was rewritten.
    """

    def __init__(self, agent_dir: Path, version: str | None = None):
        self.agent_dir = Path(agent_dir)
        self.version = str(version) if version else None
        self.version_dir: Path | None = None
        self.kind: str | None = None  # "stream" or "csv"
        self._tail = None
        self._monitor_stamp = None
        self._monitor_selection = None

    def newest_version(self) -> Path | None:
        if self.version is not None:
            path = self.agent_dir / self.version
            return path if (path / "events.jsonl").exists() or (path / "metrics.csv").exists() else None
        candidates = [
            d for d in self.agent_dir.glob("version_*") if (d / "events.jsonl").exists() or (d / "metrics.csv").exists()
        ]
        return max(candidates, key=_version_number) if candidates else None

    def read(self) -> tuple[list[dict], bool]:
        version_dir = self.newest_version()
        if version_dir is None:
            return [], False
        kind = "stream" if (version_dir / "events.jsonl").exists() else "csv"
        reset = False
        if version_dir != self.version_dir or kind != self.kind:
            reset = self.version_dir is not None
            self.version_dir, self.kind = version_dir, kind
            self._tail = (
                StreamTail(version_dir / "events.jsonl")
                if kind == "stream"
                else MetricsTail(version_dir / "metrics.csv")
            )
        if kind == "stream":
            records, rewritten = self._tail.read()
        else:
            rows, rewritten = self._tail.read()
            records = [_csv_record(row) for row in rows]
            # CSV is an archival source. Apply live exclusions when it is used
            # as a fallback so hidden metrics cannot reappear in the monitor.
            snapshot = self.agent_dir.parent / "config.yaml"
            stamp = (snapshot.stat().st_mtime_ns, snapshot.stat().st_size) if snapshot.is_file() else None
            if self._monitor_selection is None or stamp != self._monitor_stamp:
                import yaml
                from src.app.telemetry.monitors import monitors_from_config
                try:
                    config = yaml.safe_load(snapshot.read_text(encoding="utf-8")) if stamp else {}
                    self._monitor_selection = monitors_from_config(config)
                except (OSError, ValueError, yaml.YAMLError):
                    self._monitor_selection = monitors_from_config({})
                self._monitor_stamp = stamp
            records = [filtered for rec in records if (filtered := self._monitor_selection.filter_record(rec)) is not None]
        return records, reset or rewritten


def _csv_record(row: dict[str, float]) -> dict:
    """A metrics.csv row as a metrics record (no timestamp: the CSV has none)."""
    data = {k: v for k, v in row.items() if k != "step"}
    return {"type": "metrics", "step": row.get("step"), "data": data, "source": "csv"}
