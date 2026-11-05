"""What exists on disk: environments, experiments, finished and running runs, their metrics, plots and artifacts.

Paths come from the caller (the Results panel), so every group, experiment and agent name is checked to be a
plain folder name, and run files are only read from inside that run's own results folders.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import os
import shutil
import tempfile
import zipfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml

from . import LOGS_DIR, PLOTS_DIR, PROJECT_ROOT
from .metrics import latest_version_file, read_metrics_table

try:
    from src.usr.algorithms.agent_registry import list_registered_agents
except ImportError:  # pragma: no cover - registry is optional

    def list_registered_agents():
        return []


CONFIG_DIR = PROJECT_ROOT / "in" / "config"


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def list_environments() -> list[dict[str, Any]]:
    envs = []
    env_dir = CONFIG_DIR / "env"
    if not env_dir.is_dir():
        return envs
    for f in sorted(env_dir.rglob("*.yaml")):
        if f.name.startswith("_") or "/_" in f.as_posix():
            continue
        try:
            rel_name = f.relative_to(env_dir).with_suffix("").as_posix()
            envs.append({"name": rel_name, "stem": f.stem, "config": yaml.safe_load(f.read_text(encoding="utf-8"))})
        except (OSError, yaml.YAMLError):
            pass
    return envs


def list_methods() -> dict[str, Any]:
    return {"registered_agents": list_registered_agents(), "method_styles": {}}


def list_experiments() -> list[dict[str, Any]]:
    """Experiment configs, skipping internal _base / template files."""
    exp_dir = CONFIG_DIR / "experiment"
    exps = []
    for f in sorted(exp_dir.glob("**/*.yaml")):
        if f.name.startswith("_") or f.parent.name == "methods" or "/methods/" in f.as_posix():
            continue
        rel_path = f.relative_to(exp_dir)
        try:
            exps.append(
                {
                    "path": rel_path.as_posix(),
                    "name": rel_path.with_suffix("").as_posix(),
                    "config": yaml.safe_load(f.read_text(encoding="utf-8")),
                }
            )
        except (OSError, yaml.YAMLError):
            pass
    return exps


# agent_summary results by agent folder, valid while the files they were read from are unchanged.
_SUMMARY_CACHE: dict[Path, tuple[tuple, dict[str, Any] | None]] = {}
# Per-run files, keyed by path and reused until their stamp changes (see _cached_by_stamp)
_CONFIG_SUMMARY_CACHE: dict[Path, tuple[tuple, dict[str, Any]]] = {}
_RUN_METADATA_CACHE: dict[Path, tuple[tuple, dict[str, Any]]] = {}

# The caches above are also kept on disk between sessions: without them, the first list_runs of a session
# parses every metrics.csv (~9 s for 900 agents). Every entry keeps the stamp it was read at and is only
# reused while the file still matches it, as in memory, so a run that changed while the app was closed is
# read again. The file is a convenience: if it can't be read or written, listing works as before.
_PERSISTED_CACHES = {"agents": _SUMMARY_CACHE, "configs": _CONFIG_SUMMARY_CACHE, "metadata": _RUN_METADATA_CACHE}
_PERSISTED_FORMAT = 1
_persisted_loaded_for: Path | None = None  # the LOGS_DIR whose cache file has been read this session
_cache_dirty = False  # an entry was computed since the cache file was last written


def _summary_cache_file() -> Path:
    return LOGS_DIR.parent / ".cache" / "catalog-summaries.json"


def _as_tuples(value):
    """JSON turns the stamps' tuples into lists; turn them back so stamps compare equal."""
    return tuple(_as_tuples(v) for v in value) if isinstance(value, list) else value


def _load_persisted_summaries() -> None:
    global _persisted_loaded_for
    if _persisted_loaded_for == LOGS_DIR:
        return
    _persisted_loaded_for = LOGS_DIR
    try:
        data = json.loads(_summary_cache_file().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(data, dict) or data.get("format") != _PERSISTED_FORMAT or data.get("logs_dir") != str(LOGS_DIR):
        return
    for name, cache in _PERSISTED_CACHES.items():
        entries = data.get(name)
        if not isinstance(entries, dict):
            continue
        for key, entry in entries.items():
            if isinstance(entry, list) and len(entry) == 2:
                cache.setdefault(Path(key), (_as_tuples(entry[0]), entry[1]))  # this session's entries win


def _save_persisted_summaries(seen: dict[str, set[Path]]) -> None:
    """Keep only entries for paths this listing saw (runs deleted since drop out), and write the file
    if anything changed. Written to a temporary file and renamed, so a reader never sees half of it."""
    global _cache_dirty
    pruned = False
    for name, cache in _PERSISTED_CACHES.items():
        for key in [k for k in cache if k not in seen[name]]:
            del cache[key]
            pruned = True
    if not (_cache_dirty or pruned):
        return
    data: dict[str, Any] = {"format": _PERSISTED_FORMAT, "logs_dir": str(LOGS_DIR)}
    for name, cache in _PERSISTED_CACHES.items():
        entries = {}
        for key, (stamp, value) in cache.items():
            try:
                json.dumps(value, allow_nan=True)  # values JSON can't hold (a YAML date) stay in memory only
            except (TypeError, ValueError):
                continue
            entries[str(key)] = [stamp, value]
        data[name] = entries
    target = _summary_cache_file()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data), encoding="utf-8")
        os.replace(temporary, target)
        _cache_dirty = False
    except OSError:
        pass  # e.g. a read-only results folder: the next session reads the files again


def _file_stamp(path: Path | None) -> tuple:
    if path is None:
        return (None,)
    try:
        stat = path.stat()
    except OSError:
        return (str(path), None)
    return (str(path), stat.st_mtime_ns, stat.st_size)


def agent_summary(agent_dir: Path) -> dict[str, Any] | None:
    """Timing and reward summary for one agent of a run, or None if it never logged anything.

    Cached until the newest runtime.json or metrics.csv changes, so listing many runs stays cheap.
    """
    runtime_file = latest_version_file(agent_dir, "runtime.json")
    metrics_file = latest_version_file(agent_dir, "metrics.csv")
    # No .resolve(): the paths come from LOGS_DIR, already absolute, and resolving three paths per agent
    # (an lstat for every path segment) was most of a repeated list_runs
    key = agent_dir
    stamp = (_file_stamp(runtime_file), _file_stamp(metrics_file))
    cached = _SUMMARY_CACHE.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]
    summary = _read_agent_summary(agent_dir, runtime_file, metrics_file)
    _SUMMARY_CACHE[key] = (stamp, summary)
    _mark_cache_dirty()
    return summary


def _mark_cache_dirty() -> None:
    global _cache_dirty
    _cache_dirty = True


def _read_agent_summary(agent_dir: Path, runtime_file: Path | None, metrics_file: Path | None) -> dict[str, Any] | None:
    if runtime_file is None and metrics_file is None:
        return None
    runtime: dict[str, Any] = {}
    if runtime_file is not None:
        try:
            runtime = json.loads(runtime_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            runtime = {}
    rewards, timesteps = [], None
    if metrics_file is not None:
        try:
            with open(metrics_file, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    # Only rows with a step count, as the GUI's charts (metric_points) use: Lightning also
                    # repeats each evaluation in an epoch row without one.
                    transitions = _float(row.get("transitions"))
                    if transitions is None:
                        continue
                    timesteps = max(timesteps or 0, round(transitions))
                    reward = _float(row.get("eval/reward"))
                    if reward is not None:
                        rewards.append(reward)
        except (OSError, csv.Error):
            pass
    return {
        "name": agent_dir.name,
        "started": runtime.get("start_time_iso"),
        "finished": runtime.get("end_time_iso"),
        "training_time_seconds": _float(runtime.get("training_time_seconds")),
        "timesteps": timesteps,
        "best_reward": max(rewards) if rewards else None,
        "latest_reward": rewards[-1] if rewards else None,
    }


def _cached_by_stamp(cache: dict, path: Path, load):
    """load(path), reused until the file's mtime or size changes (or it appears or disappears)."""
    stamp = _file_stamp(path)
    cached = cache.get(path)
    if cached is not None and cached[0] == stamp:
        return copy.deepcopy(cached[1])
    value = load(path)
    cache[path] = (stamp, value)
    _mark_cache_dirty()
    return copy.deepcopy(value)  # callers get their own copy; the cached one must not change


def run_config_summary(run_dir: Path) -> dict[str, Any]:
    """The fields of a run's saved Hydra config (config.yaml) that the Results browser shows.

    Cached per file: listing runs re-read every run's config.yaml, and YAML parsing was about two thirds
    of a repeated list_runs."""
    return _cached_by_stamp(_CONFIG_SUMMARY_CACHE, run_dir / "config.yaml", _read_config_summary)


def _read_run_metadata(runtime_file: Path) -> dict[str, Any]:
    try:
        return json.loads(runtime_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _read_config_summary(config_file: Path) -> dict[str, Any]:
    try:
        cfg = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(cfg, dict):
        return {}
    env = cfg.get("env") if isinstance(cfg.get("env"), dict) else {}
    methods = cfg.get("methods") if isinstance(cfg.get("methods"), dict) else {}
    return {
        "env": env.get("name"),
        "env_id": env.get("env_id"),
        "seed": cfg.get("seed"),
        "total_timesteps": cfg.get("total_timesteps"),
        "paradigm": cfg.get("paradigm"),
        "methods": {
            name: {key: settings.get(key) for key in ("agent", "model", "lr", "batch_size", "gamma")}
            for name, settings in methods.items()
            if isinstance(settings, dict)
        },
    }


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _result_replicates(run_dir: Path):
    """Execution records identify seed versions separately from a trial's objective version."""
    found = {}
    root = LOGS_DIR.parent / "executions" / run_dir.parent.name / run_dir.name
    for path in sorted(root.glob("*/manifest.json")):
        manifest = _read_json(path)
        parent = manifest.get("parent") or {}
        for run in manifest.get("runs", []):
            if run.get("experiment_id") != run_dir.name or not run.get("version"):
                continue
            agent = run.get("agent") or run.get("method")
            if not agent:
                continue
            found[(agent, run["version"])] = {
                "seed": run.get("training_seed"), "status": run.get("status"),
                **({"trial_version": parent.get("log_version"), "study": parent.get("study"),
                    "trial_number": parent.get("trial_number")} if parent.get("kind") == "trial" else {}),
            }
    return found


def _result_snapshot(run_dir: Path, replicates=None):
    """Published versions and checkpoint folders; active trials never enter Results."""
    from src.app.pipeline.trial_artifacts import trial_checkpoint_dir

    versions, terminal_trials, trial_states = {}, set(), {}
    if replicates is None:
        replicates = _result_replicates(run_dir)
    checkpoint_root = LOGS_DIR.parent / "checkpoints" / run_dir.parent.name / run_dir.name
    active_methods = set()
    for path in [*checkpoint_root.glob("*/tuning_summary.json"),
                 *checkpoint_root.glob("*/sweeps/*/tuning_summary.json")]:
        summary = _read_json(path)
        method = summary.get("log_method") or path.relative_to(checkpoint_root).parts[0]
        if summary.get("stop_reason") == "running":
            active_methods.add(method)
        for trial in summary.get("trials", []):
            terminal = trial.get("state") in ("complete", "pruned", "fail")
            if trial.get("log_version"):
                trial_states[(method, trial["log_version"])] = terminal
            if terminal and summary.get("study") and isinstance(trial.get("id"), int) and trial["id"] >= 0:
                directory = trial_checkpoint_dir(checkpoint_root / path.relative_to(checkpoint_root).parts[0],
                                                 summary["study"], trial["id"])
                terminal_trials.add(directory.relative_to(checkpoint_root).parts)
    complete_methods = set()
    in_progress = bool(active_methods)
    if run_dir.is_dir():
        for agent_dir in run_dir.iterdir():
            if not agent_dir.is_dir():
                continue
            all_versions = [p for p in agent_dir.glob("version_*") if p.is_dir()]
            completed = []
            for version in all_versions:
                runtime = _read_json(version / "runtime.json")
                key = (agent_dir.name, version.name)
                terminal = trial_states.get(key)
                if key in replicates:
                    terminal = replicates[key]["status"] in ("completed", "failed", "cancelled")
                if terminal is True or (terminal is None and not version.name.startswith("version_sweep_") and
                                        (runtime.get("end_time_iso") or runtime.get("completed_at"))):
                    completed.append(version.name)
            if completed:
                versions[agent_dir.name] = sorted(completed, key=lambda name: (_version_number(name), name))
            if all_versions and len(completed) == len(all_versions) and agent_dir.name not in active_methods:
                complete_methods.add(agent_dir.name)
            elif all_versions:
                in_progress = True
    return versions, complete_methods, terminal_trials, in_progress


def list_runs(completed_only: bool = False) -> list[dict[str, Any]]:
    """Every experiment folder under results/logs, however it was launched (GUI, CLI or SLURM).

    `agents` lists each agent that logged something, with its timing and reward summary;
    `finished` is true once every agent has recorded an end time.
    With `completed_only`, summarize only finalized versions; an ongoing experiment
    can expose its finished trials while withholding its active ones.
    """
    runs = []
    if not LOGS_DIR.exists():
        return runs
    # Older repeated executions gave every replicate its own experiment ID.
    # They remain available in the Executions browser, but should not flood the
    # ordinary experiment list as if every seed were a separate experiment.
    replicate_experiments = set()
    execution_root = LOGS_DIR.parent / "executions"
    for manifest_path in execution_root.glob("*/*/*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            result_id = manifest.get("result_experiment_id")
            if result_id:
                continue
            group = manifest.get("group")
            for replicate in manifest.get("runs", []):
                experiment_id = replicate.get("experiment_id")
                if experiment_id and experiment_id != manifest.get("experiment_id"):
                    replicate_experiments.add((group, experiment_id))
        except (OSError, ValueError):
            continue
    _load_persisted_summaries()
    seen: dict[str, set[Path]] = {name: set() for name in _PERSISTED_CACHES}
    for run_dir in LOGS_DIR.glob("*/*"):
        if not run_dir.is_dir():
            continue
        if (run_dir.parent.name, run_dir.name) in replicate_experiments:
            continue
        metadata = _cached_by_stamp(_RUN_METADATA_CACHE, run_dir / "runtime.json", _read_run_metadata)
        agent_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir())
        seen["metadata"].add(run_dir / "runtime.json")
        seen["configs"].add(run_dir / "config.yaml")
        seen["agents"].update(agent_dirs)
        if completed_only:
            versions, _, _, in_progress = _result_snapshot(run_dir)
            agents = []
            for name, names in versions.items():
                version = run_dir / name / names[-1]
                if not (version / "metrics.csv").is_file():
                    continue
                summary = _read_agent_summary(run_dir / name, version / "runtime.json", version / "metrics.csv")
                summary["finished"] = summary["finished"] or _read_json(version / "runtime.json").get("completed_at") or True
                agents.append(summary)
            if not agents:
                continue
        else:
            agents = [summary for agent_dir in agent_dirs if (summary := agent_summary(agent_dir)) is not None]
        runs.append(
            {
                "group": run_dir.parent.name,
                "experiment_id": run_dir.name,
                "metadata": metadata,
                "config": run_config_summary(run_dir),
                "agents": agents,
                **({"completed_versions": versions, "in_progress": in_progress} if completed_only else {}),
                **({"has_run_logs": True} if any(run_dir.glob("*.stderr.log")) or
                   any(run_dir.glob("*.combined.log")) else {}),
                "finished": bool(agents) and all(agent["finished"] for agent in agents),
            }
        )
    _save_persisted_summaries(seen)
    return runs


def check_path_segment(part: str) -> None:
    """Reject names that could step outside results/ (e.g. '..', a slash or a drive letter)."""
    if not part or part in (".", "..") or any(c in part for c in "/\\\x00") or ":" in part:
        raise ValueError(f"Invalid path segment: {part!r}")


def _version_number(name: str) -> int:
    suffix = name.removeprefix("version_")
    return int(suffix) if suffix.isdigit() else -1


def run_metrics(
    group: str, experiment_id: str, agent: str, version: str | None = None, since: int = 0,
    completed_only: bool = False,
) -> dict[str, Any]:
    """Every row of an agent's metrics.csv (as text), with every column the engine logged.

    `version` picks a version_N folder (default: the newest). `since` skips rows the caller already has,
    so a client following a run that is still training reads only new rows; `total_rows` says where to
    resume. Raises FileNotFoundError if there are no such metrics, ValueError for a bad name.
    """
    for part in (group, experiment_id, agent):
        check_path_segment(part)
    agent_dir = LOGS_DIR / group / experiment_id / agent
    versions = sorted(
        (p.name for p in agent_dir.glob("version_*") if (p / "metrics.csv").is_file()), key=_version_number
    )
    if completed_only:
        published, _, _, _ = _result_snapshot(agent_dir.parent)
        versions = [name for name in versions if name in published.get(agent, [])]
        if version is None:
            version = versions[-1] if versions else None
        name = version if version and version.startswith("version_") else f"version_{version}"
        if name not in versions:
            raise FileNotFoundError(f"No completed metrics for {group}/{experiment_id}/{agent}")
    if version is None:
        metrics_file = latest_version_file(agent_dir, "metrics.csv")
    else:
        name = version if version.startswith("version_") else f"version_{version}"
        check_path_segment(name)
        metrics_file = agent_dir / name / "metrics.csv"
    if metrics_file is None or not metrics_file.exists():
        raise FileNotFoundError(f"No metrics for {group}/{experiment_id}/{agent}")
    rows, columns, total = _metrics_rows(metrics_file, max(int(since), 0))
    return {
        "metrics": rows,
        "columns": columns,
        "total_rows": total,
        "version": metrics_file.parent.name,
        "versions": versions,
        "source": str(metrics_file),
    }


class _MetricsTail:
    """Where the last read of one metrics.csv stopped, so a later read can start there."""

    __slots__ = ("inode", "header", "rows", "last_row_start", "end")

    def __init__(self, inode, header, rows, last_row_start, end):
        self.inode, self.header, self.rows, self.last_row_start, self.end = inode, header, rows, last_row_start, end


# metrics.csv path -> where its last read stopped. A run being followed is polled with since=N every few
# seconds; re-reading the whole file each time cost ~0.5 s per poll at 100k rows, and grew with the run.
_METRICS_TAILS: OrderedDict[Path, _MetricsTail] = OrderedDict()
_METRICS_TAILS_LIMIT = 64
_METRICS_TAILS_LOCK = threading.Lock()


def _parse_rows(header_text: str, body: bytes) -> list[dict[str, str]]:
    """Rows of `body` (complete CSV lines) as DictReader yields them for a file with this header."""
    text = header_text + body.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def _metrics_rows(path: Path, since: int) -> tuple[list[dict[str, str]], list[str], int]:
    """(rows from `since` on, columns, total rows) of a metrics.csv; only complete lines count.

    A row still being written (no newline yet) is left for the next read. Reads continue from where the
    previous read of this file stopped when they can: same file (inode), same header, not shrunk, and
    `since` at or after its last row (the GUI re-fetches its last row, which may have been half-written).
    Anything else (a first load, a rewrite with new columns, a replaced file) reads the file whole.
    """
    key = path.resolve()
    try:
        stat = path.stat()
        with open(path, "rb") as f:
            header_line = f.readline()
            if not header_line.endswith(b"\n"):
                return [], [], 0  # the header itself is still being written
            header_text = header_line.decode("utf-8", errors="replace")
            header = next(csv.reader([header_text.rstrip("\r\n")]), [])
            with _METRICS_TAILS_LOCK:
                tail = _METRICS_TAILS.get(key)
            resumable = (
                tail is not None
                and tail.inode == stat.st_ino
                and tail.header == header
                and stat.st_size >= tail.end
                and tail.rows >= 1
                and since >= tail.rows - 1
            )
            if resumable:
                start, first_row = (
                    (tail.last_row_start, tail.rows - 1) if since == tail.rows - 1 else (tail.end, tail.rows)
                )
            else:
                start, first_row = len(header_line), 0
            f.seek(start)
            body = f.read()
    except OSError as exc:
        raise FileNotFoundError(f"Cannot read {path}: {exc}") from exc

    complete = body[: body.rfind(b"\n") + 1]
    rows = _parse_rows(header_text, complete)
    total = first_row + len(rows)
    if rows:
        # The byte offset where the last complete row starts, so the next read can re-serve it
        last_line = complete[:-1].rfind(b"\n") + 1
        last_row_start = start + last_line
    else:
        last_row_start = tail.last_row_start if resumable else start
    with _METRICS_TAILS_LOCK:
        _METRICS_TAILS[key] = _MetricsTail(stat.st_ino, header, total, last_row_start, start + len(complete))
        _METRICS_TAILS.move_to_end(key)
        while len(_METRICS_TAILS) > _METRICS_TAILS_LIMIT:
            _METRICS_TAILS.popitem(last=False)
    return rows[max(since - first_row, 0) :], header, total


# ── Run manifest: every artifact a run produced ───────────────────────────────

# Top-level folders of results/ that hold per-run output as <root>/<group>/<experiment_id>/...
RESULT_ROOTS = ("logs", "plots", "checkpoints", "tensorboard", "datasets")
MANIFEST_LIMIT = 5000

_KIND_BY_SUFFIX = {
    ".csv": "table",
    ".tsv": "table",
    ".png": "figure",
    ".jpg": "figure",
    ".jpeg": "figure",
    ".svg": "figure",
    ".pdf": "figure",
    ".md": "report",
    ".txt": "report",
    ".html": "report",
    ".yaml": "config",
    ".yml": "config",
    ".json": "metadata",
    ".ckpt": "checkpoint",
    ".pt": "checkpoint",
    ".pth": "checkpoint",
    ".zip": "checkpoint",
    ".npz": "cache",
    ".npy": "cache",
    ".pkl": "cache",
    ".parquet": "cache",
    ".log": "log",
}


def results_dir() -> Path:
    """results/, the parent of results/logs (follows LOGS_DIR, so tests can point it elsewhere)."""
    return LOGS_DIR.parent


def _artifact_kind(root: str, path: Path) -> str:
    if root == "logs" and path.name == "metrics.csv":
        return "metrics"
    if root == "tensorboard" and path.name.startswith("events.out.tfevents"):
        return "tensorboard"
    return _KIND_BY_SUFFIX.get(path.suffix.lower(), "other")


def _run_dirs(group: str, experiment_id: str) -> dict[str, Path]:
    """The existing results/<root>/<group>/<experiment_id> folders of one run, by root."""
    check_path_segment(group)
    check_path_segment(experiment_id)
    base = results_dir()
    dirs = {root: base / root / group / experiment_id for root in RESULT_ROOTS}
    return {root: path for root, path in dirs.items() if path.is_dir()}



def _mutable_run_dirs(group: str, experiment_id: str) -> dict[str, Path]:
    """Locate run folders without following links outside the results tree."""
    dirs = _run_dirs(group, experiment_id)
    base = results_dir().resolve()
    for path in dirs.values():
        if any(part.is_symlink() for part in (path, path.parent, path.parent.parent)) or not path.resolve().is_relative_to(base):
            raise ValueError(f"Unsafe run folder: {path}")
    if not dirs:
        raise FileNotFoundError(f"Run not found: {group}/{experiment_id}")
    return dirs


def _archive_dirs(dirs: dict[str, Path], destination: str) -> None:
    """Write an archive atomically, preserving paths relative to results/."""
    target = Path(destination).absolute()
    if any(target.resolve().is_relative_to(path.resolve()) for path in dirs.values()):
        raise ValueError("Archive destination cannot be inside the selected results")
    fd, temporary = tempfile.mkstemp(prefix=".theta-archive-", suffix=".zip", dir=target.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for run_dir in dirs.values():
                for path in sorted(run_dir.rglob("*")):
                    if path.is_symlink():
                        raise ValueError(f"Cannot archive symbolic link: {path}")
                    if path.is_file():
                        archive.write(path, path.relative_to(results_dir()))
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def archive_run(group: str, experiment_id: str, destination: str) -> None:
    """Write a ZIP of all results folders for a run, without removing the originals."""
    _archive_dirs(_mutable_run_dirs(group, experiment_id), destination)


def _mutable_group_dirs(group: str) -> dict[str, Path]:
    check_path_segment(group)
    base = results_dir()
    dirs = {root: base / root / group for root in RESULT_ROOTS}
    dirs = {root: path for root, path in dirs.items() if path.is_dir()}
    if not dirs:
        raise FileNotFoundError(f"Group not found: {group}")
    for path in dirs.values():
        if (path.is_symlink() or path.parent.is_symlink()
                or not path.resolve().is_relative_to(base.resolve())):
            raise ValueError(f"Unsafe group folder: {path}")
    return dirs


def archive_group(group: str, destination: str) -> None:
    """Archive all of a group's results, including runs not shown in the tree."""
    _archive_dirs(_mutable_group_dirs(group), destination)


def delete_group(group: str) -> None:
    """Delete all of a group's folders across the results roots."""
    for path in _mutable_group_dirs(group).values():
        shutil.rmtree(path)


def delete_run(group: str, experiment_id: str) -> None:
    """Delete the run's folders across the results roots."""
    for path in _mutable_run_dirs(group, experiment_id).values():
        shutil.rmtree(path)

def run_manifest(group: str, experiment_id: str, completed_only: bool = False) -> dict[str, Any]:
    """Every file a run produced across results/{logs,plots,checkpoints,tensorboard,datasets}.

    Each artifact has a `path` relative to results/ (read it with read_run_file), a `kind` (metrics, table,
    figure, report, config, metadata, checkpoint, cache, tensorboard, log, other), and the `agent` and
    `version` folders it sits under, when it has them. `agents` maps each agent with logged metrics to its
    version folders, oldest first. Raises FileNotFoundError if the run has no folders.
    """
    dirs = _run_dirs(group, experiment_id)
    if not dirs:
        raise FileNotFoundError(f"Run not found: {group}/{experiment_id}")
    base = results_dir()
    run_logs = LOGS_DIR / group / experiment_id
    replicates = _result_replicates(run_logs)
    published, complete_methods, terminal_trials, in_progress = (
        _result_snapshot(run_logs, replicates) if completed_only else ({}, set(), set(), False)
    )
    if completed_only and not published:
        raise FileNotFoundError(f"No completed results for {group}/{experiment_id}")
    whole_run_finished = completed_only and not in_progress

    def visible(root, parts):
        if not completed_only:
            return True
        version = next((p for p in parts if p.startswith("version_")), None)
        if version:
            return version in published.get(parts[0], [])
        if "sweeps" in parts and any(p.startswith("trial_") for p in parts):
            return any(tuple(parts[:len(folder)]) == folder for folder in terminal_trials)
        if len(parts) == 1:
            return parts[0] == "config.yaml" or whole_run_finished
        if root == "checkpoints" and parts[-1] == "tuning_summary.json" and (
            len(parts) == 2 or (len(parts) == 4 and parts[1] == "sweeps")
        ):
            # A finished study remains available even if another study of this method
            # is active (including an older attempt left with a running snapshot).
            summary = _read_json(dirs[root].joinpath(*parts))
            trials = summary.get("trials")
            return bool(summary.get("study") and summary.get("stop_reason") != "running"
                        and isinstance(trials, list) and trials
                        and all(t.get("state") in ("complete", "pruned", "fail") for t in trials))
        return parts[0] in complete_methods

    artifacts: list[dict[str, Any]] = []
    truncated = False
    for root, run_dir in dirs.items():
        # Prune active version directories before walking their files (notably TensorBoard logs).
        for directory, children, files in os.walk(run_dir):
            relative = Path(directory).relative_to(run_dir).parts
            children[:] = sorted(name for name in children if not name.startswith("version_") or visible(root, (*relative, name)))
            for filename in sorted(files):
                path = Path(directory) / filename
                parts = path.relative_to(run_dir).parts
                if not visible(root, parts):
                    continue
                if len(artifacts) >= MANIFEST_LIMIT:
                    truncated = True
                    break
                stat = path.stat()
                artifacts.append(
                    {
                        "path": path.relative_to(base).as_posix(),
                        "root": root,
                        "kind": _artifact_kind(root, path),
                        "agent": parts[0] if len(parts) > 1 else None,
                        "version": next((p for p in parts[1:-1] if p.startswith("version_")), None),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
            if len(artifacts) >= MANIFEST_LIMIT:
                truncated = True
                break
    agents: dict[str, list[str]] = {}
    for item in artifacts:
        if item["kind"] == "metrics" and item["agent"] and item["version"]:
            agents.setdefault(item["agent"], []).append(item["version"])
    return {
        "group": group,
        "experiment_id": experiment_id,
        "agents": {name: sorted(versions, key=_version_number) for name, versions in sorted(agents.items())},
        "version_details": {agent: {version: details for (name, version), details in replicates.items()
                                    if name == agent and version in versions}
                            for agent, versions in agents.items()},
        "artifacts": artifacts,
        "truncated": truncated,
    }


def read_run_file(group: str, experiment_id: str, path: str) -> bytes:
    """One artifact of a run, by the `path` its manifest lists (relative to results/).

    Only files inside this run's own results/<root>/<group>/<experiment_id> folders are read: anything
    else raises PermissionError. A missing file raises FileNotFoundError.
    """
    dirs = _run_dirs(group, experiment_id)
    target = (results_dir().resolve() / path).resolve()
    if not any(target.is_relative_to(d.resolve()) for d in dirs.values()):
        raise PermissionError("Path is outside this run's results")
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return target.read_bytes()


def list_plots(group: str, experiment_id: str) -> list[str]:
    plot_dir = PLOTS_DIR / group / experiment_id
    return sorted(f.name for f in plot_dir.glob("*") if f.is_file()) if plot_dir.exists() else []
