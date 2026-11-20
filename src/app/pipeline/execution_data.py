"""Versioned, read-only access to repeated execution evaluation data."""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def replicate_label(manifest, run):
    """Stable, collision-free source identity for experiment and nested trial seeds."""
    parent = manifest.get("parent") or {}
    method = str(run.get("method", "run"))
    if parent.get("kind") == "trial":
        method += f" · {parent['study']} · trial {parent['trial_number']}"
    return f"{method} · seed {run.get('training_seed')}"



def synced_execution_sources(project_root, group, experiment):
    """Rebase cluster seed feeds onto the exact downloaded execution directory."""
    from src.app.sites.storage import results_relative, safe_path
    relative = str(group) + '/' + str(experiment)
    manifests = safe_path(project_root, results_relative(project_root, 'executions/' + relative))
    logs = safe_path(project_root, results_relative(project_root, 'logs/' + relative))
    found, sources = False, []
    for path in sorted(manifests.glob('*/manifest.json')):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        found = True
        for run in manifest.get('runs', []):
            if run.get('status') != 'running' or not run.get('version'):
                continue
            agent = str(run.get('agent') or run.get('method', 'run'))
            directory = safe_path(logs, agent)
            sources.append((replicate_label(manifest, run), directory, run['version']))
    return found, sources

def trial_execution(logs_root, method, trial):
    """The seed execution a trial's log version links to, or None."""
    version = trial.get("log_version")
    from src.app.pipeline.result_paths import LEGACY_TRIAL_VERSION
    from src.app.pipeline.trial_artifacts import trial_logs
    if "id" not in trial and (match := LEGACY_TRIAL_VERSION.fullmatch(version or "")):
        trial = dict(trial, id=int(match[1]))
    method_dir = Path(logs_root) / method
    directory = trial_logs(method_dir, trial)
    if directory == method_dir:
        return None
    try:
        pointer = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
        path = Path(pointer["manifest"])
        path = path if path.is_absolute() else directory / path
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None


def study_execution_states(summary, logs_root, method, *, executions_root=None):
    """Enrich a study snapshot with actual seed-worker states from shared records."""
    trials = []
    for trial in summary.get("trials", []):
        execution = trial_execution(logs_root, method, trial)
        if execution:
            trial = dict(trial, execution=execution)
            runs = execution.get("runs", [])
            if trial.get("state") in ("running", "pending"):
                states = {run.get("status") for run in runs}
                trial["state"] = "running" if "running" in states or not states.intersection({"queued", "pending"}) else "pending"
                starts = [run["execution_started"] for run in runs if run.get("execution_started") and run.get("status") == "running"]
                trial["execution_started"] = min(starts) if starts else trial.get("execution_started")
        trials.append(trial)
    # Cluster telemetry/manifests are published more frequently than database
    # exports. Preserve the study snapshot's values, but admit a newer active
    # trial so monitor navigation does not disappear between exports.
    known = {trial.get('id') for trial in trials}
    for path in sorted(Path(executions_root).glob('*/manifest.json')) if executions_root is not None else ():
        try:
            execution = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        parent = execution.get('parent') or {}
        number = parent.get('trial_number')
        if (parent.get('kind') != 'trial' or parent.get('study') != summary.get('study')
                or not isinstance(number, int) or number in known):
            continue
        active = [run for run in execution.get('runs', [])
                  if run.get('method') == method and run.get('status') == 'running']
        if not active:
            continue
        starts = [run['execution_started'] for run in active if run.get('execution_started')]
        trials.append({'id': number, 'state': 'running', 'log_version': parent.get('log_version'),
                       'execution_started': min(starts) if starts else None,
                       'parameters': {}, 'value': None, 'execution': execution})
        known.add(number)
    trials.sort(key=lambda trial: trial['id'])
    return dict(summary, trials=trials)


class ExecutionData:
    """Read a manifest and its per-run event streams without modifying artifacts."""

    version = 1

    def __init__(self, manifest_path, *, root=None):
        """``root`` is the workspace that the manifest's relative run paths belong to."""
        from src.app.workspace import WorkspacePaths
        self.path = Path(manifest_path)
        self.root = Path(root) if root is not None else WorkspacePaths.for_process().root
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported execution manifest version")

    def runs(self):
        return tuple(dict(run) for run in self.manifest.get("runs", []))

    def events(self, run):
        path = self.root / (run.get("event_path") or Path(run["logs"]) / "evaluation_events.jsonl")
        if not path.exists():
            return []
        events = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid evaluation event at {path}:{line_number}") from exc
                if event.get("schema_version") != 1 or event.get("run_id") != run["run_id"]:
                    raise ValueError(f"Invalid evaluation event identity at {path}:{line_number}")
                events.append(event)
        return events

    def aggregate(self, method, grid=None):
        """Mean and sample SD of per-run episode means on a common step grid.

        The default grid is the union of observed steps. Linear interpolation is
        restricted to each run's observed range. Failed and unfinished runs are
        retained in the manifest but excluded from this aggregate.
        """
        curves = []
        for run in self.runs():
            if run.get("method") != method or run.get("status") != "completed":
                continue
            points = {}
            for event in self.events(run):
                outcomes = event.get("episode_outcomes") or []
                if event.get("metric") != "eval/reward" or not outcomes:
                    continue
                values = [float(x) for x in outcomes]
                step = float(event["training_step"])
                if math.isfinite(step) and all(math.isfinite(x) for x in values):
                    points[step] = statistics.mean(values)
            if points:
                curves.append(sorted(points.items()))
        steps = sorted(set(float(x) for x in grid)) if grid is not None else sorted({x for curve in curves for x, _ in curve})
        result = []
        for step in steps:
            values = []
            for curve in curves:
                if step < curve[0][0] or step > curve[-1][0]:
                    continue
                for i, (x, y) in enumerate(curve):
                    if step == x:
                        values.append(y)
                        break
                    if x > step:
                        x0, y0 = curve[i - 1]
                        values.append(y0 + (y - y0) * (step - x0) / (x - x0))
                        break
            if values:
                result.append({"step": step, "mean": statistics.mean(values), "n": len(values),
                               "std": statistics.stdev(values) if len(values) >= 2 else None})
        return result


def trial_worker_results(paths, group, experiment, method, log_version):
    """Return seed-indexed worker execution records, including failed workers."""
    logs = paths.results / "logs" / str(group) / str(experiment)
    execution = trial_execution(logs, method, {"log_version": log_version}) or {}
    records = {}
    for run in execution.get("runs", ()):
        try:
            path = (paths.root / run["checkpoints"] / "execution.json").resolve()
            if not path.is_relative_to(paths.results.resolve()):
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(record, dict):
                records[run["training_seed"]] = record
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return records
