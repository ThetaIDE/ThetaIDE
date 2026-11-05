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


def trial_execution(logs_root, method, trial):
    """Read a shared execution, adapting the earlier objective-runner artifacts read-only.

    Compatibility stays at the data boundary; queue/monitor never understand
    objective implementations or launch their own workers.
    """
    version = trial.get("log_version")
    if not version:
        return None
    directory = Path(logs_root) / method / version
    try:
        pointer = json.loads((directory / "execution.json").read_text(encoding="utf-8"))
        path = Path(pointer["manifest"])
        path = path if path.is_absolute() else directory / path
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    try:
        old = json.loads((directory / "objective.json").read_text(encoding="utf-8"))
        runtime = old["config"]["theta_tuning"]
        runs = []
        for index, sample in enumerate(old["runs"]):
            if sample is None:
                continue
            path = Path(sample["artifacts"]["logs"])
            runs.append({"run_id": f"{version}:{index}", "method": method, "agent": path.parent.name,
                         "training_seed": sample["seed"], "status": sample["status"],
                         "logs": str(path.parent.parent), "version": path.name,
                         "output_log": str(path / "training.log"), "legacy_execution": True})
        return {"schema_version": 1, "runs": runs, "parent": {
            "kind": "trial", "method": method, "study": runtime["study_name"],
            "trial_number": runtime["trial_number"], "log_version": version}}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def study_execution_states(summary, logs_root, method):
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
    return dict(summary, trials=trials)


class ExecutionData:
    """Read a manifest and its per-run event streams without modifying artifacts."""

    version = 1

    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported execution manifest version")

    def runs(self):
        return tuple(dict(run) for run in self.manifest.get("runs", []))

    def events(self, run):
        path = Path(run.get("event_path") or (Path(run["logs"]) / "evaluation_events.jsonl"))
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


class LegacyExecutionData:
    """Expose one existing log directory as a single-run execution.

    Legacy scalar telemetry does not establish episode outcomes, so this
    adapter deliberately returns no episode events and no aggregate curve.
    Existing plotters can continue reading the original log directory.
    """
    version = 1

    def __init__(self, group, experiment_id, method, logs_root="results/logs"):
        self.run = {"run_id": f"legacy:{group}/{experiment_id}/{method}", "method": method,
                    "status": "unknown", "logs": str(Path(logs_root) / group / experiment_id),
                    "experiment_id": experiment_id, "legacy": True}
        self.manifest = {"schema_version": 1, "group": group, "experiment_id": experiment_id,
                         "legacy": True, "runs": [self.run]}

    def runs(self):
        return (dict(self.run),)

    def events(self, run):
        return []

    def aggregate(self, method, grid=None):
        return []
