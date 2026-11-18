"""Adapt completed shared executions to the objective API; never launch training."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
from src.app.objectives.api import RunResult, TrialResults, validate_objective


def _records(path):
    """JSON lines; a seed still writing may leave an incomplete final line."""
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [json.loads(line) for line in lines[:-1]]
    if lines:
        try:
            records.append(json.loads(lines[-1]))
        except json.JSONDecodeError:
            pass
    return records


def read_run(seed, status, directory, elapsed, error=None):
    metrics, events = {}, []
    directory = Path(directory)
    csv_path = directory / "metrics.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                values = {}
                for name, value in row.items():
                    try:
                        number = float(value)
                        if math.isfinite(number):
                            values[name] = number
                    except (ValueError, TypeError):
                        continue
                metrics.update(values)
                events.append({"step": values.get("step"), "data": values})
    stream_path = directory / "events.jsonl"
    if stream_path.is_file():
        from src.app.telemetry.records import decode, metrics_row
        for line in stream_path.read_text(encoding="utf-8").splitlines():
            event = decode(line)
            if event and event["type"] == "metrics":
                events.append(event)
                metrics.update(metrics_row(event))
    event_path = directory / "evaluation_events.jsonl"
    if event_path.is_file():
        events.extend(_records(event_path))
    return RunResult(seed, status, metrics, events, {"logs": str(directory)}, elapsed, error)


def read_runs(execution, root):
    runs = []
    for run in execution["runs"]:
        directory = Path(root) / run["logs"] / run["agent"] / run["version"]
        result = read_run(run["training_seed"], run["status"], directory,
                          run.get("elapsed_seconds", 0), run.get("error"))
        result.artifacts["checkpoints"] = run["checkpoints"]
        runs.append(result)
    return runs


class ProgressScorer:
    """Scores a running seed panel with the objective's intermediate function.

    The objective source is loaded once per trial, so every checkpoint of a trial
    is scored by the same code.
    """

    def __init__(self, component, params, config, root):
        self.component, self.params, self.config, self.root = component, params, config, Path(root)
        self.module = component.load_source()
        self._fingerprint = None

    def _state(self, execution):
        state = []
        for run in execution["runs"]:
            directory = self.root / run["logs"] / run["agent"] / run["version"]
            for name in ("metrics.csv", "events.jsonl", "evaluation_events.jsonl"):
                try:
                    info = (directory / name).stat()
                    state.append((run["run_id"], name, info.st_size, info.st_mtime_ns))
                except OSError:
                    continue
            state.append((run["run_id"], run["status"]))
        return state

    def score(self, manifest_path):
        """The objective's progress, or None when nothing changed or it has no checkpoint yet."""
        execution = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        fingerprint = self._state(execution)
        if fingerprint == self._fingerprint:
            return None
        self._fingerprint = fingerprint
        results = TrialResults(read_runs(execution, self.root), self.config)
        return self.component.intermediate(results, self.params, module=self.module)


def score_execution(cfg, manifest_path, output, artifact_root, *, paths=None):
    from src.app.workspace import WorkspacePaths
    root = (paths or WorkspacePaths.for_process()).root
    component, params = validate_objective(cfg)
    execution = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = OmegaConf.to_container(cfg, resolve=True)
    runs = read_runs(execution, root)
    record = {"schema_version": 1, "objective": config["objective"], "params": params,
              "config": config, "plan": execution["plan"], "execution_manifest": str(manifest_path),
              "runs": [asdict(run) for run in runs]}
    path = Path(output) / "objective.json"
    def save():
        path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (Path(artifact_root) / "objective.json").write_text(path.read_text(), encoding="utf-8")
    try:
        result = component.evaluate(TrialResults(runs, config), params)
        record["result"] = asdict(result)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise
    save()
    return result, path
