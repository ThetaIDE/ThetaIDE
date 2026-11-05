"""Adapt completed shared executions to the objective API; never launch training."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf
from src.app.objectives.api import RunResult, TrialResults, validate_objective


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
        events.extend(json.loads(line) for line in event_path.read_text().splitlines() if line.strip())
    return RunResult(seed, status, metrics, events, {"logs": str(directory)}, elapsed, error)


def score_execution(cfg, manifest_path, output, artifact_root):
    component, params = validate_objective(cfg)
    execution = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    config = OmegaConf.to_container(cfg, resolve=True)
    runs = []
    for run in execution["runs"]:
        directory = Path(run["logs"]) / run["agent"] / run["version"]
        result = read_run(run["training_seed"], run["status"], directory,
                          run.get("elapsed_seconds", 0), run.get("error"))
        result.artifacts["checkpoints"] = run["checkpoints"]
        runs.append(result)
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
