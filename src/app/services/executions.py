"""Local execution browser data, kept separate from legacy run catalog APIs."""
from pathlib import Path
import json

from src.app.pipeline.execution_data import ExecutionData


def list_executions(completed_only=False):
    root = Path("results/executions")
    paths = sorted(root.glob("*/*/*/manifest.json"), reverse=True)
    if completed_only:
        completed = []
        for path in paths:
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if any(run.get("status") == "completed" for run in manifest.get("runs", [])):
                completed.append(path)
        paths = completed
    return [str(path) for path in paths]


def read_execution(path, completed_only=False):
    root = Path("results/executions").resolve()
    manifest = Path(path).resolve()
    if root not in manifest.parents or manifest.name != "manifest.json":
        raise ValueError("Invalid execution manifest path")
    data = ExecutionData(manifest)
    if completed_only:
        data.manifest["runs"] = [run for run in data.runs() if run.get("status") == "completed"]
    runs = list(data.runs())
    events, curves, errors = {}, {}, []
    for run in runs:
        try:
            events[run["run_id"]] = data.events(run)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            events[run["run_id"]] = []
    for method in dict.fromkeys(run["method"] for run in runs):
        try:
            curves[method] = data.aggregate(method)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            curves[method] = []
    return {"manifest": data.manifest, "events": events, "curves": curves, "errors": errors}
