"""Local execution browser data, kept separate from legacy run catalog APIs."""
from pathlib import Path

from src.app.pipeline.execution_data import ExecutionData


def list_executions():
    root = Path("results/executions")
    return [str(path) for path in sorted(root.glob("*/*/*/manifest.json"), reverse=True)]


def read_execution(path):
    root = Path("results/executions").resolve()
    manifest = Path(path).resolve()
    if root not in manifest.parents or manifest.name != "manifest.json":
        raise ValueError("Invalid execution manifest path")
    data = ExecutionData(manifest)
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
