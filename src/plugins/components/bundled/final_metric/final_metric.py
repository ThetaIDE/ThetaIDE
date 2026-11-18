"""Average the final recorded metric across independent training runs."""
from src.app.api.tuning import ObjectiveProgress, ObjectiveResult


def evaluate(results, params):
    values = []
    for run in results.runs:
        if run.status != "completed":
            raise ValueError(f"Training seed {run.seed} did not complete: {run.error}")
        values.append(float(run.metrics[params["metric"]]))
    return ObjectiveResult(sum(values) / len(values), {"per_run": values})


def intermediate(results, params):
    """Mean across seeds of the metric's latest observation that every seed has logged."""
    for run in results.runs:
        if run.status in ("failed", "cancelled"):
            raise ValueError(f"Training seed {run.seed} did not complete: {run.error}")
    progress = results.common_progress(params["metric"], axis="step")
    if progress is None:
        return None
    step, _ = progress
    values = [run.curve(params["metric"], axis="step")[step][1] for run in results.runs]
    return ObjectiveProgress(step, sum(values) / len(values), {"per_run": values})
