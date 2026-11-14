"""Average the final recorded metric across independent training runs."""
from src.app.api.tuning import ObjectiveResult


def evaluate(results, params):
    values = []
    for run in results.runs:
        if run.status != "completed":
            raise ValueError(f"Training seed {run.seed} did not complete: {run.error}")
        values.append(float(run.metrics[params["metric"]]))
    return ObjectiveResult(sum(values) / len(values), {"per_run": values})
