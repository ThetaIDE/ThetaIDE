"""Read-only, headless study summaries shared by queues and desktop monitors."""
from pathlib import Path
import json
import math


def local_study_summaries(project_root, group, experiment_id, *, started=None):
    """Read normalized provider exports; never open live tuner state.

    ``started`` excludes files left by an earlier launch of the same recipe.
    Historical exports without a schema marker remain readable.
    """
    root = Path(project_root) / 'results/checkpoints' / str(group) / str(experiment_id)
    found = {}
    for path in sorted(root.glob('*/sweeps/*/tuning_summary.json')) + sorted(root.glob('*/tuning_summary.json')):
        try:
            if started is not None and path.stat().st_mtime < started:
                continue
            data = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if not valid_study_summary(data):
            continue
        method = path.relative_to(root).parts[0]
        found[(method, data['study'])] = data
    return [(method, data) for (method, _), data in found.items()]


def valid_study_summary(data):
    if not isinstance(data, dict) or type(data.get('schema_version', 1)) is not int or data.get('schema_version', 1) != 1:
        return False
    if not isinstance(data.get('study'), str) or not data['study'] or not isinstance(data.get('trials'), list):
        return False
    if data.get('component') is not None and (not isinstance(data['component'], str) or not data['component']):
        return False
    if 'direction' in data and data['direction'] not in ('minimize', 'maximize'):
        return False
    for key in ('requested_trials', 'initial_trials', 'future_trials', 'pending_trials', 'best_trial_id'):
        if data.get(key) is not None and (type(data[key]) is not int or data[key] < 0):
            return False
    importance = data.get('parameter_importance', {})
    if not isinstance(importance, dict) or any(type(value) not in (int, float) or not math.isfinite(value) for value in importance.values()):
        return False
    ids = set()
    for trial in data['trials']:
        if not isinstance(trial, dict) or type(trial.get('id')) is not int or trial['id'] < 0 or trial['id'] in ids:
            return False
        ids.add(trial['id'])
        if not isinstance(trial.get('state'), str) or not trial['state']:
            return False
        if not isinstance(trial.get('parameters', {}), dict) or not isinstance(trial.get('intermediate_values', {}), dict):
            return False
        for key in ('value', 'duration_seconds'):
            if trial.get(key) is not None and (type(trial[key]) not in (int, float) or not math.isfinite(trial[key])):
                return False
    return True


def study_summaries(project_root, run, *, paths=None):
    from src.app.pipeline.execution_data import study_execution_states
    backend = (run or {}).get("backend") or {}
    if not backend.get("group") or not backend.get("experiment_id"):
        return []
    executions = (Path(project_root) / 'results/executions' / backend['group'] / backend['experiment_id']
                  if (run or {}).get('remote') else None)
    found = {}
    for method, data in local_study_summaries(project_root, backend["group"], backend["experiment_id"]):
        logs = Path(project_root) / "results/logs" / backend["group"] / backend["experiment_id"]
        found[(method, data["study"])] = study_execution_states(
            data, logs, data.get("log_method") or method, executions_root=executions)
    # Older compute snapshots omitted commits for live tuning summaries. Recover
    # study metadata from the downloaded database export, never the live database.
    remote = (run or {}).get('remote') or {}
    execution_id = remote.get('execution_id')
    if not found and execution_id:
        from src.app.pipeline.exceptions import ConfigurationError
        from src.app.pipeline.tuner_execution import read_remote_snapshots
        try:
            logs = Path(project_root) / 'results/logs' / backend['group'] / backend['experiment_id']
            for method, data in read_remote_snapshots(project_root, remote, paths=paths):
                found[(method, data['study'])] = study_execution_states(
                    data, logs, data.get('log_method') or method, executions_root=executions)
        except (OSError, ValueError, RuntimeError, ConfigurationError, ImportError):
            pass  # A partially pulled snapshot is retried after its stamp changes.
    return [(method, data) for (method, _), data in found.items()]

