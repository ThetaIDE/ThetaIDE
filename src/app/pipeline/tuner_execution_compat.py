"""Bounded metadata-only reader for pre-portable-execution site receipts.

Old receipts reserved the Optuna SQLite path even for fixed jobs. Other legacy
local-only tuners retain their original overrides without gaining remote hooks.
This reader is never used for new jobs, which explicitly store a plan or null.
"""
from .tuner_execution import TunerArtifact, TunerExecutionPlan, TunerLaunchBindings


class LegacyTunerExecutionPlan(TunerExecutionPlan):
    """In-memory marker; old receipts are never rewritten to add this marker."""


def fixed_bindings(cfg, plan, runtime):
    if not isinstance(plan, LegacyTunerExecutionPlan) or cfg.get("tuning") is not None:
        return None
    environment = {}
    if plan.storage_kind == "managed":
        environment["THETA_SQLITE_ROOT"] = str((runtime / plan.artifact("study").path).parent)
    return TunerLaunchBindings(environment=environment)


def legacy_config(record):
    """Adapt only old receipt configuration, without editing saved bytes."""
    from copy import deepcopy
    config = deepcopy(record.get('config') or {})
    tuning = config.get('tuning')
    if tuning is not None and not isinstance(tuning, dict):
        raise ValueError('Legacy receipt tuning must be a mapping')
    plan = legacy_plan(record)
    if tuning is not None:
        if plan is not None and tuning.get('component') in (None, ''):
            tuning['component'] = plan.component
    return config


def legacy_plan(record, *, paths=None):
    if record.get('schema_version', 1) != 1:
        raise ValueError('Unsupported legacy site receipt schema')
    config = record.get('config') or {}
    tuning = config.get('tuning') or {}
    if (tuning.get('component') or 'optuna') != 'optuna':
        return None
    options = (tuning.get('options') or {}).get('optuna') or {}
    identity = record['execution_id']
    managed = record.get('tuning_storage_kind', 'sqlite') != 'external'
    return LegacyTunerExecutionPlan('optuna', identity,
        artifacts=(TunerArtifact('study', f'results/optuna/{identity}/study.sqlite', True),) if managed else (),
        storage_kind='managed' if managed else 'external',
        concurrency=options.get('n_jobs', tuning.get('n_jobs', 1)),
        trials=options.get('n_trials', tuning.get('n_trials', 40)),
        distributions=('optuna', 'sqlalchemy'),
        continuation_notes='reinitialized from configured seed; suggestion sequence can differ')
