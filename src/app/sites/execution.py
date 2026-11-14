"""Scheduler-neutral receipts, ownership and finalization; the selected scheduler component executes jobs."""
from __future__ import annotations

import re
import subprocess  # noqa: F401  (API-1 tests and adapters patch execution.subprocess)
from pathlib import Path

from .storage import digest, read_json, safe_path, write_json

TERMINAL = {'completed', 'failed', 'cancelled', 'timeout', 'out_of_memory', 'node_fail', 'preempted', 'boot_fail', 'deadline', 'revoked'}


def close_unfinished_runs(runtime, record, status, *, paths=None):
    """Called only after allocation-owned producers have stopped, before publication."""
    directory = safe_path(runtime, 'results/executions/' + record['group'] + '/' + record['result_experiment_id'])
    for path in directory.glob('*/manifest.json'):
        manifest = read_json(path)
        for run in manifest.get('runs', []):
            if run.get('status') in ('queued', 'pending', 'running'):
                run.update(status=status, error='Allocation ended before this seed completed')
        write_json(path, manifest)
    if record['config'].get('tuning') is not None:
        from src.app.pipeline.tuners import get_tuner
        from src.app.pipeline.tuner_execution import record_config
        from src.app.workspace import WorkspacePaths
        hook = getattr(get_tuner(record_config(record), paths=paths or WorkspacePaths(runtime)), 'finalize_interrupted', None)
        if callable(hook):
            hook(Path(runtime), record)


def receipt_path(root, execution_id):
    if not re.fullmatch(r'[a-f0-9]{32}', execution_id): raise ValueError('Invalid execution ID')
    return Path(root) / 'results/slurm' / execution_id / 'submission.json'


def _regular_output(path):
    if path.name.endswith(('-wal', '-shm', '-journal')): return False
    try: return path.is_file() and not path.is_symlink()
    except FileNotFoundError: return False


def owned_files(root, record):
    """Select only output trees owned by this UUID, including nested seed executions."""
    root = Path(root)
    prefix = record['result_experiment_id']
    for category in ('logs', 'checkpoints', 'plots', 'datasets', 'executions'):
        parent = safe_path(root, f'results/{category}/{record["group"]}')
        if not parent.is_dir(): continue
        # Preserve recipe subdirectories and allow variability's stamped suffix.
        parent = parent / str(Path(prefix).parent)
        if not parent.is_dir(): continue
        for directory in parent.iterdir():
            if directory.name != Path(prefix).name and not directory.name.startswith(Path(prefix).name + '_'): continue
            if directory.is_dir() and not directory.is_symlink():
                for path in directory.rglob('*'):
                    if _regular_output(path):
                        yield safe_path(root, path.relative_to(root).as_posix())
    from src.app.pipeline.tuner_execution import record_plan
    plan = record_plan(record)
    for artifact in plan.artifacts if plan else ():
        path = safe_path(root, artifact.path)
        if _regular_output(path):
            yield path
    directory = receipt_path(root, record['execution_id']).parent
    for path in directory.iterdir():
        if _regular_output(path) and path.suffix != '.lock' and path.name not in ('submission.json', 'artifacts.json'):
            yield path


def finalize(root, execution_id):
    path = receipt_path(root, execution_id)
    record = read_json(path)
    if record.get('state') not in TERMINAL or record.get('scheduler_state') not in TERMINAL:
        raise ValueError('Finalization requires confirmed scheduler termination')
    if record.get('execution_policy'):
        publication = read_json(Path(record['execution_policy']['work_root']) / 'results/slurm' / execution_id / 'publication.json')
        if not publication.get('sealed'):
            raise ValueError('Payload is unsealed; explicit scheduled repair is required')
        write_json(path.parent / 'artifacts.json', {'execution_id': execution_id, 'files': publication['files']})
        record.update(artifacts_finalized=True)
        write_json(path, record)
        return record
    artifacts = []
    for item in sorted(set(owned_files(root, record))):
        artifacts.append({'path': item.relative_to(root).as_posix(), 'size': item.stat().st_size,
                          'sha256': digest(item)})
    write_json(path.parent / 'artifacts.json', {'execution_id': execution_id, 'files': artifacts})
    record.update(artifacts_finalized=True)
    write_json(path, record)
    return record


# Provider-neutral execution entry points. Historic schema-v1 receipts use the
# bounded scheduler selection reader in scheduler.py; new receipts name their owner.
def validate_allocation(cfg, *, paths=None):
    from .scheduler_api import get_scheduler
    return get_scheduler(cfg=cfg, paths=paths).validate_allocation(cfg, paths=paths)


def submit(root, experiment, overrides, cfg):
    from .scheduler_api import get_scheduler
    from src.app.workspace import WorkspacePaths
    return get_scheduler(cfg=cfg, paths=WorkspacePaths(root)).submit(root, experiment, overrides, cfg)


def run(root, execution_id):
    from .scheduler_api import get_scheduler
    from src.app.workspace import WorkspacePaths
    record = read_json(receipt_path(root, execution_id))
    return get_scheduler(record=record, paths=WorkspacePaths(root)).run(root, execution_id)


def refresh_status(root, record):
    from .scheduler_api import get_scheduler
    from src.app.workspace import WorkspacePaths
    return get_scheduler(record=record, paths=WorkspacePaths(root)).refresh_status(root, record)


def refresh_many(root, records, *, minimum_seconds=10):
    from .scheduler_api import get_scheduler, scheduler_selector
    from src.app.workspace import WorkspacePaths
    groups = {}
    for record in records:
        groups.setdefault(scheduler_selector(record=record), []).append(record)
    for group in groups.values():
        get_scheduler(record=group[0], paths=WorkspacePaths(root)).refresh_many(root, group, minimum_seconds=minimum_seconds)
    return records


def cancel(root, record):
    from .scheduler_api import get_scheduler
    from src.app.workspace import WorkspacePaths
    return get_scheduler(record=record, paths=WorkspacePaths(root)).cancel(root, record)


def __getattr__(name):
    if name in ('worker_threads', 'accounting_details'):
        from .scheduler_api import scheduler_object
        return scheduler_object('slurm', 'execution.py:' + name)
    raise AttributeError(name)
