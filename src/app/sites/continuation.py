"""Explicit study continuation; never model/optimizer checkpoint resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .storage import digest, read_json, safe_path, write_json


def compatibility(config):
    # Conservative identity: allow only attempt/schedule/display changes. Provider
    # source identity is pinned independently. Objective/search-space remain exact.
    config = json.loads(json.dumps(config))
    for key in ('hydra', 'site', 'resources', 'experiment_id', 'experiment_name', 'group',
                'study_continuation', 'dry_run', 'plot_only'):
        config.pop(key, None)
    options = ((config.get('tuning') or {}).get('options') or {}).get('optuna') or {}
    for key in ('n_trials', 'n_jobs', 'storage', 'study_name', 'study_mode', 'timeout_seconds'):
        options.pop(key, None)
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def pin(root, record, request):
    """Under the project mutation lock, pin parent and exclusive lineage owner."""
    from .execution import receipt_path, refresh_status, TERMINAL
    previous = refresh_status(root, read_json(receipt_path(root, request['execution_id'])))
    if previous.get('scheduler_state') not in TERMINAL:
        raise ValueError('Continuation requires confirmed prior scheduler termination')
    if not previous.get('execution_policy') or previous.get('remote_cleaned_at'):
        raise ValueError('Prior execution has no retained managed publication')
    work = Path(previous['execution_policy']['work_root'])
    publication = read_json(work / 'results/slurm' / previous['execution_id'] / 'publication.json')
    matches = [item for item in publication.get('files', []) if item['category'] == 'database' and item['sha256'] == request['export_sha256']]
    if len(matches) != 1: raise ValueError('Selected verified database export is missing')
    if compatibility(record['config']) != compatibility(previous['config']):
        raise ValueError('Continuation objective/search-space/provider configuration is incompatible')
    if record['source_sha256'] != previous['source_sha256']:
        raise ValueError('Continuation requires identical source/dependency pins; migration is explicit')
    lineage = previous.get('continuation', {}).get('lineage_id', previous['execution_id'])
    for path in (Path(root) / 'results/slurm').glob('*/submission.json'):
        other = read_json(path)
        if other['execution_id'] == record['execution_id']: continue
        if other.get('continuation', {}).get('lineage_id') == lineage:
            other = refresh_status(root, other)
            if other.get('scheduler_state') not in TERMINAL and other.get('state') != 'submission_failed':
                raise ValueError('Study lineage already has an active/unknown continuation owner')
            # Refuse branching from an obsolete parent after another owner finished.
            if other['execution_id'] != previous['execution_id'] and other.get('state') != 'submission_failed':
                raise ValueError('Continue the latest lineage execution; branching/duplicate ownership is unsupported')
    return {'lineage_id': lineage, 'parent_execution_id': previous['execution_id'],
            'dependencies_versions': publication.get('dependencies'),
            'export': matches[0], 'work_root': str(work), 'study_name': request['study_name'],
            'parent_result_experiment_id': previous['result_experiment_id'],
            'dependencies': [item for item in publication.get('files', []) if item['category'] == 'checkpoints'],
            'sampler_restart': 'reinitialized from configured seed; suggestion sequence can differ',
            'training_resume': False}


def restore(runtime, record):
    lineage = record['continuation']
    from .publication import dependency_versions
    if lineage.get('dependencies_versions') != dependency_versions():
        raise ValueError('Continuation runtime dependency versions differ; explicit migration is required')
    target = Path(runtime) / 'results/optuna' / record['execution_id'] / 'study.sqlite'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.restoring')
    try:
        with temporary.open('xb') as output:
            for chunk in lineage['export']['chunks']:
                source = safe_path(lineage['work_root'], chunk['source'])
                if digest(source) != chunk['sha256']: raise ValueError('Continuation object checksum mismatch')
                with source.open('rb') as stream:
                    __import__('shutil').copyfileobj(stream, output, 1024 * 1024)
        if digest(temporary) != lineage['export']['sha256']: raise ValueError('Continuation export checksum mismatch')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    dependency_bytes = sum(item['size'] for item in lineage['dependencies'])
    if dependency_bytes > record['execution_policy']['max_input_bytes']:
        raise ValueError('Continuation checkpoint dependencies exceed max_input_bytes')
    for item in lineage['dependencies']:
        relative = item['path'].replace('/' + lineage['parent_result_experiment_id'] + '/', '/' + record['result_experiment_id'] + '/', 1)
        destination = safe_path(runtime, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as output:
            for chunk in item['chunks']:
                source = safe_path(lineage['work_root'], chunk['source'])
                if digest(source) != chunk['sha256']: raise ValueError('Checkpoint dependency checksum mismatch')
                with source.open('rb') as stream: __import__('shutil').copyfileobj(stream, output, 1024 * 1024)
        if digest(destination) != item['sha256']: raise ValueError('Checkpoint dependency checksum mismatch')
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(target)) as db:
        if db.execute('PRAGMA quick_check').fetchone() != ('ok',): raise ValueError('Invalid continuation database')
    import optuna
    storage = optuna.storages.RDBStorage('sqlite:///' + str(target))
    try:
        study = optuna.load_study(storage=storage, study_name=lineage['study_name'])
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.RUNNING:
                study._storage.set_trial_user_attr(trial._trial_id, 'theta_interruption',
                    {'parent_execution_id': lineage['parent_execution_id'], 'policy': 'fail; retry uses new trial number'})
                study.tell(trial.number, state=optuna.trial.TrialState.FAIL)
        study.set_user_attr('theta_continuation', {key: lineage[key] for key in ('parent_execution_id', 'sampler_restart', 'training_resume')})
    finally:
        storage.engine.dispose()
    write_json(Path(runtime) / 'results/slurm' / record['execution_id'] / 'lineage.json', lineage)
