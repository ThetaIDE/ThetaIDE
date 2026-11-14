"""Explicit study continuation; never model/optimizer checkpoint resume."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .storage import digest, read_json, safe_path, write_json


def compatibility(config, *, paths=None):
    # Conservative identity: allow only attempt/schedule/display changes. Provider
    # source identity is pinned independently. Objective/search-space remain exact.
    config = json.loads(json.dumps(config))
    for key in ('hydra', 'site', 'resources', 'experiment_id', 'experiment_name', 'group',
                'study_continuation', 'dry_run', 'plot_only'):
        config.pop(key, None)
    if config.get('tuning') is not None:
        from src.app.pipeline.tuners import tuner_name
        from src.app.pipeline.tuner_execution import TunerExecutionPlan, remote_tuner
        tuner = remote_tuner(TunerExecutionPlan(tuner_name(config), '0' * 32), paths=paths)
        config['tuning'] = tuner.continuation_config(config['tuning'])
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
    from src.app.pipeline.tuner_execution import record_plan, record_config
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(root)
    prior_plan, next_plan = record_plan(previous, paths=paths), record_plan(record, paths=paths)
    if not prior_plan or not next_plan or prior_plan.component != next_plan.component:
        raise ValueError('Continuation requires the same portable tuner provider')
    artifacts = {item.path: item for item in prior_plan.artifacts if item.continuation}
    matches = [item for item in publication.get('files', []) if item['path'] in artifacts and item['sha256'] == request['export_sha256']]
    if len(matches) != 1: raise ValueError('Selected verified database export is missing')
    if not next_plan.artifact(artifacts[matches[0]['path']].id).continuation:
        raise ValueError('Selected tuner artifact cannot be continued by this execution')
    if compatibility(record_config(record), paths=paths) != compatibility(record_config(previous), paths=paths):
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
            'artifact_id': artifacts[matches[0]['path']].id, 'export': matches[0], 'work_root': str(work), 'study_name': request['study_name'],
            'parent_result_experiment_id': previous['result_experiment_id'],
            'dependencies': [item for item in publication.get('files', []) if item['category'] == 'checkpoints'],
            'sampler_restart': next_plan.continuation_notes,
            'training_resume': False}


def restore(runtime, record):
    lineage = record['continuation']
    from .publication import dependency_versions
    from src.app.pipeline.tuner_execution import record_plan, remote_tuner
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(runtime)
    plan = record_plan(record, paths=paths)
    if plan is None:
        raise ValueError('Continuation requires a tuner execution plan')
    if lineage.get('dependencies_versions') != dependency_versions(record, paths=paths):
        raise ValueError('Continuation runtime dependency versions differ; explicit migration is required')
    dependency_bytes = lineage['export']['size'] + sum(item['size'] for item in lineage['dependencies'])
    if dependency_bytes > record['execution_policy']['max_input_bytes']:
        raise ValueError('Continuation state/checkpoint dependencies exceed max_input_bytes')
    artifact = plan.artifact(lineage.get('artifact_id', next((item.id for item in plan.artifacts if item.continuation), '')))
    if not artifact.continuation:
        raise ValueError('Tuner artifact does not support continuation')
    target = safe_path(runtime, artifact.path)
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
    remote_tuner(plan, paths=paths).restore_continuation(record['config'], plan=plan,
        runtime=Path(runtime), lineage=lineage)
    write_json(Path(runtime) / 'results/slurm' / record['execution_id'] / 'lineage.json', lineage)
