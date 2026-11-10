"""Declarative site storage policy and bounded input staging (no shell expansion)."""
from __future__ import annotations

import os
import pwd
import shutil
import string
from pathlib import Path

from .storage import safe_path, workspace_id, write_json

DEFAULTS = {
    'schema_version': 1, 'runtime': 'scratch', 'metrics_seconds': 60,
    'database_seconds': 300, 'database_min_seconds': 30, 'stage_out_seconds': 120,
    'chunk_bytes': 8 * 1024 * 1024, 'tick_bytes': 32 * 1024 * 1024,
    'max_pending_bytes': 2 * 1024**3, 'max_queue_files': 128, 'max_input_bytes': 2 * 1024**3, 'max_input_files': 4096,
    'scratch_reserve_bytes': 256 * 1024**2, 'max_runtime_bytes': 8 * 1024**3,
    'max_runtime_files': 100000, 'max_lag_seconds': 600,
    'max_published_bytes': 16 * 1024**3, 'max_source_bytes': 2 * 1024**3,
    'diagnostic_bytes': 128 * 1024**2, 'diagnostic_files': 1024,
    'snapshot_bytes': 8 * 1024**2, 'stream_flush_seconds': 5,
    'system_interval': 10, 'inputs': [], 'diagnostic_strict': False,
}


def expand(value, root):
    fields = {'user': pwd.getpwuid(os.getuid()).pw_name, 'uid': str(os.getuid()),
              'project_id': workspace_id(root), 'control_root': str(Path(root).resolve())}
    for _, name, spec, conversion in string.Formatter().parse(str(value)):
        if name and (name not in fields or spec or conversion):
            raise ValueError(f'Unsupported path placeholder: {name}')
    path = Path(str(value).format(**fields))
    if not path.is_absolute() or '..' in path.parts or path == Path('/'):
        raise ValueError('Site work/scratch roots must be absolute non-root paths')
    return str(path)


def resolve(cfg, root):
    raw = (cfg.get('site') or {}).get('execution')
    if not raw: return None
    if raw.get('schema_version') != 1 or raw.get('runtime') != 'scratch':
        raise ValueError('Unsupported site execution policy')
    unknown = set(raw) - set(DEFAULTS) - {'work_root', 'scratch_root'}
    if unknown: raise ValueError('Unknown execution policy fields: ' + ', '.join(sorted(unknown)))
    from omegaconf import OmegaConf
    values = OmegaConf.to_container(raw, resolve=True) if OmegaConf.is_config(raw) else dict(raw)
    policy = {**DEFAULTS, **values}
    for key in DEFAULTS:
        if key in ('runtime', 'inputs', 'diagnostic_strict'): continue
        integer = key.endswith(('_bytes', '_files')) or key == 'schema_version'
        if (type(policy[key]) is not int if integer else type(policy[key]) not in (int, float)) or policy[key] <= 0:
            raise ValueError(f'execution.{key} must be positive')
    if type(policy['diagnostic_strict']) is not bool: raise ValueError('diagnostic_strict must be boolean')
    if policy['tick_bytes'] < policy['chunk_bytes']:
        raise ValueError('tick_bytes must fit at least one chunk')
    policy['work_root'] = expand(raw['work_root'], root)
    policy['scratch_root'] = expand(raw['scratch_root'], root)
    control, work, scratch = map(Path, (root, policy['work_root'], policy['scratch_root']))
    if work == control or scratch == control or work == scratch or work.is_relative_to(scratch) or scratch.is_relative_to(work):
        raise ValueError('Control, shared work, and scratch roots must be distinct')
    return policy


def stage_inputs(root, destination, policy):
    """Copy only declared files/trees; fresh mtimes respect shared-storage purge policy."""
    files, total = [], 0
    for mapping in policy['inputs']:
        if not str(mapping['target']).startswith(('in/datasets/', 'in/inputs/')):
            raise ValueError('Declared inputs must target in/datasets/ or in/inputs/')
        source = safe_path(root, mapping['source'])
        target = safe_path(destination, mapping['target'])
        if not source.exists(): raise ValueError(f'Missing declared input: {source}')
        paths = [source] if source.is_file() else sorted(source.rglob('*'))
        for item in paths:
            if item.is_symlink(): raise ValueError('Input symlinks are not supported; declare a copy')
            if not item.is_file(): continue
            output = target if source.is_file() else target / item.relative_to(source)
            if output.exists(): raise ValueError(f'Input destination collision: {output}')
            total += item.stat().st_size
            if total > policy['max_input_bytes']: raise ValueError('Declared inputs exceed max_input_bytes')
            files.append((item, output))
            if len(files) > policy['max_input_files']: raise ValueError('Declared inputs exceed max_input_files')
    if shutil.disk_usage(destination.parent).free < total + policy['scratch_reserve_bytes']:
        raise ValueError('Insufficient free storage for declared inputs')
    metadata = []
    for source, target in files:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        metadata.append({'path': target.relative_to(destination).as_posix(), 'original_mtime_ns': source.stat().st_mtime_ns})
    write_json(destination / '.theta-inputs.json', {'files': metadata, 'bytes': total})


def assert_local_scratch(path):
    path = Path(path)
    if not path.is_dir() or not os.access(path, os.W_OK):
        raise ValueError('Required job-local scratch is unavailable; no shared-storage fallback')
    mounts = Path('/proc/mounts')
    if mounts.exists():
        matches = []
        for line in mounts.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 3 and path.resolve().is_relative_to(fields[1]):
                matches.append((len(fields[1]), fields[2]))
        if matches and max(matches)[1] in {'nfs', 'nfs4', 'cifs', 'smbfs', 'lustre', 'gpfs'}:
            raise ValueError('Scratch must be node-local; network filesystems are unsupported for live SQLite')


def own_work_root(root, project):
    directory = Path(root)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / '.theta-project.json'
    try:
        with marker.open('x') as stream:
            __import__('json').dump({'project_id': project}, stream)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        from .storage import read_json
        if read_json(marker).get('project_id') != project:
            raise ValueError('Shared work root belongs to a different project')
