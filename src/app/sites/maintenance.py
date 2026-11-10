"""Ownership/reference based maintenance; legacy files are inventory-only."""
import re
import shutil
import time
from pathlib import Path

from .storage import read_json, transfer_lock, workspace_id, safe_path


def inventory(root):
    root = Path(root)
    return {'legacy': [str(path.relative_to(root)) for pattern in ('results/slurm/*/workspace', 'results/slurm/.sync/*', '.thetaide/uploads/*')
                       for path in root.glob(pattern) if not (path / '.theta-managed.json').is_file()],
            'note': 'Unclassified legacy paths are preserved. Export live legacy studies on their owning compute node; do not open NFS WAL from login.'}


def collect(root, *, apply=False, temporary_only=False):
    """Only expired owned leases and unreferenced managed archives are candidates."""
    root = Path(root)
    with transfer_lock(root):
        project = workspace_id(root)
        records = [read_json(path) for path in (root / 'results/slurm').glob('*/submission.json')]
        pinned = {record.get('source_sha256') for record in records if not record.get('remote_cleaned_at') and record.get('state') != 'submission_failed'}
        pinned.add(read_json(root / '.thetaide/source.json').get('sha256'))
        candidates = []
        for directory in (list((root / 'results/slurm/.sync').glob('*')) + list((root / '.thetaide/uploads').glob('*')))[:100]:
            try: safe_path(root, directory.relative_to(root).as_posix())
            except ValueError: continue
            lease = read_json(directory / '.theta-managed.json')
            if lease.get('project_id') == project and lease.get('expires_at', float('inf')) < time.time():
                candidates.append(directory)
        store_owned = read_json(root / '.thetaide/sources/ownership.json').get('project_id') == project
        for archive in ([] if temporary_only or not store_owned else (root / '.thetaide/sources').glob('*.zip')):
            if not archive.is_symlink() and re.fullmatch(r'[a-f0-9]{64}', archive.stem) and archive.stem not in pinned:
                # Only the archive store is managed; no generic source-tree collection.
                candidates.append(archive)
        result = []
        for path in candidates:
            size = path.stat().st_size if path.is_file() else sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())
            result.append({'path': str(path.relative_to(root)), 'bytes': size})
            if apply:
                if path.is_file(): path.unlink()
                elif not path.is_symlink(): shutil.rmtree(path)
        return {'apply': apply, 'candidates': result, **inventory(root)}


def active_leases(root):
    return [path for path in (Path(root) / 'results/slurm/.sync').glob('*/.theta-managed.json')
            if read_json(path).get('expires_at', 0) >= time.time()]
