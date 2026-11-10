"""Freeze portable source files; datasets and Python environments have separate lifecycles."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .storage import digest, safe_path, workspace_id, write_json

SOURCE_ROOTS = ('src', 'frontend', 'hydra_plugins', 'plot', 'in/config')
SOURCE_FILES = ('run_pipeline.py', 'pyproject.toml', 'setup.py', 'requirements.txt', 'README.md', 'LICENSE.md', 'MANIFEST.in',
                '.thetaide/provider_overrides.json', '.thetaide/workspace.json')
EXCLUDED = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache',
            '.DS_Store', '.env', '.aws', '.ssh'}


def source_files(root):
    root = Path(root)
    files = set()
    for name in SOURCE_FILES:
        path = safe_path(root, name)
        if path.is_file(): files.add(name)
    for name in SOURCE_ROOTS:
        directory = safe_path(root, name)
        if not directory.is_dir(): continue
        for path in directory.rglob('*'):
            relative = path.relative_to(root)
            if any(part in EXCLUDED for part in relative.parts): continue
            if path.is_symlink():
                raise ValueError(f'Source symlink needs an explicit cluster copy: {relative}')
            if path.is_file() and path.suffix not in ('.pyc', '.pyo'):
                files.add(relative.as_posix())
    return sorted(files)


def freeze(root, destination):
    root, destination = Path(root).resolve(), Path(destination)
    project = workspace_id(root)
    files = source_files(root)
    hashes = {}
    for relative in files:
        source = safe_path(root, relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(target)
    value = hashlib.sha256(''.join(f'{key}\0{value}\n' for key, value in hashes.items()).encode()).hexdigest()
    result = {'project_id': project, 'sha256': value, 'files': hashes}
    write_json(destination / '.thetaide/source.json', result)
    return result
