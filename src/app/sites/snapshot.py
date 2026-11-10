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


def archive(root, destination):
    """Build locally during project push; archives contain regular source files only."""
    import zipfile
    with __import__('tempfile').TemporaryDirectory(prefix='theta-source-') as directory:
        source = freeze(root, directory)
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in [*source['files'], '.thetaide/source.json']:
                bundle.write(Path(directory) / name, name)
    return source


def extract(archive_path, destination, *, max_bytes=256 * 1024**2):
    """Validate the entire archive before writing. Never extract links or traversal."""
    import stat
    import zipfile
    with zipfile.ZipFile(archive_path) as bundle:
        seen, total = set(), 0
        for item in bundle.infolist():
            safe_path(destination, item.filename)
            mode = item.external_attr >> 16
            if item.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError('Source archives accept regular files only')
            if item.filename in seen: raise ValueError('Duplicate archive member')
            seen.add(item.filename)
            total += item.file_size
            if total > max_bytes: raise ValueError('Source archive exceeds size limit')
        for item in bundle.infolist():
            target = safe_path(destination, item.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    from .storage import read_json
    source = read_json(Path(destination) / '.thetaide/source.json')
    if set(seen) != set(source['files']) | {'.thetaide/source.json'}:
        raise ValueError('Source archive manifest mismatch')
    for name, expected in source['files'].items():
        if digest(safe_path(destination, name)) != expected:
            raise ValueError('Source archive checksum mismatch')
    return source
