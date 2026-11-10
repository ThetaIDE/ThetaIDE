"""Small durable JSON records. Credentials are owned by OpenSSH, never by Theta."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except FileNotFoundError:
        return {} if default is None else default


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def workspace_id(root):
    path = Path(root) / '.thetaide/workspace.json'
    if not path.exists():
        # Exclusive creation also handles two GUI panels initializing together.
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open('x') as stream:
                json.dump({'id': uuid.uuid4().hex}, stream)
        except FileExistsError:
            pass
    value = read_json(path)['id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
        raise ValueError('Invalid persistent workspace ID')
    return value


def safe_path(root, relative):
    """Do not follow destination symlinks or accept paths outside the project."""
    if (not isinstance(relative, str) or not relative or '\\' in relative or '\x00' in relative
            or '\n' in relative or '\r' in relative or Path(relative).is_absolute()
            or any(part in ('', '.', '..') for part in relative.split('/'))):
        raise ValueError(f'Invalid project-relative path: {relative!r}')
    root = Path(root).resolve()
    path = root / relative
    if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root and parent.is_relative_to(root)):
        raise ValueError(f'Symlinks cannot be transferred: {relative}')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Path escapes project')
    return path


@contextmanager
def transfer_lock(root):
    """An OS lock releases automatically after a crash; the file is intentionally retained."""
    import fcntl
    path = Path(root) / '.thetaide/site-transfer.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another project transfer is in progress') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def record_lock(path):
    """Serialize receipt updates from the submitter, batch worker and status helper."""
    import fcntl
    lock = Path(path).with_suffix('.lock')
    with lock.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
