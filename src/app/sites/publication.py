"""Compute-owned immutable publication. Login helpers only read its manifests.

Objects are bounded chunks; a manifest entry becomes visible only after all its
chunks are closed and copied. Scheduler termination is a separate contract.
"""
from __future__ import annotations

import hashlib
from contextlib import closing
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

from .storage import read_json, safe_path, write_json

SCHEMA = 'theta.publication/v1'
SMALL = {'.json', '.jsonl', '.csv', '.yaml', '.yml', '.log', '.out', '.err', '.sbatch'}


def committed(path, position=None):
    """Producer hook: call only after an atomic save/close, never before writing."""
    directory = os.environ.get('THETA_COMMIT_DIR')
    if not directory: return
    path = Path(path).absolute()
    root = Path(os.environ['THETA_OUTPUT_ROOT']).resolve()
    if not path.is_relative_to(root) or not path.is_file(): return
    info = path.stat()
    name = hashlib.sha256(str(path).encode()).hexdigest()
    write_json(Path(directory) / (name + '.json'), {'path': str(path), 'size': info.st_size,
                                                 'mtime_ns': info.st_mtime_ns, 'position': position})


def request_database_export(position=None):
    path = os.environ.get('THETA_DB_EXPORT_REQUEST')
    if path: write_json(path, {'requested_ns': time.time_ns(), 'position': position})


def admission():
    path = os.environ.get('THETA_PUBLICATION_STOP')
    if path and Path(path).exists():
        raise RuntimeError('Publication lag/storage budget exhausted; no new trials admitted')


def sqlite_export(source, target, *, seconds=20):
    """Bounded same-node backup, closed DELETE-mode snapshot, integrity checked."""
    deadline = time.monotonic() + seconds
    def progress(*_):
        if time.monotonic() > deadline: raise TimeoutError('SQLite backup deadline exceeded')
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with closing(sqlite3.connect(Path(source).as_uri() + '?mode=ro', uri=True, timeout=2)) as src:
            with closing(sqlite3.connect(target, timeout=2)) as dest:
                src.backup(dest, pages=128, progress=progress, sleep=0.01)
                dest.execute('PRAGMA journal_mode=DELETE')
                if dest.execute('PRAGMA quick_check').fetchone() != ('ok',):
                    raise ValueError('Exported SQLite integrity check failed')
    except BaseException:
        target.unlink(missing_ok=True)
        raise


class Publisher:
    def __init__(self, runtime, work, record, policy, *, clock=time.monotonic):
        self.runtime, self.work = Path(runtime), Path(work)
        self.record, self.policy, self.clock = record, policy, clock
        self.directory = self.work / 'results/slurm' / record['execution_id']
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / 'publication.json'
        self.manifest = {'schema': SCHEMA, 'execution_id': record['execution_id'],
                         'project_id': record['project_id'], 'generation': 0, 'sealed': False,
                         'files': [], 'durable': {}, 'dependencies': dependency_versions()}
        self.entries, self.stamps, self.pending = {}, {}, {}
        self.append_prefixes = {}
        self.last_metrics = self.last_database = -float('inf')
        self.last_request = None
        self.finalizing = self.final_database = False
        self.deferred = 0
        self.last_progress = self.clock()
        self.stop_path = self.runtime / '.thetaide/publication-stop'
        self.object_bytes = 0
        self.budget_stamp = None
        self.last_gc = -float('inf')
        self.counters = {'source_bytes_read': 0, 'object_bytes_written': 0, 'objects_written': 0}

    def _start(self, path, relative, category, position=None):
        try: info = path.stat()
        except FileNotFoundError: return False
        stamp = (info.st_size, info.st_mtime_ns)
        if self.stamps.get(relative) == stamp or relative in self.pending: return False
        if len(self.pending) >= self.policy['max_queue_files']:
            priority = {'database': 0, 'metrics': 1, 'checkpoints': 2, 'diagnostics': 3}
            worst = max(self.pending, key=lambda key: priority[self.pending[key]['category']])
            if priority[category] < priority[self.pending[worst]['category']]:
                self.pending.pop(worst)['stream'].close()
            else:
                self.deferred += 1
                return False
        limit = stamp[0]
        if path.suffix in ('.jsonl', '.csv') and limit:
            with path.open('rb') as source:
                source.seek(max(0, limit - 1024 * 1024))
                tail = source.read(min(limit, 1024 * 1024))
            if not tail.endswith(b'\n'):
                last = tail.rfind(b'\n')
                limit = max(0, stamp[0] - len(tail) + last + 1) if last >= 0 else 0
        self.pending[relative] = {'path': path, 'stamp': stamp, 'limit': limit, 'stream': path.open('rb'),
                                  'chunks': [], 'hash': hashlib.sha256(), 'size': 0,
                                  'category': category, 'position': position}
        item = self.pending[relative]
        prefix = self.append_prefixes.get(relative)
        if prefix and limit > prefix['source_size'] and path.stat().st_ino == prefix['inode']:
            # Verify a bounded head before reusing an append-only prefix. CSV
            # header/schema rewrites invalidate it and trigger a fresh snapshot.
            first = prefix['chunks'][0]
            head = item['stream'].read(first['size'])
            self.counters['source_bytes_read'] += len(head)
            if hashlib.sha256(head).hexdigest() == first['sha256']:
                item.update(chunks=list(prefix['chunks']), hash=prefix['hash'].copy(), size=prefix['length'], prefix=prefix)
            item['stream'].seek(item['size'])
        return True

    def _object(self, data):
        sha = hashlib.sha256(data).hexdigest()
        relative = f'results/slurm/{self.record["execution_id"]}/objects/{sha}'
        target = safe_path(self.work, relative)
        if not target.exists():
            if self.object_bytes + len(data) > self.policy['max_published_bytes']:
                self.stop_path.touch()
                raise RuntimeError('Publication object store exceeds max_published_bytes; active leases may pin obsolete objects')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name('.' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('xb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
            self.object_bytes += len(data)
            self.counters['object_bytes_written'] += len(data)
            self.counters['objects_written'] += 1
        return {'source': relative, 'sha256': sha, 'size': len(data)}

    def discover(self, final=False):
        from .execution import owned_files
        now = self.clock()
        self.deferred = 0
        final = final or self.finalizing
        budget = self.runtime / '.thetaide/diagnostic-budget.json'
        if budget.exists() and budget.stat().st_mtime_ns != self.budget_stamp:
            destination = self.runtime / 'results/slurm' / self.record['execution_id'] / 'diagnostic-budget.json'
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(budget, destination)
            self.budget_stamp = budget.stat().st_mtime_ns
        paths = list(owned_files(self.runtime, self.record))
        stable = []
        total = 0
        for path in paths:
            try: total += path.stat().st_size
            except FileNotFoundError: continue  # WAL/SHM and producer temporary files can disappear
            stable.append(path)
        paths = stable
        if total > self.policy['max_runtime_bytes'] or len(paths) > self.policy['max_runtime_files']:
            self.stop_path.touch()
            raise RuntimeError('Execution storage budget exceeded')
        due_metrics = final or now - self.last_metrics >= self.policy['metrics_seconds']
        commits = {}
        for marker in (self.runtime / '.thetaide/commits').glob('*.json'):
            item = read_json(marker)
            commits[item['path']] = item
        paths.sort(key=lambda path: (2 if '/monitors/' in str(path) else 1 if 'checkpoints' in path.parts else 0, str(path)))
        for path in paths:
            relative = path.relative_to(self.runtime).as_posix()
            if path.name.startswith('.') or path.name.endswith(('-wal', '-shm', '-journal')): continue
            if path.suffix in ('.sqlite', '.db', '.sqlite3'): continue
            category = 'checkpoints' if relative.startswith('results/checkpoints/') else ('diagnostics' if '/monitors/' in relative else 'metrics')
            small = path.suffix in SMALL and category != 'checkpoints'
            try: info = path.stat()
            except FileNotFoundError: continue
            stamp = (info.st_size, info.st_mtime_ns)
            marker = commits.get(str(path.absolute()), {})
            commit = (marker.get('size'), marker.get('mtime_ns')) == stamp
            # Snapshot metadata is committed after its NPZ; neither is read halfway.
            if '/monitors/' in relative and path.suffix == '.npz':
                meta = path.with_suffix('.json')
                meta_marker = commits.get(str(meta.absolute()), {})
                commit = commit or (meta.exists() and (meta_marker.get('size'), meta_marker.get('mtime_ns')) == (meta.stat().st_size, meta.stat().st_mtime_ns))
            if final or (small and due_metrics) or commit:
                self._start(path, relative, category, marker.get('position'))
        if due_metrics: self.last_metrics = now
        request = read_json(self.runtime / '.thetaide/db-export-request.json')
        requested = request.get('requested_ns') != self.last_request and bool(request)
        database = self.runtime / 'results/optuna' / self.record['execution_id'] / 'study.sqlite'
        due_db = (final and not self.final_database) or (not final and (now - self.last_database >= self.policy['database_seconds'] or (requested and now - self.last_database >= self.policy['database_min_seconds'])))
        relative = f'results/optuna/{self.record["execution_id"]}/study.sqlite'
        if due_db and database.exists() and relative not in self.pending:
            target = self.runtime / '.thetaide' / ('export-' + uuid.uuid4().hex + '.sqlite')
            sqlite_export(database.resolve(), target)
            self._start(target, relative, 'database', request.get('position'))
            if final: self.final_database = True
            self.last_database, self.last_request = now, request.get('requested_ns')
        pending_bytes = sum(max(0, item['stamp'][0] - item['size']) for item in self.pending.values())
        if pending_bytes > self.policy['max_pending_bytes'] or (self.pending and now - self.last_progress > self.policy['max_lag_seconds']):
            self.stop_path.touch()
            raise RuntimeError('Publication backlog exceeds configured limit')

    def tick(self, *, final=False):
        # One bounded publisher turn across the project; all SQLite work stays on
        # the calling allocation's compute node, even while this lock is shared.
        from .storage import record_lock
        control = self.record.get('control_root')
        if control:
            with record_lock(Path(control) / '.thetaide/publication'):
                return self._tick(final=final)
        return self._tick(final=final)

    def _tick(self, *, final=False):
        self.discover(final)
        remaining = self.policy['tick_bytes']
        # Every tick returns to status/database/checkpoints before diagnostic chunks.
        priority = {'database': 0, 'metrics': 1, 'checkpoints': 2, 'diagnostics': 3}
        for relative in sorted(list(self.pending), key=lambda key: priority[self.pending[key]['category']]):
            if self.pending[relative]['category'] == 'diagnostics' and self.pending[relative]['limit'] <= self.policy['chunk_bytes']: continue
            item = self.pending[relative]
            while remaining > 0:
                amount = min(self.policy['chunk_bytes'], remaining, item['limit'] - item['size'])
                data = item['stream'].read(amount)
                self.counters['source_bytes_read'] += len(data)
                remaining -= len(data)
                if data:
                    item['hash'].update(data)
                    item['size'] += len(data)
                    item['chunks'].append(self._object(data))
                    if item['path'].suffix in ('.jsonl', '.csv', '.log', '.out', '.err') and item['size'] % self.policy['chunk_bytes'] == 0:
                        item['prefix'] = {'chunks': list(item['chunks']), 'hash': item['hash'].copy(),
                                          'length': item['size'], 'inode': item['path'].stat().st_ino,
                                          'source_size': item['stamp'][0]}
                    self.last_progress = self.clock()
                if not data:
                    item['stream'].close()
                    try:
                        info = item['path'].stat()
                        current = (info.st_size, info.st_mtime_ns)
                    except FileNotFoundError: current = None
                    append_only = item['path'].suffix in ('.jsonl', '.csv', '.log', '.out', '.err')
                    if current != item['stamp'] and not (append_only and current and current[0] >= item['stamp'][0]):
                        del self.pending[relative]  # changed while reading; no manifest commit
                        break
                    entry = {'path': relative, 'size': item['size'], 'sha256': item['hash'].hexdigest(),
                             'chunks': item['chunks'], 'category': item['category']}
                    self.entries[relative], self.stamps[relative] = entry, item['stamp']
                    if item.get('prefix'): self.append_prefixes[relative] = item['prefix']
                    self.manifest['durable'][item['category']] = {'at': time.time(), 'path': relative,
                                                                'sha256': entry['sha256'], 'position': item['position']}
                    if item['category'] == 'database': item['path'].unlink()
                    del self.pending[relative]
                    break
            if remaining <= 0: break
        # Pack small, newly committed diagnostics into one bounded raw bundle.
        # Explicit ranges preserve the local file layout without archive extraction.
        bundled, data, used = [], [], 0
        for relative, item in list(self.pending.items()):
            size = item['limit']
            if item['category'] != 'diagnostics' or size > self.policy['chunk_bytes'] or item['size']: continue
            if used + size > min(remaining, self.policy['chunk_bytes']): continue
            payload = item['stream'].read(size)
            item['stream'].close()
            del self.pending[relative]
            try: info = item['path'].stat()
            except FileNotFoundError: continue
            if (info.st_size, info.st_mtime_ns) != item['stamp']: continue
            self.counters['source_bytes_read'] += len(payload)
            bundled.append((relative, item, used, payload))
            data.append(payload)
            used += len(payload)
        if bundled:
            chunk = self._object(b''.join(data))
            for relative, item, offset, payload in bundled:
                self.entries[relative] = {'path': relative, 'size': len(payload),
                    'sha256': hashlib.sha256(payload).hexdigest(), 'category': 'diagnostics',
                    'chunks': [{**chunk, 'offset': offset, 'length': len(payload)}]}
                self.stamps[relative] = item['stamp']
            self.last_progress = self.clock()
        value = {**self.manifest, 'files': list(self.entries.values()),
                 'counters': dict(self.counters), 'pending': len(self.pending)}
        if value != self.manifest or not self.manifest_path.exists():
            value['generation'] += 1
            self.manifest = value
            write_json(self.manifest_path, self.manifest)
        self.collect_obsolete()

    def collect_obsolete(self):
        control = self.record.get('control_root')
        if not control or self.clock() - self.last_gc < 60: return
        from .storage import transfer_lock
        from .maintenance import active_leases
        try:
            with transfer_lock(control):
                if active_leases(control): return
                referenced = {chunk['source'] for item in self.entries.values() for chunk in item['chunks']}
                referenced.update(chunk['source'] for item in self.pending.values() for chunk in item['chunks'])
                removed = 0
                for path in (self.directory / 'objects').glob('*'):
                    if path.relative_to(self.work).as_posix() not in referenced and not path.is_symlink():
                        self.object_bytes -= path.stat().st_size
                        path.unlink()
                        removed += 1
                        if removed >= 100: break
                self.last_gc = self.clock()
        except RuntimeError:
            pass  # an active mutation/transfer pins the old generation

    def finish(self, *, seconds):
        deadline = self.clock() + seconds
        # Discover once after all producers have closed, then drain bounded chunks.
        self.finalizing = True
        # A periodic backup already queued may precede the last completed trial.
        # Supersede it with a fresh post-close backup before sealing the payload.
        for relative, item in list(self.pending.items()):
            if item['category'] == 'database':
                item['stream'].close()
                item['path'].unlink(missing_ok=True)
                del self.pending[relative]
        self.final_database = False
        self.tick(final=True)
        while self.pending or self.deferred:
            if self.clock() > deadline: raise TimeoutError('Final stage-out allowance exceeded')
            self.tick()  # DB is already coalesced; immutable files are skipped by stamp
        self.manifest.update(sealed=True, sealed_at=time.time())
        write_json(self.manifest_path, self.manifest)

    def close(self):
        for item in self.pending.values(): item['stream'].close()


def dependency_versions():
    from importlib.metadata import version, PackageNotFoundError
    import platform
    versions = {'python': platform.python_version()}
    for name in ('optuna', 'sqlalchemy', 'torch', 'numpy'):
        try: versions[name] = version(name)
        except PackageNotFoundError: versions[name] = None
    return versions
