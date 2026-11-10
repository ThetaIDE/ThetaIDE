"""Conservative local admission and process-tree memory accounting.

Reservations are estimates, not OS isolation. Capacity controls new starts;
admitted workers run to completion without scheduler-driven restarts.
"""
from __future__ import annotations

import math
import os
import json
import hashlib
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager, closing

import psutil

GIB = 1024 ** 3


@dataclass(frozen=True)
class ResourceBudget:
    cpu_percent: int = 75
    memory_percent: int = 75
    threads_per_job: int = 2
    memory_gb_per_job: float = .75  # internal cold-start fallback, never a user reservation
    max_parallel: int = 0  # zero lets resource budgets determine concurrency

    @classmethod
    def from_settings(cls, values):
        def number(key, default, low, high):
            try:
                value = float(values.get(key, default))
                return min(high, max(low, value)) if math.isfinite(value) else default
            except (TypeError, ValueError):
                return default
        return cls(int(number('cpu_percent', 75, 1, 100)),
                   int(number('memory_percent', 75, 10, 95)),
                   int(number('threads_per_job', 2, 1, 1024)),
                   .75,
                   int(number('max_parallel', 0, 0, 100000)))

    def cpu_slots(self, cores):
        return max(1, math.floor(cores * self.cpu_percent / 100))

    def threads(self, cores):
        return min(self.threads_per_job, self.cpu_slots(cores))

    def admission_reason(self, *, cores, total, available, active_memory, accelerator=False,
                         active_accelerators=False, active_threads=None, reservations=None,
                         next_memory=None, next_cpu=None):
        threads_used = sum(active_threads) if active_threads is not None else len(active_memory) * self.threads(cores)
        cpu_target = max(1, cores * min(100, self.cpu_percent + 5) / 100)
        if threads_used + (next_cpu if next_cpu is not None else self.threads(cores)) > cpu_target:
            return 'Waiting for CPU capacity within your training budget.'
        reservation = next_memory if next_memory is not None else self.memory_gb_per_job * GIB
        limit = total * min(95, self.memory_percent + 5) / 100
        # Admission accounts for Theta's own commitments. Host pressure is
        # telemetry, not a second hidden budget controlled by other apps.
        reserved = reservations if reservations is not None else [reservation] * len(active_memory)
        demand = sum(max(claim, rss) for claim, rss in zip(reserved, active_memory))
        if demand + reservation > limit:
            return 'Waiting for RAM: estimated demand exceeds your system usage target.'
        return ''


def host_capacity():
    memory = psutil.virtual_memory()
    cores = os.cpu_count() or 1
    try:
        cores = min(cores, len(psutil.Process().cpu_affinity()))
    except (AttributeError, psutil.Error):
        pass
    return cores, memory.total, memory.available


def tree_memory(pid):
    if not pid:
        return 0
    parent = psutil.Process(pid)
    processes = [parent, *parent.children(recursive=True)]
    total = 0
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
    return total


def managed_environment(threads):
    return {**{key: str(threads) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
             'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
            'THETA_MANAGED_THREADS': str(threads), 'THETA_MANAGED_SERIAL': '1'}


def configure_worker_threads():
    """Apply current workspace settings in each new worker, before ML imports."""
    if os.environ.get('THETA_RESOURCE_POOL'):
        budget = ResourceBudget.from_settings(load_training_settings(os.environ.get('THETAIDE_SETTINGS_FILE')))
        os.environ.update(managed_environment(budget.threads(host_capacity()[0])))


def workload_estimate(config):
    """A rough initial estimate based on engine, model size and local dataset size."""
    model = config.get('model') or {}
    model = model if hasattr(model, 'get') else {'name': str(model)}
    agent = config.get('agent') or {}
    agent = agent if hasattr(agent, 'get') else {'name': str(agent)}
    neural = model.get('neural') or {}
    layers = model.get('hidden_sizes') or (neural.get('hidden_sizes') if hasattr(neural, 'get') else None) or [64, 64]
    widths = [64, 64]
    try:
        widths = [int(x) for x in layers]
        inputs = int(model.get('input_dim', 128))
        params = sum(a * b + b for a, b in zip([inputs, *widths], widths))
    except (ValueError, TypeError):
        params = 65536
    dataset = config.get('dataset_path')
    dataset_bytes = 0
    if dataset:
        try:
            dataset_bytes = Path(str(dataset)).stat().st_size * 3
        except OSError:
            pass
    engine = str(config.get('entrypoint') or agent.get('algorithm') or agent.get('name') or '')
    replay = 0
    activations = 0
    try:
        # Replay storage depends on observations; use a rough default if the
        # component does not expose its observation width.
        replay = int(agent.get('buffer_size', 0)) * int(model.get('input_dim', 32)) * 8
        batch = int(agent.get('batch_size', config.get('batch_size', 64)))
        activations = batch * sum(widths) * 32
    except (ValueError, TypeError):
        pass
    variable_memory = params * 32 + dataset_bytes + replay + activations
    memory = (.55 if 'sb3' in engine else .8) * GIB + variable_memory
    identity = {'engine': engine, 'model': model, 'env': config.get('env'), 'dataset': dataset,
                'batch_size': agent.get('batch_size', config.get('batch_size')), 'buffer_size': agent.get('buffer_size')}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return {'key': key, 'memory': memory, 'minimum_memory': .1 * GIB + variable_memory, 'cpu': 1.0}


def uses_accelerator(config):
    if (config.get('trainer') or {}).get('accelerator') == 'cpu':
        return False
    methods = config.get('methods')
    if hasattr(methods, 'items'):
        enabled = [method for name, method in methods.items()
                   if name != 'params' and hasattr(method, 'get') and method.get('enabled', True)]
        if enabled:
            return any(uses_accelerator(method) for method in enabled)
    agent, model = config.get('agent') or {}, config.get('model') or {}
    algorithm = str(agent.get('algorithm') or agent.get('name') or '') if hasattr(agent, 'get') else str(agent)
    native_policy = model.get('sb3_native_policy', False) if hasattr(model, 'get') else False
    # Theta's SB3 adapter selects CPU inside the worker, without a user device request.
    if 'sb3' in algorithm and not native_policy:
        return False
    return True


def load_training_settings(path):
    import tomllib
    values = {}
    for candidate in [Path.home() / '.config/thetaide/settings.toml', Path(path) if path else None]:
        if candidate is None:
            continue
        try:
            with candidate.open('rb') as stream:
                section = tomllib.load(stream).get('training', {})
                if isinstance(section, dict):
                    values.update(section)
        except (OSError, ValueError):
            pass
    return values


def orchestration_workers(planned):
    """Coordinator threads derived from the admission estimator's CPU floor.

    This avoids allocating a thread for every item in a large seed plan;
    ResourcePool, rather than thread count, determines training concurrency.
    """
    cores, _total, _available = host_capacity()
    # Use physical capacity, so changing the soft target during a run cannot
    # leave a smaller executor acting as an accidental concurrency ceiling.
    cpu_budget = max(1, cores)
    return max(1, min(planned, math.ceil(cpu_budget / .25) + 1))


def learned_usage(path):
    """A lightweight read-only summary for the settings page."""
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=.1)) as db:
        count, memory, cpu = db.execute('SELECT COUNT(*), AVG(memory), AVG(cpu) FROM profiles').fetchone()
        active, used_memory, used_cpu = db.execute('SELECT COUNT(*), SUM(MAX(memory,rss)), SUM(cpu) FROM tasks').fetchone()
    return {'profiles': count, 'memory': (memory or .75 * GIB) * 1.15, 'cpu': max(.25, (cpu or 1.) * 1.1),
            'active': active, 'used_memory': used_memory or 0, 'used_cpu': used_cpu or 0}


class ResourceCapacityRetry(RuntimeError):
    """Legacy retry signal; built-in training launchers no longer emit it."""


class ResourcePool:
    """Shared local task estimates and leases, across experiments and trials.

    Workers share workspace CPU/RAM budgets and a configurable worker cap.
    """
    def __init__(self, path, settings_path=None):
        self.path = Path(path)
        self.settings_path = settings_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS profiles (key TEXT PRIMARY KEY, memory REAL, cpu REAL, samples INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS controls (key TEXT PRIMARY KEY, value INTEGER)')
            db.execute('CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, key TEXT, pid INTEGER, birth REAL, '
                       'memory REAL, cpu REAL, accelerator INTEGER, created REAL, rss REAL DEFAULT 0, '
                       'last_cpu REAL DEFAULT 0, last_sample REAL DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS waiters (id TEXT PRIMARY KEY, position TEXT, pid INTEGER, birth REAL)')
            db.execute('CREATE INDEX IF NOT EXISTS waiters_position ON waiters(position)')
            columns = {row[1] for row in db.execute('PRAGMA table_info(tasks)')}
            for column in ('peak', 'floor', 'admitted_memory', 'retry', 'reserved_cpu'):
                if column not in columns:
                    db.execute(f'ALTER TABLE tasks ADD COLUMN {column} REAL DEFAULT 0')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        try:
            with db:
                yield db
        finally:
            db.close()

    def estimate(self, spec):
        with self.connect() as db:
            row = db.execute('SELECT memory, cpu, samples FROM profiles WHERE key=?', (spec['key'],)).fetchone()
        if row:
            return max(spec.get('minimum_memory', .1 * GIB), row[0] * 1.15), max(.25, row[1] * 1.1), row[2]
        return spec['memory'], spec.get('cpu', 1.), 0

    def _sample(self, db):
        db.row_factory = sqlite3.Row
        tasks = db.execute('SELECT * FROM tasks').fetchall()
        alive = []
        now = time.time()
        for task in tasks:
            try:
                process = psutil.Process(task['pid'])
                if abs(process.create_time() - task['birth']) > .01 or process.status() == psutil.STATUS_ZOMBIE:
                    raise psutil.NoSuchProcess(task['pid'])
                members = [process, *process.children(recursive=True)]
                rss, seconds = 0, 0
                for member in members:
                    try:
                        rss += member.memory_info().rss
                        timing = member.cpu_times()
                        seconds += timing.user + timing.system
                    except psutil.NoSuchProcess:
                        pass
                elapsed = now - task['last_sample']
                measured = bool(task['last_sample'] and elapsed >= .5)
                cpu = max(0, seconds - task['last_cpu']) / elapsed if measured else task['cpu']
                estimate_cpu = max(task['reserved_cpu'], .25, task['cpu'] * .8 + cpu * .2) if measured else task['cpu']
                peak = max(task['peak'], rss)
                warmed = now - task['created'] >= 5
                # Hold the original reservation for the whole run. Quiet
                # startup must not lend its headroom to another worker.
                memory = max(task['memory'], task['admitted_memory'], task['floor'], peak * 1.15)
                # Learn after imports/warmup, rather than mistaking an empty
                # starting interpreter for the size of a training workload.
                if warmed:
                    db.execute('INSERT INTO profiles VALUES (?, ?, ?, 1) ON CONFLICT(key) DO UPDATE SET '
                               'memory=MAX(memory,excluded.memory), cpu=cpu*.8+excluded.cpu*.2, samples=samples+1',
                               (task['key'], peak, cpu))
                prime = not task['last_sample'] or measured
                db.execute('UPDATE tasks SET rss=?, last_cpu=?, last_sample=?, memory=?, '
                           'cpu=?,peak=? WHERE id=?',
                           (rss, seconds if prime else task['last_cpu'], now if prime else task['last_sample'],
                            memory, estimate_cpu, peak, task['id']))
                alive.append(dict(task, rss=rss, memory=memory, cpu=estimate_cpu))
            except psutil.NoSuchProcess:
                # Short runs may finish before warmup. Keep the peak already
                # observed so their next seed does not start from scratch.
                if task['peak'] > 0:
                    db.execute('INSERT INTO profiles VALUES (?, ?, ?, 1) ON CONFLICT(key) DO UPDATE SET '
                               'memory=MAX(memory,excluded.memory), cpu=cpu*.8+excluded.cpu*.2, samples=samples+1',
                               (task['key'], task['peak'], task['cpu']))
                db.execute('DELETE FROM tasks WHERE id=?', (task['id'],))
        return alive

    def enqueue(self, count=1, *, parent=None):
        """Reserve FIFO positions before starting coordinators or worker threads.

        Replacing a coordinator position with its children is atomic, so a fast
        later coordinator cannot jump ahead while an earlier one imports code.
        """
        pid = os.getpid()
        birth = psutil.Process(pid).create_time()
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT position FROM waiters WHERE id=?', (parent,)).fetchone() if parent else None
            if row:
                prefix = row[0] + '.'
            else:
                db.execute("INSERT INTO controls VALUES('queue_sequence',1) ON CONFLICT(key) DO UPDATE SET value=value+1")
                sequence = db.execute("SELECT value FROM controls WHERE key='queue_sequence'").fetchone()[0]
                prefix = f'{sequence:020d}.'
            tickets = [uuid.uuid4().hex for _ in range(count)]
            db.executemany('INSERT INTO waiters VALUES(?,?,?,?)',
                           [(ticket, prefix + f'{index:020d}', pid, birth) for index, ticket in enumerate(tickets)])
            if parent:
                db.execute('DELETE FROM waiters WHERE id=?', (parent,))
        return tickets

    def dequeue(self, ticket):
        with self.connect() as db:
            db.execute('DELETE FROM waiters WHERE id=?', (ticket,))

    def _first_waiter(self, db):
        for ticket, pid, birth in db.execute('SELECT id,pid,birth FROM waiters ORDER BY position').fetchall():
            try:
                process = psutil.Process(pid)
                if abs(process.create_time() - birth) < .01 and process.status() != psutil.STATUS_ZOMBIE:
                    return ticket
            except psutil.NoSuchProcess:
                pass
            db.execute('DELETE FROM waiters WHERE id=?', (ticket,))
        return None

    def acquire(self, spec, *, accelerator=False, reserve=True, capacity=None, ticket=None):
        settings = load_training_settings(self.settings_path)
        budget = ResourceBudget.from_settings(settings)
        cores, total, available = capacity or host_capacity()
        memory, cpu, samples = self.estimate(spec)
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            first = self._first_waiter(db)
            if ticket and first and first != ticket:
                return None, 'Waiting for earlier queued training jobs.'
            paused = db.execute("SELECT value FROM controls WHERE key='paused'").fetchone()
            if paused and paused[0]:
                return None, 'Training starts are paused.'
            active = self._sample(db)
            # Coordinators do not consume worker slots. All leaf workers share
            # this atomic cap, including methods, trials and training seeds.
            if spec.get('cpu', 1) > 0 and budget.max_parallel and len(active) >= budget.max_parallel:
                return None, f'Waiting for worker limit ({len(active)}/{budget.max_parallel} active).'
            cpu_reservation = min(max(cpu, budget.threads(cores)), budget.cpu_slots(cores))
            reason = budget.admission_reason(
                cores=cores, total=total, available=available,
                active_memory=[t['rss'] for t in active],
                reservations=[t['memory'] for t in active], active_threads=[t['cpu'] for t in active],
                next_memory=memory, next_cpu=cpu_reservation if spec.get('cpu', 1) > 0 else 0,
                accelerator=accelerator, active_accelerators=any(t['accelerator'] for t in active))
            if reason or not reserve:
                return None, reason
            token = uuid.uuid4().hex
            pid = os.getpid()
            db.execute('INSERT INTO tasks(id,key,pid,birth,memory,cpu,accelerator,created,floor,admitted_memory,reserved_cpu) VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                       (token, spec['key'], pid, psutil.Process(pid).create_time(), memory,
                        cpu_reservation, accelerator, time.time(), spec.get('minimum_memory', .1 * GIB), memory, cpu_reservation))
            if ticket:
                db.execute('DELETE FROM waiters WHERE id=?', (ticket,))
            return token, ''

    def attach(self, token, pid):
        try:
            birth = psutil.Process(pid).create_time()
        except psutil.NoSuchProcess:
            return  # a tiny task can finish before the launcher attaches it
        with self.connect() as db:
            db.execute('UPDATE tasks SET pid=?,birth=?,last_cpu=0,last_sample=0 WHERE id=?',
                       (pid, birth, token))

    def release(self, token):
        with self.connect() as db:
            self._sample(db)
            db.execute('DELETE FROM tasks WHERE id=?', (token,))

    def release_coordinator_reservation(self, pid):
        """Upgrade compatibility: an older launcher may have leased the coordinator.

        Resolved seed workers now acquire their own leases. Drop only this
        coordinator's legacy reservation before any children are launched.
        """
        with self.connect() as db:
            db.execute('DELETE FROM tasks WHERE pid=? AND birth=?',
                       (pid, psutil.Process(pid).create_time()))

    def should_retry(self, token):
        """Compatibility for older launchers: resource growth never evicts a run."""
        return False

    def sample(self):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            self._sample(db)

    def set_paused(self, paused):
        with self.connect() as db:
            db.execute("INSERT INTO controls VALUES('paused',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (int(paused),))

    @contextmanager
    def slot(self, spec, *, accelerator=False, check_cancel=None, ticket=None, on_wait=None):
        last_reason = None
        token = None
        ticket = ticket or self.enqueue()[0]
        with self.connect() as db:
            row = db.execute('SELECT position FROM waiters WHERE id=?', (ticket,)).fetchone()
        retry = False
        try:
            while token is None:
                if check_cancel:
                    check_cancel()
                token, reason = self.acquire(spec, accelerator=accelerator, ticket=ticket)
                if reason != last_reason:
                    print(f'[System usage] {reason or "Starting training."}', flush=True)
                    last_reason = reason
                    if reason and on_wait:
                        on_wait(reason)
                if token is None:
                    time.sleep(.5)
            yield token
        except ResourceCapacityRetry as exc:
            # Keep the original place: moving a retry behind not-yet-dispatched
            # seeds can deadlock a bounded coordinator thread pool.
            if row:
                pid = os.getpid()
                with self.connect() as db:
                    db.execute('INSERT OR REPLACE INTO waiters VALUES(?,?,?,?)',
                               (ticket, row[0], pid, psutil.Process(pid).create_time()))
                retry = True
                exc.resource_ticket = ticket
            raise
        finally:
            if not retry:
                self.dequeue(ticket)
            if token:
                self.release(token)
