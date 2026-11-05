"""Desktop client of the detached training and queue runner.

The runner owns QProcesses and writes logs/status while no desktop is connected.
The desktop rebuilds telemetry from durable files and tails them while open.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from PyQt6.QtCore import QObject, QTimer
from frontend.runner_manager import JobManager as RunnerManager, ACTIVE_STATUSES, FINAL_STATUSES
from src.app.services import JOBS_DIR, LOGS_DIR, PROJECT_ROOT
from src.app.services.config_service import LaunchError, prepare_launch
from frontend.job_runner import write_json


class JobManager(RunnerManager):
    def __init__(self, store_path=None, python=None, project_root=None, logs_dir=None,
                 parent=None, settings_path=None, enabled_plugins_provider=None):
        QObject.__init__(self, parent)
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.logs_dir = Path(logs_dir or LOGS_DIR).resolve()
        store_path = Path(store_path or JOBS_DIR / 'jobs.db').resolve()
        self.jobs_dir = store_path.parent
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.ipc_dir = self.jobs_dir / 'runner'
        self.ipc_dir.mkdir(mode=0o700, exist_ok=True)
        self.enabled_plugins_provider = enabled_plugins_provider
        self.jobs = {}
        self.queue_order = []
        self.queue_running = False
        self.max_parallel = None
        self._feeds = {}
        self._streamed_hardware = set()
        self._logs = {}
        self._log_offsets = {}
        self._last_hardware = {}
        self._finished_scanned = set()
        self._queue = {}
        self._closed = False
        self._heartbeat = self.ipc_dir / (uuid.uuid4().hex + ".client")
        self._heartbeat.touch()
        config = {'store_path': str(store_path), 'python': python,
                  'project_root': str(self.project_root), 'logs_dir': str(self.logs_dir),
                  'settings_path': str(settings_path) if settings_path else None,
                  'ipc_dir': str(self.ipc_dir)}
        # A unique config avoids concurrent desktop startups overwriting each other's file.
        config_path = self.ipc_dir / (uuid.uuid4().hex + '.config')
        write_json(config_path, config)
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
        options = {'start_new_session': True} if os.name != 'nt' else {
            'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        with open(self.ipc_dir / 'runner.log', 'ab') as output:
            subprocess.Popen([sys.executable, '-m', 'frontend.job_runner', str(config_path)],
                             cwd=source_root, env=env, stdin=subprocess.DEVNULL,
                             stdout=output, stderr=output, **options)
        self._refresh(emit=False)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    def _rpc(self, method, *args, **kwargs):
        path = self.ipc_dir / (uuid.uuid4().hex + '.request')
        data = {'method': method, 'args': args, 'kwargs': kwargs}
        if self.enabled_plugins_provider is not None:
            data['plugins'] = self.enabled_plugins_provider()
        write_json(path, data)
        reply = path.with_suffix('.reply')
        deadline = time.monotonic() + 30
        while not reply.exists():
            if time.monotonic() > deadline:
                raise LaunchError('Training runner did not respond; see ' + str(self.ipc_dir / 'runner.log'))
            time.sleep(.02)
        result = json.loads(reply.read_text())
        reply.unlink(missing_ok=True)
        if 'error' in result:
            raise LaunchError(result['error'], conflict=result.get('conflict', False))
        return result['result']

    def _refresh(self, emit=True, cached=False):
        if cached:
            data = json.loads((self.ipc_dir / "state.json").read_text())
            if time.time() - data.get("updated", 0) > 5:
                raise LaunchError("Training runner is unavailable")
        else:
            data = self._rpc("snapshot")
        old = self.jobs
        self.jobs = {j['job_id']: j for j in data['jobs']}
        # Reconcile worker states locally, including when an older detached runner is still alive.
        previous_queue = self._queue
        self._queue = data['queue']
        if self._queue.get('hierarchy_version', 0) < 2:
            self._queue = dict(self._queue, active=[
                item for job in self.jobs.values() if job.get('status') in ACTIVE_STATUSES
                for item in self._execution_queue_items(self.public(job))])
        self.queue_order = [j['job_id'] for j in self._queue['queued']]
        self.queue_running = self._queue['running']
        self.max_parallel = data['max_parallel']
        if emit:
            for key, job in self.jobs.items():
                if old.get(key) != job:
                    self.job_changed.emit(self.public(job))
            if previous_queue != self._queue:
                self.queue_changed.emit(self._queue)

    def _poll(self):
        if self._closed:
            return
        try:
            self._heartbeat.touch()
            self._refresh(cached=True)
            for job_id, job in self.jobs.items():
                # Replay once on reconnect, including jobs finished while the desktop was closed.
                if job.get('started') and (job.get('status') in ACTIVE_STATUSES or job_id not in self._finished_scanned):
                    self._scan_metrics(job_id)
                    self._sample_hardware(job_id)
                    path = self.jobs_dir / f'{job_id}.log'
                    offset = self._log_offsets.get(job_id, 0)
                    try:
                        with path.open('rb') as stream:
                            stream.seek(offset)
                            raw = stream.read()
                        end = raw.rfind(b'\n') + 1
                        self._log_offsets[job_id] = offset + end
                        lines = raw[:end].decode('utf-8', errors='replace').splitlines()
                        if lines:
                            self._logs[job_id] = (self._logs.get(job_id, []) + lines)[-20000:]
                            self.job_output.emit(job_id, lines)
                    except OSError:
                        pass
                    if job.get("status") in FINAL_STATUSES:
                        self._finished_scanned.add(job_id)
        except (LaunchError, OSError, ValueError):
            # Leave running jobs intact when communication is temporarily unavailable.
            return

    def _scan_metrics(self, job_id):
        # A first read replaces persisted desktop history rather than appending duplicates.
        job = self.jobs[job_id]
        for label, *_path, status in self._run_sources(job):
            if status in ('queued', 'pending'):
                continue
            if (job_id, label) not in self._feeds:
                self.job_records.emit(job_id, label, [], True)
                self.job_metrics.emit(job_id, label, [], True)
        super()._scan_metrics(job_id)

    def launch(self, experiment, overrides=None, queue=False, overwrite=False):
        self._refresh()
        plan = prepare_launch(experiment, overrides, overwrite=overwrite,
                              active_jobs=[j for j in self.jobs.values() if j.get('status') in ('queued', *ACTIVE_STATUSES)])
        result = self._rpc('launch', experiment, overrides, queue=queue, overwrite=overwrite, _plan=plan)
        self._refresh()
        return result

    def cancel(self, job_id, target=""):
        if job_id not in self.jobs:
            raise KeyError(job_id)
        result = self._rpc('cancel', job_id, target)
        self._refresh()
        return result

    def move(self, job_id, position):
        result = self._rpc('move', job_id, position)
        self._refresh()
        return result

    def start_queue(self):
        result = self._rpc('start_queue')
        self._refresh()
        return result

    def pause_queue(self):
        result = self._rpc('pause_queue')
        self._refresh()
        return result

    def set_max_parallel(self, count):
        self._rpc('set_max_parallel', count)
        self._refresh()

    def queue_state(self):
        return self._queue

    def focus_replicate(self, job_id, replicate_label):
        self._scan_metrics(job_id)

    def shutdown(self):
        """Disconnect the desktop; the runner and its queue keep running."""
        self._closed = True
        self.timer.stop()
        self._heartbeat.unlink(missing_ok=True)
