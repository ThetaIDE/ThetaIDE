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
from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from frontend.runner_manager import JobManager as RunnerManager, ACTIVE_STATUSES, FINAL_STATUSES
from src.app.workspace import WorkspacePaths
from src.app.services.config_service import LaunchError, prepare_launch
from frontend.job_runner import write_json


class JobManager(RunnerManager):
    readiness_changed = pyqtSignal(bool, str)

    def __init__(self, store_path=None, python=None, project_root=None, logs_dir=None,
                 parent=None, settings_path=None, enabled_plugins_provider=None, defer_telemetry=False):
        QObject.__init__(self, parent)
        paths = WorkspacePaths(project_root) if project_root else WorkspacePaths.for_process()
        self.project_root = paths.root
        self.logs_dir = Path(logs_dir or paths.results / 'logs').resolve()
        store_path = Path(store_path or paths.results / 'jobs/jobs.db').resolve()
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
        self._following_jobs = set()
        self._telemetry_pending = {}
        self._telemetry_read_limits = {"max_records": 64, "max_bytes": 256 * 1024}
        self._streaming_enabled = not defer_telemetry
        self._poll_cursor = 0
        self._queue = {"running": False, "active": [], "queued": [], "finished": []}
        self._closed = False
        self.ready = False
        self.readiness_error = "Connecting to training runner…"
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
        # Runs execute experiment and Hub component code: they get no access to an MCP server's tools.
        for name in ('THETA_MCP_URL', 'THETA_MCP_TOKEN'):
            env.pop(name, None)
        source_root = str(Path(__file__).resolve().parent.parent)
        env['PYTHONPATH'] = source_root + os.pathsep + env.get('PYTHONPATH', '')
        options = {'start_new_session': True} if os.name != 'nt' else {
            'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
        with open(self.ipc_dir / 'runner.log', 'ab') as output:
            try:
                subprocess.Popen([sys.executable, '-m', 'frontend.job_runner', str(config_path)],
                                 cwd=source_root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=output, **options)
            except OSError as exc:
                self.readiness_error = f"Could not start training runner: {exc}"
        self._request_readiness()
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self._poll)
        self.timer.start()

    def _send_request(self, method, *args, **kwargs):
        path = self.ipc_dir / (uuid.uuid4().hex + '.request')
        data = {'method': method, 'args': args, 'kwargs': kwargs}
        if self.enabled_plugins_provider is not None:
            data['plugins'] = self.enabled_plugins_provider()
        write_json(path, data)
        return path

    def _request_readiness(self):
        self._readiness_request = self._send_request("snapshot")
        self._readiness_deadline = time.monotonic() + 30

    def _set_readiness(self, ready, error=""):
        changed = self.ready != ready or self.readiness_error != error
        self.ready, self.readiness_error = ready, error
        self.timer.setInterval(1000 if ready else 50)
        if changed:
            from .startup import mark
            mark("runner_ready" if ready else "runner_unavailable")
            self.readiness_changed.emit(ready, error)

    def _rpc(self, method, *args, **kwargs):
        if not self.ready:
            raise LaunchError(self.readiness_error or "Training runner is connecting")
        path = self._send_request(method, *args, **kwargs)
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
        self._apply_snapshot(data, emit)

    def _apply_snapshot(self, data, emit=True):
        # Validate completely before replacing the last known-good state.
        jobs = {job["job_id"]: job for job in data["jobs"]}
        queue = dict(data["queue"])
        if not isinstance(queue.get("running"), bool) or not all(
            isinstance(queue.get(key), list) for key in ("active", "queued", "finished")
        ):
            raise ValueError("Invalid runner queue snapshot")
        queue_order = [job["job_id"] for job in queue["queued"]]
        max_parallel = data["max_parallel"]
        if queue.get('hierarchy_version', 0) < 2:
            queue["active"] = [item for job in jobs.values() if job.get('status') in ACTIVE_STATUSES
                               for item in self._execution_queue_items(self.public(job))]
        old, previous_queue = self.jobs, self._queue
        self.jobs, self._queue = jobs, queue
        self._following_jobs.intersection_update(jobs)
        self._following_jobs.update(job_id for job_id, job in jobs.items() if job.get("status") in ACTIVE_STATUSES)
        self._telemetry_pending = {key: value for key, value in self._telemetry_pending.items() if key in jobs}
        self.queue_order = queue_order
        self.queue_running = queue["running"]
        self.max_parallel = max_parallel
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
            if not self.ready:
                reply = self._readiness_request.with_suffix(".reply")
                if reply.is_file():
                    response = json.loads(reply.read_text())
                    reply.unlink(missing_ok=True)
                    if "error" in response:
                        raise LaunchError(response["error"])
                    self._apply_snapshot(response["result"])
                    self._set_readiness(True)
                    return  # state.json may be published just after this reply.
                elif time.monotonic() >= self._readiness_deadline:
                    self._readiness_request.unlink(missing_ok=True)
                    self._set_readiness(False, "Training runner did not respond; see " + str(self.ipc_dir / "runner.log"))
                    self._request_readiness()
                if not self.ready:
                    return
            self._refresh(cached=True)
        except (LaunchError, OSError, ValueError, KeyError, TypeError) as exc:
            # Retain jobs while reconnecting; an unavailable runner must never
            # cause the desktop to mark its jobs interrupted.
            if self.ready:
                self._request_readiness()
            self._set_readiness(False, str(exc))
            return
        if not self._streaming_enabled:
            return
        deadline = time.monotonic() + .025
        items = list(self.jobs.items())
        if items:
            cursor = self._poll_cursor % len(items)
            items = items[cursor:] + items[:cursor]
        try:
            for job_id, job in items:
                self._poll_cursor += 1
                # Archives belong to Results. Only follow jobs observed live by
                # this desktop, including their final tail after completion.
                if job.get('started') and (job.get('status') in ACTIVE_STATUSES or
                        (job_id in self._following_jobs and job_id not in self._finished_scanned)):
                    self._scan_metrics(job_id)
                    self._sample_hardware(job_id)
                    path = self.jobs_dir / f'{job_id}.log'
                    offset = self._log_offsets.get(job_id, 0)
                    try:
                        with path.open('rb') as stream:
                            stream.seek(offset)
                            from src.app.services.metrics import _read_batch
                            raw, log_pending = _read_batch(stream, max_bytes=64 * 1024)
                            self._telemetry_pending[job_id] = self._telemetry_pending.get(job_id, False) or log_pending
                        end = raw.rfind(b'\n') + 1
                        self._log_offsets[job_id] = offset + end
                        lines = raw[:end].decode('utf-8', errors='replace').splitlines()
                        if lines:
                            self._logs[job_id] = (self._logs.get(job_id, []) + lines)[-20000:]
                            self.job_output.emit(job_id, lines)
                    except OSError:
                        pass
                    if job.get("status") in FINAL_STATUSES and not self._telemetry_pending.get(job_id):
                        self._finished_scanned.add(job_id)
                if time.monotonic() >= deadline:
                    break
            self.timer.setInterval(50 if any(self._telemetry_pending.values()) or len(items) > 0 and time.monotonic() >= deadline else 1000)
        except (LaunchError, OSError, ValueError, KeyError, TypeError):
            # A bad telemetry/log file does not mean the queue owner disconnected.
            self.timer.setInterval(1000)
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
        self._telemetry_pending[job_id] = any(
            getattr(feed, "pending", False) for (owner, _), feed in self._feeds.items() if owner == job_id)

    def start_streaming(self):
        self._streaming_enabled = True

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

    def replay_history(self, job_ids):
        """Replay telemetry that arrived before saved desktop records were loaded."""
        for job_id in job_ids:
            if not job_id:
                continue
            self._finished_scanned.discard(job_id)
            for key in list(self._feeds):
                if key[0] == job_id:
                    self._feeds.pop(key, None)
            self._log_offsets.pop(job_id, None)
            self._logs.pop(job_id, None)

    def focus_replicate(self, job_id, replicate_label):
        self._scan_metrics(job_id)

    def shutdown(self):
        """Disconnect the desktop; the runner and its queue keep running."""
        self._closed = True
        self.timer.stop()
        self._heartbeat.unlink(missing_ok=True)
        self._readiness_request.unlink(missing_ok=True)
        self._readiness_request.with_suffix(".reply").unlink(missing_ok=True)
