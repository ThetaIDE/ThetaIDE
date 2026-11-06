"""Detached queue owner. Requests and replies are atomic files in a private workspace directory."""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path
from PyQt6.QtCore import QCoreApplication, QLockFile, QTimer
from frontend.runner_manager import JobManager


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value), encoding='utf-8')
    temporary.replace(path)


class RunnerLifecycle:
    """Check shutdown independently of queue/IPC failures and limit error output."""

    def __init__(self, directory, manager, app):
        self.directory = directory
        self.manager = manager
        self.app = app
        self.last_request = time.monotonic()
        self.last_error = float('-inf')

    def should_stop(self):
        # Temporary test/preview workspaces can be removed before the idle grace
        # period expires. Never keep writing to their now-unlinked log files.
        if not self.directory.is_dir():
            return True
        connected = False
        for path in self.directory.glob('*.client'):
            try:
                connected = connected or time.time() - path.stat().st_mtime < 60
            except FileNotFoundError:
                pass
        return (not self.manager.is_busy() and not connected
                and time.monotonic() - self.last_request > 60)

    def update(self, tick):
        try:
            # A failed snapshot or resource update must not bypass idle shutdown.
            if self.should_stop():
                self.app.quit()
                return
            tick()
        except Exception:
            now = time.monotonic()
            if now - self.last_error >= 60:
                self.last_error = now
                logging.exception('Runner update failed')


def main():
    config_path = Path(sys.argv[1])
    config = json.loads(config_path.read_text())
    config_path.unlink(missing_ok=True)
    directory = Path(config.pop('ipc_dir'))
    app = QCoreApplication(sys.argv[:1])
    lock = QLockFile(str(directory / 'owner.lock'))
    if not lock.tryLock(0):
        return
    manager = JobManager(**config)
    allowed = {'launch', 'cancel', 'move', 'start_queue', 'pause_queue', 'set_max_parallel'}
    lifecycle = RunnerLifecycle(directory, manager, app)
    last_snapshot = [0.0]
    def tick():
        manager.resource_tick()
        for request in sorted(directory.glob('*.request')):
            lifecycle.last_request = time.monotonic()
            try:
                data = json.loads(request.read_text())
                method = data['method']
                if method == 'snapshot':
                    result = {'jobs': list(manager.jobs.values()), 'queue': manager.queue_state(),
                              'max_parallel': manager.max_parallel}
                elif method in allowed:
                    plugins = data.get('plugins')
                    if plugins is not None:
                        manager.enabled_plugins_provider = lambda p=plugins: p
                    result = getattr(manager, method)(*data.get('args', []), **data.get('kwargs', {}))
                else:
                    raise ValueError('Unknown runner command')
                reply = {'result': result}
            except Exception as exc:
                reply = {'error': str(exc), 'conflict': getattr(exc, 'conflict', False)}
            write_json(request.with_suffix('.reply'), reply)
            request.unlink(missing_ok=True)
        if time.monotonic() - last_snapshot[0] >= 1:
            last_snapshot[0] = time.monotonic()
            write_json(directory / 'state.json',
                       {'jobs': list(manager.jobs.values()), 'queue': manager.queue_state(),
                        'max_parallel': manager.max_parallel, 'updated': time.time()})
            # Reconcile files even when filesystem notifications were coalesced.
            manager._on_path_changed('')
        # Remove replies left behind by a client that exited during a request.
        for reply in directory.glob('*.reply'):
            try:
                if time.time() - reply.stat().st_mtime > 120:
                    reply.unlink(missing_ok=True)
            except FileNotFoundError:
                pass
    def safe_tick():
        lifecycle.update(tick)

    timer = QTimer()
    timer.timeout.connect(safe_tick)
    timer.start(100)
    app.exec()


if __name__ == '__main__':
    main()
