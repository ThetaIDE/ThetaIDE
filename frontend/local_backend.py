"""Config composition and the run catalog, called in-process instead of over HTTP.

Catalog/file calls run serially on a background thread. The desktop isolates
validation in a private interpreter with a separate serial queue; the default
in-process mode retains one shared queue for Hydra. Callbacks receive (data,
error) on the GUI thread.
"""
from __future__ import annotations

from typing import Any, Callable
import time

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from src.app.services import catalog, config_service

Callback = Callable[[Any, "str | None"], None]


class _Relay(QObject):
    """Carries a worker's result back to the GUI thread (signals are queued across threads)."""

    done = pyqtSignal(object, object)


class _Call(QRunnable):
    def __init__(self, fn, args, relay, current=None):
        super().__init__()
        self.fn, self.args, self.relay = fn, args, relay
        self.current = current

    def run(self):
        try:
            data, error = (self.fn(*self.args), None) if self.current is None or self.current() else (None, None)
        except Exception as exc:  # reported to the caller, like an HTTP error used to be
            data, error = None, str(exc).strip() or type(exc).__name__
        try:
            self.relay.done.emit(data, error)
        except RuntimeError:
            pass


class LocalBackend(QObject):
    def __init__(self, parent=None, *, isolate_validation=False):
        super().__init__(parent)
        self._validator = None
        if isolate_validation:
            from .validation_process import ValidationProcess
            self._validator = ValidationProcess()
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._validation_pool = QThreadPool(self) if isolate_validation else None
        if self._validation_pool is not None:
            self._validation_pool.setMaxThreadCount(1)
        self._relays: set[_Relay] = set()
        self._compose_generation = 0
        # A call can still be running when the window that asked for it is closed. Its result must then
        # be dropped: calling back into deleted widgets raises inside a slot, and PyQt aborts on that.
        self._alive = True
        self.destroyed.connect(self._mark_destroyed)

    def _mark_destroyed(self, *_):
        self.shutdown()

    def shutdown(self):
        self._alive = False
        self.cancel_compose()
        if self._validator is not None:
            self._validator.shutdown()

    def _submit(self, fn, *args, callback: Callback, current=None, pool=None):
        relay = _Relay()
        self._relays.add(relay)

        def deliver(data, error, relay=relay):
            self._relays.discard(relay)
            if not self._alive or (current is not None and not current()):
                return
            try:
                callback(data, error)
            except RuntimeError as exc:  # the caller's widgets were deleted while the call ran
                if "has been deleted" not in str(exc):
                    raise

        relay.done.connect(deliver)
        (self.pool if pool is None else pool).start(_Call(fn, args, relay, current))

    def config_schema(self, callback: Callback):
        if self._validator is not None:
            self._submit(self._validator.call, "schema", callback=callback, pool=self._validation_pool)
        else:
            self._submit(config_service.config_schema, callback=callback)

    def compose(self, experiment: str, overrides: list[str], callback: Callback, draft_yaml: str | None = None):
        self.cancel_compose()
        generation = self._compose_generation
        fn = config_service.compose_report if self._validator is None else lambda *args: self._validator.call("compose", *args)
        self._submit(fn, experiment, list(overrides), draft_yaml, callback=callback,
                     current=lambda: self._alive and generation == self._compose_generation, pool=self._validation_pool)

    def cancel_compose(self):
        """Discard queued and late validation results; never interrupt Hydra mid-call."""
        self._compose_generation += 1

    def list_runs(self, callback: Callback, *, completed_only=False):
        self._submit(lambda: {"runs": catalog.list_runs(completed_only=completed_only)}, callback=callback)

    def list_executions(self, callback: Callback, *, completed_only=False):
        from src.app.services.executions import list_executions
        self._submit(list_executions, completed_only, callback=callback)

    def read_execution(self, path: str, callback: Callback, *, completed_only=False):
        from src.app.services.executions import read_execution
        self._submit(read_execution, path, completed_only, callback=callback)

    def archive_run(self, group: str, experiment_id: str, destination: str, callback: Callback):
        self._submit(catalog.archive_run, group, experiment_id, destination, callback=callback)

    def delete_run(self, group: str, experiment_id: str, callback: Callback):
        self._submit(catalog.delete_run, group, experiment_id, callback=callback)

    def rename_results(self, kind: str, group: str, experiment_id: str, new_name: str, callback: Callback):
        from src.app.services.result_rename import rename_results
        self._submit(rename_results, kind, group, experiment_id, new_name, callback=callback)

    def archive_group(self, group: str, destination: str, callback: Callback):
        self._submit(catalog.archive_group, group, destination, callback=callback)

    def delete_group(self, group: str, callback: Callback):
        self._submit(catalog.delete_group, group, callback=callback)

    def run_metrics(self, group: str, experiment_id: str, agent: str, callback: Callback,
                    version: str | None = None, since: int = 0, *, completed_only=False):
        self._submit(catalog.run_metrics, group, experiment_id, agent, version, since, completed_only, callback=callback)

    def run_manifest(self, group: str, experiment_id: str, callback: Callback, *, completed_only=False):
        """Every artifact a run produced (catalog.run_manifest)."""
        self._submit(catalog.run_manifest, group, experiment_id, completed_only, callback=callback)

    def run_file(self, group: str, experiment_id: str, path: str, callback: Callback):
        """One artifact's bytes, by the path its manifest lists (catalog.read_run_file)."""
        self._submit(catalog.read_run_file, group, experiment_id, path, callback=callback)

    def rename_run_file(self, group: str, experiment_id: str, path: str, new_name: str, callback: Callback):
        self._submit(catalog.rename_run_file, group, experiment_id, path, new_name, callback=callback)

    def wait(self, msecs: int = 30000) -> bool:
        """Block until queued calls finish (tests and shutdown)."""
        started = time.monotonic()
        if not self.pool.waitForDone(msecs):
            return False
        if self._validation_pool is None:
            return True
        remaining = -1 if msecs < 0 else max(0, msecs - int((time.monotonic() - started) * 1000))
        return self._validation_pool.waitForDone(remaining)
