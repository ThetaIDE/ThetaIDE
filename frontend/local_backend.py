"""Config composition and the run catalog, called in-process instead of over HTTP.

Each call runs on one background thread (Hydra's global state is not thread-safe, so calls never
overlap) and its callback receives (data, error) on the GUI thread, like the old network client.
"""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal

from src.app.services import catalog, config_service

Callback = Callable[[Any, "str | None"], None]


class _Relay(QObject):
    """Carries a worker's result back to the GUI thread (signals are queued across threads)."""

    done = pyqtSignal(object, object)


class _Call(QRunnable):
    def __init__(self, fn, args, relay):
        super().__init__()
        self.fn, self.args, self.relay = fn, args, relay

    def run(self):
        try:
            data, error = self.fn(*self.args), None
        except Exception as exc:  # reported to the caller, like an HTTP error used to be
            data, error = None, str(exc).strip() or type(exc).__name__
        try:
            self.relay.done.emit(data, error)
        except RuntimeError:
            pass


class LocalBackend(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)
        self._relays: set[_Relay] = set()
        # A call can still be running when the window that asked for it is closed. Its result must then
        # be dropped: calling back into deleted widgets raises inside a slot, and PyQt aborts on that.
        self._alive = True
        self.destroyed.connect(self._mark_destroyed)

    def _mark_destroyed(self, *_):
        self._alive = False

    def _submit(self, fn, *args, callback: Callback):
        relay = _Relay()
        self._relays.add(relay)

        def deliver(data, error, relay=relay):
            self._relays.discard(relay)
            if not self._alive:
                return
            try:
                callback(data, error)
            except RuntimeError as exc:  # the caller's widgets were deleted while the call ran
                if "has been deleted" not in str(exc):
                    raise

        relay.done.connect(deliver)
        self.pool.start(_Call(fn, args, relay))

    def config_schema(self, callback: Callback):
        self._submit(config_service.config_schema, callback=callback)

    def compose(self, experiment: str, overrides: list[str], callback: Callback):
        self._submit(config_service.compose_report, experiment, list(overrides), callback=callback)

    def list_runs(self, callback: Callback):
        self._submit(lambda: {"runs": catalog.list_runs()}, callback=callback)

    def archive_run(self, group: str, experiment_id: str, destination: str, callback: Callback):
        self._submit(catalog.archive_run, group, experiment_id, destination, callback=callback)

    def delete_run(self, group: str, experiment_id: str, callback: Callback):
        self._submit(catalog.delete_run, group, experiment_id, callback=callback)

    def archive_group(self, group: str, destination: str, callback: Callback):
        self._submit(catalog.archive_group, group, destination, callback=callback)

    def delete_group(self, group: str, callback: Callback):
        self._submit(catalog.delete_group, group, callback=callback)

    def run_metrics(self, group: str, experiment_id: str, agent: str, callback: Callback,
                    version: str | None = None, since: int = 0):
        self._submit(catalog.run_metrics, group, experiment_id, agent, version, since, callback=callback)

    def run_manifest(self, group: str, experiment_id: str, callback: Callback):
        """Every artifact a run produced (catalog.run_manifest)."""
        self._submit(catalog.run_manifest, group, experiment_id, callback=callback)

    def run_file(self, group: str, experiment_id: str, path: str, callback: Callback):
        """One artifact's bytes, by the path its manifest lists (catalog.read_run_file)."""
        self._submit(catalog.read_run_file, group, experiment_id, path, callback=callback)

    def wait(self, msecs: int = 30000) -> bool:
        """Block until queued calls finish (tests and shutdown)."""
        return self.pool.waitForDone(msecs)
