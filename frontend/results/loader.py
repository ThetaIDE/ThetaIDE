"""Fetches run listings, manifests, metrics and artifact files for the Results panel, and caches them."""

from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal

from .data import Manifest, MetricTable, Source

FILE_CACHE_SIZE = 40  # artifact previews kept in memory


class ResultsLoader(QObject):
    """Reads results/ for the Results panel through the window's LocalBackend.

    `backend` runs src/app/services.catalog on its background thread and calls back with (data, error):
    list_runs(cb), run_manifest(group, experiment_id, cb), run_metrics(group, experiment_id, agent, cb,
    version=, since=) and run_file(group, experiment_id, path, cb).
    """

    runs_listed = pyqtSignal(list)  # catalog.list_runs items
    manifest_loaded = pyqtSignal(object)  # Manifest
    table_loaded = pyqtSignal(object)  # Source whose MetricTable is now in self.tables
    file_loaded = pyqtSignal(str, str)  # run key, artifact path now in self.files
    file_failed = pyqtSignal(str, str, str)  # run key, artifact path, error
    failed = pyqtSignal(str)

    def __init__(self, backend, parent=None):
        super().__init__(parent)
        self.backend = backend
        self.manifests: dict[str, Manifest] = {}
        self.tables: dict[Source, MetricTable] = {}
        self.table_versions: dict[Source, str | None] = {}  # version folder each table was read from
        self.files: dict[tuple[str, str], bytes] = {}
        self._pending: set = set()
        self._generation = 0

    def _current(self, callback):
        """Ignore replies to requests from before the most recent refresh."""
        generation = self._generation

        def received(*args):
            if generation == self._generation:
                callback(*args)

        return received

    def list_runs(self):
        self.backend.list_runs(self._current(self._runs_received), completed_only=True)

    def _runs_received(self, data, error):
        if error:
            self.failed.emit(error)
            return
        self.runs_listed.emit([entry for entry in (data or {}).get("runs") or [] if entry.get("finished")])

    def load_manifest(self, group: str, experiment_id: str):
        key = f"{group}/{experiment_id}"
        if key in self.manifests or ("manifest", key) in self._pending:
            return
        self._pending.add(("manifest", key))
        self.backend.run_manifest(group, experiment_id, self._current(lambda data, error: self._manifest_received(key, data, error)),
                                  completed_only=True)

    def _manifest_received(self, key, data, error):
        self._pending.discard(("manifest", key))
        if error:
            self.failed.emit(f"{key}: {error}")
            return
        manifest = Manifest.from_api(data)
        self.manifests[key] = manifest
        self.manifest_loaded.emit(manifest)

    def load_table(self, source: Source):
        """Fetch a source's metrics unless cached or already on the way; table_loaded fires when ready."""
        if source in self.tables:
            self.table_loaded.emit(source)
            return
        if source in self._pending:
            return
        self._pending.add(source)
        self._fetch_metrics(source, 0, self._current(lambda data, error: self._table_received(source, data, error)))

    def _fetch_metrics(self, source: Source, since: int, callback):
        self.backend.run_metrics(source.group, source.experiment_id, source.agent, callback,
                                 version=source.version or None, since=since, completed_only=True)

    def _table_received(self, source, data, error):
        self._pending.discard(source)
        if error:
            self.failed.emit(f"{source.label}: {error}")
            return
        self.tables[source] = MetricTable.from_rows(data.get("metrics") or [], data.get("columns"))
        self.table_versions[source] = data.get("version")
        self.table_loaded.emit(source)

    def load_file(self, run_key: str, path: str):
        """Fetch one artifact's bytes (`path` as its manifest lists it); file_loaded fires when ready."""
        key = (run_key, path)
        if key in self.files:
            self.file_loaded.emit(run_key, path)
            return
        if key in self._pending:
            return
        self._pending.add(key)
        group, experiment_id = run_key.split("/", 1)
        self.backend.run_file(group, experiment_id, path, self._current(lambda data, error: self._file_received(key, data, error)))

    def _file_received(self, key, data, error):
        self._pending.discard(key)
        if error:
            self.file_failed.emit(*key, str(error))
            self.failed.emit(f"{key[1]}: {error}")
            return
        self.files[key] = data
        while len(self.files) > FILE_CACHE_SIZE:  # keep memory bounded: drop the oldest preview
            self.files.pop(next(iter(self.files)))
        self.file_loaded.emit(*key)

    def clear(self):
        """Forget cached data, so the next requests re-read results/ (e.g. after Refresh)."""
        self._generation += 1
        self._pending.clear()
        self.manifests.clear()
        self.tables.clear()
        self.table_versions.clear()
        self.files.clear()
