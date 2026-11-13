"""Load desktop run summaries without blocking the first window paint."""
from PyQt6.QtCore import pyqtSignal

from .local_backend import LocalBackend


class RunHistoryLoader(LocalBackend):
    loaded = pyqtSignal(list, list)

    def __init__(self, store, parent=None):
        super().__init__(parent)
        self.store = store
        self._started = False

    def start(self):
        if not self._started:
            self._started = True
            self._submit(self.store.load_summaries, callback=self._received)

    def _received(self, data, error):
        if error:
            self.loaded.emit([], [f"Could not load run history: {error}"])
        else:
            self.loaded.emit(*data)

    def shutdown(self):
        self._alive = False  # A late worker may finish, but cannot mutate a closing window.
