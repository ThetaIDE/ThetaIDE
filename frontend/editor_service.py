"""Host-level file routing shared by editor plugins and future UI actions."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices


class EditorService(QObject):
    providersChanged = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._providers: dict[str, tuple[str, Callable]] = {}

    def register(self, provider_id: str, name: str, open_file: Callable) -> None:
        if provider_id == "system" or provider_id in self._providers:
            raise ValueError(f"Editor provider already registered: {provider_id}")
        self._providers[provider_id] = (name, open_file)
        self.providersChanged.emit()

    def unregister(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)
        self.providersChanged.emit()

    def providers(self) -> list[tuple[str, str]]:
        return [("system", "System editor"), *[(key, value[0]) for key, value in self._providers.items()]]

    @property
    def default_editor(self) -> str:
        return self.settings.default_text_editor

    def open_file(self, path, line: int | None = None, column: int | None = None) -> bool:
        """Open a local file; line and column are one-based when supported."""
        path = Path(path).expanduser().resolve()
        provider = self._providers.get(self.default_editor)
        if provider is not None:
            # A provider failure must not silently send the file to another application.
            return bool(provider[1](path, line=line, column=column))
        return QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
