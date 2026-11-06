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
        self._folders = {}
        self._prepare_copy = {}
        self._providers: dict[str, tuple[str, Callable]] = {}

    def register(self, provider_id: str, name: str, open_file: Callable, open_folder: Callable | None = None, prepare_copy: Callable | None = None) -> None:
        if provider_id == "system" or provider_id in self._providers:
            raise ValueError(f"Editor provider already registered: {provider_id}")
        self._providers[provider_id] = (name, open_file)
        if open_folder:
            self._folders[provider_id] = open_folder
        if prepare_copy:
            self._prepare_copy[provider_id] = prepare_copy
        self.providersChanged.emit()

    def unregister(self, provider_id: str) -> None:
        self._providers.pop(provider_id, None)
        self._folders.pop(provider_id, None)
        self._prepare_copy.pop(provider_id, None)
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

    def prepare_component_copy(self) -> bool:
        return all(callback() for callback in self._prepare_copy.values())

    def open_folder(self, root, initial_file=None, owned_paths=()) -> bool:
        root = Path(root).resolve()
        provider = self._folders.get(self.default_editor)
        if provider:
            return bool(provider(root, initial_file=initial_file, owned_paths=owned_paths))
        if self.default_editor in self._providers:
            return self.open_file(initial_file or root / "COMPONENT_CONTEXT.md")
        from .component_actions import open_in_editor
        return open_in_editor(root)
