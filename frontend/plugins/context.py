"""PluginContext host API facade providing controlled IDE access to plugins."""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import QObject, QProcess
from PyQt6.QtWidgets import QWidget

from src.app.services.processes import kill_process_tree
from .commands import Command
from .slots import ExtensionSlot

if TYPE_CHECKING:
    from ..app import Window


class _SectionStore:
    """``get``/``set`` access to one section of the host settings."""

    def __init__(self, settings, section: str):
        self._settings = settings
        self._section = section

    def _check(self, section: str) -> None:
        if section != self._section:
            raise PermissionError(f"Plugin settings access is limited to the {self._section!r} section")

    def get(self, section: str, *keys: str, default: Any = None) -> Any:
        self._check(section)
        return self._settings.get(section, *keys, default=default)

    def set(self, section: str, *keys_and_value: Any) -> None:
        self._check(section)
        self._settings.set(section, *keys_and_value)


class PluginContext:
    """The host execution context passed to a Plugin when it is activated.

    This is the supported host facade of Theta SDK 1.0 (``src.app.api.desktop``).
    Plugins use its services rather than attributes of the host window; every
    registration made through it is undone by ``cleanup``.
    """

    def __init__(self, plugin_id: str, window: Window, storage_file: Path, *, activated_explicitly: bool = False):
        self.plugin_id = plugin_id
        self._window = window
        self._storage_file = storage_file
        self._activated_explicitly = bool(activated_explicitly)
        self._editor_registered = False
        self._appearance_listeners: List[Callable[[], None]] = []
        self._hub_listeners: Dict[Callable[[], None], Callable[..., None]] = {}
        self._registered_tabs: List[str] = []
        self._experiment_listeners: List[Callable[[Optional[str], Optional[str]], None]] = []
        self._storage_cache: Optional[Dict[str, Any]] = None  # lazy-loaded on first access
        self._tracked_processes: List[QProcess] = []
        self._contributed_actions: List[Tuple[str, str]] = []  # (slot_id, action_id)
        self._registered_commands: List[str] = []

    @property
    def window(self) -> Window:
        """The main IDE window instance.

        Transitional and unsupported: window attributes are private host
        implementation. Use the context services instead; bundled plugins do.
        """
        return self._window

    @property
    def activated_explicitly(self) -> bool:
        """True when the user enabled the plugin now, False when restoring saved state."""
        return self._activated_explicitly

    def parent_widget(self) -> Optional[QWidget]:
        """Host widget to parent dialogs and Qt objects to. Treat it as an opaque QWidget."""
        return self._window if isinstance(self._window, QWidget) else None

    # ── Sidebar & UI Contributions ──────────────────────────────────────────

    def add_sidebar_tab(
        self,
        tab_id: str,
        widget: QWidget,
        title: str,
        icon_name: Optional[str] = None,
        short_label: Optional[str] = None,
    ) -> None:
        """Add a panel to the IDE's primary vertical sidebar.
        
        Args:
            tab_id: Unique string identifier for this tab.
            widget: QWidget to display when tab is selected.
            title: Full display title (and tooltip).
            icon_name: SVG icon name in frontend/icons/ or absolute path to an .svg.
            short_label: Short text label under the icon in the sidebar (defaults to title).
        """
        resolved_icon = icon_name
        if resolved_icon and hasattr(self._window, "plugin_manager"):
            manifest = self._window.plugin_manager.manifests.get(self.plugin_id)
            if manifest and manifest.plugin_dir:
                p_cand = Path(resolved_icon)
                builtin_icon_dir = Path(__file__).resolve().parent.parent / "icons"
                if not (p_cand.is_file() or (builtin_icon_dir / f"{resolved_icon}.svg").is_file()):
                    for cand in [
                        manifest.plugin_dir / resolved_icon,
                        manifest.plugin_dir / f"{resolved_icon}.svg",
                        manifest.plugin_dir / f"{self.plugin_id}.svg",
                        manifest.plugin_dir / f"{manifest.id}.svg",
                    ]:
                        if cand.is_file():
                            resolved_icon = str(cand)
                            break

        if hasattr(self._window, "tabs") and self._window.tabs is not None:
            self._window.tabs.addTab(
                widget=widget,
                text=title,
                icon=resolved_icon,
                short=short_label or title,
                tab_id=tab_id,
            )
            if tab_id not in self._registered_tabs:
                self._registered_tabs.append(tab_id)

    def remove_sidebar_tab(self, tab_id: str) -> None:
        """Remove a contributed panel from the sidebar."""
        if hasattr(self._window, "tabs") and self._window.tabs is not None:
            self._window.tabs.removeTab(tab_id)
        if tab_id in self._registered_tabs:
            self._registered_tabs.remove(tab_id)

    def select_sidebar_tab(self, tab_id: str) -> None:
        """Switch the active view to the specified sidebar tab."""
        if hasattr(self._window, "tabs") and self._window.tabs is not None:
            entry = self._window.tabs.tabs.get(tab_id)
            if entry and "widget" in entry:
                self._window.tabs.setCurrentWidget(entry["widget"])

    def open_settings(self, page: Optional[str] = None) -> None:
        """Open Settings (not a sidebar tab, so select_sidebar_tab can't), on `page` (e.g. "hotkeys") if given."""
        if hasattr(self._window, "show_settings"):
            self._window.show_settings(page)

    def open_component_hub(self) -> None:
        """Open the Component Hub inside the Components panel."""
        panel = getattr(self._window, "components_panel", None)
        if panel is not None and hasattr(panel, "show_hub"):
            self._window.tabs.setCurrentWidget(panel)
            panel.show_hub()

    # ── Extension Slots ──────────────────────────────────────────────────────

    def contribute_to_slot(
        self,
        slot_id: str,
        action_id: str,
        factory_or_widget: Union[Callable[[], Optional[QWidget]], QWidget],
        priority: int = 100,
    ) -> None:
        """Contribute a widget or action button to a named UI extension slot."""
        if hasattr(self._window, "extension_slots") and self._window.extension_slots is not None:
            slot = self._window.extension_slots.get_slot(slot_id)
            if slot:
                slot.add_action(action_id, factory_or_widget, priority=priority)
                pair = (slot_id, action_id)
                if pair not in self._contributed_actions:
                    self._contributed_actions.append(pair)

    def remove_from_slot(self, slot_id: str, action_id: str) -> None:
        """Remove a previously contributed action from an extension slot."""
        if hasattr(self._window, "extension_slots") and self._window.extension_slots is not None:
            slot = self._window.extension_slots.get_slot(slot_id, create=False)
            if slot:
                slot.remove_action(action_id)
        pair = (slot_id, action_id)
        if pair in self._contributed_actions:
            self._contributed_actions.remove(pair)

    # ── Commands & Hotkeys ───────────────────────────────────────────────────

    def register_command(
        self,
        command_id: str,
        title: str,
        callback: Callable[[], None],
        default_hotkey: Optional[str] = None,
        keywords: Tuple[str, ...] = (),
        category: str = "Plugin",
        icon: Optional[str] = None,
        detail: str = "",
    ) -> Command:
        """Register an executable command discoverable via launcher and hotkeys."""
        cmd = Command(
            command_id=command_id,
            title=title,
            callback=callback,
            default_hotkey=default_hotkey,
            keywords=keywords,
            category=category,
            icon=icon,
            plugin_id=self.plugin_id,
            detail=detail,
        )
        if hasattr(self._window, "command_registry") and self._window.command_registry is not None:
            self._window.command_registry.register_command(cmd)
            if command_id not in self._registered_commands:
                self._registered_commands.append(command_id)
        return cmd

    def unregister_command(self, command_id: str) -> None:
        """Unregister a command previously contributed by this plugin."""
        if hasattr(self._window, "command_registry") and self._window.command_registry is not None:
            self._window.command_registry.unregister_command(command_id)
        if command_id in self._registered_commands:
            self._registered_commands.remove(command_id)

    # ── Process Management & Supervision ─────────────────────────────────────

    def track_process(self, process: QProcess) -> QProcess:
        """Register a QProcess to be tracked and cleanly terminated on plugin deactivation or IDE exit."""
        if process not in self._tracked_processes:
            self._tracked_processes.append(process)
            try:
                process.destroyed.connect(lambda: self._untrack_process(process))
            except Exception:
                pass
        return process

    def _untrack_process(self, process: QProcess) -> None:
        if process in self._tracked_processes:
            self._tracked_processes.remove(process)

    def spawn_process(
        self,
        program: str,
        arguments: Optional[List[str]] = None,
        working_dir: Optional[Union[str, Path]] = None,
        parent: Optional[QObject] = None,
    ) -> QProcess:
        """Create and track a QProcess configured with the given executable and arguments."""
        proc = QProcess(parent or self._window)
        proc.setProgram(program)
        if arguments:
            proc.setArguments(arguments)
        if working_dir:
            proc.setWorkingDirectory(str(working_dir))
        return self.track_process(proc)

    # ── Workspace & Environment ──────────────────────────────────────────────

    def workspace_paths(self):
        """The host's selected `WorkspacePaths` (workspace, state and application roots).

        Plugins resolve research inputs, results and state from this object rather than
        from the process CWD or application module constants.
        """
        paths = getattr(self._window, "workspace_paths", None)
        if paths is None:
            from src.app.workspace import WorkspacePaths
            paths = WorkspacePaths.for_process()
        return paths

    def get_workspace_dir(self) -> Path:
        """Return the root workspace directory."""
        return self.workspace_paths().root

    def get_data_dir(self) -> Path:
        """Return the ThetaIDE internal data/storage directory."""
        return self.workspace_paths().data_dir

    def log(self, message: str) -> None:
        """Log a message to the IDE console / status log."""
        prefix = f"[{self.plugin_id}]"
        if hasattr(self._window, "log") and callable(self._window.log):
            self._window.log(f"{prefix} {message}")
        else:
            print(f"{prefix} {message}")

    def show_status_message(self, message: str, timeout_ms: int = 4000) -> None:
        """Display a transient message on the main window status bar."""
        if hasattr(self._window, "statusBar"):
            status_bar = self._window.statusBar()
            if status_bar:
                status_bar.showMessage(message, timeout_ms)

    def mark_startup(self, event: str, **details: Any) -> None:
        """Record a startup-trace event (a no-op unless startup tracing is enabled)."""
        from ..startup import mark
        mark(event, plugin=self.plugin_id, **details)

    # ── Editors, Appearance & Configuration ──────────────────────────────────

    def register_editor(
        self,
        name: str,
        open_file: Callable[..., bool],
        open_folder: Optional[Callable[..., bool]] = None,
        prepare_copy: Optional[Callable[[], bool]] = None,
        *,
        make_default: bool = False,
    ) -> None:
        """Offer this plugin as a text-editor provider for host file actions.

        ``open_file(path, line=None, column=None)`` opens one file; ``open_folder(root, **kw)``
        opens a component folder; ``prepare_copy()`` may refuse a component copy while
        unsaved buffers exist. ``make_default`` selects this provider as the user's
        default editor (do so only when the user enabled the plugin explicitly).
        """
        service = getattr(self._window, "editor_service", None)
        if service is None:
            return
        service.register(self.plugin_id, name, open_file, open_folder, prepare_copy)
        self._editor_registered = True
        settings = getattr(self._window, "settings_manager", None)
        if make_default and settings is not None:
            settings.default_text_editor = self.plugin_id

    def unregister_editor(self) -> None:
        """Withdraw this plugin's editor provider."""
        service = getattr(self._window, "editor_service", None)
        if service is not None and self._editor_registered:
            service.unregister(self.plugin_id)
        self._editor_registered = False

    def editor_preference(self, key: str, default: Any = None) -> Any:
        """A user preference from the host's editor settings page (e.g. ``wrap_lines``)."""
        settings = getattr(self._window, "settings_manager", None)
        return default if settings is None else settings.get("editor", key, default=default)

    def set_editor_preference(self, key: str, value: Any) -> None:
        """Update a host editor preference; the host settings page shows the change."""
        settings = getattr(self._window, "settings_manager", None)
        if settings is not None:
            settings.set("editor", key, value)

    def ui_zoom(self) -> float:
        """The user's interface zoom factor."""
        settings = getattr(self._window, "settings_manager", None)
        return float(getattr(settings, "zoom", 1.0) or 1.0)

    def add_appearance_listener(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` after the theme or appearance settings change."""
        if callback in self._appearance_listeners:
            return
        for source in self._appearance_sources():
            source.changed.connect(callback)
        self._appearance_listeners.append(callback)

    def remove_appearance_listener(self, callback: Callable[[], None]) -> None:
        """Stop calling a listener added with ``add_appearance_listener``."""
        if callback not in self._appearance_listeners:
            return
        self._appearance_listeners.remove(callback)
        for source in self._appearance_sources():
            try:
                source.changed.disconnect(callback)
            except (TypeError, RuntimeError):
                pass

    def _appearance_sources(self):
        return [source for source in (getattr(self._window, "theme_manager", None),
                                      getattr(self._window, "settings_manager", None))
                if source is not None and hasattr(source, "changed")]

    def tree_toolbar_store(self):
        """Settings store for shared tree toolbars (pass as ``settings_manager`` to
        ``TreeToolbarPersistence`` widgets), or None when the host keeps no settings.
        Access is limited to the host's tree-toolbar section."""
        settings = getattr(self._window, "settings_manager", None)
        return None if settings is None else _SectionStore(settings, "filetree")

    # ── Component Hub ────────────────────────────────────────────────────────

    def hub_components(self) -> List[Dict[str, Any]]:
        """Hub index entries (``id``, ``name``, ``kind``, ``tags``, ``description``).

        Uses the loaded index, else the last cached copy; empty when neither exists.
        """
        client = getattr(self._window, "hub_client", None)
        if client is None:
            return []
        records = [{"id": c.id, "name": c.name, "kind": c.kind, "tags": list(c.tags or []),
                    "description": c.description} for c in (getattr(client, "components", None) or [])]
        cache_file = getattr(client, "cache_file", None)
        if not records and cache_file and Path(cache_file).is_file():
            try:
                data = json.loads(Path(cache_file).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            records = [dict(item) for item in (data.get("components", []) if isinstance(data, dict) else [])
                       if isinstance(item, dict) and item.get("id")]
        return records

    def add_hub_index_listener(self, callback: Callable[[], None]) -> None:
        """Call ``callback()`` whenever the Hub index finishes loading."""
        client = getattr(self._window, "hub_client", None)
        if client is None or not hasattr(client, "indexLoaded") or callback in self._hub_listeners:
            return
        relay = lambda *_args: callback()  # noqa: E731 - disconnected in remove/cleanup
        client.indexLoaded.connect(relay)
        self._hub_listeners[callback] = relay

    def remove_hub_index_listener(self, callback: Callable[[], None]) -> None:
        """Stop calling a listener added with ``add_hub_index_listener``."""
        relay = self._hub_listeners.pop(callback, None)
        client = getattr(self._window, "hub_client", None)
        if relay is not None and client is not None:
            try:
                client.indexLoaded.disconnect(relay)
            except (TypeError, RuntimeError):
                pass

    def is_managed_file(self, path: Union[str, Path]) -> bool:
        """True for files of an installed (non-user) package; edit a user override instead."""
        client = getattr(self._window, "hub_client", None)
        return bool(client is not None and client.installer.is_installed_file(path))

    def config_catalog(self, force_reload: bool = False):
        """The host's parsed configuration catalog for the selected workspace, or None."""
        from ..config_panel.catalogs import config_tree
        return config_tree(force_reload)

    def notify_config_files_changed(self) -> None:
        """Tell the host that configuration files changed on disk, so catalogs reload."""
        from ..config_panel.catalogs import invalidate_config_tree
        invalidate_config_tree()

    # ── Experiment State & Event Subscriptions ───────────────────────────────

    def get_current_experiment(self) -> Optional[str]:
        """Return the currently selected experiment name, e.g. 'group/experiment'."""
        return getattr(self._window, "current_experiment", None)

    def get_active_config_path(self) -> Optional[Path]:
        """Return the Path to the currently active config file, if any."""
        path = getattr(self._window, "active_config_path", None)
        return Path(path) if path else None

    def add_experiment_change_listener(
        self, callback: Callable[[Optional[str], Optional[str]], None]
    ) -> None:
        """Register a callback for when the active experiment config changes.
        
        Callback signature: callback(experiment_name: str | None, file_path: str | None)
        """
        if callback not in self._experiment_listeners:
            self._experiment_listeners.append(callback)

    def remove_experiment_change_listener(
        self, callback: Callable[[Optional[str], Optional[str]], None]
    ) -> None:
        """Unregister an experiment change listener."""
        if callback in self._experiment_listeners:
            self._experiment_listeners.remove(callback)

    def notify_experiment_changed(
        self, exp_name: Optional[str], file_path: Optional[str]
    ) -> None:
        """Internal dispatch called by Window when user selects a new config."""
        for listener in list(self._experiment_listeners):
            try:
                listener(exp_name, file_path)
            except Exception as exc:
                self.log(f"Error in experiment change listener: {exc}")

    # ── Plugin-Scoped Persistence ────────────────────────────────────────────

    def _read_storage(self) -> Dict[str, Any]:
        if self._storage_cache is not None:
            return self._storage_cache
        if not self._storage_file.exists():
            self._storage_cache = {}
            return self._storage_cache
        try:
            self._storage_cache = json.loads(self._storage_file.read_text(encoding="utf-8"))
        except Exception:
            self._storage_cache = {}
        return self._storage_cache

    def _write_storage(self, data: Dict[str, Any]) -> None:
        self._storage_cache = data
        try:
            self._storage_file.parent.mkdir(parents=True, exist_ok=True)
            self._storage_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self.log(f"Failed to save plugin storage: {exc}")

    def get_setting(self, key: str, default: Any = None) -> Any:
        """Retrieve a saved setting value for this plugin."""
        data = self._read_storage()
        return data.get(key, default)

    def set_setting(self, key: str, value: Any) -> None:
        """Save a setting value for this plugin."""
        data = self._read_storage()
        data[key] = value
        self._write_storage(data)

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Remove any remaining tabs, slots, commands, and safely terminate tracked processes."""
        try:
            import psutil
        except ImportError:
            psutil = None

        for proc in list(self._tracked_processes):
            if proc.state() == QProcess.ProcessState.NotRunning:
                continue
            pid = int(proc.processId())
            if pid > 0 and psutil is not None:
                try:
                    parent = psutil.Process(pid)
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except Exception:
                            pass
                    try:
                        parent.kill()
                    except Exception:
                        pass
                except Exception:
                    pass
            elif pid > 0:
                try:
                    import signal
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
            try:
                if proc.state() != QProcess.ProcessState.NotRunning:
                    proc.kill()
                proc.waitForFinished(3000)
            except Exception as exc:
                self.log(f"Error terminating child process via Qt: {exc}")
        self._tracked_processes.clear()

        # 2. Unregister UI tabs
        for tab_id in list(self._registered_tabs):
            self.remove_sidebar_tab(tab_id)

        # 3. Remove contributed slot actions
        for slot_id, action_id in list(self._contributed_actions):
            self.remove_from_slot(slot_id, action_id)

        # 4. Remove registered commands
        for cmd_id in list(self._registered_commands):
            self.unregister_command(cmd_id)

        # 5. Withdraw editor providers and appearance listeners
        self.unregister_editor()
        for callback in list(self._appearance_listeners):
            self.remove_appearance_listener(callback)
        for callback in list(self._hub_listeners):
            self.remove_hub_index_listener(callback)

        # 6. Clear listeners and caches
        self._experiment_listeners.clear()
        self._storage_cache = None
