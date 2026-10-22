"""Plugin discovery, lifecycle management, and persistence for ThetaIDE."""
from __future__ import annotations
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from .base import Plugin, PluginManifest
from .context import PluginContext
from .dependencies import check_missing_dependencies, load_requirements_from_file

if TYPE_CHECKING:
    from ..app import Window
    from ..settings import SettingsManager


class PluginManager(QObject):
    """Manages discovery, activation, deactivation, and settings for plugins."""

    pluginStateChanged = pyqtSignal(str, bool)  # (plugin_id, enabled)

    def __init__(self, window: Window, plugin_dirs: Optional[List[Path]] = None,
                 settings_manager: Optional["SettingsManager"] = None):
        super().__init__()
        self.window = window
        self.plugin_dirs = plugin_dirs or [Path(__file__).parent / "core", Path(__file__).parent]
        self._settings = settings_manager

        # Legacy fallback file — used only when settings_manager is unavailable.
        store = getattr(window, "store", None)
        root_dir = getattr(store, "root", getattr(store, "data_dir", Path("."))) if store else Path(".")
        self._legacy_state_file = Path(root_dir) / ".plugins.json"

        self.manifests: Dict[str, PluginManifest] = {}
        self.instances: Dict[str, Plugin] = {}
        self.contexts: Dict[str, PluginContext] = {}
        self.enabled_states: Dict[str, bool] = {}
        self._states_loaded: bool = False

    # ── Discovery ────────────────────────────────────────────────────────────

    def discover(self) -> None:
        """Scan configured plugin directories for plugin.json manifests.

        Installed plugins are defined by their filesystem presence.
        Plugins found on disk are registered. If a plugin is removed from disk,
        its stale enabled state is pruned automatically.
        """
        if not self._states_loaded:
            self._load_states()
            self._states_loaded = True

        self.manifests.clear()
        discovered_paths = set()
        for pdir in self.plugin_dirs:
            if not pdir.exists():
                continue
            candidates = list(pdir.glob("*/plugin.json"))
            if (pdir / "core").is_dir():
                candidates.extend((pdir / "core").glob("*/plugin.json"))

            for manifest_path in candidates:
                resolved = manifest_path.resolve()
                if resolved in discovered_paths:
                    continue
                discovered_paths.add(resolved)
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if data.get("kind", "plugin") != "plugin":
                        continue
                    pid = data["id"]
                    is_core = bool(
                        data.get("core", False)
                        or data.get("is_core", False)
                        or "core" in manifest_path.parts
                        or manifest_path.is_relative_to(Path(__file__).parent)
                    )
                    manifest = PluginManifest(
                        id=pid,
                        name=data.get("name", pid),
                        version=data.get("version", "0.1.0"),
                        description=data.get("description", ""),
                        author=data.get("author", "ThetaIDE Team"),
                        default_enabled=data.get("default_enabled", False),
                        icon=data.get("icon"),
                        entry_point=data.get("entry_point", "Plugin"),
                        plugin_dir=manifest_path.parent,
                        is_core=is_core,
                        extra=data,
                    )
                    self.manifests[pid] = manifest
                except Exception as exc:
                    self.window.log(f"Failed to load plugin manifest at {manifest_path}: {exc}")

        # Drop enabled state for any plugin that no longer exists on disk
        installed = set(self.manifests.keys())
        for pid in list(self.enabled_states.keys()):
            if pid not in installed:
                del self.enabled_states[pid]

        # Plugins discovered for the first time default to their manifest setting
        for pid, manifest in self.manifests.items():
            if pid not in self.enabled_states:
                self.enabled_states[pid] = bool(manifest.default_enabled)

    # ── State persistence ────────────────────────────────────────────────────

    def _load_states(self) -> None:
        """Load enabled plugin state from settings.toml (preferred)
        or legacy .plugins.json (fallback when no SettingsManager)."""
        if self._settings is not None:
            enabled_list = self._settings.plugins_enabled
            disabled_list = self._settings.get("plugins", "disabled", default=[])
            self.enabled_states = {pid: True for pid in enabled_list}
            for pid in disabled_list:
                self.enabled_states[pid] = False
            self._migrate_legacy_states()
        else:
            self._load_legacy_states()

    def _load_legacy_states(self) -> None:
        """Read enabled state from legacy .plugins.json."""
        saved: dict = {}
        if self._legacy_state_file.exists():
            try:
                saved = json.loads(self._legacy_state_file.read_text(encoding="utf-8"))
            except Exception as exc:
                self.window.log(f"Failed to read plugin state file: {exc}")
        saved_enabled = saved.get("enabled", saved) if isinstance(saved, dict) else {}
        self.enabled_states = {
            k: bool(v) for k, v in saved_enabled.items()
            if k not in ("uninstalled", "enabled")
        }

    def _migrate_legacy_states(self) -> None:
        """One-time migration: pull enabled states from .plugins.json into settings.toml."""
        if not self._legacy_state_file.exists():
            return
        try:
            saved = json.loads(self._legacy_state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        saved_enabled = saved.get("enabled", saved) if isinstance(saved, dict) else {}
        migrated = False
        for k, v in saved_enabled.items():
            if k not in ("uninstalled", "enabled") and k not in self.enabled_states:
                self.enabled_states[k] = bool(v)
                migrated = True
        if migrated:
            self.save_states()

    def save_states(self) -> None:
        """Persist enabled plugin state to settings.toml."""
        enabled_list = sorted(pid for pid, on in self.enabled_states.items() if on)
        disabled_list = sorted(pid for pid, on in self.enabled_states.items() if not on)
        if self._settings is not None:
            try:
                self._settings.set("plugins", "enabled", enabled_list)
                self._settings.set("plugins", "disabled", disabled_list)
            except Exception as exc:
                self.window.log(f"Failed to save plugin states to settings.toml: {exc}")
        else:
            try:
                self._legacy_state_file.parent.mkdir(parents=True, exist_ok=True)
                payload = {"enabled": dict(self.enabled_states)}
                self._legacy_state_file.write_text(
                    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
                )
            except OSError as exc:
                self.window.log(f"Failed to save plugin states: {exc}")

    # ── Install / Uninstall ──────────────────────────────────────────────────

    def uninstall_plugin(self, plugin_id: str) -> None:
        """Deactivate a plugin, remove its directory from disk, and update states."""
        manifest = self.manifests.get(plugin_id)
        if manifest and getattr(manifest, "is_core", False):
            self.window.log(f"Cannot uninstall core plugin: {plugin_id}")
            return

        self.disable_plugin(plugin_id)
        manifest = self.manifests.pop(plugin_id, None)
        self.enabled_states.pop(plugin_id, None)

        plugin_dir = manifest.plugin_dir if manifest else None
        if plugin_dir and plugin_dir.exists():
            import shutil
            shutil.rmtree(plugin_dir, ignore_errors=True)
        else:
            store = getattr(self.window, "store", None)
            root_dir = (
                getattr(store, "root", getattr(store, "data_dir", None)) if store
                else getattr(self.window, "data_dir", None)
            )
            if root_dir:
                import shutil
                target = Path(root_dir) / "plugins" / plugin_id
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)

        self.save_states()

    def mark_uninstalled(self, plugin_id: str) -> None:
        """Deprecated alias: use uninstall_plugin() instead."""
        self.uninstall_plugin(plugin_id)

    def unmark_uninstalled(self, plugin_id: str) -> None:
        """Deprecated no-op: installation is tracked by filesystem presence."""
        pass

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def initialize_plugins(self) -> None:
        """Activate plugins that are currently enabled in the saved state."""
        for pid, is_enabled in list(self.enabled_states.items()):
            if is_enabled:
                self.enable_plugin(pid)

    def is_plugin_enabled(self, plugin_id: str) -> bool:
        """Check if a plugin is currently enabled."""
        return self.enabled_states.get(plugin_id, False)

    def is_plugin_active(self, plugin_id: str) -> bool:
        """Check if a plugin successfully activated and contributed its runtime features."""
        return plugin_id in self.instances

    def enable_plugin(self, plugin_id: str) -> bool:
        """Dynamically activate and mount a plugin."""
        if plugin_id not in self.manifests:
            self.window.log(f"Cannot enable unknown plugin: {plugin_id}")
            return False

        if plugin_id in self.instances:
            return True  # Already active

        manifest = self.manifests[plugin_id]
        plugin_dir = manifest.plugin_dir

        # Verify declared dependencies before loading Python module
        declared_deps = []
        if isinstance(manifest.extra, dict):
            declared_deps.extend(manifest.extra.get("dependencies", {}).get("pip", []))
        if plugin_dir:
            req_file = plugin_dir / "requirements.txt"
            if req_file.exists():
                declared_deps.extend(load_requirements_from_file(req_file))

        missing = check_missing_dependencies(declared_deps)
        if missing:
            self.window.log(
                f"Cannot activate plugin '{manifest.name}' ({plugin_id}): "
                f"Missing Python dependencies: {', '.join(missing)}. "
                f"Please install them via pip or the Theta Hub."
            )
            return False

        try:
            # Load the plugin module.
            try:
                module = importlib.import_module(f"frontend.plugins.{plugin_id}")
            except (ImportError, ModuleNotFoundError):
                try:
                    module = importlib.import_module(f"frontend.plugins.core.{plugin_id}")
                except (ImportError, ModuleNotFoundError):
                    init_file = plugin_dir / "__init__.py" if plugin_dir else None
                    if init_file and init_file.exists():
                        mod_name = f"theta_plugin_{plugin_id}"
                        spec = importlib.util.spec_from_file_location(
                            mod_name, init_file, submodule_search_locations=[str(plugin_dir)]
                        )
                        if spec is None or spec.loader is None:
                            raise ImportError(f"Cannot load spec from {init_file}")
                        module = importlib.util.module_from_spec(spec)
                        module.__path__ = [str(plugin_dir)]
                        sys.modules[mod_name] = module
                        sys.modules[f"frontend.plugins.{plugin_id}"] = module
                        spec.loader.exec_module(module)
                    else:
                        raise

            plugin_cls = getattr(module, manifest.entry_point)
            instance: Plugin = plugin_cls(manifest)

            store = getattr(self.window, "store", None)
            root_dir = (
                getattr(store, "root", getattr(store, "data_dir", ".")) if store else "."
            )
            storage_file = Path(root_dir) / "plugins" / f"{plugin_id}.json"
            context = PluginContext(plugin_id, self.window, storage_file)

            instance.activate(context)

            self.instances[plugin_id] = instance
            self.contexts[plugin_id] = context
            self.enabled_states[plugin_id] = True
            self.save_states()
            self.pluginStateChanged.emit(plugin_id, True)
            self.window.log(f"Plugin activated: {manifest.name} ({plugin_id})")
            return True

        except Exception as exc:
            self.window.log(f"Failed to activate plugin {plugin_id}: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Dynamically deactivate and unmount a plugin."""
        if plugin_id not in self.manifests:
            return False

        instance = self.instances.pop(plugin_id, None)
        context = self.contexts.pop(plugin_id, None)

        if instance:
            try:
                instance.deactivate()
            except Exception as exc:
                self.window.log(f"Error during deactivation of plugin {plugin_id}: {exc}")

        if context:
            try:
                context.cleanup()
            except Exception as exc:
                self.window.log(f"Error cleaning up context for {plugin_id}: {exc}")

        self.enabled_states[plugin_id] = False
        self.save_states()
        self.pluginStateChanged.emit(plugin_id, False)
        self.window.log(f"Plugin deactivated: {self.manifests[plugin_id].name} ({plugin_id})")
        return True

    def shutdown(self) -> None:
        """Safely deactivate all running plugins on application exit without changing saved enabled states."""
        for pid in list(self.instances.keys()):
            instance = self.instances.pop(pid, None)
            context = self.contexts.pop(pid, None)
            if instance:
                try:
                    instance.deactivate()
                except Exception as exc:
                    self.window.log(f"Error during shutdown deactivation of plugin {pid}: {exc}")
            if context:
                try:
                    context.cleanup()
                except Exception as exc:
                    self.window.log(f"Error cleaning up context during shutdown for {pid}: {exc}")
        self.instances.clear()
        self.contexts.clear()

    # ── Events ───────────────────────────────────────────────────────────────

    def notify_experiment_changed(
        self, exp_name: Optional[str], file_path: Optional[str]
    ) -> None:
        """Dispatch experiment changes to all active plugin contexts."""
        for context in self.contexts.values():
            context.notify_experiment_changed(exp_name, file_path)
