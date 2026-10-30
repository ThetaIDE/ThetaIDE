"""Unified settings management for Theta-IDE.

A single ``settings.toml`` file is the canonical source of truth for all user
preferences (theme, sidebar layout, enabled plugins, backend URL, etc.).

Two locations are merged in priority order:
  1. User global: ``~/.config/thetaide/settings.toml``
  2. Workspace:   ``<data_dir>/settings.toml``

Workspace values override user-global values for every key they define.

The file is watched with ``QFileSystemWatcher`` so that editing it in an
external editor (Neovim, nano, VS Code …) applies changes live without a
restart.

GUI interactions that mutate a preference call :meth:`SettingsManager.set`,
which writes the change back into the workspace ``settings.toml`` and emits
:attr:`changed`, which in turn the rest of the app listens to.
"""
from __future__ import annotations

import copy
import re
import time
import tomllib
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QFileSystemWatcher, QObject, pyqtSignal


# ---------------------------------------------------------------------------
# Default settings — these are written verbatim into a fresh settings.toml
# (with the comments intact) on first run.
# ---------------------------------------------------------------------------

_DEFAULT_TOML = """\
# =============================================================================
# Theta-IDE Settings  (settings.toml)
# =============================================================================
# Edit this file directly in your favourite text editor; Theta-IDE will pick up
# changes automatically without restarting.
#
# Priority order (workspace overrides user-global):
#   ~/.config/thetaide/settings.toml   — user-global defaults
#   <workspace>/.thetaide/settings.toml — this file (workspace overrides)

[appearance]
# Active color theme.  Built-in choices:
#   "Gruvbox Dark", "Gruvbox Light", "Nord", "Dracula", "Catppuccin",
#   "Catppuccin Macchiato", "Catppuccin Latte", "Paper",
#   "Apollo", "Athena", "Ares", "Dionysus", "Poseidon"
theme = "Catppuccin"

# Use the Theta logo instead of the gear on the Settings sidebar button.
settings_button_logo = false

# Display zoom factor (1.0 = 100%). Use Cmd++ / Cmd+- to resize.
zoom = 1.0

# Show all resolved group defaults in the experiment configuration panel.
# When disabled (false), only explicit overrides and configured methods are shown.
show_resolved_defaults = false

# Show default hyperparameters for algorithms in experiment methods
show_method_agent_defaults = false

# Show default architecture parameters for models in experiment methods
show_method_model_defaults = false

# Show all default parameters across methods and experiments
show_all_default_params = false

[sidebar]
# Canonical panel IDs, in left-to-right display order.
order = [
    "components",
    "config",
    "workflows",
    "monitor",
    "results",
    "plots",
    "tensorboard",
    "terminal",
    "console",
]

# Subset of order that is shown by default.  Hidden panels are still available
# via View → <Panel Name>.
visible = [
    "components",
    "config",
    "results",
    "terminal",
    "console",
]

[plugins]
# List of plugin IDs that are currently enabled.
enabled = []

[components]
# Prompt for confirmation when deleting a component from the components panel.
confirm_delete = true

[hotkeys]
# Enable custom keyboard shortcuts.
enabled = true

# Primary action key (leader / chord modifier).
# Press or hold this key, then press a number key to switch panes.
# Default: "ctrl+b" (tmux-style prefix). Also supports: "caps_lock", "alt", "ctrl", "meta", "ctrl+tab", etc.
action_key = "ctrl+b"

# Leader timeout in seconds for sequential presses (e.g. tap Action key, then press number).
leader_timeout = 1.5

# When true, the Terminal pane takes precedence over IDE action hotkeys so tmux
# and shell keybindings (e.g. tmux prefix, ctrl/caps shortcuts) work uninterrupted.
terminal_precedence = true

# Pane navigation shortcuts (Action key + number):
# Maps numbers [0, 1, 2, ...] to pane IDs.
#   0 -> "settings"
#   1 -> "components"
#   2 -> "config" (experiments)
#   3 -> "monitor"
#   4 -> "results"
#   5 -> "plots"
#   6 -> "tensorboard"
#   7 -> "terminal"
#   8 -> "console"
panes = [
    "settings",
    "components",
    "config",
    "monitor",
    "results",
    "plots",
    "tensorboard",
    "terminal",
    "console",
]

[terminal]
# Shell for the Terminal pane, as a command line. Empty = detect automatically
# (Windows: PowerShell 7, then Windows PowerShell, then cmd; Linux/macOS: your login shell).
# Examples: "pwsh -NoLogo", "cmd.exe", "C:/Program Files/Git/bin/bash.exe --login", "/bin/zsh -l"
shell = ""

# Folder new shells start in. Empty = the folder ThetaIDE was launched from.
cwd = ""

# Put the backend venv (venv/) first on PATH, so python, pytest and run_pipeline.py
# use the project's environment without activating it by hand.
activate_venv = true

[shortcuts]
# Application-wide keyboard shortcuts (customizable in Settings -> Hotkeys).
zoom_in = "Ctrl++"
zoom_out = "Ctrl+-"
reset_zoom = "Ctrl+0"
open_launcher = "Ctrl+Space"
open_docs = "F1"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict that is *override* merged on top of *base* (deep)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _parse_toml_file(path: Path) -> dict:
    """Read and parse a TOML file; return an empty dict on any error."""
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def _write_toml_file(path: Path, data: dict) -> None:
    """Serialise *data* as minimal TOML and write it atomically.

    This is a best-effort writer that handles the flat-to-two-level structure
    used by Theta-IDE settings.  It does **not** preserve user comments (that
    would require ``tomlkit``), but it never corrupts the file.
    """
    lines: list[str] = []
    for section, value in data.items():
        if not isinstance(value, dict):
            # Top-level scalar (unusual but tolerated)
            lines.append(f"{section} = {_toml_value(value)}")
        else:
            scalars = {k: v for k, v in value.items() if not isinstance(v, dict)}
            subdicts = {k: v for k, v in value.items() if isinstance(v, dict)}
            if scalars or not subdicts:
                lines.append(f"\n[{section}]")
                for k, v in scalars.items():
                    lines.append(f"{k} = {_toml_value(v)}")
            for sub, subval in subdicts.items():
                lines.append(f"\n[{section}.{sub}]")
                for k, v in subval.items():
                    lines.append(f"{k} = {_toml_value(v)}")
    content = "\n".join(lines).lstrip("\n") + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(content, encoding="utf-8")
    # On Windows, replacing a file another process briefly holds open (antivirus, the search indexer, a
    # second ThetaIDE watching it) fails with "Access is denied"; it clears within moments, so retry.
    for attempt in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025 * (attempt + 1))


def _toml_value(v: Any) -> str:
    """Convert a Python value to its TOML literal representation."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        for char, code in (("\n", "\\n"), ("\r", "\\r"), ("\t", "\\t")):
            escaped = escaped.replace(char, code)
        return '"' + escaped + '"'
    if isinstance(v, list):
        items = ", ".join(_toml_value(i) for i in v)
        return f"[{items}]"
    if isinstance(v, dict):
        # Inline table, e.g. a sidebar bin: { name = "Training", members = ["monitor"], expanded = true }
        items = ", ".join(f"{_toml_key(k)} = {_toml_value(val)}" for k, val in v.items())
        return "{ " + items + " }" if items else "{}"
    return repr(v)  # fallback


def _toml_key(key: Any) -> str:
    key = str(key)
    return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else _toml_value(key)


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------

class SettingsManager(QObject):
    """Single source of truth for all Theta-IDE user preferences.

    Signals
    -------
    changed
        Emitted whenever any preference changes, whether caused by an in-app
        GUI action or a live file-system edit.
    """

    changed = pyqtSignal()

    def __init__(self, data_dir: Path, parent: QObject | None = None) -> None:
        super().__init__(parent)

        # Canonical workspace settings file sits one level above data_dir
        # (e.g. data_dir=.thetaide/runs  →  .thetaide/settings.toml),
        # but stays inside data_dir for custom or temporary directories.
        p = Path(data_dir)
        if p.name == "runs" or p.parent.name == ".thetaide":
            self._workspace_path = p.parent / "settings.toml"
        else:
            self._workspace_path = p / "settings.toml"
        # User-global settings file (lower priority)
        self._user_path = Path.home() / ".config" / "thetaide" / "settings.toml"

        # Merged, in-memory view of all settings
        self._data: dict = {}

        # Initialise defaults then load from disk
        self._bootstrap_workspace_file()
        self._reload()

        # Watch both files for live edits
        self._watcher = QFileSystemWatcher(self)
        self._watcher.fileChanged.connect(self._on_file_changed)
        self._watch_files()

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    def get(self, *keys: str, default: Any = None) -> Any:
        """Return a (possibly nested) value by dot-path keys.

        Example::

            mgr.get("appearance", "theme")
        """
        node: Any = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, None)
            if node is None:
                return default
        return node

    def section(self, key: str) -> dict:
        """Return a whole top-level section as a dict (or empty dict)."""
        return dict(self._data.get(key, {}))

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def set(self, *keys_and_value: Any) -> None:
        """Set a (possibly nested) preference and persist to the workspace file.

        The last positional argument is the value; all preceding arguments are
        the key path.

        Example::

            mgr.set("appearance", "theme", "Nord")
            mgr.set("sidebar", "visible", ["components", "config"])
        """
        if len(keys_and_value) < 2:  # noqa: PLR2004
            raise ValueError("set() requires at least one key and a value.")
        *keys, value = keys_and_value

        # Update the in-memory merged view
        node = self._data
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value

        # Persist to the workspace file (load → patch → write)
        on_disk = _parse_toml_file(self._workspace_path)
        disk_node = on_disk
        for k in keys[:-1]:
            disk_node = disk_node.setdefault(k, {})
        disk_node[keys[-1]] = value

        # Temporarily pause the watcher so our own write doesn't re-trigger reload
        self._watcher.removePath(str(self._workspace_path))
        try:
            _write_toml_file(self._workspace_path, on_disk)
        finally:
            self._watcher.addPath(str(self._workspace_path))

        self.changed.emit()

    def persist_key(self, section: str, key: str, value: Any) -> None:
        """Persist a single preference to disk without emitting the ``changed`` signal.

        Useful for debounced settings (like live display scaling) where the UI has
        already applied the change and re-evaluating the full settings cascade is unnecessary.
        """
        node = self._data.setdefault(section, {})
        node[key] = value

        on_disk = _parse_toml_file(self._workspace_path)
        disk_node = on_disk.setdefault(section, {})
        if disk_node.get(key) != value:
            disk_node[key] = value
            self._watcher.removePath(str(self._workspace_path))
            try:
                _write_toml_file(self._workspace_path, on_disk)
            finally:
                self._watcher.addPath(str(self._workspace_path))

    def persist_many(self, section: str, values: dict[str, Any]) -> None:
        """Persist internal state values together without emitting ``changed``."""
        if not values:
            return
        self._data.setdefault(section, {}).update(values)
        on_disk = _parse_toml_file(self._workspace_path)
        disk_section = on_disk.setdefault(section, {})
        changed = any(disk_section.get(key) != value for key, value in values.items())
        if not changed:
            return
        disk_section.update(values)
        self._watcher.removePath(str(self._workspace_path))
        try:
            _write_toml_file(self._workspace_path, on_disk)
        finally:
            self._watcher.addPath(str(self._workspace_path))

    # ------------------------------------------------------------------
    # Convenience helpers for the most-touched sections
    # ------------------------------------------------------------------

    @property
    def theme(self) -> str:
        return self.get("appearance", "theme", default="Catppuccin")

    @property
    def settings_button_logo(self) -> bool:
        return bool(self.get("appearance", "settings_button_logo", default=False))

    @property
    def zoom(self) -> float:
        try:
            return float(self.get("appearance", "zoom", default=1.0))
        except (ValueError, TypeError):
            return 1.0

    @zoom.setter
    def zoom(self, val: float) -> None:
        self.set("appearance", "zoom", round(float(val), 2))

    @property
    def show_resolved_defaults(self) -> bool:
        return bool(self.get("appearance", "show_resolved_defaults", default=False))

    @show_resolved_defaults.setter
    def show_resolved_defaults(self, val: bool) -> None:
        self.set("appearance", "show_resolved_defaults", bool(val))

    @property
    def visual_layer_editor(self) -> bool:
        return bool(self.get("appearance", "visual_layer_editor", default=True))

    @visual_layer_editor.setter
    def visual_layer_editor(self, val: bool) -> None:
        self.set("appearance", "visual_layer_editor", bool(val))

    @property
    def show_method_agent_defaults(self) -> bool:
        return bool(self.get("appearance", "show_method_agent_defaults", default=False))

    @show_method_agent_defaults.setter
    def show_method_agent_defaults(self, val: bool) -> None:
        self.set("appearance", "show_method_agent_defaults", bool(val))

    @property
    def show_method_model_defaults(self) -> bool:
        return bool(self.get("appearance", "show_method_model_defaults", default=False))

    @show_method_model_defaults.setter
    def show_method_model_defaults(self, val: bool) -> None:
        self.set("appearance", "show_method_model_defaults", bool(val))

    @property
    def show_all_default_params(self) -> bool:
        return bool(self.get("appearance", "show_all_default_params", default=False))

    @show_all_default_params.setter
    def show_all_default_params(self, val: bool) -> None:
        self.set("appearance", "show_all_default_params", bool(val))

    @property
    def sidebar_order(self) -> list[str]:
        v = self.get("sidebar", "order", default=[])
        return list(v) if isinstance(v, list) else []

    @property
    def sidebar_visible(self) -> list[str]:
        v = self.get("sidebar", "visible", default=[])
        return list(v) if isinstance(v, list) else []

    @property
    def plugins_enabled(self) -> list[str]:
        v = self.get("plugins", "enabled", default=[])
        return list(v) if isinstance(v, list) else []

    @property
    def confirm_delete_component(self) -> bool:
        return bool(self.get("components", "confirm_delete", default=True))

    @confirm_delete_component.setter
    def confirm_delete_component(self, val: bool) -> None:
        self.set("components", "confirm_delete", bool(val))

    @property
    def hotkeys_enabled(self) -> bool:
        return bool(self.get("hotkeys", "enabled", default=True))

    @property
    def action_key(self) -> str:
        return str(self.get("hotkeys", "action_key", default="ctrl+b"))

    @property
    def hotkeys_leader_timeout(self) -> float:
        try:
            return float(self.get("hotkeys", "leader_timeout", default=1.5))
        except (ValueError, TypeError):
            return 1.5

    @property
    def hotkeys_terminal_precedence(self) -> bool:
        return bool(self.get("hotkeys", "terminal_precedence", default=True))

    @property
    def terminal_shell(self) -> str:
        return str(self.get("terminal", "shell", default="") or "").strip()

    @property
    def terminal_cwd(self) -> str:
        return str(self.get("terminal", "cwd", default="") or "").strip()

    @property
    def terminal_activate_venv(self) -> bool:
        return bool(self.get("terminal", "activate_venv", default=True))

    @property
    def hotkey_bindings(self) -> dict[str, str]:
        v = self.get("hotkeys", "bindings", default=None)
        if isinstance(v, dict):
            return dict(v)
        # Default bindings derived from hotkey_panes
        panes = self.hotkey_panes
        return {pid: f"Action + {i}" for i, pid in enumerate(panes)}

    @property
    def hotkey_panes(self) -> list[str]:
        v = self.get("hotkeys", "panes", default=None)
        if isinstance(v, list) and v:
            return list(v)
        # Default order: settings, followed by sidebar panels in order
        return [
            "settings",
            "components",
            "config",
            "monitor",
            "results",
            "plots",
            "tensorboard",
            "terminal",
            "console",
        ]

    @property
    def shortcuts(self) -> dict[str, str]:
        v = self.get("shortcuts", default=None)
        return dict(v) if isinstance(v, dict) else {}

    @property
    def workspace_settings_path(self) -> Path:
        """Absolute path to the workspace ``settings.toml``."""
        return self._workspace_path


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bootstrap_workspace_file(self) -> None:
        """Write the default settings.toml if it doesn't exist yet."""
        if not self._workspace_path.exists():
            self._workspace_path.parent.mkdir(parents=True, exist_ok=True)
            self._workspace_path.write_text(_DEFAULT_TOML, encoding="utf-8")

    def _reload(self) -> None:
        """Merge user-global and workspace files into the in-memory view."""
        user_data = _parse_toml_file(self._user_path)
        workspace_data = _parse_toml_file(self._workspace_path)
        self._data = _deep_merge(user_data, workspace_data)

    def _watch_files(self) -> None:
        for p in (self._user_path, self._workspace_path):
            if p.exists():
                self._watcher.addPath(str(p))

    def _on_file_changed(self, _path: str) -> None:
        """Called by QFileSystemWatcher when any watched settings file changes."""
        # Some editors (Vim, Neovim) replace the file via rename; re-watch.
        self._watcher.removePath(str(self._workspace_path))
        self._watcher.removePath(str(self._user_path))
        old_data = copy.deepcopy(self._data)
        self._reload()
        self._watch_files()
        if old_data != self._data:
            self.changed.emit()
