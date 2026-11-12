"""Hotkey and action-key manager for Theta-IDE.

Supports two pane-switching modes for any configured action key:

  - **Leader / modal**: tap Action key, then press a digit [0-9] within
    ``leader_timeout`` seconds.  Displayed in the status bar while active.
  - **Chorded**: hold Action key and press a digit simultaneously.

Default action key: Ctrl+B (tmux-style prefix).

    Action + 0 → Settings
    Action + 1 → Components
    Action + 2 → Experiment (config)
    Action + 3 → Training monitor
    Action + 4 → Results browser
    Action + 5 → Plot viewer
    Action + 6 → TensorBoard
    Action + 7 → Job queue
    Action + 8 → Terminal
    Action + 9 → Console

Configure the action key, leader timeout, terminal precedence, and pane
mappings in settings.toml under [hotkeys].
"""
from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QWidget

if TYPE_CHECKING:
    from .app import Window
    from .settings import SettingsManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pane name aliases
# ---------------------------------------------------------------------------

PANE_ALIASES: dict[str, str] = {
    "settings": "settings",
    "settings_about": "settings",  # the combined pane's old name, still accepted in settings.toml
    "preferences": "settings",
    "components": "components",
    "config": "config",
    "experiment": "config",
    "experiments": "config",
    "builder": "config",
    "monitor": "monitor",
    "training_monitor": "monitor",
    "results": "results",
    "results_browser": "results",
    "history": "results",
    "plots": "plots",
    "plot": "plots",
    "plot_viewer": "plots",
    "tensorboard": "tensorboard",
    "tb": "tensorboard",
    "queue": "queue",
    "job_queue": "queue",
    "terminal": "terminal",
    "term": "terminal",
    "console": "console",
    "documentation": "documentation",
    "docs": "documentation",
    "doc": "documentation",
}

# ---------------------------------------------------------------------------
# Key codes that are pure modifier keys.
# When the action key is one of these, matching is done by key code alone
# (the OS sets the key as its own modifier, so requiring an *additional*
# modifier would never fire).
# ---------------------------------------------------------------------------

_MODIFIER_KEY_CODES: frozenset[Qt.Key] = frozenset({
    Qt.Key.Key_Control,
    Qt.Key.Key_Shift,
    Qt.Key.Key_Alt,
    Qt.Key.Key_Meta,
    Qt.Key.Key_AltGr,
    Qt.Key.Key_Super_L,
    Qt.Key.Key_Super_R,
    Qt.Key.Key_CapsLock,
})


# ---------------------------------------------------------------------------
# macOS Ctrl ↔ Meta swap helpers
#
# Qt remaps physical keys on macOS so that user-visible names stay intuitive:
#
#   Physical key  │ Qt Key code   │ Qt Modifier
#   ──────────────┼───────────────┼─────────────────
#   Control  (⌃)  │ Key_Meta      │ MetaModifier      ← "ctrl" in Theta
#   Command  (⌘)  │ Key_Control   │ ControlModifier   ← "cmd" / "meta"
#   Option   (⌥)  │ Key_Alt       │ AltModifier
#
# These helpers return the correct Qt value per-platform so that "ctrl+b"
# always means the physical Control+B the user intends (tmux default).
# ---------------------------------------------------------------------------

def get_ctrl_key() -> Qt.Key:
    """Qt key code for the physical Control key (⌃ on macOS)."""
    return Qt.Key.Key_Meta if sys.platform == "darwin" else Qt.Key.Key_Control

def get_ctrl_modifier() -> Qt.KeyboardModifier:
    """Qt modifier flag for the physical Control key."""
    return Qt.KeyboardModifier.MetaModifier if sys.platform == "darwin" else Qt.KeyboardModifier.ControlModifier

def get_meta_key() -> Qt.Key:
    """Qt key code for the physical Command/Meta key (⌘ on macOS)."""
    return Qt.Key.Key_Control if sys.platform == "darwin" else Qt.Key.Key_Meta

def get_meta_modifier() -> Qt.KeyboardModifier:
    """Qt modifier flag for the physical Command/Meta key."""
    return Qt.KeyboardModifier.ControlModifier if sys.platform == "darwin" else Qt.KeyboardModifier.MetaModifier

_ctrl_key = get_ctrl_key
_ctrl_mod = get_ctrl_modifier
_meta_key = get_meta_key
_meta_mod = get_meta_modifier


# ---------------------------------------------------------------------------
# parse_action_key
# ---------------------------------------------------------------------------

def parse_action_key(key_name: str) -> tuple[Optional[Qt.Key], Optional[Qt.KeyboardModifier]]:
    """Parse a human-readable action-key string into ``(Qt.Key, modifier | None)``.

    Names always refer to *physical* keys, regardless of platform:

      ``ctrl``        → physical Control key  (⌃ on macOS)
      ``cmd``/``meta``→ physical Command/Meta key  (⌘ on macOS)
      ``alt``         → physical Option/Alt key  (⌥ on macOS)
      ``ctrl+b``      → physical Control + B  (the tmux default)

    Qt swaps Ctrl and Meta internally on macOS; the helpers above handle
    that transparently so callers never need to think about it.
    """
    cleaned = key_name.strip().lower().replace(" ", "_")

    # Single-token lookup — always resolves to the *physical* key
    _SINGLE: dict[str, tuple[Qt.Key, Optional[Qt.KeyboardModifier]]] = {
        "caps_lock": (Qt.Key.Key_CapsLock, None),
        "capslock":  (Qt.Key.Key_CapsLock, None),
        "caps":      (Qt.Key.Key_CapsLock, None),
        "ctrl":      (get_ctrl_key(),      get_ctrl_modifier()),
        "control":   (get_ctrl_key(),      get_ctrl_modifier()),
        "alt":       (Qt.Key.Key_Alt,       Qt.KeyboardModifier.AltModifier),
        "option":    (Qt.Key.Key_Alt,       Qt.KeyboardModifier.AltModifier),
        "meta":      (get_meta_key(),      get_meta_modifier()),
        "cmd":       (get_meta_key(),      get_meta_modifier()),
        "command":   (get_meta_key(),      get_meta_modifier()),
        "super":     (get_meta_key(),      get_meta_modifier()),
        "win":       (get_meta_key(),      get_meta_modifier()),
        "shift":     (Qt.Key.Key_Shift,     Qt.KeyboardModifier.ShiftModifier),
        "space":     (Qt.Key.Key_Space,     None),
        "spacebar":  (Qt.Key.Key_Space,     None),
        "tab":       (Qt.Key.Key_Tab,       None),
        "escape":    (Qt.Key.Key_Escape,    None),
        "esc":       (Qt.Key.Key_Escape,    None),
    }
    if cleaned in _SINGLE:
        return _SINGLE[cleaned]

    # Compound "modifier+key" (e.g. "ctrl+b", "alt+space", "cmd+grave")
    if "+" in cleaned or (cleaned.count("-") >= 1 and cleaned != "-"):
        delim = "+" if "+" in cleaned else "-"
        parts = [p.strip() for p in cleaned.split(delim)]
        mod = Qt.KeyboardModifier.NoModifier
        key: Optional[Qt.Key] = None
        for p in parts:
            if p in ("ctrl", "control"):
                mod |= _ctrl_mod()
            elif p in ("alt", "option"):
                mod |= Qt.KeyboardModifier.AltModifier
            elif p in ("meta", "cmd", "command", "super", "win"):
                mod |= _meta_mod()
            elif p in ("shift",):
                mod |= Qt.KeyboardModifier.ShiftModifier
            else:
                for attr in (f"Key_{p.capitalize()}", f"Key_{p.upper()}"):
                    if hasattr(Qt.Key, attr):
                        key = getattr(Qt.Key, attr)
                        break
        if key is not None:
            return key, mod if mod != Qt.KeyboardModifier.NoModifier else None

    # Fallback: bare Qt.Key name
    for attr in (f"Key_{cleaned.capitalize()}", f"Key_{cleaned.upper()}"):
        if hasattr(Qt.Key, attr):
            return getattr(Qt.Key, attr), None

    logger.warning("Unknown action_key %r; falling back to Ctrl+B.", key_name)
    return Qt.Key.Key_B, _ctrl_mod()


# ---------------------------------------------------------------------------
# HotkeyManager
# ---------------------------------------------------------------------------

class HotkeyManager(QObject):
    """Application-wide event filter that intercepts Action+digit pane switches.

    Every configured action key is handled identically — there is no
    special-casing of specific keys.  The two activation modes are:

    1. **Leader (modal)**: tap Action key → status bar prompt appears →
       press a digit within ``leader_timeout`` seconds → switch pane.
    2. **Chord**: hold Action key, press digit before releasing → switch pane.

    Pressing Action a second time (or pressing Escape) cancels leader mode.
    Auto-repeated key events are ignored for the action trigger.
    """

    def __init__(self, window: "Window", settings_manager: "SettingsManager") -> None:
        super().__init__(window)
        self.window = window
        self.settings_manager = settings_manager

        # Leader / chord state
        self._leader_active = False
        self._action_key_held = False  # True while action key is physically down

        self._leader_timer = QTimer(self)
        self._leader_timer.setSingleShot(True)
        self._leader_timer.timeout.connect(self._on_leader_timeout)

        self.reload_settings()
        self.settings_manager.changed.connect(self.reload_settings)

        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        if hasattr(self.window, "installEventFilter"):
            self.window.installEventFilter(self)

    def reload_settings(self) -> None:
        """Re-read hotkey configuration from the settings manager."""
        self.enabled = self.settings_manager.hotkeys_enabled
        self.action_key_name = self.settings_manager.action_key
        self.action_key_code, self.action_modifier = parse_action_key(self.action_key_name)
        self.leader_timeout = self.settings_manager.hotkeys_leader_timeout
        self.terminal_precedence = self.settings_manager.hotkeys_terminal_precedence

        # Reset transient state on any settings change
        self._action_key_held = False
        self._cancel_leader()

    def cleanup(self) -> None:
        """Remove event filters cleanly on window shutdown."""
        self._cancel_leader()
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        if hasattr(self.window, "removeEventFilter"):
            self.window.removeEventFilter(self)

    # ── Terminal detection ────────────────────────────────────────────────

    def is_in_terminal(self, watched: Optional[QObject] = None) -> bool:
        """Return True if the user is currently inside the terminal pane."""
        terminal_panel = getattr(self.window, "terminal_panel", None)
        if terminal_panel is None:
            return False

        tabs = getattr(self.window, "tabs", None)
        if tabs and hasattr(tabs, "currentWidget"):
            try:
                if tabs.currentWidget() is not terminal_panel:
                    return False
                return True
            except RuntimeError:
                return False

        if watched is not None and isinstance(watched, QWidget):
            if watched is terminal_panel or terminal_panel.isAncestorOf(watched):
                return True

        app = QApplication.instance()
        if app:
            focus_w = app.focusWidget()
            if focus_w and (focus_w is terminal_panel or terminal_panel.isAncestorOf(focus_w)):
                return True

        return False

    # ── Key matching ──────────────────────────────────────────────────────

    def _is_action_trigger(self, event: QKeyEvent) -> bool:
        """Return True if *event* matches the configured action key.

        Modifier-only keys (CapsLock, Meta, Ctrl, …) are matched by key code
        alone — no separate modifier check needed, because the OS reports the
        key itself as its own modifier.

        Compound keys (e.g. Ctrl+B) require both the key code and the modifier.
        """
        key = event.key()
        mods = event.modifiers()

        # Digit keys are navigation targets, never the action trigger
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return False
        if self.action_key_code is None:
            return False
        if key != self.action_key_code:
            return False

        # Modifier-only key: key code match is sufficient
        if self.action_key_code in _MODIFIER_KEY_CODES:
            return True

        # Compound key: the required modifier must be held
        if self.action_modifier:
            return bool(mods & self.action_modifier)

        # Plain key with no modifier required (allow keypad variants)
        return not bool(mods & ~Qt.KeyboardModifier.KeypadModifier)

    def _is_modifier_key(self, key: Qt.Key) -> bool:
        """Return True if *key* is a pure modifier key (Ctrl, Shift, Alt, …)."""
        return key in _MODIFIER_KEY_CODES

    def is_in_hotkey_editor(self, watched: Optional[QObject] = None) -> bool:
        """Return True if a hotkey or shortcut editor field is currently focused or being edited."""
        app = QApplication.instance()
        focus_w = app.focusWidget() if app else None
        for w in (watched, focus_w):
            if w is not None:
                if getattr(w, "_is_hotkey_editor", False):
                    return True
                cname = w.__class__.__name__
                if cname in ("HotkeyComboEdit", "MenuShortcutEdit", "QKeySequenceEdit"):
                    return True
        return False

    # ── Event filter ──────────────────────────────────────────────────────

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        # This filter sees every event in the app (a theme switch alone sends ~100k style/polish
        # events), so bail out on anything that isn't a key event before doing any other work.
        etype = event.type()
        if etype not in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            return False
        if not self.enabled:
            return super().eventFilter(watched, event)

        # When actively recording or editing a hotkey/shortcut, do not intercept keys
        if self.is_in_hotkey_editor(watched):
            self._cancel_leader()
            self._action_key_held = False
            return super().eventFilter(watched, event)

        # When the terminal pane is active, let it handle all keys uninterrupted
        if self.terminal_precedence and self.is_in_terminal(watched):
            return super().eventFilter(watched, event)

        if etype == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            return self._handle_key_press(watched, event)
        if etype == QEvent.Type.KeyRelease and isinstance(event, QKeyEvent):
            return self._handle_key_release(watched, event)

        return super().eventFilter(watched, event)

    def _handle_key_press(self, watched: QObject, event: QKeyEvent) -> bool:
        key = event.key()

        # Shift+digit (e.g. '!' from Shift+1) must not trigger navigation
        if bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            return False

        # ── Step 1: action key pressed ────────────────────────────────────
        if self._is_action_trigger(event):
            if event.isAutoRepeat():
                return True  # swallow auto-repeat silently
            if self._leader_active:
                # Second tap → cancel leader
                self._cancel_leader()
            else:
                self._action_key_held = True
                self._activate_leader()
            return True

        # ── Step 2: pure modifier keys never disturb leader state ─────────
        if self._is_modifier_key(key):
            return False

        # ── Step 3: only proceed if action mode is active ─────────────────
        is_modifier_chord = bool(self.action_modifier and (event.modifiers() & self.action_modifier))
        if not (self._leader_active or self._action_key_held or is_modifier_chord):
            return False

        # ── Step 4: Escape cancels action mode ────────────────────────────
        if key == Qt.Key.Key_Escape and self.action_key_name not in ("escape", "esc"):
            self._action_key_held = False
            self._cancel_leader()
            return True

        # ── Step 5: target key (digit or letter) → switch pane ────────────
        pane_id: Optional[str] = None
        target_char = event.text().strip().upper()

        # 5a. Match against explicit hotkeys.bindings (e.g. "Action + 1", "Action + T")
        if target_char:
            combo_to_check = f"Action + {target_char}".upper()
            bindings = self.settings_manager.get("hotkeys", "bindings", default=None)
            if isinstance(bindings, dict):
                for pid, c in bindings.items():
                    c_clean = str(c).strip().upper()
                    if not c_clean.startswith("ACTION"):
                        c_clean = f"Action + {c_clean}".upper()
                    if c_clean == combo_to_check:
                        pane_id = PANE_ALIASES.get(pid.lower(), pid)
                        break

        # 5b. Match digit fallback via hotkeys.panes
        if not pane_id:
            num: Optional[int] = None
            if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
                num = key - Qt.Key.Key_0
            elif target_char.isdigit() and len(target_char) == 1:
                num = int(target_char)
            if num is not None:
                pane_id = self.get_pane_id_by_index(num)

        if pane_id:
            self.switch_to_pane(pane_id)
            self._action_key_held = False
            self._cancel_leader()
            return True

        # ── Step 6: any other key → cancel leader, let event pass through ──
        self._action_key_held = False
        self._cancel_leader()
        return False

    def _handle_key_release(self, watched: QObject, event: QKeyEvent) -> bool:
        key = event.key()

        # Action trigger released
        if self._is_action_trigger(event):
            self._action_key_held = False
            # Do NOT cancel leader — tap-then-type still needs it active
            return True

        # Key code released (e.g. B released when Ctrl was already released)
        if self.action_key_code is not None and key == self.action_key_code:
            self._action_key_held = False
            return False

        # Modifier component released (e.g. Ctrl released in Ctrl+B chord)
        if self.action_modifier is not None and self._is_modifier_key(key):
            self._action_key_held = False

        return False

    # ── Leader mode ───────────────────────────────────────────────────────

    def _activate_leader(self) -> None:
        self._leader_active = True
        if self.leader_timeout > 0:
            self._leader_timer.start(int(self.leader_timeout * 1000))
        sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
        if sb:
            key_label = self.action_key_name.replace("_", "+").title()
            timeout_ms = int(self.leader_timeout * 1000) if self.leader_timeout > 0 else 0
            sb.showMessage(
                f"Action [{key_label}]: Press [0-9] to switch pane  (Esc or {key_label} to cancel)…",
                timeout_ms,
            )

    def _cancel_leader(self) -> None:
        self._leader_active = False
        self._leader_timer.stop()
        sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
        if sb and sb.currentMessage().startswith("Action ["):
            sb.clearMessage()

    def _on_leader_timeout(self) -> None:
        self._cancel_leader()

    # ── Pane navigation ───────────────────────────────────────────────────

    def get_pane_id_by_index(self, index: int) -> Optional[str]:
        """Resolve a digit [0-9] to a canonical pane ID.

        Priority:
          1. Explicit ``hotkeys.bindings`` mapping (including unassigned digits)
          2. Legacy ``hotkeys.panes`` list when bindings have not been migrated
          3. Default: 0 → settings, 1-9 → sidebar tabs in order
        """
        bindings = self.settings_manager.get("hotkeys", "bindings", default=None)
        if isinstance(bindings, dict):
            target = f"ACTION + {index}"
            for pid, combo in bindings.items():
                normalized = str(combo).strip().upper()
                if not normalized.startswith("ACTION"):
                    normalized = f"ACTION + {normalized}"
                if normalized == target:
                    return PANE_ALIASES.get(pid.lower(), pid)
            return None

        configured = self.settings_manager.hotkey_panes
        if 0 <= index < len(configured):
            raw = configured[index]
            if not raw or str(raw).lower() in ("none", "unassigned", "disabled", ""):
                return None
            return PANE_ALIASES.get(raw.lower(), raw)

        # Fallback: 0 → settings, rest follow tab_order
        if index == 0:
            return "settings"
        tabs = getattr(self.window, "tabs", None)
        tab_order = getattr(tabs, "tab_order", [])
        if 0 <= (index - 1) < len(tab_order):
            raw = tab_order[index - 1]
            return PANE_ALIASES.get(raw.lower(), raw)

        return None

    def switch_to_pane_by_index(self, index: int) -> bool:
        """Switch to the pane mapped to *index*."""
        pane_id = self.get_pane_id_by_index(index)
        if not pane_id:
            sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
            if sb:
                sb.showMessage(f"No pane mapped to Action + {index}", 2500)
            return False
        return self.switch_to_pane(pane_id, index=index)

    def switch_to_pane(self, pane_id: str, index: Optional[int] = None) -> bool:
        """Switch the view to *pane_id*, unhiding it if necessary."""
        pane_id = PANE_ALIASES.get(pane_id.lower(), pane_id)
        key_label = f"Action + {index}" if index is not None else "Hotkey"

        # Special case: Settings lives outside the normal tab stack
        if pane_id == "settings":
            panel = getattr(self.window, "settings_panel", None)
            if panel is not None and hasattr(self.window, "tabs"):
                self.window.tabs.setCurrentWidget(panel)
                sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
                if sb:
                    sb.showMessage(f"Switched to {pane_id.title()} ({key_label})", 3000)
                return True
            return False

        tabs = getattr(self.window, "tabs", None)
        if not tabs or not hasattr(tabs, "tabs"):
            return False

        if pane_id not in tabs.tabs:
            return False

        # If the tab is hidden, make it visible first
        if not tabs.is_tab_visible(pane_id):
            tabs.set_tab_visible(pane_id, True)
            if hasattr(self.window, "pane_sliders") and pane_id in self.window.pane_sliders:
                slider = self.window.pane_sliders[pane_id]
                slider.blockSignals(True)
                slider.setChecked(True)
                slider.blockSignals(False)
            if hasattr(self.window, "save_layout"):
                self.window.save_layout()

        entry = tabs.tabs[pane_id]
        tabs.setCurrentWidget(entry["widget"])
        display_name = entry.get("short") or entry.get("text") or pane_id.capitalize()
        sb = self.window.statusBar() if hasattr(self.window, "statusBar") else None
        if sb:
            sb.showMessage(f"Switched to {display_name} ({key_label})", 3000)
        return True
