"""Organized, tabbed Settings view: category side-tabs on the left, the chosen page on the right."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from PyQt6.QtCore import QByteArray, QEvent, QKeyCombination, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QKeyEvent, QKeySequence, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.services import PROJECT_ROOT

from .hotkeys import parse_action_key
from .theme import current_theme, theme_color
from .widgets import SmoothScrollArea, ToggleSlider, label

if TYPE_CHECKING:
    from .app import Window

ICON_DIR = Path(__file__).resolve().parent / "icons"


def get_settings_icon(name: str, size: int = 20) -> QIcon:
    """Render icons/<name>.svg with current theme's text and accent colors."""
    p = Path(name)
    if p.exists() and p.is_file():
        source_path = p
    elif (ICON_DIR / f"{name}.svg").exists():
        source_path = ICON_DIR / f"{name}.svg"
    elif (ICON_DIR / name).exists():
        source_path = ICON_DIR / name
    else:
        return QIcon()

    source = source_path.read_text(encoding="utf-8")
    ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    icon = QIcon()

    colors = {
        (QIcon.Mode.Normal, QIcon.State.Off): "muted",
        (QIcon.Mode.Active, QIcon.State.Off): "text",
        (QIcon.Mode.Normal, QIcon.State.On): "accent",
        (QIcon.Mode.Active, QIcon.State.On): "accent",
    }
    for (mode, state), role in colors.items():
        rendered_src = source.replace("currentColor", theme_color(role)).encode("utf-8")
        renderer = QSvgRenderer(QByteArray(rendered_src))
        pixmap = QPixmap(round(size * ratio), round(size * ratio))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        icon.addPixmap(pixmap, mode, state)
    return icon


TAB_SPECS = [
    {
        "id": "appearance",
        "title": "Appearance",
        "subtitle": "Themes, sidebar panel navigation & display options",
        "icon": "appearance",
    },
    {
        "id": "core_plugins",
        "title": "Core Plugins",
        "subtitle": "Built-in core extensions bundled natively into Theta-IDE",
        "icon": "plugins_core",
    },
    {
        "id": "community_plugins",
        "title": "Community Plugins",
        "subtitle": "Installed plugins, extensions, and Community Hub",
        "icon": "plugins_community",
    },
    {
        "id": "hotkeys",
        "title": "Hotkeys",
        "subtitle": "Action-key leader navigation & keyboard shortcuts",
        "icon": "hotkeys",
    },
    {
        "id": "backend",
        "title": "Training Pipeline",
        "subtitle": "The Python environment that composes configs and runs training",
        "icon": "backend",
    },
    {
        "id": "terminal",
        "title": "Terminal",
        "subtitle": "Shell, start folder and Python environment for the Terminal pane",
        "icon": "terminal",
    },
    {
        "id": "editor",
        "title": "Text Editor",
        "subtitle": "Default editor and text editing preferences",
        "icon": "editor",
    },
    {
        "id": "storage",
        "title": "Workspace & Storage",
        "subtitle": "Data roots, UI layout reset, and settings.toml",
        "icon": "storage",
    },
]

ALL_TAB_SPECS = TAB_SPECS
 
DEFAULT_PANES = [
    "settings",
    "components",
    "config",
    "monitor",
    "results",
    "plots",
    "tensorboard",
    "queue",
    "terminal",
    "console",
]

AVAILABLE_PANES = [
    ("settings", "Settings"),
    ("components", "Components"),
    ("config", "Experiment Builder"),
    ("monitor", "Training Monitor"),
    ("results", "Results Browser"),
    ("plots", "Plot Viewer"),
    ("queue", "Job Queue"),
    ("terminal", "Terminal"),
    ("console", "Console"),
    ("none", "— Unassigned / Disabled —"),
]

ACTION_KEY_PRESETS = [
    ("ctrl+b", "Ctrl + B / Caps Lock (tmux default)"),
    ("caps_lock", "Caps Lock (fast modal key)"),
    ("ctrl+space", "Ctrl + Space"),
    ("alt+space", "Alt + Space"),
    ("ctrl+a", "Ctrl + A (Screen default)"),
    ("ctrl+x", "Ctrl + X (Emacs default)"),
    ("alt", "Alt / Option (hold chord)"),
    ("ctrl", "Control (hold chord)"),
    ("meta", "Command / Meta (⌘ / Win)"),
    ("shift", "Shift (hold chord)"),
]

DEFAULT_MENU_SHORTCUTS = [
    ("launch_training", "Launch training", "F5"),
    ("stop_run", "Stop training", "Shift+F5"),
    ("new_experiment", "New document / experiment", "Ctrl+N"),
    ("open_text_file", "Open text file", "Ctrl+O"),
    ("reload_components", "Reload component code", "Ctrl+Alt+R"),
    ("export_config", "Save As / Export YAML…", "Ctrl+Shift+S"),
    ("save_config", "Save document / configuration", "Ctrl+S"),
    ("add_to_queue", "Add to queue", "Ctrl+Shift+Q"),
    ("toggle_queue", "Start or pause queue", "Ctrl+Shift+R"),
    ("start_demo", "Start simulated demo", "Ctrl+F5"),
    ("zoom_in", "Zoom in (increase text size)", "Ctrl++"),
    ("zoom_out", "Zoom out (decrease text size)", "Ctrl+-"),
    ("reset_zoom", "Reset zoom (default text size)", "Ctrl+0"),
    ("open_launcher", "Open launcher (panels, plugins, settings, commands)", "Ctrl+Space"),
    ("open_docs", "Open the documentation", "F1"),
    ("quit", "Quit", "Ctrl+Q"),
]


# Sidebar panes the app always has, in their default order; plugin tabs follow them.
BUILT_IN_PANES = (
    ("components", "Components"),
    ("config", "Experiment Builder"),
    ("workflows", "Workflows"),
    ("monitor", "Training Monitor"),
    ("results", "Results Browser"),
    ("plots", "Plot Viewer"),
    ("queue", "Job Queue"),
    ("terminal", "Terminal"),
    ("console", "Console"),
)

SIDE_TAB_ICON = 20      # px
SIDE_TAB_ICON_GAP = 8   # px of space between a settings tab's icon and its label


class SettingsSideTabButton(QToolButton):
    """Button in the left settings navigation bar."""

    def __init__(self, tab_id: str, title: str, icon_name: str, parent=None):
        super().__init__(parent)
        self.tab_id = tab_id
        self.title_text = title
        self.icon_name = icon_name
        self.setObjectName("settingsSideTab")
        self.setCheckable(True)
        self.setAutoRaise(True)
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.setText(title.replace("&", "&&"))  # a lone & is a mnemonic marker: "Workspace _Storage"
        self.setToolTip(f"Configure {title}")
        self.setIconSize(QSize(SIDE_TAB_ICON + SIDE_TAB_ICON_GAP, SIDE_TAB_ICON))
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFixedHeight(38)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_icon()

    def refresh_icon(self):
        icon = get_settings_icon(self.icon_name, SIDE_TAB_ICON)
        if not icon.isNull():
            self.setIcon(_with_trailing_gap(icon))


def _with_trailing_gap(icon: QIcon) -> QIcon:
    """The icon on a canvas SIDE_TAB_ICON_GAP px wider, transparent on the right, so the
    tab's text sits a little further from it (QToolButton has no icon-text spacing setting)."""
    ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    padded = QIcon()
    for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected, QIcon.Mode.Disabled):
        for state in (QIcon.State.Off, QIcon.State.On):
            glyph = icon.pixmap(QSize(SIDE_TAB_ICON, SIDE_TAB_ICON), ratio, mode, state)
            canvas = QPixmap(round((SIDE_TAB_ICON + SIDE_TAB_ICON_GAP) * ratio), round(SIDE_TAB_ICON * ratio))
            canvas.setDevicePixelRatio(ratio)
            canvas.fill(Qt.GlobalColor.transparent)
            painter = QPainter(canvas)
            painter.drawPixmap(0, 0, glyph)
            painter.end()
            padded.addPixmap(canvas, mode, state)
    return padded


class DynamicStackedWidget(QStackedWidget):
    """QStackedWidget sized by its current page only.

    A plain QStackedWidget's layout takes its minimum height from the tallest page, so inside a
    scroll area every short page gained the Hotkeys page's height of empty space to scroll into.
    Hidden pages get the Ignored size policy, which the stacked layout skips. Width is
    unconstrained to prevent child sizeHint blowup in the QScrollArea.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.currentChanged.connect(self._fit_current_page)

    def addWidget(self, widget):
        index = super().addWidget(widget)
        self._fit_current_page()
        return index

    def _fit_current_page(self, _index=None):
        current = self.currentWidget()
        for i in range(self.count()):
            page = self.widget(i)
            if page is current:
                page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
            else:
                page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.updateGeometry()
        # The stack sits in the scroll area's viewport, which has no layout, so the new size hint
        # never reaches the scroll area on its own; ask it to re-fit the page now.
        area = self.parentWidget().parentWidget() if self.parentWidget() else None
        if isinstance(area, QScrollArea):
            QApplication.sendEvent(area, QEvent(QEvent.Type.LayoutRequest))

    def minimumSizeHint(self) -> QSize:
        curr = self.currentWidget()
        h = curr.minimumSizeHint().height() if curr is not None else 0
        return QSize(0, h)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        # QStackedLayout.heightForWidth returns the tallest of *all* pages, and the scroll area
        # sizes its content from it whenever pages hold word-wrapped labels. Answer for the
        # current page only.
        curr = self.currentWidget()
        if curr is None:
            return 0
        hfw = curr.heightForWidth(width) if curr.hasHeightForWidth() else -1
        return max(hfw, curr.minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        curr = self.currentWidget()
        h = curr.sizeHint().height() if curr is not None else 0
        return QSize(0, h)


class SettingsDetailWindow(QFrame):
    """The settings page area: the active category's header over its scrollable page.

    The page fills the content area up to a readable width (MAX_PAGE_WIDTH); wider windows leave
    the rest empty rather than stretching every row across the screen.
    """

    MAX_PAGE_WIDTH = 920

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("settingsOverlayLayer")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.current_tab_id = "appearance"

        self.outer_layout = QHBoxLayout(self)
        self.outer_layout.setContentsMargins(0, 0, 0, 0)
        self.outer_layout.setSpacing(0)

        self.card = QFrame(self)
        self.card.setObjectName("settingsDetailCard")
        self.card.setMaximumWidth(self.MAX_PAGE_WIDTH)
        self.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(24, 20, 24, 20)
        card_layout.setSpacing(14)

        # Header: the category's icon, title and subtitle
        header_row = QHBoxLayout()
        header_row.setSpacing(12)

        self.icon_label = QLabel()
        self.icon_label.setFixedSize(28, 28)
        self.icon_label.setScaledContents(True)
        header_row.addWidget(self.icon_label)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        self.title_label = label("Settings", "heading")
        self.title_label.setStyleSheet("font-size: 20px; font-weight: 700;")
        title_layout.addWidget(self.title_label)

        self.subtitle_label = label("", "muted")
        self.subtitle_label.setStyleSheet("font-size: 12px;")
        title_layout.addWidget(self.subtitle_label)
        header_row.addLayout(title_layout, 1)

        card_layout.addLayout(header_row)

        # Subtle separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("settingsSeparator")
        card_layout.addWidget(sep)

        # Content Area: Scrollable stacked pages (vertical only, no horizontal scrollbars)
        self.scroll_area = SmoothScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setObjectName("settingsScrollArea")
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.verticalScrollBar().setSingleStep(16)
        if hasattr(self.scroll_area, "viewport") and self.scroll_area.viewport():
            self.scroll_area.viewport().setAutoFillBackground(True)

        self.pages_stack = DynamicStackedWidget()
        self.scroll_area.setWidget(self.pages_stack)
        card_layout.addWidget(self.scroll_area, 1)

        self.outer_layout.addWidget(self.card, 1)
        self.outer_layout.addStretch(0)

    def set_header(self, title: str, subtitle: str, icon_name: str):
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        icon = get_settings_icon(icon_name, 28)
        if not icon.isNull():
            pix = icon.pixmap(28, 28)
            self.icon_label.setPixmap(pix)
            self.icon_label.setVisible(True)
        else:
            self.icon_label.setVisible(False)


def normalize_hotkey_combo(text: str) -> str:
    """Normalize user input (e.g. '1', 'Action + 1', 't', 'Action + T') to canonical 'Action + <KEY>' or empty."""
    cleaned = text.strip()
    if not cleaned or cleaned.lower() in ("none", "unassigned", "-", "disabled", "clear", ""):
        return ""
    lower = cleaned.lower()
    if lower.startswith("action"):
        cleaned = cleaned[6:]
    elif lower.startswith("leader"):
        cleaned = cleaned[6:]
    # Strip any leading/trailing spaces and plus signs (e.g. " + 1" -> "1", "+ C" -> "C")
    cleaned = cleaned.strip(" +").strip()
    cleaned = cleaned.upper()
    if not cleaned:
        return ""
    return f"Action + {cleaned}"


def normalize_menu_shortcut(shortcut_str: str) -> str:
    """Normalize user-entered shortcut string for Qt QKeySequence.

    Replaces 'cmd' and 'command' with 'Ctrl' so that macOS command shortcuts
    are properly parsed by QKeySequence on all platforms.
    """
    if not shortcut_str:
        return ""
    import re
    return re.sub(r"(?i)\b(cmd|command)\b", "Ctrl", str(shortcut_str).strip())


class HotkeyComboEdit(QLineEdit):
    """Interactive hotkey input field for pane navigation chords.

    Accepts keyboard input directly (e.g. pressing '1' sets 'Action + 1',
    pressing 'T' sets 'Action + T', Backspace/Delete clears).
    """

    comboChanged = pyqtSignal(str)
    _is_hotkey_editor = True

    def __init__(self, canonical_combo: str = "", parent=None):
        super().__init__(parent)
        self._canonical_combo = canonical_combo
        self.setText(canonical_combo)
        self.setPlaceholderText("Unassigned (click to set)")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setStyleSheet("font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 600;")
        self.editingFinished.connect(self._on_editing_finished)

    def get_canonical_combo(self) -> str:
        return self._canonical_combo

    def set_canonical_combo(self, combo: str):
        self._canonical_combo = combo
        self.blockSignals(True)
        self.setText(combo)
        self.blockSignals(False)

    def keyPressEvent(self, event):
        key = event.key()
        # Ignore pure modifiers alone
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            super().keyPressEvent(event)
            return

        # Clear on Backspace or Delete
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            self.set_canonical_combo("")
            self.comboChanged.emit("")
            return

        # Cancel / exit focus on Escape
        if key == Qt.Key.Key_Escape:
            self.clearFocus()
            return

        # Check for digit (0-9)
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            digit = str(key - Qt.Key.Key_0)
            combo = f"Action + {digit}"
            self.set_canonical_combo(combo)
            self.comboChanged.emit(combo)
            return

        # Check for letter/printable text
        text = event.text().strip().upper()
        if text and len(text) == 1 and (text.isalnum() or text in "+-_/"):
            combo = f"Action + {text}"
            self.set_canonical_combo(combo)
            self.comboChanged.emit(combo)
            return

        super().keyPressEvent(event)

    def _on_editing_finished(self):
        normalized = normalize_hotkey_combo(self.text())
        if normalized != self._canonical_combo:
            self.set_canonical_combo(normalized)
            self.comboChanged.emit(normalized)

class MenuShortcutEdit(QLineEdit):
    """Interactive hotkey input field for application menu shortcuts.

    When clicked or focused, enters listening mode to capture a single key combo,
    overwriting whatever was there previously. Backspace/Delete clears the shortcut,
    Escape cancels and restores the previous shortcut.
    """

    keySequenceChanged = pyqtSignal(QKeySequence)
    comboChanged = pyqtSignal(str)
    _is_hotkey_editor = True

    def __init__(self, key_sequence: QKeySequence | str = "", parent=None):
        super().__init__(parent)
        self._is_recording = False
        self._canonical_sc = ""
        self._has_conflict = False
        self.setReadOnly(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setPlaceholderText("Unassigned (click to record)")
        if key_sequence:
            self.setKeySequence(key_sequence)
        else:
            self._update_appearance()

    def keySequence(self) -> QKeySequence:
        return QKeySequence(self._canonical_sc) if self._canonical_sc else QKeySequence()

    def setKeySequence(self, seq: QKeySequence | str):
        if isinstance(seq, QKeySequence):
            sc_str = seq.toString(QKeySequence.SequenceFormat.PortableText)
        else:
            sc_str = str(seq or "")
        norm = normalize_menu_shortcut(sc_str)
        self._canonical_sc = norm
        self._is_recording = False
        self.setText(norm)
        if not norm:
            self.setPlaceholderText("Unassigned (click to record)")
        self._update_appearance()
        if not self.signalsBlocked():
            self.keySequenceChanged.emit(self.keySequence())
            self.comboChanged.emit(norm)

    def set_conflict_style(self, has_conflict: bool):
        self._has_conflict = has_conflict
        self._update_appearance()

    def _update_appearance(self):
        if self._is_recording:
            self.setStyleSheet(
                "font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 600; "
                "border: 1.5px solid #8aadf4; background: rgba(138, 173, 244, 0.15); "
                "color: #b7bdf8; border-radius: 4px; padding: 4px 8px;"
            )
        elif self._has_conflict:
            self.setStyleSheet(
                "font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 700; "
                "border: 1.5px solid #fb4934; background: rgba(251, 73, 52, 0.15); "
                "color: #ff9999; border-radius: 4px; padding: 4px 8px;"
            )
        else:
            self.setStyleSheet(
                "font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 600; "
                "border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 4px; padding: 4px 8px;"
            )

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self._start_recording()

    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._start_recording()

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if self._is_recording:
            self._stop_recording(revert=True)

    def _start_recording(self):
        self._is_recording = True
        self.setPlaceholderText("Press key combination… (Esc to cancel)")
        self.setText("")
        self._update_appearance()

    def _stop_recording(self, revert: bool = False):
        self._is_recording = False
        if revert:
            self.setText(self._canonical_sc)
        if not self._canonical_sc:
            self.setPlaceholderText("Unassigned (click to record)")
        self._update_appearance()

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mods = event.modifiers()

        # 1. Escape cancels recording
        if key == Qt.Key.Key_Escape:
            self._stop_recording(revert=True)
            self.clearFocus()
            event.accept()
            return

        # 2. Backspace or Delete without modifiers clears the shortcut
        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and not (mods & ~Qt.KeyboardModifier.KeypadModifier):
            self.setKeySequence("")
            self.clearFocus()
            event.accept()
            return

        # 3. Pure modifier keys: show live modifier feedback while waiting for combination key
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Shift, Qt.Key.Key_Alt, Qt.Key.Key_Meta):
            parts = []
            if mods & Qt.KeyboardModifier.ControlModifier:
                parts.append("Ctrl")
            if mods & Qt.KeyboardModifier.AltModifier:
                parts.append("Alt")
            if mods & Qt.KeyboardModifier.ShiftModifier:
                parts.append("Shift")
            if mods & Qt.KeyboardModifier.MetaModifier:
                parts.append("Meta")
            if parts:
                self.setText(" + ".join(parts) + " + …")
            event.accept()
            return

        # 4. Main key pressed with or without modifiers
        comb = event.keyCombination()
        seq = QKeySequence(comb)
        sc_str = seq.toString(QKeySequence.SequenceFormat.PortableText)
        norm = normalize_menu_shortcut(sc_str)

        if norm:
            self.setKeySequence(norm)
            self.clearFocus()
            event.accept()
            return

        super().keyPressEvent(event)


class SettingsView(QWidget):
    """Complete, organized settings view: category side-tabs on the left, the chosen page beside them."""

    def __init__(self, window: Window, parent=None):
        super().__init__(parent)
        self.window = window
        self.active_tab_id: str | None = None
        self.tab_buttons: Dict[str, SettingsSideTabButton] = {}
        self.pane_hotkey_edits: Dict[str, HotkeyComboEdit] = {}
        self.pane_conflict_labels: Dict[str, QLabel] = {}
        self.menu_shortcut_edits: Dict[str, MenuShortcutEdit] = {}
        self.menu_conflict_labels: Dict[str, QLabel] = {}
        self.notification_toggles: Dict[str, ToggleSlider] = {}
        self.slider_term: ToggleSlider | None = None
        self.conflict_banner: QFrame | None = None
        self.conflict_banner_label: QLabel | None = None
        self.menu_conflict_banner: QFrame | None = None
        self.menu_conflict_banner_label: QLabel | None = None

        # Main horizontal layout: Side-tab on the left, Content stack on the right
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── 1. Left Side-Tab ───────────────────────────────────────────────────
        self.sidebar_frame = QFrame()
        self.sidebar_frame.setObjectName("settingsSidebar")
        self.sidebar_frame.setFixedWidth(220)

        sb_layout = QVBoxLayout(self.sidebar_frame)
        sb_layout.setContentsMargins(10, 16, 10, 14)
        sb_layout.setSpacing(6)

        # Navigation tabs list
        for spec in TAB_SPECS:
            tid = spec["id"]
            btn = SettingsSideTabButton(tid, spec["title"], spec["icon"], self.sidebar_frame)
            btn.clicked.connect(lambda _, t=tid: self.on_tab_button_clicked(t))
            self.tab_buttons[tid] = btn
            sb_layout.addWidget(btn)

        sb_layout.addStretch()

        layout.addWidget(self.sidebar_frame)

        # ── 2. Right: the active category's page ─────────────────────────────
        self.settings_window = SettingsDetailWindow(self)
        layout.addWidget(self.settings_window, 1)

        # ── 3. Build Pages inside the Settings Window ─────────────────────────
        self.page_widgets: Dict[str, QWidget] = {}
        self.build_pages()

        # Apply initial styling
        self.refresh_styles()

        # Keep hotkeys & settings in sync with live settings changes
        if hasattr(self.window, "settings_manager") and self.window.settings_manager:
            self.window.settings_manager.changed.connect(self.sync_from_settings)

    # ── Page Builders ────────────────────────────────────────────────────────

    def build_pages(self):
        """Construct the settings pages inside the settings window stack."""
        stack = self.settings_window.pages_stack

        # Page 0: Appearance
        p_appearance = self._build_appearance_page()
        stack.addWidget(p_appearance)
        self.page_widgets["appearance"] = p_appearance

        # Page 1: Core Plugins
        p_core = self._build_core_plugins_page()
        stack.addWidget(p_core)
        self.page_widgets["core_plugins"] = p_core

        # Page 2: Community Plugins
        p_comm = self._build_community_plugins_page()
        stack.addWidget(p_comm)
        self.page_widgets["community_plugins"] = p_comm

        # Page 3: Hotkeys
        p_hotkeys = self._build_hotkeys_page()
        stack.addWidget(p_hotkeys)
        self.page_widgets["hotkeys"] = p_hotkeys

        # Page 4: Training pipeline
        p_backend = self._build_backend_page()
        stack.addWidget(p_backend)
        self.page_widgets["backend"] = p_backend

        # Page 5: Terminal
        p_terminal = self._build_terminal_page()
        stack.addWidget(p_terminal)
        self.page_widgets["terminal"] = p_terminal

        p_editor = self._build_editor_page()
        stack.addWidget(p_editor)
        self.page_widgets["editor"] = p_editor

        # Page 6: Workspace & Storage
        p_storage = self._build_storage_page()
        stack.addWidget(p_storage)
        self.page_widgets["storage"] = p_storage


    def _build_appearance_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        # Card 1: Active Theme
        theme_card = QFrame()
        theme_card.setObjectName("card")
        tc_layout = QVBoxLayout(theme_card)
        tc_layout.setContentsMargins(16, 14, 16, 14)
        tc_layout.setSpacing(10)

        tc_layout.addWidget(label("Color Theme & Palette", "cardTitle"))
        tc_sub = label("Select an active color palette or create custom theme roles.", "muted")
        tc_sub.setWordWrap(True)
        tc_layout.addWidget(tc_sub)

        theme_row = QHBoxLayout()
        theme_row.addWidget(label("Active Theme:", "muted"))
        self.window.settings_theme_select = QComboBox()
        for name in self.window.theme_manager.themes():
            self.window.settings_theme_select.addItem(name)
        self.window.settings_theme_select.setCurrentText(self.window.theme_manager.active["name"])
        self.window.settings_theme_select.currentTextChanged.connect(self.window.settings_theme_selected)
        self.window.settings_theme_select.setMaximumWidth(320)
        theme_row.addWidget(self.window.settings_theme_select, 1)

        btn_builder = QPushButton("Customize palette…")
        btn_builder.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_builder.clicked.connect(self.window.show_theme_builder)
        theme_row.addWidget(btn_builder)
        theme_row.addStretch()
        tc_layout.addLayout(theme_row)

        settings_icon_row = QHBoxLayout()
        settings_icon_row.setSpacing(10)
        self.window.settings_button_logo_toggle = ToggleSlider(
            checked=self.window.settings_manager.settings_button_logo
        )
        self.window.settings_button_logo_toggle.setToolTip(
            "Show the Theta logo on the Settings button instead of the gear"
        )
        self.window.settings_button_logo_toggle.toggled.connect(
            lambda enabled: self.window.settings_manager.set(
                "appearance", "settings_button_logo", bool(enabled)
            )
        )
        settings_icon_row.addWidget(self.window.settings_button_logo_toggle)
        settings_icon_label = label("Use Theta logo for the Settings button", "muted")
        settings_icon_label.setWordWrap(True)
        settings_icon_row.addWidget(settings_icon_label, 1)
        tc_layout.addLayout(settings_icon_row)
        layout.addWidget(theme_card)

        # Card 2: Display & Text Scaling (Zoom)
        zoom_card = QFrame()
        zoom_card.setObjectName("card")
        zc_layout = QVBoxLayout(zoom_card)
        zc_layout.setContentsMargins(16, 14, 16, 14)
        zc_layout.setSpacing(10)

        zc_layout.addWidget(label("Display & Text Scaling", "cardTitle"))
        zc_sub = label("Resize interface text and elements. Use Cmd++ / Cmd+- or bind custom hotkeys in Hotkeys.", "muted")
        zc_sub.setWordWrap(True)
        zc_layout.addWidget(zc_sub)

        current_zoom = getattr(self.window, "zoom_factor", 1.0)
        current_pct = int(round(current_zoom * 100))

        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(10)

        btn_zoom_out = QPushButton("−")
        btn_zoom_out.setToolTip("Zoom out (decrease text size)")
        btn_zoom_out.setFixedSize(32, 28)
        btn_zoom_out.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_zoom_out.clicked.connect(self.window.zoom_out)
        zoom_row.addWidget(btn_zoom_out)

        self.window.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.window.zoom_slider.setRange(60, 240)
        self.window.zoom_slider.setSingleStep(10)
        self.window.zoom_slider.setValue(current_pct)
        self.window.zoom_slider.setToolTip("Drag to adjust display scaling factor (60% – 240%)")
        self.window.zoom_slider.valueChanged.connect(lambda val: self.window.set_zoom(val / 100.0))
        zoom_row.addWidget(self.window.zoom_slider, 1)

        btn_zoom_in = QPushButton("+")
        btn_zoom_in.setToolTip("Zoom in (increase text size)")
        btn_zoom_in.setFixedSize(32, 28)
        btn_zoom_in.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_zoom_in.clicked.connect(self.window.zoom_in)
        zoom_row.addWidget(btn_zoom_in)

        self.window.zoom_percent_label = label(f"{current_pct}%", "badge")
        self.window.zoom_percent_label.setFixedWidth(54)
        self.window.zoom_percent_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        zoom_row.addWidget(self.window.zoom_percent_label)

        btn_reset_zoom = QPushButton("Reset (100%)")
        btn_reset_zoom.setToolTip("Reset text and interface zoom to 100%")
        btn_reset_zoom.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_zoom.clicked.connect(self.window.reset_zoom)
        zoom_row.addWidget(btn_reset_zoom)

        zc_layout.addLayout(zoom_row)
        layout.addWidget(zoom_card)

        # Card 3: Sidebar Panels
        sidebar_card = QFrame()
        sidebar_card.setObjectName("card")
        sb_layout = QVBoxLayout(sidebar_card)
        sb_layout.setContentsMargins(16, 14, 16, 14)
        sb_layout.setSpacing(10)

        sb_layout.addWidget(label("Sidebar Panels Visibility", "cardTitle"))
        sb_sub = label("Choose which navigation panels are visible in the left sidebar.", "muted")
        sb_sub.setWordWrap(True)
        sb_layout.addWidget(sb_sub)

        self.panes_grid = QGridLayout()
        self.panes_grid.setHorizontalSpacing(24)
        self.panes_grid.setVerticalSpacing(8)
        sb_layout.addLayout(self.panes_grid)
        self.rebuild_pane_switches()
        if getattr(self.window, "tabs", None) is not None:
            # Plugins add their tabs after this page is built, and can be enabled or disabled later:
            # keep a switch for every tab, and each switch in step with its tab.
            self.window.tabs.tabsChanged.connect(self.rebuild_pane_switches)
            self.window.tabs.tabVisibilityChanged.connect(self._sync_pane_switch)

        auto_hide_row = QHBoxLayout()
        auto_hide_info = QVBoxLayout()
        auto_hide_info.setSpacing(1)
        auto_hide_title = label("Auto-hide sidebar")
        auto_hide_title.setStyleSheet("font-weight: 500; font-size: 12px;")
        auto_hide_info.addWidget(auto_hide_title)
        auto_hide_sub = label("Tuck the sidebar into a thin left edge; hover the edge to slide it out.", "muted")
        auto_hide_sub.setWordWrap(True)
        auto_hide_info.addWidget(auto_hide_sub)
        auto_hide_row.addLayout(auto_hide_info, 1)
        self.window.auto_hide_slider = ToggleSlider(
            checked=bool(self.window.settings_manager.get("sidebar", "auto_hide", default=False)))
        self.window.auto_hide_slider.setToolTip("Hide the sidebar until you hover the left edge")
        self.window.auto_hide_slider.setAccessibleName("Toggle sidebar auto-hide")
        self.window.auto_hide_slider.toggled.connect(lambda on: self.window.tabs.set_auto_hide(on))
        auto_hide_row.addWidget(self.window.auto_hide_slider)
        sb_layout.addLayout(auto_hide_row)

        sb_btn_row = QHBoxLayout()
        btn_reset_sidebar = QPushButton("Restore default sidebar")
        btn_reset_sidebar.setToolTip("Show all panels and restore original sidebar order")
        btn_reset_sidebar.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_sidebar.clicked.connect(self.window.reset_sidebar_layout)
        sb_btn_row.addWidget(btn_reset_sidebar)
        sb_btn_row.addStretch()
        sb_layout.addLayout(sb_btn_row)

        layout.addWidget(sidebar_card)

        # Card 3: Experiment Panel Preferences
        exp_card = QFrame()
        exp_card.setObjectName("card")
        ec_layout = QVBoxLayout(exp_card)
        ec_layout.setContentsMargins(16, 14, 16, 14)
        ec_layout.setSpacing(8)

        ec_layout.addWidget(label("Experiment Panel Configuration", "cardTitle"))
        ec_sub = label("Control how configurations and inherited defaults are displayed in the Experiment panel.", "muted")
        ec_sub.setWordWrap(True)
        ec_layout.addWidget(ec_sub)

        row_defaults = QHBoxLayout()
        row_defaults.addWidget(label("Show Resolved Defaults (Experiment Panel)", "muted"), 1)
        self.defaults_toggle = ToggleSlider(checked=self.window.settings_manager.show_resolved_defaults)
        self.defaults_toggle.toggled.connect(self._on_show_resolved_defaults_toggled)
        self.window.toggle_resolved_defaults = self.defaults_toggle
        row_defaults.addWidget(self.defaults_toggle)
        ec_layout.addLayout(row_defaults)

        row_agent_defs = QHBoxLayout()
        row_agent_defs.addWidget(label("Show Defaults for Algorithms in Methods", "muted"), 1)
        self.method_agent_defaults_toggle = ToggleSlider(checked=self.window.settings_manager.show_method_agent_defaults)
        self.method_agent_defaults_toggle.toggled.connect(self._on_show_method_agent_defaults_toggled)
        row_agent_defs.addWidget(self.method_agent_defaults_toggle)
        ec_layout.addLayout(row_agent_defs)

        row_model_defs = QHBoxLayout()
        row_model_defs.addWidget(label("Show Defaults for Models in Methods", "muted"), 1)
        self.method_model_defaults_toggle = ToggleSlider(checked=self.window.settings_manager.show_method_model_defaults)
        self.method_model_defaults_toggle.toggled.connect(self._on_show_method_model_defaults_toggled)
        row_model_defs.addWidget(self.method_model_defaults_toggle)
        ec_layout.addLayout(row_model_defs)

        row_all_defs = QHBoxLayout()
        row_all_defs.addWidget(label("Show All Default Parameters", "muted"), 1)
        self.all_defaults_toggle = ToggleSlider(checked=self.window.settings_manager.show_all_default_params)
        self.all_defaults_toggle.toggled.connect(self._on_show_all_default_params_toggled)
        row_all_defs.addWidget(self.all_defaults_toggle)
        ec_layout.addLayout(row_all_defs)

        layout.addWidget(exp_card)

        layout.addStretch()
        return container

    def rebuild_pane_switches(self):
        """One visibility switch per sidebar tab: the built-in panes first, then any a plugin added."""
        grid = getattr(self, "panes_grid", None)
        if grid is None:
            return
        # Hide what's taken out as well as deleting it: deleteLater only runs once the event loop does,
        # and plugins add their tabs (and so rebuild this) while the window is still being built. Until
        # then the old labels and switches stayed visible, piled up at the card's top-left corner.
        while grid.count():
            item = grid.takeAt(0)
            row = item.layout()
            while row is not None and row.count():
                child = row.takeAt(0)
                if child.widget() is not None:
                    child.widget().hide()
                    child.widget().deleteLater()
            if row is not None:
                row.deleteLater()  # takeAt hands the row layout to us; nothing else frees it
            if item.widget() is not None:
                item.widget().hide()
                item.widget().deleteLater()

        tabs = getattr(self.window, "tabs", None)
        present = tabs.tabs if tabs is not None else {}
        pane_metadata = [(pid, name) for pid, name in BUILT_IN_PANES if pid in present]
        known = {pid for pid, _ in pane_metadata}
        for tid, entry in present.items():
            if tid not in known:
                pane_metadata.append((tid, entry.get("text") or entry.get("short") or tid.capitalize()))

        self.window.pane_sliders = {}
        per_column = (len(pane_metadata) + 1) // 2
        for i, (pid, name) in enumerate(pane_metadata):
            item_row = QHBoxLayout()
            item_row.setSpacing(8)
            title_lbl = label(name)
            title_lbl.setWordWrap(True)
            title_lbl.setStyleSheet("font-weight: 500; font-size: 12px;")
            item_row.addWidget(title_lbl, 1)

            slider = ToggleSlider(checked=tabs.is_tab_visible(pid))
            slider.setToolTip(f"Show or hide {name} in the sidebar")
            slider.setAccessibleName(f"Toggle {name} visibility in sidebar")
            slider.toggled.connect(lambda chk, p=pid: self.window.on_pane_slider_toggled(p, chk))
            self.window.pane_sliders[pid] = slider
            item_row.addWidget(slider)

            grid.addLayout(item_row, i % per_column, i // per_column)
        if hasattr(self.window, "update_pane_sliders_state"):
            self.window.update_pane_sliders_state()

    def _sync_pane_switch(self, tab_id, visible):
        """A tab was shown or hidden elsewhere (layout restore, plugin, hotkey): move its switch to match."""
        slider = getattr(self.window, "pane_sliders", {}).get(tab_id)
        if slider is not None and slider.isChecked() != visible:
            slider.blockSignals(True)
            slider.setChecked(visible)
            slider.blockSignals(False)

    def _on_show_resolved_defaults_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager") and self.window.settings_manager:
            self.window.settings_manager.show_resolved_defaults = checked

    def _on_show_method_agent_defaults_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager") and self.window.settings_manager:
            self.window.settings_manager.show_method_agent_defaults = checked

    def _on_show_method_model_defaults_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager") and self.window.settings_manager:
            self.window.settings_manager.show_method_model_defaults = checked

    def _on_show_all_default_params_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager") and self.window.settings_manager:
            self.window.settings_manager.show_all_default_params = checked

    def _build_core_plugins_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("card")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(18, 16, 18, 16)
        c_layout.setSpacing(12)

        c_layout.addWidget(label("Built-in Core Plugins", "cardTitle"))
        sub = label("Core extensions bundled natively into Theta-IDE.", "muted")
        sub.setWordWrap(True)
        c_layout.addWidget(sub)

        # Status badge pill
        status_box = QHBoxLayout()
        engine_badge = label("● Core Engine v1.0  ·  Ready", "badge")
        engine_badge.setStyleSheet("font-weight: 600; padding: 4px 10px;")
        status_box.addWidget(engine_badge)
        status_box.addStretch()
        c_layout.addLayout(status_box)

        desc_lbl = label(
            "Foundational pipeline runners, experiment builder, and telemetry engines\n"
            "operate directly via the Theta-IDE core framework.",
            "muted",
        )
        desc_lbl.setWordWrap(True)
        c_layout.addWidget(desc_lbl)

        # Dynamic core plugins grid populated by window.refresh_plugins_ui()
        self.window.core_plugins_grid = QVBoxLayout()
        self.window.core_plugins_grid.setSpacing(8)
        c_layout.addLayout(self.window.core_plugins_grid)

        if hasattr(self.window, "refresh_plugins_ui"):
            self.window.refresh_plugins_ui()

        layout.addWidget(card)
        layout.addStretch()
        return container

    def _build_community_plugins_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        plugins_card = QFrame()
        plugins_card.setObjectName("card")
        pc_layout = QVBoxLayout(plugins_card)
        pc_layout.setContentsMargins(18, 16, 18, 16)
        pc_layout.setSpacing(12)

        header_row = QHBoxLayout()
        header_row.addWidget(label("Community Plugins & Extensions", "cardTitle"), 1)

        btn_browse_hub = QPushButton("Browse Community Hub…")
        btn_browse_hub.setToolTip("Explore and install community plugins and extensions")
        btn_browse_hub.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_browse_hub.clicked.connect(lambda: self.window.open_hub("plugin", mode="plugins"))
        header_row.addWidget(btn_browse_hub)
        pc_layout.addLayout(header_row)

        sub = label("Install, configure, or remove community-created plugins.", "muted")
        sub.setWordWrap(True)
        pc_layout.addWidget(sub)

        # Dynamic plugins grid populated by window.refresh_plugins_ui()
        self.window.plugins_grid = QVBoxLayout()
        self.window.plugins_grid.setSpacing(8)
        self.window.plugin_sliders = {}
        pc_layout.addLayout(self.window.plugins_grid)
        self.window.refresh_plugins_ui()

        layout.addWidget(plugins_card)

        # ── Card 2: Component configuration lifecycle ───────────────────────
        hub_card = QFrame()
        hub_card.setObjectName("card")
        hc_layout = QVBoxLayout(hub_card)
        hc_layout.setContentsMargins(18, 16, 18, 16)
        hc_layout.setSpacing(12)

        hc_layout.addWidget(label("Component configurations", "cardTitle"))
        hc_sub = label(
            "Configs are hidden when their component is uninstalled and restored when it is installed again. "
            "Your edits and copied profiles stay in local IDE storage while hidden.",
            "muted",
        )
        hc_sub.setWordWrap(True)
        hc_layout.addWidget(hc_sub)

        layout.addWidget(hub_card)
        layout.addStretch()
        return container

    def _build_hotkeys_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(14)

        # ── Card 1: Action Key & Terminal Precedence ──────────────────────────
        action_card = QFrame()
        action_card.setObjectName("card")
        ak_layout = QVBoxLayout(action_card)
        ak_layout.setContentsMargins(18, 16, 18, 16)
        ak_layout.setSpacing(12)

        ak_layout.addWidget(label("Action Key & Terminal Precedence", "cardTitle"))
        sub = label("Select the primary leader key used to trigger navigation chords, and manage terminal precedence.", "muted")
        ak_layout.addWidget(sub)

        # 1. Action key selector row
        ak_row = QHBoxLayout()
        ak_lbl = label("Primary Action Key (Leader):", "muted")
        ak_lbl.setFixedWidth(190)
        ak_row.addWidget(ak_lbl)

        self.window.settings_action_key_combo = QComboBox()
        for k, title in ACTION_KEY_PRESETS:
            self.window.settings_action_key_combo.addItem(title, k)
        cur_action_key = self.window.settings_manager.action_key.lower()
        idx = self.window.settings_action_key_combo.findData(cur_action_key)
        if idx >= 0:
            self.window.settings_action_key_combo.setCurrentIndex(idx)
        else:
            custom_title = f"{cur_action_key.replace('_', '+').upper()} (Custom)"
            self.window.settings_action_key_combo.addItem(custom_title, cur_action_key)
            self.window.settings_action_key_combo.setCurrentText(custom_title)
        self.window.settings_action_key_combo.currentIndexChanged.connect(self._on_action_preset_changed)
        self.window.settings_action_key_combo.setMaximumWidth(320)
        ak_row.addWidget(self.window.settings_action_key_combo, 1)
        ak_row.addStretch()
        ak_layout.addLayout(ak_row)

        # 2. Terminal precedence toggle
        row_term = QHBoxLayout()
        term_vbox = QVBoxLayout()
        term_vbox.setSpacing(2)
        term_title = label("Terminal Precedence")
        term_title.setStyleSheet("font-weight: 500;")
        term_vbox.addWidget(term_title)
        term_sub = label("Pass action keys directly to terminal when terminal pane is focused.", "muted")
        term_vbox.addWidget(term_sub)
        row_term.addLayout(term_vbox, 1)

        self.slider_term = ToggleSlider(checked=self.window.settings_manager.hotkeys_terminal_precedence)
        self.slider_term.toggled.connect(self._on_terminal_precedence_toggled)
        row_term.addWidget(self.slider_term)
        ak_layout.addLayout(row_term)

        layout.addWidget(action_card)

        # ── Card 2: Interactive Pane Navigation Hotkey Menu ───────────────────
        pane_card = QFrame()
        pane_card.setObjectName("card")
        pc_layout = QVBoxLayout(pane_card)
        pc_layout.setContentsMargins(18, 16, 18, 16)
        pc_layout.setSpacing(12)

        pc_layout.addWidget(label("Pane Navigation Hotkey Menu", "cardTitle"))
        pc_sub = label(
            "Assign hotkeys (e.g. Action + 1, Action + T) to IDE panels. "
            "Click an input and press a key to record, or Backspace to clear. "
            "Hotkeys persist across panel reordering, sidebar toggling, and plugin states.",
            "muted",
        )
        pc_layout.addWidget(pc_sub)

        # Conflict Warning Banner (hidden by default; shown on shortcut collision)
        self.conflict_banner = QFrame()
        self.conflict_banner.setObjectName("hotkeyConflictBanner")
        self.conflict_banner.setStyleSheet(
            "QFrame#hotkeyConflictBanner {"
            "  background-color: rgba(251, 73, 52, 0.12);"
            "  border: 1.5px solid #fb4934;"
            "  border-radius: 6px;"
            "  padding: 10px 14px;"
            "}"
        )
        banner_layout = QHBoxLayout(self.conflict_banner)
        banner_layout.setContentsMargins(8, 6, 8, 6)
        banner_layout.setSpacing(10)

        self.conflict_banner_label = label("", "danger")
        self.conflict_banner_label.setWordWrap(True)
        self.conflict_banner_label.setStyleSheet("color: #ff9999; font-size: 12px; line-height: 1.4;")
        banner_layout.addWidget(self.conflict_banner_label, 1)

        self.conflict_banner.setVisible(False)
        pc_layout.addWidget(self.conflict_banner)

        # Pane Rows List
        self.pane_hotkey_edits = {}
        self.pane_conflict_labels = {}

        current_bindings = self.window.settings_manager.hotkey_bindings
        configured_panes = self.window.settings_manager.hotkey_panes
        all_panes = self._get_all_tracked_panes()

        rows_layout = QVBoxLayout()
        rows_layout.setSpacing(8)

        for pane_info in all_panes:
            pid = pane_info["id"]
            title = pane_info["title"]
            status_text = pane_info["status"]
            status_type = pane_info["status_type"]

            row_frame = QFrame()
            row_frame.setObjectName("hotkeyRowFrame")
            row_frame.setStyleSheet(
                "QFrame#hotkeyRowFrame {"
                "  background: rgba(255, 255, 255, 0.03);"
                "  border: 1px solid rgba(255, 255, 255, 0.07);"
                "  border-radius: 6px;"
                "}"
                "QFrame#hotkeyRowFrame:hover {"
                "  background: rgba(255, 255, 255, 0.05);"
                "  border-color: rgba(255, 255, 255, 0.14);"
                "}"
            )
            rf_layout = QVBoxLayout(row_frame)
            rf_layout.setContentsMargins(12, 10, 12, 10)
            rf_layout.setSpacing(4)

            top_row = QHBoxLayout()
            top_row.setSpacing(10)

            # Left: Title and ID
            title_box = QVBoxLayout()
            title_box.setSpacing(2)
            t_lbl = label(title)
            t_lbl.setStyleSheet("font-weight: 600; font-size: 13px;")
            title_box.addWidget(t_lbl)

            sub_lbl = label(f"ID: {pid}", "muted")
            sub_lbl.setStyleSheet("font-size: 11px; font-family: Menlo, Monaco, 'Consolas', monospace; opacity: 0.75;")
            title_box.addWidget(sub_lbl)
            top_row.addLayout(title_box, 1)

            # Center: Status pill badge
            status_badge = label(status_text, "badge")
            if status_type == "success":
                status_badge.setStyleSheet("background: rgba(166, 218, 149, 0.15); color: #a6da95; border: 1px solid rgba(166, 218, 149, 0.3); font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;")
            elif status_type == "warning":
                status_badge.setStyleSheet("background: rgba(238, 212, 159, 0.15); color: #eed49f; border: 1px solid rgba(238, 212, 159, 0.3); font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;")
            elif status_type == "accent":
                status_badge.setStyleSheet("background: rgba(138, 173, 244, 0.15); color: #8aadf4; border: 1px solid rgba(138, 173, 244, 0.3); font-size: 10px; font-weight: 600; padding: 2px 8px; border-radius: 10px;")
            else:
                status_badge.setStyleSheet("background: rgba(255, 255, 255, 0.08); color: #a5adcb; border: 1px solid rgba(255, 255, 255, 0.12); font-size: 10px; font-weight: 500; padding: 2px 8px; border-radius: 10px;")
            top_row.addWidget(status_badge)

            # Right: HotkeyComboEdit + Clear button
            init_combo = current_bindings.get(pid) or ""
            if not init_combo and pid in configured_panes:
                idx = configured_panes.index(pid)
                init_combo = f"Action + {idx}"

            combo_edit = HotkeyComboEdit(canonical_combo=init_combo)
            combo_edit.setFixedWidth(160)
            combo_edit.setToolTip("Click to record key (e.g. 1, T), or Backspace to clear")
            combo_edit.comboChanged.connect(lambda c, p=pid: self._on_pane_hotkey_changed(p, c))
            self.pane_hotkey_edits[pid] = combo_edit
            top_row.addWidget(combo_edit)

            btn_clear = QToolButton()
            btn_clear.setText("✕")
            btn_clear.setToolTip(f"Clear shortcut for {title}")
            btn_clear.setFixedSize(28, 28)
            btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_clear.setStyleSheet("QToolButton { border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 4px; color: #a5adcb; } QToolButton:hover { border-color: #ed8796; color: #ed8796; }")
            btn_clear.clicked.connect(lambda _, e=combo_edit: (e.set_canonical_combo(""), e.comboChanged.emit("")))
            top_row.addWidget(btn_clear)

            rf_layout.addLayout(top_row)

            # Per-row conflict label
            conflict_lbl = label("", "danger")
            conflict_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #fb4934; margin-top: 2px;")
            conflict_lbl.setVisible(False)
            self.pane_conflict_labels[pid] = conflict_lbl
            rf_layout.addWidget(conflict_lbl)

            rows_layout.addWidget(row_frame)

        pc_layout.addLayout(rows_layout)

        # Pane reset row
        pane_btn_row = QHBoxLayout()
        btn_reset_panes = QPushButton("Restore default pane shortcuts")
        btn_reset_panes.setToolTip("Reset all panels to default Action + [0–9] mapping")
        btn_reset_panes.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_panes.clicked.connect(self._reset_pane_mappings_to_default)
        pane_btn_row.addWidget(btn_reset_panes)
        pane_btn_row.addStretch()
        pc_layout.addLayout(pane_btn_row)

        layout.addWidget(pane_card)

        # ── Card 3: Global Menu & Application Shortcuts ───────────────────────
        menu_card = QFrame()
        menu_card.setObjectName("card")
        mc_layout = QVBoxLayout(menu_card)
        mc_layout.setContentsMargins(18, 16, 18, 16)
        mc_layout.setSpacing(12)

        mc_layout.addWidget(label("Application Menu Shortcuts", "cardTitle"))
        mc_sub = label(
            "Global shortcuts accessible anywhere in Theta-IDE. Click any shortcut to record a new key combination:",
            "muted",
        )
        mc_layout.addWidget(mc_sub)

        # Conflict Banner for Card 3
        self.menu_conflict_banner = QFrame()
        self.menu_conflict_banner.setObjectName("menuConflictBanner")
        self.menu_conflict_banner.setStyleSheet(
            "QFrame#menuConflictBanner {"
            "  background-color: rgba(251, 73, 52, 0.12);"
            "  border: 1.5px solid #fb4934;"
            "  border-radius: 6px;"
            "  padding: 8px 12px;"
            "}"
        )
        m_banner_layout = QHBoxLayout(self.menu_conflict_banner)
        m_banner_layout.setContentsMargins(8, 4, 8, 4)
        self.menu_conflict_banner_label = label("", "danger")
        self.menu_conflict_banner_label.setWordWrap(True)
        self.menu_conflict_banner_label.setStyleSheet("color: #ff9999; font-size: 12px; line-height: 1.4;")
        m_banner_layout.addWidget(self.menu_conflict_banner_label, 1)
        self.menu_conflict_banner.setVisible(False)
        mc_layout.addWidget(self.menu_conflict_banner)

        m_grid = QGridLayout()
        m_grid.setHorizontalSpacing(16)
        m_grid.setVerticalSpacing(10)

        self.menu_shortcut_edits = {}
        self.menu_conflict_labels = {}
        half = (len(DEFAULT_MENU_SHORTCUTS) + 1) // 2
        for i, (aid, title, default_sc) in enumerate(DEFAULT_MENU_SHORTCUTS):
            col_offset = 0 if i < half else 2
            row = i if i < half else i - half

            t_lbl = label(title)
            t_lbl.setStyleSheet("font-weight: 500; font-size: 12px;")

            current_sc = self.window.settings_manager.get("shortcuts", aid, default=default_sc)
            if current_sc is None:
                current_sc = default_sc
            seq_edit = MenuShortcutEdit(normalize_menu_shortcut(current_sc))
            seq_edit.setProperty("_action_id", aid)
            seq_edit.setProperty("_default_sc", default_sc)
            seq_edit.setToolTip("Click to record shortcut (e.g. Ctrl+Space, F5), or Backspace to clear")
            seq_edit.keySequenceChanged.connect(lambda seq, a=aid: self._on_menu_shortcut_changed(a, seq))
            self.menu_shortcut_edits[aid] = seq_edit

            btn_clear = QToolButton()
            btn_clear.setText("✕")
            btn_clear.setToolTip(f"Clear shortcut for {title}")
            btn_clear.setFixedSize(26, 26)
            btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_clear.setStyleSheet(
                "QToolButton { border: 1px solid rgba(255, 255, 255, 0.12); border-radius: 4px; color: #a5adcb; } "
                "QToolButton:hover { border-color: #ed8796; color: #ed8796; }"
            )
            btn_clear.clicked.connect(lambda _, e=seq_edit: e.setKeySequence(""))

            edit_box = QHBoxLayout()
            edit_box.setSpacing(6)
            edit_box.addWidget(seq_edit, 1)
            edit_box.addWidget(btn_clear)

            warn_lbl = label("", "danger")
            warn_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #fb4934; margin-top: 1px;")
            warn_lbl.setVisible(False)
            self.menu_conflict_labels[aid] = warn_lbl

            cell_layout = QVBoxLayout()
            cell_layout.setSpacing(2)
            cell_layout.addLayout(edit_box)
            cell_layout.addWidget(warn_lbl)

            m_grid.addWidget(t_lbl, row, col_offset)
            m_grid.addLayout(cell_layout, row, col_offset + 1)

        mc_layout.addLayout(m_grid)

        # Menu reset row
        menu_btn_row = QHBoxLayout()
        btn_reset_menu = QPushButton("Restore default menu shortcuts")
        btn_reset_menu.setToolTip("Reset all application menu shortcuts to their factory defaults")
        btn_reset_menu.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_menu.clicked.connect(self._reset_menu_shortcuts_to_default)
        menu_btn_row.addWidget(btn_reset_menu)
        menu_btn_row.addStretch()
        mc_layout.addLayout(menu_btn_row)

        layout.addWidget(menu_card)
        layout.addStretch()

        # Run initial conflict check
        self._validate_hotkey_conflicts()

        return container

    def _get_pane_title(self, pane_id: str) -> str:
        """Resolve a human-readable title for a pane ID."""
        for pid, title in AVAILABLE_PANES:
            if pid == pane_id:
                return title
        if hasattr(self.window, "tabs") and hasattr(self.window.tabs, "tabs") and pane_id in self.window.tabs.tabs:
            t = self.window.tabs.tabs[pane_id].get("text")
            if t:
                return t
        if hasattr(self.window, "plugin_manager") and hasattr(self.window.plugin_manager, "plugins"):
            for p in self.window.plugin_manager.plugins.values():
                if getattr(p, "id", None) == pane_id or getattr(p, "panel_id", None) == pane_id:
                    return getattr(p, "name", pane_id.replace("_", " ").title())
        return pane_id.replace("_", " ").title()

    def _get_all_tracked_panes(self) -> List[dict]:
        """Aggregate all panels across built-ins, open tabs, sidebar config, plugins, and saved bindings."""
        sm = self.window.settings_manager if hasattr(self.window, "settings_manager") else None
        tracked = []
        seen = set()

        present_tabs = getattr(getattr(self.window, "tabs", None), "tabs", {})

        # 1. Canonical built-in panels. Optional plugin panes appear below only
        # while their plugin has actually contributed a tab.
        for pid, title in AVAILABLE_PANES:
            if pid == "none" or pid in seen or (pid not in present_tabs and pid != "settings"):
                continue
            seen.add(pid)
            tracked.append({
                "id": pid,
                "title": title,
                "is_builtin": True,
            })

        # 2. Open / dynamic tabs
        if hasattr(self.window, "tabs"):
            tab_order = getattr(self.window.tabs, "tab_order", [])
            for pid in tab_order:
                if pid not in seen and pid != "none":
                    seen.add(pid)
                    tracked.append({
                        "id": pid,
                        "title": self._get_pane_title(pid),
                        "is_builtin": False,
                    })

        # 3. Sidebar order panels from settings
        if sm:
            sidebar_order = sm.get("sidebar", "order", default=[])
            if isinstance(sidebar_order, list):
                for pid in sidebar_order:
                    if pid not in seen and pid != "none" and pid in present_tabs:
                        seen.add(pid)
                        tracked.append({
                            "id": pid,
                            "title": self._get_pane_title(pid),
                            "is_builtin": False,
                        })

        # 4. Plugins (active and inactive)
        if hasattr(self.window, "plugin_manager") and hasattr(self.window.plugin_manager, "plugins"):
            for plugin_id, plugin in self.window.plugin_manager.plugins.items():
                pid = getattr(plugin, "panel_id", None) or plugin_id
                if pid not in seen and pid != "none":
                    seen.add(pid)
                    tracked.append({
                        "id": pid,
                        "title": getattr(plugin, "name", pid.replace("_", " ").title()),
                        "is_builtin": False,
                    })

        # 5. Any previously bound pane IDs from settings.toml
        if sm:
            saved_bindings = sm.get("hotkeys", "bindings", default={})
            if isinstance(saved_bindings, dict):
                for pid in saved_bindings.keys():
                    if pid not in seen and pid != "none" and (pid == "settings" or pid in present_tabs):
                        seen.add(pid)
                        tracked.append({
                            "id": pid,
                            "title": self._get_pane_title(pid),
                            "is_builtin": False,
                        })

            saved_panes = sm.get("hotkeys", "panes", default=[])
            if isinstance(saved_panes, list):
                for pid in saved_panes:
                    if pid not in seen and pid != "none" and (pid == "settings" or pid in present_tabs):
                        seen.add(pid)
                        tracked.append({
                            "id": pid,
                            "title": self._get_pane_title(pid),
                            "is_builtin": False,
                        })

        # Decorate each pane with current status (Active, Hidden, Inactive Plugin, etc.)
        for pane in tracked:
            pid = pane["id"]
            if pid == "settings":
                pane["status"] = "Global Panel"
                pane["status_type"] = "accent"
            elif hasattr(self.window, "tabs") and hasattr(self.window.tabs, "has_tab") and self.window.tabs.has_tab(pid):
                pane["status"] = "Active"
                pane["status_type"] = "success"
            elif sm:
                sidebar_visible = sm.get("sidebar", "visible", default=[])
                enabled_plugins = sm.get("plugins", "enabled", default=[])
                if isinstance(sidebar_visible, list) and pid not in sidebar_visible and pid in sm.get("sidebar", "order", default=[]):
                    pane["status"] = "Hidden in sidebar"
                    pane["status_type"] = "muted"
                elif isinstance(enabled_plugins, list) and pid not in enabled_plugins and hasattr(self.window, "plugin_manager"):
                    pane["status"] = "Inactive plugin"
                    pane["status_type"] = "warning"
                else:
                    pane["status"] = "Active"
                    pane["status_type"] = "success"
            else:
                pane["status"] = "Active"
                pane["status_type"] = "success"

        return tracked

    def _get_menu_action_title(self, action_id: str) -> str:
        """Resolve a human-readable title for a menu action ID."""
        for aid, title, _ in DEFAULT_MENU_SHORTCUTS:
            if aid == action_id:
                return title
        return action_id.replace("_", " ").title()

    def _validate_hotkey_conflicts(self) -> bool:
        """Scan all pane hotkey assignments and application menu shortcuts for duplicates/conflicts,
        updating warning banners, per-row alerts, and highlighting conflicting fields.

        Returns True if any conflicts are detected, False otherwise.
        """
        # combo_key -> list of (category, identifier, title)
        combo_map: Dict[str, List[tuple[str, str, str]]] = {}
        combo_display: Dict[str, str] = {}

        # 1. Pane hotkeys
        for pid, edit in self.pane_hotkey_edits.items():
            raw_text = edit.text().strip()
            combo = normalize_hotkey_combo(raw_text)
            if combo:
                title = self._get_pane_title(pid)
                key = combo.upper()
                if key not in combo_display:
                    combo_display[key] = combo
                combo_map.setdefault(key, []).append(("pane", pid, title))

        # 2. Action key (e.g. "CTRL+B")
        sm = getattr(self.window, "settings_manager", None)
        if sm and getattr(sm, "action_key", None):
            action_key_str = sm.action_key.strip().upper().replace("_", "+")
            if action_key_str in ("CTRL+B", "ALT", "CTRL", "META", "CAPS+LOCK"):
                if action_key_str not in combo_display:
                    combo_display[action_key_str] = action_key_str
                combo_map.setdefault(action_key_str, []).append(("action_key", "action_key", f"Action Key ({action_key_str})"))

        # 3. Application menu shortcuts
        for aid, edit in self.menu_shortcut_edits.items():
            seq = edit.keySequence()
            sc_str = seq.toString(QKeySequence.SequenceFormat.PortableText).strip()
            norm = normalize_menu_shortcut(sc_str)
            if norm:
                title = self._get_menu_action_title(aid)
                key = norm.upper()
                if key not in combo_display:
                    combo_display[key] = norm
                combo_map.setdefault(key, []).append(("menu", aid, title))

        conflicts = {c: items for c, items in combo_map.items() if len(items) > 1}
        has_conflicts = len(conflicts) > 0

        # Update Top Pane Conflict Banner (Card 2)
        if hasattr(self, "conflict_banner") and self.conflict_banner:
            if has_conflicts:
                conflict_details = []
                for combo_key, items in sorted(conflicts.items()):
                    disp = combo_display.get(combo_key, combo_key)
                    names = " and ".join(title for _, _, title in items)
                    conflict_details.append(f"• <b>{disp}</b> is assigned to <b>{names}</b>")
                self.conflict_banner_label.setText(
                    "<b>Hotkey conflict:</b> Multiple actions or panels share the same shortcut:<br>"
                    + "<br>".join(conflict_details)
                    + "<br><span style='font-size: 11px; opacity: 0.8;'>Conflicting shortcuts will only navigate to the first matching target.</span>"
                )
                self.conflict_banner.setVisible(True)
            else:
                self.conflict_banner.setVisible(False)

        # Update Menu Conflict Banner (Card 3)
        menu_conflicts = {c: items for c, items in conflicts.items() if any(cat == "menu" for cat, _, _ in items)}
        if hasattr(self, "menu_conflict_banner") and self.menu_conflict_banner:
            if menu_conflicts:
                menu_details = []
                for combo_key, items in sorted(menu_conflicts.items()):
                    disp = combo_display.get(combo_key, combo_key)
                    names = " and ".join(title for _, _, title in items)
                    menu_details.append(f"• <b>{disp}</b> is assigned to <b>{names}</b>")
                self.menu_conflict_banner_label.setText(
                    "<b>Shortcut conflict:</b> Multiple actions share the same shortcut:<br>"
                    + "<br>".join(menu_details)
                )
                self.menu_conflict_banner.setVisible(True)
            else:
                self.menu_conflict_banner.setVisible(False)

        # Update Pane hotkey per-row styling and warnings
        conflicting_pids = {target_id for items in conflicts.values() for cat, target_id, _ in items if cat == "pane"}
        for pid, edit in self.pane_hotkey_edits.items():
            warn_lbl = self.pane_conflict_labels.get(pid)
            if pid in conflicting_pids:
                raw_text = edit.text().strip()
                combo = normalize_hotkey_combo(raw_text).upper()
                other_titles = [t for cat, target_id, t in combo_map.get(combo, []) if target_id != pid]
                other_str = ", ".join(other_titles)

                edit.setStyleSheet(
                    "font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 700; "
                    "border: 1.5px solid #fb4934; background: rgba(251, 73, 52, 0.15); "
                    "color: #ff9999; border-radius: 4px; padding: 4px 8px;"
                )
                if warn_lbl:
                    warn_lbl.setText(f"Conflict: also assigned to {other_str}")
                    warn_lbl.setVisible(True)
            else:
                edit.setStyleSheet(
                    "font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px; font-weight: 600; "
                    "border: 1px solid rgba(255, 255, 255, 0.15); border-radius: 4px; padding: 4px 8px;"
                )
                if warn_lbl:
                    warn_lbl.setText("")
                    warn_lbl.setVisible(False)

        # Update Menu shortcut per-row styling and warnings
        conflicting_aids = {target_id for items in conflicts.values() for cat, target_id, _ in items if cat == "menu"}
        for aid, edit in self.menu_shortcut_edits.items():
            warn_lbl = self.menu_conflict_labels.get(aid)
            if aid in conflicting_aids:
                seq = edit.keySequence()
                sc_str = seq.toString(QKeySequence.SequenceFormat.PortableText).strip()
                norm = normalize_menu_shortcut(sc_str).upper()
                other_titles = [t for cat, target_id, t in combo_map.get(norm, []) if target_id != aid]
                other_str = ", ".join(other_titles)
                edit.set_conflict_style(True)
                if warn_lbl:
                    warn_lbl.setText(f"Conflict: also assigned to {other_str}")
                    warn_lbl.setVisible(True)
            else:
                edit.set_conflict_style(False)
                if warn_lbl:
                    warn_lbl.setText("")
                    warn_lbl.setVisible(False)

        return has_conflicts

    def _on_action_preset_changed(self, index: int):
        if not hasattr(self.window, "settings_action_key_combo"):
            return
        val = self.window.settings_action_key_combo.itemData(index)
        if val and hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("hotkeys", "action_key", val)
            self._validate_hotkey_conflicts()
            if hasattr(self.window, "statusBar") and self.window.statusBar():
                self.window.statusBar().showMessage(f"Action Key set to {val.replace('_', '+').upper()}", 3000)

    def _on_terminal_precedence_toggled(self, checked: bool):
        if hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("hotkeys", "terminal_precedence", checked)

    def _on_pane_hotkey_changed(self, pane_id: str, combo_str: str):
        normalized = normalize_hotkey_combo(combo_str)
        sm = getattr(self.window, "settings_manager", None)
        if sm:
            bindings = dict(sm.hotkey_bindings)
            if normalized:
                bindings[pane_id] = normalized
            elif pane_id in bindings:
                del bindings[pane_id]
            sm.set("hotkeys", "bindings", bindings)

            # Update legacy hotkey_panes for digits 0-9
            panes = list(sm.hotkey_panes)
            while len(panes) < 10:
                panes.append("none")

            for d in range(10):
                target_combo = f"Action + {d}".upper()
                assigned_pid = next((p for p, c in bindings.items() if c.upper() == target_combo), None)
                if assigned_pid:
                    panes[d] = assigned_pid
                elif panes[d] == pane_id and not normalized:
                    panes[d] = "none"
            sm.set("hotkeys", "panes", panes)

        # Validate conflicts across all inputs
        self._validate_hotkey_conflicts()

        if hasattr(self.window, "statusBar") and self.window.statusBar():
            title = self._get_pane_title(pane_id)
            if normalized:
                self.window.statusBar().showMessage(f"Shortcut for {title} set to {normalized}", 3000)
            else:
                self.window.statusBar().showMessage(f"Shortcut for {title} cleared", 3000)

    def _reset_pane_mappings_to_default(self):
        sm = getattr(self.window, "settings_manager", None)
        default_bindings = {DEFAULT_PANES[i]: f"Action + {i}" for i in range(len(DEFAULT_PANES))}
        if sm:
            sm.set("hotkeys", "bindings", default_bindings)
            sm.set("hotkeys", "panes", list(DEFAULT_PANES))

        for pid, edit in self.pane_hotkey_edits.items():
            expected = default_bindings.get(pid, "")
            edit.set_canonical_combo(expected)

        self._validate_hotkey_conflicts()

        if hasattr(self.window, "statusBar") and self.window.statusBar():
            self.window.statusBar().showMessage("Restored default pane navigation shortcuts.", 3000)

    def _on_menu_shortcut_changed(self, action_id: str, sequence: QKeySequence):
        seq_str = sequence.toString(QKeySequence.SequenceFormat.PortableText)
        if hasattr(self.window, "settings_manager"):
            self.window.settings_manager.set("shortcuts", action_id, seq_str)
        if hasattr(self.window, "menu_actions") and action_id in self.window.menu_actions:
            action = self.window.menu_actions[action_id]
            norm = normalize_menu_shortcut(seq_str)
            if norm:
                seq = QKeySequence(norm)
                if action_id == "zoom_in":
                    shortcuts = [seq]
                    if norm.endswith("+"):
                        shortcuts.append(QKeySequence(norm[:-1] + "="))
                    elif norm.endswith("="):
                        shortcuts.append(QKeySequence(norm[:-1] + "+"))
                    action.setShortcuts(shortcuts)
                else:
                    action.setShortcut(seq)
            else:
                action.setShortcut(QKeySequence())

        # Validate conflicts across all inputs
        has_conflicts = self._validate_hotkey_conflicts()

        if hasattr(self.window, "statusBar") and self.window.statusBar():
            title = self._get_menu_action_title(action_id)
            warn_lbl = self.menu_conflict_labels.get(action_id)
            if has_conflicts and warn_lbl and warn_lbl.isVisible():
                self.window.statusBar().showMessage(f"Warning for {title}: {warn_lbl.text()}", 4000)
            elif seq_str:
                self.window.statusBar().showMessage(f"Shortcut for {title} set to {seq_str}", 3000)
            else:
                self.window.statusBar().showMessage(f"Shortcut for {title} cleared", 3000)

    def _reset_menu_shortcuts_to_default(self):
        for aid, _, default_sc in DEFAULT_MENU_SHORTCUTS:
            norm = normalize_menu_shortcut(default_sc)
            seq = QKeySequence(norm)
            if aid in self.menu_shortcut_edits:
                self.menu_shortcut_edits[aid].blockSignals(True)
                self.menu_shortcut_edits[aid].setKeySequence(seq)
                self.menu_shortcut_edits[aid].blockSignals(False)
            if hasattr(self.window, "settings_manager"):
                self.window.settings_manager.set("shortcuts", aid, default_sc)
            if hasattr(self.window, "menu_actions") and aid in self.window.menu_actions:
                action = self.window.menu_actions[aid]
                if aid == "zoom_in":
                    shortcuts = [seq]
                    if norm.endswith("+"):
                        shortcuts.append(QKeySequence(norm[:-1] + "="))
                    elif norm.endswith("="):
                        shortcuts.append(QKeySequence(norm[:-1] + "+"))
                    action.setShortcuts(shortcuts)
                else:
                    action.setShortcut(seq)
        self._validate_hotkey_conflicts()
        if hasattr(self.window, "statusBar") and self.window.statusBar():
            self.window.statusBar().showMessage("Restored default application menu shortcuts.", 3000)

    def sync_from_settings(self):
        """Keep UI in sync with settings_manager without triggering feedback loops."""
        sm = getattr(self.window, "settings_manager", None)
        if not sm:
            return

        # 1. Terminal precedence
        if hasattr(self, "slider_term") and self.slider_term:
            self.slider_term.blockSignals(True)
            self.slider_term.setChecked(sm.hotkeys_terminal_precedence)
            self.slider_term.blockSignals(False)

        # 2. Action key combo
        if hasattr(self.window, "settings_action_key_combo") and self.window.settings_action_key_combo:
            combo = self.window.settings_action_key_combo
            cur_key = sm.action_key.lower()
            idx = combo.findData(cur_key)
            combo.blockSignals(True)
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                custom_title = f"{cur_key.replace('_', '+').upper()} (Custom)"
                combo.addItem(custom_title, cur_key)
                combo.setCurrentIndex(combo.count() - 1)
            combo.blockSignals(False)

        # 3. Pane navigation hotkey edits
        if hasattr(self, "pane_hotkey_edits") and self.pane_hotkey_edits:
            bindings = sm.hotkey_bindings
            for pid, edit in self.pane_hotkey_edits.items():
                expected = bindings.get(pid, "")
                if edit.get_canonical_combo() != expected:
                    edit.set_canonical_combo(expected)
            self._validate_hotkey_conflicts()

        # 4. Menu shortcut edits
        if hasattr(self, "menu_shortcut_edits") and self.menu_shortcut_edits:
            for aid, edit in self.menu_shortcut_edits.items():
                default_sc = edit.property("_default_sc") or ""
                sc = sm.get("shortcuts", aid, default=default_sc)
                if sc is None:
                    sc = default_sc
                norm = normalize_menu_shortcut(sc)
                seq = QKeySequence(norm)
                if edit.keySequence() != seq:
                    edit.blockSignals(True)
                    edit.setKeySequence(seq)
                    edit.blockSignals(False)
            self._validate_hotkey_conflicts()

        # 5. Zoom controls sync
        zoom_val = getattr(self.window, "zoom_factor", sm.zoom)
        zoom_pct = int(round(zoom_val * 100))
        if hasattr(self.window, "zoom_slider") and self.window.zoom_slider:
            self.window.zoom_slider.blockSignals(True)
            self.window.zoom_slider.setValue(zoom_pct)
            self.window.zoom_slider.blockSignals(False)
        if hasattr(self.window, "zoom_percent_label") and self.window.zoom_percent_label:
            self.window.zoom_percent_label.setText(f"{zoom_pct}%")

        # Settings sidebar icon
        if hasattr(self.window, "settings_button_logo_toggle") and self.window.settings_button_logo_toggle:
            self.window.settings_button_logo_toggle.blockSignals(True)
            self.window.settings_button_logo_toggle.setChecked(sm.settings_button_logo)
            self.window.settings_button_logo_toggle.blockSignals(False)

        # 7. Experiment panel resolved defaults toggle
        if hasattr(self, "defaults_toggle") and self.defaults_toggle:
            is_def = bool(sm.get("appearance", "show_resolved_defaults", default=False))
            self.defaults_toggle.blockSignals(True)
            self.defaults_toggle.setChecked(is_def)
            self.defaults_toggle.blockSignals(False)

        if hasattr(self, "method_agent_defaults_toggle") and self.method_agent_defaults_toggle:
            is_ag = bool(sm.get("appearance", "show_method_agent_defaults", default=False))
            self.method_agent_defaults_toggle.blockSignals(True)
            self.method_agent_defaults_toggle.setChecked(is_ag)
            self.method_agent_defaults_toggle.blockSignals(False)

        if hasattr(self, "method_model_defaults_toggle") and self.method_model_defaults_toggle:
            is_mo = bool(sm.get("appearance", "show_method_model_defaults", default=False))
            self.method_model_defaults_toggle.blockSignals(True)
            self.method_model_defaults_toggle.setChecked(is_mo)
            self.method_model_defaults_toggle.blockSignals(False)

        if hasattr(self, "all_defaults_toggle") and self.all_defaults_toggle:
            is_all = bool(sm.get("appearance", "show_all_default_params", default=False))
            self.all_defaults_toggle.blockSignals(True)
            self.all_defaults_toggle.setChecked(is_all)
            self.all_defaults_toggle.blockSignals(False)



    def _build_terminal_page(self) -> QWidget:
        """Shell, start folder and venv for the Terminal pane, saved to [terminal] in settings.toml."""
        from .terminal.pty_session import available_shells, get_default_shell

        sm = self.window.settings_manager
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        card = QFrame()
        card.setObjectName("card")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(18, 16, 18, 16)
        c_layout.setSpacing(12)
        c_layout.addWidget(label("Shell", "cardTitle"))
        sub = label("Changes apply to the next shell you open. Use Restart terminal to apply them now.", "muted")
        sub.setWordWrap(True)
        c_layout.addWidget(sub)

        def row(caption, *widgets):
            line = QHBoxLayout()
            line.setSpacing(10)
            cap = label(caption, "muted")
            cap.setFixedWidth(110)
            line.addWidget(cap)
            for widget in widgets:
                line.addWidget(widget)
            line.addStretch()
            c_layout.addLayout(line)

        # Shell picker: auto-detect, the shells found on this machine, or a custom command line
        self.terminal_shell_combo = QComboBox()
        self.terminal_shell_combo.setMaximumWidth(320)
        default = get_default_shell()
        self.terminal_shell_combo.addItem(f"Auto-detect ({Path(default[0]).stem})", "")
        self.terminal_shell_combo.setItemData(0, " ".join(default), Qt.ItemDataRole.ToolTipRole)
        for name, command in available_shells():
            self.terminal_shell_combo.addItem(name, command)
            self.terminal_shell_combo.setItemData(self.terminal_shell_combo.count() - 1, command,
                                                  Qt.ItemDataRole.ToolTipRole)
        self.terminal_shell_combo.addItem("Custom command…", None)
        row("Shell", self.terminal_shell_combo)

        self.terminal_shell_edit = QLineEdit()
        self.terminal_shell_edit.setPlaceholderText('e.g. "C:/tools/nu.exe" or /bin/zsh -l')
        self.terminal_shell_edit.setMaximumWidth(420)
        self.terminal_shell_edit.setStyleSheet("font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px;")
        row("Command", self.terminal_shell_edit)

        self.terminal_shell_status = label("", "muted")
        self.terminal_shell_status.setWordWrap(True)
        c_layout.addWidget(self.terminal_shell_status)

        # Start folder
        self.terminal_cwd_edit = QLineEdit(sm.terminal_cwd)
        self.terminal_cwd_edit.setPlaceholderText(f"Launch folder ({Path.cwd().name})")
        self.terminal_cwd_edit.setMaximumWidth(320)
        browse = QPushButton("Browse…")
        browse.setCursor(Qt.CursorShape.PointingHandCursor)
        browse.clicked.connect(self._browse_terminal_cwd)
        row("Start folder", self.terminal_cwd_edit, browse)

        # Project venv
        venv_line = QHBoxLayout()
        venv_info = QVBoxLayout()
        venv_info.setSpacing(1)
        venv_title = label("Use the project venv")
        venv_title.setStyleSheet("font-weight: 500; font-size: 12px;")
        venv_info.addWidget(venv_title)
        venv_sub = label("Puts venv/ first on PATH, so python, pytest and run_pipeline.py use the backend environment.", "muted")
        venv_sub.setWordWrap(True)
        venv_info.addWidget(venv_sub)
        venv_line.addLayout(venv_info, 1)
        self.terminal_venv_slider = ToggleSlider(checked=sm.terminal_activate_venv)
        self.terminal_venv_slider.setAccessibleName("Toggle project venv in the terminal")
        self.terminal_venv_slider.toggled.connect(lambda on: sm.set("terminal", "activate_venv", bool(on)))
        venv_line.addWidget(self.terminal_venv_slider)
        c_layout.addLayout(venv_line)

        restart_row = QHBoxLayout()
        btn_restart = QPushButton("Restart terminal")
        btn_restart.setToolTip("Close the current shell and open a new one with these settings")
        btn_restart.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_restart.clicked.connect(self._restart_terminal)
        restart_row.addWidget(btn_restart)
        restart_row.addStretch()
        c_layout.addLayout(restart_row)

        layout.addWidget(card)
        from .site_launch import SiteSetup
        layout.addWidget(SiteSetup(self))
        layout.addStretch()

        self._load_terminal_shell(sm.terminal_shell)
        self.terminal_shell_combo.currentIndexChanged.connect(self._on_terminal_shell_picked)
        self.terminal_shell_edit.editingFinished.connect(self._on_terminal_command_edited)
        self.terminal_cwd_edit.editingFinished.connect(
            lambda: sm.set("terminal", "cwd", self.terminal_cwd_edit.text().strip()))
        return container

    def _load_terminal_shell(self, shell: str):
        """Show the saved shell: a matching preset, auto-detect, or Custom with its command line."""
        index = self.terminal_shell_combo.findData(shell) if shell else 0
        custom = index < 0
        if custom:
            index = self.terminal_shell_combo.count() - 1
        self.terminal_shell_combo.blockSignals(True)
        self.terminal_shell_combo.setCurrentIndex(index)
        self.terminal_shell_combo.blockSignals(False)
        self.terminal_shell_edit.setText(shell)
        self.terminal_shell_edit.setEnabled(custom)
        self._update_terminal_shell_status(shell)

    def _update_terminal_shell_status(self, shell: str):
        from .terminal.pty_session import parse_shell_command

        if shell and parse_shell_command(shell) is None:
            self.terminal_shell_status.setText("That program was not found, so the default shell will be used.")
            self.terminal_shell_status.setObjectName("configError")
        else:
            self.terminal_shell_status.setText("")
            self.terminal_shell_status.setObjectName("muted")
        self.terminal_shell_status.style().polish(self.terminal_shell_status)
        self.terminal_shell_status.setVisible(bool(self.terminal_shell_status.text()))

    def _on_terminal_shell_picked(self, index: int):
        command = self.terminal_shell_combo.itemData(index)
        if command is None:  # Custom: edit the command line; it is saved when editing finishes
            self.terminal_shell_edit.setEnabled(True)
            self.terminal_shell_edit.setFocus()
            return
        self.terminal_shell_edit.setText(command)
        self.terminal_shell_edit.setEnabled(False)
        self._update_terminal_shell_status(command)
        self.window.settings_manager.set("terminal", "shell", command)

    def _on_terminal_command_edited(self):
        command = self.terminal_shell_edit.text().strip()
        self._update_terminal_shell_status(command)
        self.window.settings_manager.set("terminal", "shell", command)

    def _browse_terminal_cwd(self):
        start = self.terminal_cwd_edit.text().strip() or str(Path.cwd())
        folder = QFileDialog.getExistingDirectory(self, "Terminal start folder", start)
        if folder:
            self.terminal_cwd_edit.setText(folder)
            self.window.settings_manager.set("terminal", "cwd", folder)

    def _restart_terminal(self):
        panel = getattr(self.window, "terminal_panel", None)
        if panel is None:
            return
        command, cwd, env = self.window.terminal_config()
        panel.configure(command=command, cwd=cwd, env=env)
        panel.terminal.restart_session()  # the active shell; other open shells pick it up on restart
        self.window.statusBar().showMessage("Terminal restarted with the new settings.", 4000)

    def _build_backend_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        backend_card = QFrame()
        backend_card.setObjectName("card")
        bc_layout = QVBoxLayout(backend_card)
        bc_layout.setContentsMargins(18, 16, 18, 16)
        bc_layout.setSpacing(12)

        bc_layout.addWidget(label("Training pipeline", "cardTitle"))
        bc_sub = label("ThetaIDE composes configs in-process and runs run_pipeline.py with this Python, "
                       "streaming its output and metrics. There is no server to start.", "muted")
        bc_sub.setWordWrap(True)
        bc_layout.addWidget(bc_sub)

        # Status row
        status_box = QHBoxLayout()
        status_box.setSpacing(10)
        self.window.settings_backend_status = label("Status: Checking the training pipeline…", "muted")
        self.window.settings_backend_status.setStyleSheet("font-size: 12px; font-weight: 600;")
        status_box.addWidget(self.window.settings_backend_status, 1)

        btn_test = QPushButton("Check pipeline")
        btn_test.setToolTip("Compose the base experiment to confirm the training environment is installed")
        btn_test.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_test.clicked.connect(self.window.test_backend_connection)
        status_box.addWidget(btn_test)
        bc_layout.addLayout(status_box)

        mono = "font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 12px;"
        for caption, value in (("Python:", sys.executable), ("Project:", str(PROJECT_ROOT))):
            row = QHBoxLayout()
            row.setSpacing(10)
            cap = label(caption, "muted")
            cap.setStyleSheet("font-weight: 500;")
            cap.setFixedWidth(60)
            row.addWidget(cap)
            field = QLineEdit(value)
            field.setReadOnly(True)
            field.setStyleSheet(mono)
            row.addWidget(field, 1)
            bc_layout.addLayout(row)

        tip_lbl = label('Start ThetaIDE from the project venv:  pip install -e ".[gui]"  then  python -m frontend',
                        "muted")
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 11px; font-style: italic;")
        bc_layout.addWidget(tip_lbl)

        layout.addWidget(backend_card)
        layout.addWidget(self._build_resource_budget_card())
        layout.addWidget(self._build_run_notifications_card())
        layout.addStretch()
        return container

    def _build_resource_budget_card(self):
        from src.app.services.resource_budget import ResourceBudget, host_capacity, learned_usage, GIB
        card = QFrame()
        card.setObjectName('card')
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.addWidget(label('System usage limits', 'cardTitle'))
        description = label(
            'Choose how much of your system Theta can use. Job and trial sizes are estimated automatically '
            'from the workload and learned usage. New work starts as capacity becomes available.', 'muted')
        description.setWordWrap(True)
        layout.addWidget(description)
        settings = self.window.settings_manager
        self.resource_controls = {}
        for key, title, default, minimum, maximum, suffix in (
            ('cpu_percent', 'Theta CPU budget (% of system)', 75, 1, 100, '%'),
            ('memory_percent', 'Theta RAM budget (% of system)', 75, 10, 95, '%'),
            ('max_parallel', 'Maximum training workers', 0, 0, 100000, ''),
            ('threads_per_job', 'CPU threads per worker', 2, 1, 1024, ''),
        ):
            row = QHBoxLayout()
            row.addWidget(label(title), 1)
            control = QSpinBox()
            control.setRange(minimum, maximum)
            control.setSuffix(suffix)
            if key == 'max_parallel':
                control.setSpecialValueText('Automatic')
                control.setToolTip('Shared limit across local experiments, training seeds and trials. '
                                   '1 runs one worker at a time; 0 uses available resource budgets.')
            try:
                value = float(settings.get('training', key, default=default))
                control.setValue(int(value))
            except (TypeError, ValueError, OverflowError):
                control.setValue(default)
            control.setObjectName('training_' + key)
            row.addWidget(control)
            layout.addLayout(row)
            self.resource_controls[key] = control
            control.valueChanged.connect(lambda value, k=key: settings.set('training', k, value))
        self.resource_budget_summary = label('', 'muted')
        self.resource_budget_summary.setWordWrap(True)
        layout.addWidget(self.resource_budget_summary)
        def refresh():
            for key, control in self.resource_controls.items():
                try:
                    value = float(settings.get('training', key, default=control.value()))
                    value = int(value)
                    control.blockSignals(True)
                    control.setValue(value)
                except (TypeError, ValueError, OverflowError):
                    pass
                finally:
                    control.blockSignals(False)
            values = {k: c.value() for k, c in self.resource_controls.items()}
            budget = ResourceBudget.from_settings(values)
            try:
                cores, total, available = host_capacity()
            except Exception:
                self.resource_budget_summary.setText('System measurements are unavailable. New jobs will wait.')
                return
            profiles, typical_memory, typical_cpu = 0, .75 * GIB, 1.
            active, used_memory, used_cpu = 0, 0, 0
            jobs = getattr(self.window, 'jobs', None)
            if jobs is not None:
                try:
                    usage = learned_usage(jobs.jobs_dir / 'resource_usage.db')
                    profiles, typical_memory, typical_cpu = usage['profiles'], usage['memory'], usage['cpu']
                    active, used_memory, used_cpu = usage['active'], usage['used_memory'], usage['used_cpu']
                except Exception:
                    pass  # the first run has no profile database yet
            room = max(0, total * budget.memory_percent / 100 - used_memory)
            typical_cpu = max(typical_cpu, budget.threads(cores))
            slots = max(0, min(int((budget.cpu_slots(cores) - used_cpu) / typical_cpu), int(room // typical_memory)))
            if budget.max_parallel:
                slots = min(slots, max(0, budget.max_parallel - active))
            source = f'Learned estimate ({profiles} workload profiles)' if profiles else 'Initial estimate'
            self.resource_budget_summary.setText(
                f'{source}: room for approximately {slots} more typical CPU tasks · '
                f'{total * budget.memory_percent / 100 / GIB:.1f} GiB Theta RAM target. '
                f'{active} active workers · '
                f'{"worker limit " + str(budget.max_parallel) if budget.max_parallel else "automatic worker limit"}. '
                'Per-experiment and trial limits also apply. Resource budgets can reduce concurrency further.')
        for control in self.resource_controls.values():
            control.valueChanged.connect(refresh)
        from PyQt6.QtCore import QTimer
        self.resource_budget_timer = QTimer(card)
        self.resource_budget_timer.timeout.connect(refresh)
        self.resource_budget_timer.start(2000)
        refresh()
        actions = QHBoxLayout()
        for title, method in (('Pause new starts', 'pause_queue'), ('Resume waiting jobs', 'start_queue')):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, m=method: getattr(self.window.jobs, m)())
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)
        note = label('CPU and RAM targets apply to Theta workers, with 5 percentage points of tolerance. '
                     'Overshoots hold new starts; running tasks continue. Other applications do not block '
                     'Theta admission or cause Theta to stop research jobs. '
                     'Worker limits include evaluation performed by training workers. These controls apply to local '
                     'work in this workspace; they are not OS quotas or GPU VRAM limits.', 'muted')
        note.setWordWrap(True)
        layout.addWidget(note)
        return card

    def _build_run_notifications_card(self) -> QFrame:
        """Build controls for the completion toast and its chime."""
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 16)
        card_layout.setSpacing(10)
        card_layout.addWidget(label("Run notifications", "cardTitle"))

        description = label("Choose what ThetaIDE shows when a run finishes.", "muted")
        description.setWordWrap(True)
        card_layout.addWidget(description)

        settings = self.window.settings_manager
        self.notification_toggles = {}
        for key, title in (("toast", "Show completion toast"), ("sound", "Play a chime")):
            row = QHBoxLayout()
            row.addWidget(label(title))
            row.addStretch()
            toggle = ToggleSlider(checked=bool(settings.get("notifications", key, default=True)))
            toggle.toggled.connect(lambda checked, setting=key: settings.set("notifications", setting, checked))
            row.addWidget(toggle)
            card_layout.addLayout(row)
            self.notification_toggles[key] = toggle

        try_button = QPushButton("Try notification")
        try_button.setCursor(Qt.CursorShape.PointingHandCursor)
        try_button.clicked.connect(self._try_run_notification)
        card_layout.addWidget(try_button, 0, Qt.AlignmentFlag.AlignLeft)
        return card

    def _try_run_notification(self):
        """Preview the selected run-completion notification settings."""
        from uuid import uuid4

        from .run_notifier import describe

        notifier = self.window.run_notifier
        run = {"id": f"settings-preview-{uuid4().hex}", "simulated": True,
               "config": {"name": "example_run"}, "metrics": []}
        if self.window.settings_manager.get("notifications", "toast", default=True):
            notifier.show(describe("completed", "example_run"), run)
        if self.window.settings_manager.get("notifications", "sound", default=True):
            notifier.play("completed")

    def _build_editor_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        card = QFrame()
        card.setObjectName("card")
        content = QVBoxLayout(card)
        content.setContentsMargins(18, 16, 18, 16)
        content.setSpacing(12)
        content.addWidget(label("Text Editor", "cardTitle"))
        hint = label("Choose the editor Theta uses when opening text files. Enable the Text Editor core plugin to edit files in an Editor panel.", "muted")
        hint.setWordWrap(True)
        content.addWidget(hint)
        row = QHBoxLayout()
        row.addWidget(label("Default text editor:"))
        self.default_editor_combo = QComboBox()
        self.default_editor_combo.setMinimumWidth(240)
        row.addWidget(self.default_editor_combo, 1)
        content.addLayout(row)
        wrap_row = QHBoxLayout()
        wrap_row.addWidget(label("Wrap long lines"), 1)
        self.editor_wrap_toggle = ToggleSlider()
        wrap_row.addWidget(self.editor_wrap_toggle)
        content.addLayout(wrap_row)
        self.default_editor_combo.currentIndexChanged.connect(self._select_default_editor)
        self.editor_wrap_toggle.toggled.connect(lambda value: self.window.settings_manager.set("editor", "wrap_lines", value))
        manage = QPushButton("Manage core plugins")
        manage.clicked.connect(lambda: self.show_tab("core_plugins"))
        content.addWidget(manage)
        layout.addWidget(card)
        layout.addStretch()
        self.window.editor_service.providersChanged.connect(self._refresh_editor_settings)
        self.window.settings_manager.changed.connect(self._refresh_editor_settings)
        self._refresh_editor_settings()
        return container

    def _refresh_editor_settings(self):
        sm = self.window.settings_manager
        self.default_editor_combo.blockSignals(True)
        self.default_editor_combo.clear()
        for provider_id, name in self.window.editor_service.providers():
            self.default_editor_combo.addItem(name, provider_id)
        selected = self.default_editor_combo.findData(sm.default_text_editor)
        if selected < 0:
            self.default_editor_combo.addItem(f"{sm.default_text_editor} (unavailable; uses system editor)", sm.default_text_editor)
            selected = self.default_editor_combo.count() - 1
        self.default_editor_combo.setCurrentIndex(selected)
        self.default_editor_combo.blockSignals(False)
        self.editor_wrap_toggle.blockSignals(True)
        self.editor_wrap_toggle.setChecked(bool(sm.get("editor", "wrap_lines", default=False)))
        self.editor_wrap_toggle.blockSignals(False)

    def _select_default_editor(self, index):
        value = self.default_editor_combo.itemData(index)
        if value:
            self.window.settings_manager.default_text_editor = value

    def _build_storage_page(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(12)

        storage_card = QFrame()
        storage_card.setObjectName("card")
        sc_layout = QVBoxLayout(storage_card)
        sc_layout.setContentsMargins(18, 16, 18, 16)
        sc_layout.setSpacing(12)

        sc_layout.addWidget(label("Workspace & Storage", "cardTitle"))
        sc_sub = label("Manage local experiment databases, filesystems, and layout preferences.", "muted")
        sc_sub.setWordWrap(True)
        sc_layout.addWidget(sc_sub)

        # Storage info
        info_layout = QVBoxLayout()
        info_layout.setSpacing(8)

        root_path = getattr(self.window.store, "root", "Unknown")
        row_root = QHBoxLayout()
        lbl_root = label("Data root:", "muted")
        lbl_root.setFixedWidth(90)
        row_root.addWidget(lbl_root)
        val_root = label(str(root_path))
        val_root.setWordWrap(True)
        val_root.setStyleSheet("font-family: Menlo, Monaco, 'Consolas', monospace; font-size: 11px;")
        val_root.setToolTip(str(root_path))
        row_root.addWidget(val_root, 1)
        info_layout.addLayout(row_root)

        row_runs = QHBoxLayout()
        lbl_runs = label("Database:", "muted")
        lbl_runs.setFixedWidth(90)
        row_runs.addWidget(lbl_runs)
        self.window.settings_runs_count_label = label(
            f"Total run records:  {len(getattr(self.window, 'runs', []))} runs", "muted"
        )
        self.window.settings_runs_count_label.setStyleSheet("font-weight: 500; font-size: 12px;")
        row_runs.addWidget(self.window.settings_runs_count_label, 1)
        info_layout.addLayout(row_runs)

        sc_layout.addLayout(info_layout)

        # Buttons row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_open_settings = QPushButton("Open settings.toml")
        btn_open_settings.setToolTip(
            f"Open settings file in your default editor\n{self.window.settings_manager.workspace_settings_path}"
        )
        btn_open_settings.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_settings.clicked.connect(self.window.open_settings_file)
        btn_row.addWidget(btn_open_settings)

        btn_reset_layout = QPushButton("Reset UI layout")
        btn_reset_layout.setToolTip("Restore default pane sizes and layout")
        btn_reset_layout.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_reset_layout.clicked.connect(
            lambda: (
                self.window.restoreState(self.window.default_layout),
                self.window.statusBar().showMessage("Restored default UI layout.", 4000),
            )
        )
        btn_row.addWidget(btn_reset_layout)
        btn_row.addStretch()

        sc_layout.addLayout(btn_row)

        hint = label("Changes to settings.toml are automatically hot-reloaded across the IDE.", "muted")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; font-style: italic;")
        sc_layout.addWidget(hint)

        layout.addWidget(storage_card)
        layout.addStretch()
        return container

    # ── Interaction & View Control ───────────────────────────────────────────

    def showEvent(self, event):
        """Settings always shows a page: the first category until another is chosen."""
        super().showEvent(event)
        if self.active_tab_id is None:
            self.show_tab(TAB_SPECS[0]["id"])

    def on_tab_button_clicked(self, tab_id: str):
        self.show_tab(tab_id)

    def show_tab(self, tab_id: str):
        """Activate the given category and show its page."""
        if tab_id not in self.page_widgets:
            return

        self.active_tab_id = tab_id
        for tid, btn in self.tab_buttons.items():
            btn.setChecked(tid == tab_id)

        spec = next((s for s in ALL_TAB_SPECS if s["id"] == tab_id), None)
        title = spec["title"] if spec else tab_id.title()
        subtitle = spec["subtitle"] if spec else ""
        icon_name = spec["icon"] if spec else "settings"
        self.settings_window.set_header(title, subtitle, icon_name)
        self.settings_window.current_tab_id = tab_id

        self.settings_window.pages_stack.setCurrentWidget(self.page_widgets[tab_id])
        self.settings_window.scroll_area.verticalScrollBar().setValue(0)  # every page opens at its top

    def refresh_styles(self):
        """Theme refresh: update custom styles, icons, and colors."""
        panel_col = theme_color("panel")
        surface_col = theme_color("surface")
        border_col = theme_color("border")
        accent_col = theme_color("accent")
        text_col = theme_color("text")
        muted_col = theme_color("muted")

        zoom = getattr(self.window, "zoom_factor", 1.0) if hasattr(self, "window") else 1.0
        font_12 = max(8, round(12 * zoom))
        font_20 = max(12, round(20 * zoom))

        if hasattr(self, "title_label") and self.title_label:
            self.title_label.setStyleSheet(f"font-size: {font_20}px; font-weight: 700;")
        if hasattr(self, "subtitle_label") and self.subtitle_label:
            self.subtitle_label.setStyleSheet(f"font-size: {font_12}px;")

        # Sidebar styling
        self.sidebar_frame.setStyleSheet(f"""
            QFrame#settingsSidebar {{
                background-color: {panel_col};
                border-right: 1px solid {border_col};
            }}
            QToolButton#settingsSideTab {{
                text-align: left;
                padding-left: 12px;
                border: 1px solid transparent;
                border-radius: 6px;
                background-color: transparent;
                color: {muted_col};
                font-size: {font_12}px;
                font-weight: 500;
            }}
            QToolButton#settingsSideTab:hover {{
                background-color: {surface_col};
                color: {text_col};
            }}
            QToolButton#settingsSideTab:checked {{
                background-color: {surface_col};
                border: 1px solid {accent_col};
                color: {accent_col};
                font-weight: 600;
            }}
            QFrame#settingsSidebarSeparator {{
                color: {border_col};
                background-color: {border_col};
                max-height: 1px;
                margin-top: 4px;
                margin-bottom: 4px;
            }}
        """)

        # Settings detail window styling
        self.settings_window.setStyleSheet(f"""
            QFrame#settingsOverlayLayer {{
                background-color: transparent;
                border: none;
            }}
            QFrame#settingsDetailCard {{
                background-color: {panel_col};
                border: none;
                border-right: 1px solid {border_col};
            }}
            QFrame#card {{
                background-color: {surface_col};
                border: 1px solid {border_col};
                border-radius: 8px;
            }}
            QFrame#placeholderCard {{
                background-color: {surface_col};
                border: 1px dashed {border_col};
                border-radius: 8px;
            }}
            QFrame#settingsSeparator {{
                color: {border_col};
                background-color: {border_col};
                max-height: 1px;
            }}
            QScrollArea#settingsScrollArea {{
                background-color: {panel_col};
                border: none;
            }}
            QScrollArea#settingsScrollArea > QWidget > QWidget {{
                background-color: {panel_col};
            }}
            QScrollBar:horizontal {{
                height: 0px;
                border: none;
            }}
            QKeySequenceEdit {{
                background-color: {surface_col};
                border: 1px solid {border_col};
                border-radius: 4px;
                padding: 4px 8px;
                color: {text_col};
                font-family: Menlo, Monaco, 'Consolas', monospace;
                font-size: {font_12}px;
                font-weight: 600;
            }}
            QKeySequenceEdit:focus {{
                border: 1px solid {accent_col};
            }}
        """)

        # Refresh icons on buttons
        for btn in self.tab_buttons.values():
            btn.refresh_icon()

        if self.active_tab_id:
            spec = next((s for s in ALL_TAB_SPECS if s["id"] == self.active_tab_id), None)
            if spec:
                self.settings_window.set_header(spec["title"], spec["subtitle"], spec["icon"])

