"""A rofi-style launcher: type to jump to any panel, plugin, settings page or menu command.

Open it with the "open_launcher" shortcut (Ctrl+Space by default, rebindable in Settings -> Hotkeys).
Type part of a name (letters in order are enough: "tbd" finds TensorBoard), move with the arrow keys,
Tab, Ctrl+N/P or Ctrl+J/K, open with Enter or a click, and close with Escape, a click outside, or the
shortcut again. With nothing typed, the items opened most recently come first.

Items are collected fresh each time it opens, so plugins switched on or off, hidden panes and renamed
shortcuts are always current. Matching is in plain functions (fuzzy_match, rank) so it can be tested alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from PyQt6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QIcon, QKeySequence, QPainter, QPainterPath
from PyQt6.QtWidgets import (QFrame, QLabel, QLineEdit, QListWidget, QListWidgetItem, QStyle, QStyledItemDelegate,
                             QVBoxLayout, QWidget)

from .theme import theme_color

CATEGORY_ORDER = ("Panel", "Plugin", "Settings", "Command")
RECENT_LIMIT = 12
CARD_WIDTH = 640
ROW_HEIGHT = 40
VISIBLE_ROWS = 9
ROLE_ITEM = Qt.ItemDataRole.UserRole
ROLE_MATCH = Qt.ItemDataRole.UserRole + 1


@dataclass
class LaunchItem:
    key: str  # unique and stable, e.g. "pane:monitor", "plugin:arxiv_papers", "settings:hotkeys"
    title: str
    category: str  # one of CATEGORY_ORDER
    action: Callable[[], None]
    detail: str = ""
    icon: str | None = None  # an SVG name in frontend/icons, or a path
    keywords: tuple[str, ...] = field(default_factory=tuple)
    shortcut: str = ""


# ── matching ─────────────────────────────────────────────────────────────────

def fuzzy_match(query: str, text: str):
    """(score, matched positions) if every letter of `query` appears in `text` in order, else None.

    Contiguous runs, matches at the start of the text or of a word, and short texts score higher, so
    "mon" ranks "Monitor" above "Command monitor" and "tb" finds "TensorBoard".
    """
    query, lower = query.lower(), text.lower()
    if not query:
        return 0, []
    start = lower.find(query)
    if start >= 0:  # a plain substring beats any scattered match
        at_word = start == 0 or not lower[start - 1].isalnum()
        score = 1000 + (300 if start == 0 else 150 if at_word else 0) - len(text)
        return score, list(range(start, start + len(query)))
    positions, i = [], 0
    for ch in query:
        i = lower.find(ch, i)
        if i < 0:
            return None
        positions.append(i)
        i += 1
    score = 500 - len(text)
    for n, pos in enumerate(positions):
        if pos == 0 or not lower[pos - 1].isalnum() or (text[pos].isupper() and pos > 0 and text[pos - 1].islower()):
            score += 40  # start of a word, or a capital inside one (TensorBoard)
        if n and pos == positions[n - 1] + 1:
            score += 25  # continues the previous match
        elif n:
            score -= min(20, pos - positions[n - 1])  # gap
    return score, positions


def match_item(query: str, item: LaunchItem):
    """(score, positions in the title) for every word of the query, or None if one doesn't match.

    Each word may match the title, a keyword, the category or the detail; title matches count most.
    """
    total, title_positions = 0, set()
    for word in query.split():
        best = None
        title_hit = fuzzy_match(word, item.title)
        if title_hit:
            best = title_hit[0] + 200
            title_positions.update(title_hit[1])  # highlighted even when an alias scores higher ("tb")
        for extra, weight in [(k, 100) for k in item.keywords] + [(item.category, 0)]:
            hit = fuzzy_match(word, extra) if extra else None
            if hit and (best is None or hit[0] + weight > best):
                best = hit[0] + weight
        # The description only counts as a plain substring: scattered letters across a sentence are noise.
        if best is None and item.detail and word.lower() in item.detail.lower():
            best = 300
        if best is None:
            return None
        total += best
    return total, sorted(title_positions)


def rank(items, query: str, recent=()):
    """Items matching `query`, best first, each as (item, title positions to highlight).

    With an empty query: recently opened items first (most recent first), then by category and title.
    """
    recent_index = {key: i for i, key in enumerate(recent)}
    category_index = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    if not query.strip():
        ordered = sorted(items, key=lambda it: (recent_index.get(it.key, len(recent_index)),
                                                category_index.get(it.category, 9), it.title.lower()))
        return [(it, []) for it in ordered]
    scored = []
    for it in items:
        hit = match_item(query, it)
        if hit is not None:
            bonus = 30 if it.key in recent_index else 0
            scored.append((-(hit[0] + bonus), category_index.get(it.category, 9), it.title.lower(), it, hit[1]))
    scored.sort(key=lambda row: row[:3])
    return [(row[3], row[4]) for row in scored]


# ── what the launcher lists ──────────────────────────────────────────────────

def _pane_hotkeys(window) -> dict[str, str]:
    """pane id -> "Ctrl+B 3" for panes the action-key digits reach."""
    manager = getattr(window, "hotkey_manager", None)
    if manager is None:
        return {}
    label = str(getattr(manager, "action_key_name", "ctrl+b")).replace("_", "+").title()
    out = {}
    for digit in range(10):
        try:
            pane = manager.get_pane_id_by_index(digit)
        except Exception:
            pane = None
        if pane and pane not in out:
            out[pane] = f"{label} {digit}"
    return out


def collect_items(window) -> list[LaunchItem]:
    """Every panel, plugin, settings page and menu command the window offers right now."""
    from .hotkeys import PANE_KEYWORDS

    items: list[LaunchItem] = []
    tabs = window.tabs
    hotkeys = _pane_hotkeys(window)
    aliases: dict[str, list[str]] = {}
    for alias, pane in PANE_KEYWORDS.items():
        aliases.setdefault(pane, []).append(alias.replace("_", " "))

    manager = getattr(window, "plugin_manager", None)
    plugin_of_tab = {}
    if manager is not None:
        for pid, context in getattr(manager, "contexts", {}).items():
            for tab_id in getattr(context, "_registered_tabs", []):
                plugin_of_tab[tab_id] = pid

    def open_pane(tab_id):
        hotkey_manager = getattr(window, "hotkey_manager", None)
        if hotkey_manager is not None:
            hotkey_manager.switch_to_pane(tab_id)  # also unhides a hidden pane
        else:
            tabs.setCurrentWidget(tabs.tabs[tab_id]["widget"])

    for tab_id in tabs.tab_order:
        entry = tabs.tabs[tab_id]
        notes = []
        bin_id = tabs.bin_of(tab_id) if hasattr(tabs, "bin_of") else None
        if bin_id:
            notes.append(f"in {tabs.bins[bin_id]['name']}")
        if not entry["visible"]:
            notes.append("hidden in the sidebar, opening shows it")
        pid = plugin_of_tab.get(tab_id)
        manifest = manager.manifests.get(pid) if manager is not None and pid else None
        title = entry["text"]
        items.append(LaunchItem(
            key=f"pane:{tab_id}", title=title, category="Plugin" if pid else "Panel",
            action=lambda t=tab_id: open_pane(t),
            detail="; ".join(notes) or (manifest.description if manifest else ""),
            icon=entry.get("icon"), keywords=tuple({tab_id.replace("_", " "), entry.get("short") or "",
                                                    *aliases.get(tab_id, [])} - {title, ""}),
            shortcut=hotkeys.get(tab_id, "")))

    # Plugins that are switched off: opening one switches it on and shows its panel
    if manager is not None:
        for pid, manifest in sorted(manager.manifests.items(), key=lambda kv: kv[1].name.lower()):
            if pid in manager.instances:
                continue
            items.append(LaunchItem(
                key=f"plugin:{pid}", title=f"Turn on {manifest.name}", category="Plugin",
                action=lambda p=pid: _enable_plugin(window, p),
                detail=manifest.description or "", icon=getattr(manifest, "icon", None) or "plugins_core",
                keywords=(manifest.name, pid.replace("_", " "), "enable", "plugin")))

    # Settings pages
    try:
        from .settings_view import ALL_TAB_SPECS
    except Exception:
        ALL_TAB_SPECS = []
    settings_view = getattr(window, "settings_view", None)
    for spec in ALL_TAB_SPECS:
        items.append(LaunchItem(
            key=f"settings:{spec['id']}", title=spec["title"], category="Settings",
            action=lambda tid=spec["id"]: _open_settings(window, tid),
            detail=spec.get("subtitle", ""), icon=spec.get("icon"),
            keywords=("settings", "preferences", spec["id"].replace("_", " ")),
            shortcut=hotkeys.get("settings", "") if spec is ALL_TAB_SPECS[0] else ""))
    if settings_view is None:
        items = [it for it in items if it.category != "Settings"]

    # Menu commands, with their current shortcuts
    for aid, action in getattr(window, "menu_actions", {}).items():
        if aid == "open_launcher":
            continue
        title = action.text().replace("&", "").rstrip("…").rstrip(".")
        sequence = action.shortcut()
        items.append(LaunchItem(
            key=f"command:{aid}", title=title, category="Command", action=action.trigger,
            detail="", icon="launch", keywords=(aid.replace("_", " "),),
            shortcut=sequence.toString(QKeySequence.SequenceFormat.NativeText) if not sequence.isEmpty() else ""))

    # Dynamic commands from CommandRegistry (contributed by plugins)
    command_reg = getattr(window, "command_registry", None)
    if command_reg is not None:
        for cmd in command_reg.get_commands():
            items.append(LaunchItem(
                key=f"cmd:{cmd.command_id}",
                title=cmd.title,
                category=cmd.category,
                action=cmd.callback,
                detail=cmd.detail,
                icon=cmd.icon or "launch",
                keywords=tuple(cmd.keywords) + (cmd.command_id.replace("_", " ").replace(".", " "),),
                shortcut=cmd.default_hotkey or "",
            ))

    return items


def _open_settings(window, tab_id):
    window.tabs.setCurrentWidget(window.tabs.settings_widget)
    window.settings_view.show_tab(tab_id)


def _enable_plugin(window, plugin_id):
    manager = window.plugin_manager
    slider = getattr(window, "plugin_sliders", {}).get(plugin_id)
    if slider is not None:
        slider.setChecked(True)  # the slider's handler enables it and saves the layout
    elif hasattr(window, "on_plugin_slider_toggled"):
        window.on_plugin_slider_toggled(plugin_id, True)
    else:
        manager.enable_plugin(plugin_id)
    context = manager.contexts.get(plugin_id)
    registered = getattr(context, "_registered_tabs", []) if context else []
    if registered and registered[0] in window.tabs.tabs:
        window.tabs.setCurrentWidget(window.tabs.tabs[registered[0]]["widget"])


# ── drawing ──────────────────────────────────────────────────────────────────

class _RowDelegate(QStyledItemDelegate):
    """icon · title (matched letters in the accent colour) · detail ......... shortcut  CATEGORY"""

    def __init__(self, launcher, parent=None):
        super().__init__(parent)
        self.launcher = launcher

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), ROW_HEIGHT)

    def paint(self, painter: QPainter, option, index):
        item: LaunchItem = index.data(ROLE_ITEM)
        positions = set(index.data(ROLE_MATCH) or [])
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        rect = option.rect.adjusted(6, 1, -6, -1)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if selected or option.state & QStyle.StateFlag.State_MouseOver:
            path = QPainterPath()
            path.addRoundedRect(QRectF(rect), 6, 6)
            painter.fillPath(path, QColor(theme_color("raised" if selected else "surface")))
        x = rect.left() + 10
        icon = self.launcher.icon_for(item.icon)
        if not icon.isNull():
            mode = QIcon.Mode.Normal
            state = QIcon.State.On if selected else QIcon.State.Off
            icon.paint(painter, QRect(x, rect.center().y() - 9, 18, 18), Qt.AlignmentFlag.AlignCenter, mode, state)
        x += 30

        # right side: category, then the shortcut to its left
        small = QFont(option.font)
        if option.font.pixelSize() > 0:  # the theme sets sizes in pixels (zoom)
            small.setPixelSize(max(10, round(option.font.pixelSize() * 0.85)))
        else:
            small.setPointSizeF(max(7.5, option.font.pointSizeF() * 0.85))
        small_metrics = QFontMetrics(small)
        right = rect.right() - 10
        painter.setFont(small)
        category = item.category.upper()
        cat_width = small_metrics.horizontalAdvance(category)
        painter.setPen(QColor(theme_color("accent" if selected else "comment")))
        painter.drawText(QRect(right - cat_width, rect.top(), cat_width, rect.height()),
                         Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight, category)
        right -= cat_width + 14
        if item.shortcut:
            width = small_metrics.horizontalAdvance(item.shortcut) + 12
            chip = QRectF(right - width, rect.center().y() - 10, width, 20)
            painter.setPen(QColor(theme_color("border")))
            painter.setBrush(QColor(theme_color("base")))
            painter.drawRoundedRect(chip, 4, 4)
            painter.setPen(QColor(theme_color("muted")))
            painter.drawText(chip, Qt.AlignmentFlag.AlignCenter, item.shortcut)
            right -= width + 10

        # title, letter by letter so matches can be highlighted, then the detail
        title_font = QFont(option.font)
        title_metrics = QFontMetrics(title_font)
        bold = QFont(title_font)
        bold.setWeight(QFont.Weight.Bold)
        bold_metrics = QFontMetrics(bold)
        baseline = rect.center().y() + title_metrics.ascent() // 2 - 1
        text_color, accent = QColor(theme_color("text")), QColor(theme_color("accent"))
        for i, ch in enumerate(item.title):
            if x >= right - 20:
                break
            matched = i in positions
            painter.setFont(bold if matched else title_font)
            painter.setPen(accent if matched else text_color)
            painter.drawText(x, baseline, ch)
            x += (bold_metrics if matched else title_metrics).horizontalAdvance(ch)
        if item.detail and x < right - 40:
            painter.setFont(small)
            painter.setPen(QColor(theme_color("muted")))
            detail = small_metrics.elidedText(item.detail, Qt.TextElideMode.ElideRight, right - x - 14)
            painter.drawText(QRect(x + 12, rect.top(), right - x - 12, rect.height()),
                             Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, detail)
        painter.restore()


class Launcher(QWidget):
    """Overlay over the main window: a dimmed backdrop with the search card near the top."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.setObjectName("launcherOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.hide()
        self.items: list[LaunchItem] = []
        self._icons: dict[str, QIcon] = {}

        self.card = QFrame(self)
        self.card.setObjectName("launcherCard")
        layout = QVBoxLayout(self.card)
        layout.setContentsMargins(0, 0, 0, 6)
        layout.setSpacing(0)
        self.input = QLineEdit()
        self.input.setObjectName("launcherInput")
        self.input.setPlaceholderText("Open a panel, plugin, setting or command…")
        self.input.textChanged.connect(self.refresh)
        self.input.installEventFilter(self)
        layout.addWidget(self.input)
        self.list = QListWidget()
        self.list.setObjectName("launcherList")
        self.list.setItemDelegate(_RowDelegate(self, self.list))
        self.list.setMouseTracking(True)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.list.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemClicked.connect(lambda _item: self.activate_current())
        layout.addWidget(self.list)
        self.hint = QLabel("↑↓ choose   ·   Enter open   ·   Esc close")
        self.hint.setObjectName("launcherHint")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hint)
        window.installEventFilter(self)

    # ── open / close ─────────────────────────────────────────────────────────

    def toggle(self):
        self.close_launcher() if self.isVisible() else self.open_launcher()

    def open_launcher(self):
        self._icons.clear()  # icons follow the current theme
        self.items = collect_items(self.window)
        self._place()
        self.input.blockSignals(True)
        self.input.clear()
        self.input.blockSignals(False)
        self.refresh()
        self.show()
        self.raise_()
        self.input.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def close_launcher(self):
        self.hide()

    def recent(self) -> list[str]:
        sm = getattr(self.window, "settings_manager", None)
        value = sm.get("launcher", "recent", default=[]) if sm else []
        return [str(v) for v in value] if isinstance(value, list) else []

    def _remember(self, key):
        sm = getattr(self.window, "settings_manager", None)
        if sm is None:
            return
        recent = [key] + [k for k in self.recent() if k != key]
        try:
            sm.set("launcher", "recent", recent[:RECENT_LIMIT])
        except Exception:
            pass  # remembering is a convenience; never block opening the item

    # ── list ─────────────────────────────────────────────────────────────────

    def refresh(self, *_):
        self.list.clear()
        for item, positions in rank(self.items, self.input.text(), self.recent()):
            row = QListWidgetItem()
            row.setData(ROLE_ITEM, item)
            row.setData(ROLE_MATCH, positions)
            row.setToolTip(item.detail or item.title)
            self.list.addItem(row)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.hint.setText("No matches" if not self.list.count() else "↑↓ choose   ·   Enter open   ·   Esc close")
        self._place()

    def current_item(self) -> LaunchItem | None:
        row = self.list.currentItem()
        return row.data(ROLE_ITEM) if row else None

    def move(self, step):
        count = self.list.count()
        if count:
            self.list.setCurrentRow((self.list.currentRow() + step) % count)

    def activate_current(self):
        item = self.current_item()
        if item is None:
            return
        self.close_launcher()
        self._remember(item.key)
        QTimer.singleShot(0, item.action)  # after the overlay is gone, so focus lands on what opened

    def icon_for(self, name):
        if not name:
            return QIcon()
        if name not in self._icons:
            from .sidetabs import TAB_ICON_COLORS, svg_icon
            self._icons[name] = svg_icon(name, TAB_ICON_COLORS, size=18)
        return self._icons[name]

    # ── layout and events ────────────────────────────────────────────────────

    def _place(self):
        parent = self.window
        self.setGeometry(parent.rect())
        width = min(CARD_WIDTH, max(320, parent.width() - 80))
        rows = min(VISIBLE_ROWS, max(1, self.list.count()))
        self.list.setFixedHeight(rows * ROW_HEIGHT + 6)
        self.card.setFixedWidth(width)
        self.card.adjustSize()
        top = max(40, round(parent.height() * 0.16))
        self.card.move((parent.width() - width) // 2, top)

    def eventFilter(self, obj, event):
        if obj is self.window and event.type() == QEvent.Type.Resize and self.isVisible():
            self._place()
        elif obj is self.input and event.type() == QEvent.Type.ShortcutOverride and self._is_navigation_key(event):
            # Claim the key before application shortcuts see it: Ctrl+N would otherwise run New experiment
            event.accept()
            return True
        elif obj is self.input and event.type() == QEvent.Type.KeyPress:
            key, mods = event.key(), event.modifiers()
            ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
            if key == Qt.Key.Key_Escape:
                self.close_launcher()
                return True
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self.activate_current()
                return True
            if key in (Qt.Key.Key_Down, Qt.Key.Key_Tab) or (ctrl and key in (Qt.Key.Key_N, Qt.Key.Key_J)):
                self.move(1)
                return True
            if key in (Qt.Key.Key_Up, Qt.Key.Key_Backtab) or (ctrl and key in (Qt.Key.Key_P, Qt.Key.Key_K)):
                self.move(-1)
                return True
            if key == Qt.Key.Key_PageDown:
                self.move(VISIBLE_ROWS)
                return True
            if key == Qt.Key.Key_PageUp:
                self.move(-VISIBLE_ROWS)
                return True
        elif obj is self.input and event.type() == QEvent.Type.FocusOut and self.isVisible():
            # Clicking elsewhere in the app (or another window) dismisses it, like rofi
            QTimer.singleShot(0, self._close_if_focus_left)
        return super().eventFilter(obj, event)

    @staticmethod
    def _is_navigation_key(event) -> bool:
        key, mods = event.key(), event.modifiers()
        ctrl = bool(mods & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.MetaModifier))
        plain = {Qt.Key.Key_Escape, Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Up, Qt.Key.Key_Down,
                 Qt.Key.Key_Tab, Qt.Key.Key_Backtab, Qt.Key.Key_PageUp, Qt.Key.Key_PageDown}
        return key in plain or (ctrl and key in (Qt.Key.Key_N, Qt.Key.Key_P, Qt.Key.Key_J, Qt.Key.Key_K))

    def _close_if_focus_left(self):
        focus = self.window.focusWidget() if self.window.isActiveWindow() else None
        if focus is None or not (focus is self or self.isAncestorOf(focus)):
            self.close_launcher()

    def mousePressEvent(self, event):
        if not self.card.geometry().contains(event.position().toPoint()):
            self.close_launcher()  # a click on the backdrop
        event.accept()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 90))
        # a soft painted shadow under the card (cheaper than QGraphicsDropShadowEffect)
        card = QRectF(self.card.geometry())
        for spread, alpha in ((14, 10), (9, 18), (5, 30), (2, 45)):
            path = QPainterPath()
            path.addRoundedRect(card.adjusted(-spread, -spread + 4, spread, spread + 4), 12 + spread, 12 + spread)
            painter.fillPath(path, QColor(0, 0, 0, alpha))
        painter.end()
