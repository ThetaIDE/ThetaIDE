"""Multi-tabbed document manager hosting ConfigViewer instances for Theta-IDE."""
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QMessageBox,
    QStyle,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QTabWidget,
    QWidget,
)

from .config_viewer import ConfigViewer


class PreviewTabBar(QTabBar):
    """Custom tab bar that renders preview tabs in italic text."""

    def sizeHint(self):
        hint = super().sizeHint()
        return hint.__class__(hint.width(), 30)

    def tabSizeHint(self, index):
        hint = super().tabSizeHint(index)
        return hint.__class__(hint.width(), 30)

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionTab()
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            if self.tabData(i) == "preview":
                font = self.font()
                font.setItalic(True)
                opt.font = font
            painter.drawControl(QStyle.ControlElement.CE_TabBarTab, opt)


class ConfigTabManager(QTabWidget):
    """Multi-tabbed document manager hosting independent ConfigViewer instances.

    Features:
    - True multi-instance tab architecture (each tab has its own ConfigViewer, scroll position, and state).
    - Single-click = Preview tab (italicized font, re-used unless dirty or pinned).
    - Double-click (in tree or on tab header) = Pinned permanent tab (regular font).
    - Dirty indicator (●) in tab title when modified.
    - Dirty tabs are automatically pinned so unsaved work is never overwritten.
    - Close confirmation dialog for dirty tabs.
    - Tab navigation shortcuts: Ctrl+W (close), Ctrl+Tab / Ctrl+Shift+Tab (cycle tabs).
    - Backwards-compatible proxy via current_viewer().
    - Hides the tab bar when only a single tab is open.
    - 30px height alignment matching the filetree header buttons.
    """

    current_tab_changed = pyqtSignal(str, object)  # (rel_path, viewer)
    file_renamed = pyqtSignal(str, str)            # (old_rel, new_rel)
    dirty_state_changed = pyqtSignal(str, bool)    # (rel_path, is_dirty)
    dirty_changed = pyqtSignal(bool)               # is_dirty of current viewer
    save_requested = pyqtSignal()
    config_changed = pyqtSignal()

    def __init__(self, settings_manager=None, parent=None, plugin_manager=None):
        if isinstance(settings_manager, QWidget) and parent is None:
            parent = settings_manager
            settings_manager = None
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.plugin_manager = plugin_manager
        self.setTabBar(PreviewTabBar(self))
        self.setTabsClosable(True)
        self.setMovable(True)
        self.setDocumentMode(True)
        self.setStyleSheet("""
            QTabBar {
                height: 30px;
                border: none;
            }
            QTabBar::tab {
                height: 30px;
                padding: 0 14px;
                border-bottom: 2px solid transparent;
            }
        """)
        self.tabBar().setVisible(False)

        self._tabs: dict[str, ConfigViewer] = {}  # normalized rel_path -> ConfigViewer
        self._preview_rel_path: Optional[str] = None
        self._pinned_paths: set[str] = set()

        self.auto_save = True
        self.btn_save = None

        # Fallback viewer used when no tabs are open, ensuring current_viewer() is never None
        self._fallback_viewer = ConfigViewer(
            settings_manager=self.settings_manager, parent=self, plugin_manager=self.plugin_manager
        )
        self._fallback_viewer.hide()

        self.tabCloseRequested.connect(self.close_tab_by_index)
        self.currentChanged.connect(self._on_current_changed)
        self.tabBar().tabBarDoubleClicked.connect(self._on_tab_double_clicked)

        self._setup_shortcuts()

    def tabInserted(self, index: int):
        super().tabInserted(index)
        self.tabBar().setVisible(self.count() > 1)

    def tabRemoved(self, index: int):
        super().tabRemoved(index)
        self.tabBar().setVisible(self.count() > 1)

    def _setup_shortcuts(self):
        # Close tab: Ctrl+W / Cmd+W
        self._sc_close = QShortcut(QKeySequence(QKeySequence.StandardKey.Close), self)
        self._sc_close.activated.connect(self.close_current_tab)
        self._sc_close_ctrl_w = QShortcut(QKeySequence("Ctrl+W"), self)
        self._sc_close_ctrl_w.activated.connect(self.close_current_tab)

        # Tab cycling
        self._sc_next = QShortcut(QKeySequence("Ctrl+Tab"), self)
        self._sc_next.activated.connect(self.next_tab)
        self._sc_prev = QShortcut(QKeySequence("Ctrl+Shift+Tab"), self)
        self._sc_prev.activated.connect(self.prev_tab)

    def current_viewer(self) -> ConfigViewer:
        """Return the active tab's ConfigViewer, or the fallback viewer if no tabs are open."""
        w = self.currentWidget()
        if isinstance(w, ConfigViewer):
            return w
        return self._fallback_viewer

    def refresh_component_choices(self) -> None:
        """Refresh open config forms after installed component files change."""
        viewers = list(self._tabs.values()) + [self._fallback_viewer]
        for viewer in viewers:
            if viewer.current_path or viewer.raw_data:
                viewer._render_boxes()

    def _tab_label(self, rel_path: str) -> str:
        """Derive clean tab label from relative path."""
        p = Path(rel_path)
        if p.stem == "_base":
            return f"{p.parent.name}/_base"
        return p.stem

    def _update_tab_display(self, index: int, rel_path: str, is_preview: bool, is_dirty: bool):
        """Update the tab text, tooltip, and preview state for the tab at index."""
        if index < 0 or index >= self.count():
            return
        label = self._tab_label(rel_path)
        if is_dirty:
            label = f"{label} ●"
        self.setTabText(index, label)
        self.setTabToolTip(index, rel_path)
        self.tabBar().setTabData(index, "preview" if is_preview else "pinned")
        self.tabBar().update()

    def open_file(self, file_path: Path | str, rel_path: str, pinned: bool = False) -> ConfigViewer:
        """Open a configuration file in a tab.

        If pinned is False (single click), reuses the current clean preview tab if one exists.
        If pinned is True (double click), opens or pins as a permanent tab.
        """
        norm_rel = Path(rel_path).as_posix()

        # 1. Already open in an existing tab?
        if norm_rel in self._tabs:
            viewer = self._tabs[norm_rel]
            if hasattr(viewer, "reload_data"):
                from .config_viewer import invalidate_config_tree
                invalidate_config_tree()
                viewer.reload_data(force_tree_refresh=True)
            idx = self.indexOf(viewer)
            if idx >= 0:
                self.setCurrentIndex(idx)
            if pinned:
                self.pin_tab(norm_rel)
            return viewer

        # 2. Reuse clean preview tab if available and not pinned
        if not pinned and self._preview_rel_path and self._preview_rel_path in self._tabs:
            old_preview_viewer = self._tabs[self._preview_rel_path]
            if not old_preview_viewer.is_dirty:
                old_rel = self._preview_rel_path
                del self._tabs[old_rel]
                self._preview_rel_path = norm_rel
                self._tabs[norm_rel] = old_preview_viewer

                old_preview_viewer.load_file(file_path, rel_path)
                idx = self.indexOf(old_preview_viewer)
                if idx >= 0:
                    self._update_tab_display(idx, norm_rel, is_preview=True, is_dirty=False)
                    self.setCurrentIndex(idx)
                self.current_tab_changed.emit(norm_rel, old_preview_viewer)
                return old_preview_viewer

        # 3. Create a new viewer in a new tab
        viewer = ConfigViewer(
            settings_manager=self.settings_manager, parent=self, plugin_manager=self.plugin_manager
        )
        viewer.auto_save = self.auto_save
        viewer.btn_save = self.btn_save

        viewer.config_changed.connect(self.config_changed.emit)
        viewer.save_requested.connect(self.save_requested.emit)
        viewer.dirty_state_changed.connect(lambda p, d, v=viewer: self._on_viewer_dirty_state_changed(v, d))
        viewer.dirty_changed.connect(lambda d, v=viewer: self._on_viewer_dirty_changed(v, d))
        viewer.file_renamed.connect(self._on_viewer_file_renamed)
        if hasattr(viewer, "dir_renamed"):
            viewer.dir_renamed.connect(self.on_dir_renamed)

        viewer.load_file(file_path, rel_path)

        idx = self.addTab(viewer, self._tab_label(norm_rel))
        self._tabs[norm_rel] = viewer

        if pinned:
            self._pinned_paths.add(norm_rel)
            self._update_tab_display(idx, norm_rel, is_preview=False, is_dirty=False)
        else:
            self._preview_rel_path = norm_rel
            self._update_tab_display(idx, norm_rel, is_preview=True, is_dirty=False)

        self.setCurrentIndex(idx)
        return viewer

    def pin_tab(self, rel_path: str):
        """Pin a tab by relative path so it is never overwritten by preview clicks."""
        norm_rel = Path(rel_path).as_posix()
        self._pinned_paths.add(norm_rel)
        if self._preview_rel_path == norm_rel:
            self._preview_rel_path = None
        if norm_rel in self._tabs:
            viewer = self._tabs[norm_rel]
            idx = self.indexOf(viewer)
            if idx >= 0:
                self._update_tab_display(idx, norm_rel, is_preview=False, is_dirty=viewer.is_dirty)

    def _on_tab_double_clicked(self, index: int):
        """Double-clicking a tab header pins it."""
        if index < 0 or index >= self.count():
            return
        w = self.widget(index)
        if isinstance(w, ConfigViewer) and w.current_rel_path:
            self.pin_tab(w.current_rel_path)

    def _on_viewer_dirty_changed(self, viewer: ConfigViewer, is_dirty: bool):
        """Handle dirty state change on a specific viewer."""
        rel_path = viewer.current_rel_path or ""
        norm_rel = Path(rel_path).as_posix() if rel_path else ""
        if not norm_rel:
            return

        if is_dirty:
            # Editing automatically pins the tab so changes are not lost
            self.pin_tab(norm_rel)

        idx = self.indexOf(viewer)
        if idx >= 0:
            is_preview = (self._preview_rel_path == norm_rel)
            self._update_tab_display(idx, norm_rel, is_preview=is_preview, is_dirty=is_dirty)

        self.dirty_state_changed.emit(norm_rel, is_dirty)
        if viewer == self.current_viewer():
            self.dirty_changed.emit(is_dirty)

    def _on_viewer_dirty_state_changed(self, viewer: ConfigViewer, is_dirty: bool):
        self._on_viewer_dirty_changed(viewer, is_dirty)

    def _on_viewer_file_renamed(self, old_rel: str, new_rel: str):
        """Update internal mappings when an open config is renamed."""
        old_norm = Path(old_rel).as_posix()
        new_norm = Path(new_rel).as_posix()
        if old_norm in self._tabs:
            viewer = self._tabs.pop(old_norm)
            self._tabs[new_norm] = viewer
            if self._preview_rel_path == old_norm:
                self._preview_rel_path = new_norm
            if old_norm in self._pinned_paths:
                self._pinned_paths.discard(old_norm)
                self._pinned_paths.add(new_norm)
            idx = self.indexOf(viewer)
            if idx >= 0:
                is_prev = (self._preview_rel_path == new_norm)
                self._update_tab_display(idx, new_norm, is_preview=is_prev, is_dirty=viewer.is_dirty)
        self.file_renamed.emit(old_rel, new_rel)

    def on_dir_renamed(self, old_dir_rel: str, new_dir_rel: str):
        """Update open tabs when an entire directory/group is renamed."""
        old_prefix = Path(old_dir_rel).as_posix().rstrip("/") + "/"
        new_prefix = Path(new_dir_rel).as_posix().rstrip("/") + "/"
        for old_rel in list(self._tabs.keys()):
            if old_rel.startswith(old_prefix):
                suffix = old_rel[len(old_prefix):]
                new_rel = new_prefix + suffix
                self._on_viewer_file_renamed(old_rel, new_rel)

    def _on_current_changed(self, index: int):
        """Notify listeners when the active tab switches."""
        if index >= 0:
            w = self.widget(index)
            if isinstance(w, ConfigViewer):
                rel = w.current_rel_path or ""
                self.current_tab_changed.emit(rel, w)
                self.dirty_changed.emit(w.is_dirty)
        else:
            self.current_tab_changed.emit("", self._fallback_viewer)
            self.dirty_changed.emit(False)

    def close_tab_by_index(self, index: int) -> bool:
        """Close the tab at index, prompting if dirty. Returns True if closed."""
        if index < 0 or index >= self.count():
            return False

        viewer = self.widget(index)
        if not isinstance(viewer, ConfigViewer):
            self.removeTab(index)
            return True

        rel_path = viewer.current_rel_path or ""
        norm_rel = Path(rel_path).as_posix() if rel_path else ""

        viewer.flush_pending_save()  # auto-save edits typed just before closing, without a prompt
        if viewer.is_dirty:
            name = Path(rel_path).name or "Configuration"
            res = QMessageBox.question(
                self,
                "Unsaved Changes",
                f"Save changes to '{name}' before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if res == QMessageBox.StandardButton.Save:
                if not viewer.save_to_disk():
                    return False
            elif res == QMessageBox.StandardButton.Cancel:
                return False

        # Cleanup tab tracking
        if norm_rel in self._tabs:
            del self._tabs[norm_rel]
        if self._preview_rel_path == norm_rel:
            self._preview_rel_path = None
        self._pinned_paths.discard(norm_rel)

        self.removeTab(index)
        viewer.deleteLater()

        if self.count() > 0:
            curr = self.current_viewer()
            self.current_tab_changed.emit(curr.current_rel_path or "", curr)
            self.dirty_changed.emit(curr.is_dirty)
        else:
            self.current_tab_changed.emit("", self._fallback_viewer)
            self.dirty_changed.emit(False)

        return True

    def close_current_tab(self) -> bool:
        """Close the currently active tab."""
        return self.close_tab_by_index(self.currentIndex())

    def close_tabs_matching(self, predicate: Callable[[str], bool]) -> None:
        """Close all tabs whose rel_path matches predicate without save prompts."""
        to_close = []
        for i in range(self.count()):
            w = self.widget(i)
            if isinstance(w, ConfigViewer) and w.current_rel_path and predicate(w.current_rel_path):
                to_close.append(i)
        for i in reversed(to_close):
            w = self.widget(i)
            rel = w.current_rel_path or ""
            norm = Path(rel).as_posix()
            self._tabs.pop(norm, None)
            if self._preview_rel_path == norm:
                self._preview_rel_path = None
            self._pinned_paths.discard(norm)
            self.removeTab(i)
            w.deleteLater()

        if self.count() > 0:
            curr = self.current_viewer()
            self.current_tab_changed.emit(curr.current_rel_path or "", curr)
            self.dirty_changed.emit(curr.is_dirty)
        else:
            self.current_tab_changed.emit("", self._fallback_viewer)
            self.dirty_changed.emit(False)

    def next_tab(self):
        """Cycle to the next tab."""
        if self.count() > 1:
            self.setCurrentIndex((self.currentIndex() + 1) % self.count())

    def prev_tab(self):
        """Cycle to the previous tab."""
        if self.count() > 1:
            self.setCurrentIndex((self.currentIndex() - 1 + self.count()) % self.count())
