"""Multi-tabbed document manager hosting ConfigViewer instances for Theta-IDE."""
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QFileSystemWatcher, QTimer, Qt, pyqtSignal
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
    - Double-click in the tree opens a permanent tab; double-clicking a tab pins it.
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
        self._disk_contents: dict[str, bytes] = {}
        self._file_watcher = QFileSystemWatcher(self)
        self._reload_timer = QTimer(self)
        self._reload_timer.setSingleShot(True)
        self._reload_timer.setInterval(200)
        self._reload_timer.timeout.connect(self._reload_external_changes)
        self._file_watcher.fileChanged.connect(lambda _: self._reload_timer.start())
        self._file_watcher.directoryChanged.connect(lambda _: self._reload_timer.start())
        # Some filesystems do not deliver watcher events for atomic replacement.
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(750)
        self._poll_timer.timeout.connect(self._reload_external_changes)
        self._poll_timer.start()

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
        viewers = [self.widget(i) for i in range(self.count())] + [self._fallback_viewer]
        for viewer in viewers:
            if viewer.current_path or viewer.raw_data:
                viewer._render_boxes()

    def _sync_file_watches(self):
        viewers = [self.widget(i) for i in range(self.count())]
        files = {str(v.current_path) for v in viewers if isinstance(v, ConfigViewer) and v.current_path}
        desired = {p for p in files if Path(p).is_file()} | {str(Path(p).parent) for p in files if Path(p).parent.is_dir()}
        watched = set(self._file_watcher.files()) | set(self._file_watcher.directories())
        if watched - desired:
            self._file_watcher.removePaths(list(watched - desired))
        if desired - watched:
            self._file_watcher.addPaths(list(desired - watched))
        self._disk_contents = {p: value for p, value in self._disk_contents.items() if p in files}
        for path in files:
            if path not in self._disk_contents and Path(path).is_file():
                try:
                    self._disk_contents[path] = Path(path).read_bytes()
                except OSError:
                    pass

    def _remember_saved_file(self, viewer):
        if viewer.current_path and viewer.current_path.is_file():
            self._disk_contents[str(viewer.current_path)] = viewer.current_path.read_bytes()
        self._sync_file_watches()

    def _reload_external_changes(self):
        viewers = [self.widget(i) for i in range(self.count())]
        paths = {str(v.current_path) for v in viewers if isinstance(v, ConfigViewer) and v.current_path}
        for path in paths:
            try:
                contents = Path(path).read_bytes()
            except OSError:
                continue
            if self._disk_contents.get(path) == contents:
                continue
            matching = [v for v in viewers if isinstance(v, ConfigViewer) and str(v.current_path) == path]
            if any(v.is_dirty for v in matching):
                continue  # Never overwrite an unsaved form.
            for viewer in matching:
                viewer.reload_data(force_tree_refresh=True)
            self._disk_contents[path] = contents
            if self.current_viewer() in matching:
                self.config_changed.emit()
        self._sync_file_watches()  # Atomic saves replace the watched inode.

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

    def open_file(self, file_path: Path | str, rel_path: str, pinned: bool = False,
                  new_tab: bool = False) -> ConfigViewer:
        """Open a configuration file in a tab.

        If pinned is False (single click), reuses the current clean preview tab if one exists.
        If pinned is True, opens or pins as a permanent tab.
        If new_tab is True, keeps a permanent tab, focusing it if already open.
        """
        norm_rel = Path(rel_path).as_posix()

        # 1. Already open in an existing tab?
        if norm_rel in self._tabs:
            viewer = self._tabs[norm_rel]
            if not viewer.is_dirty and hasattr(viewer, "reload_data"):
                from .config_viewer import invalidate_config_tree
                invalidate_config_tree()
                viewer.reload_data(force_tree_refresh=True)
            idx = self.indexOf(viewer)
            if idx >= 0:
                self.setCurrentIndex(idx)
            if pinned or new_tab:
                self.pin_tab(norm_rel)
            return viewer

        # 2. Reuse clean preview tab if available and not pinned
        if not new_tab and not pinned and self._preview_rel_path and self._preview_rel_path in self._tabs:
            old_preview_viewer = self._tabs[self._preview_rel_path]
            if not old_preview_viewer.is_dirty:
                old_rel = self._preview_rel_path
                self._forget_viewer(old_preview_viewer, old_rel)
                self._preview_rel_path = norm_rel
                self._tabs[norm_rel] = old_preview_viewer

                old_preview_viewer.load_file(file_path, rel_path)
                idx = self.indexOf(old_preview_viewer)
                if idx >= 0:
                    self._update_tab_display(idx, norm_rel, is_preview=True, is_dirty=False)
                    self.setCurrentIndex(idx)
                self.current_tab_changed.emit(norm_rel, old_preview_viewer)
                self._sync_file_watches()
                return old_preview_viewer

        # 3. Create a new viewer in a new tab
        viewer = ConfigViewer(
            settings_manager=self.settings_manager, parent=self, plugin_manager=self.plugin_manager
        )
        viewer.auto_save = self.auto_save
        viewer.btn_save = self.btn_save

        viewer.config_changed.connect(self.config_changed.emit)
        viewer.save_requested.connect(lambda v=viewer: self._remember_saved_file(v))
        viewer.save_requested.connect(self.save_requested.emit)
        viewer.dirty_state_changed.connect(lambda p, d, v=viewer: self._on_viewer_dirty_state_changed(v, d))
        viewer.dirty_changed.connect(lambda d, v=viewer: self._on_viewer_dirty_changed(v, d))
        viewer.file_renamed.connect(self._on_viewer_file_renamed)
        if hasattr(viewer, "dir_renamed"):
            viewer.dir_renamed.connect(self.on_dir_renamed)

        viewer.load_file(file_path, rel_path)

        idx = self.addTab(viewer, self._tab_label(norm_rel))
        self._tabs.setdefault(norm_rel, viewer)

        if pinned or new_tab:
            self._pinned_paths.add(norm_rel)
            self._update_tab_display(idx, norm_rel, is_preview=False, is_dirty=False)
        else:
            self._preview_rel_path = norm_rel
            self._update_tab_display(idx, norm_rel, is_preview=True, is_dirty=False)

        self.setCurrentIndex(idx)
        self._sync_file_watches()
        return viewer

    def pin_tab(self, rel_path: str):
        """Pin the tracked tab for a relative path."""
        viewer = self._tabs.get(Path(rel_path).as_posix())
        if viewer is not None:
            self._pin_viewer(viewer)

    def _pin_viewer(self, viewer: ConfigViewer):
        norm_rel = Path(viewer.current_rel_path).as_posix()
        self._pinned_paths.add(norm_rel)
        if self._preview_rel_path == norm_rel and self._tabs.get(norm_rel) is viewer:
            self._preview_rel_path = None
        idx = self.indexOf(viewer)
        if idx >= 0:
            self._update_tab_display(idx, norm_rel, is_preview=False, is_dirty=viewer.is_dirty)

    def _on_tab_double_clicked(self, index: int):
        """Double-clicking a tab header pins it."""
        if index < 0 or index >= self.count():
            return
        w = self.widget(index)
        if isinstance(w, ConfigViewer) and w.current_rel_path:
            self._pin_viewer(w)

    def _on_viewer_dirty_changed(self, viewer: ConfigViewer, is_dirty: bool):
        """Handle dirty state change on a specific viewer."""
        rel_path = viewer.current_rel_path or ""
        norm_rel = Path(rel_path).as_posix() if rel_path else ""
        was_preview = self._preview_rel_path == norm_rel and self._tabs.get(norm_rel) is viewer
        if not norm_rel:
            return

        if is_dirty:
            # Editing automatically pins the tab so changes are not lost
            self._pin_viewer(viewer)

        idx = self.indexOf(viewer)
        if idx >= 0:
            is_preview = (self._preview_rel_path == norm_rel and self._tabs.get(norm_rel) is viewer)
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
        viewers = [self.widget(i) for i in range(self.count())]
        matching = [v for v in viewers if isinstance(v, ConfigViewer) and
                    Path(v.current_rel_path or "").as_posix() in (old_norm, new_norm)]
        tracked = self._tabs.pop(old_norm, None)
        if tracked is not None:
            self._tabs[new_norm] = tracked
        if self._preview_rel_path == old_norm:
            self._preview_rel_path = new_norm
        if old_norm in self._pinned_paths:
            self._pinned_paths.discard(old_norm)
            self._pinned_paths.add(new_norm)
        for viewer in matching:
            if Path(viewer.current_rel_path or "").as_posix() == old_norm:
                viewer.current_path = Path(str(viewer.current_path)[:-len(old_norm)]) / new_norm
                viewer.current_rel_path = new_norm
            idx = self.indexOf(viewer)
            self._update_tab_display(idx, new_norm,
                                     self._preview_rel_path == new_norm and self._tabs.get(new_norm) is viewer,
                                     viewer.is_dirty)
        self.file_renamed.emit(old_rel, new_rel)
        self._sync_file_watches()

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
        self._forget_viewer(viewer, norm_rel)

        self.removeTab(index)
        viewer.deleteLater()
        self._sync_file_watches()

        if self.count() > 0:
            curr = self.current_viewer()
            self.current_tab_changed.emit(curr.current_rel_path or "", curr)
            self.dirty_changed.emit(curr.is_dirty)
        else:
            self.current_tab_changed.emit("", self._fallback_viewer)
            self.dirty_changed.emit(False)

        return True

    def _forget_viewer(self, viewer: ConfigViewer, norm_rel: str):
        if self._tabs.get(norm_rel) is viewer:
            replacement = next((self.widget(i) for i in range(self.count())
                                if self.widget(i) is not viewer and
                                isinstance(self.widget(i), ConfigViewer) and
                                self.widget(i).current_rel_path == norm_rel), None)
            if replacement is None:
                self._tabs.pop(norm_rel, None)
            else:
                self._tabs[norm_rel] = replacement
            if self._preview_rel_path == norm_rel:
                self._preview_rel_path = None
        if norm_rel not in self._tabs:
            self._pinned_paths.discard(norm_rel)

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
            self._forget_viewer(w, norm)
            self.removeTab(i)
            w.deleteLater()

        self._sync_file_watches()

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
