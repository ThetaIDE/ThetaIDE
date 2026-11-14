"""Native, lazy filesystem views for the editor and its related component files."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import QDir, QSize, QSortFilterProxyModel, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QLabel,
    QSizePolicy,
    QSplitter,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

try:  # QFileSystemModel moved to QtGui in newer Qt 6 releases.
    from PyQt6.QtGui import QFileSystemModel
except ImportError:
    from PyQt6.QtWidgets import QFileSystemModel

from src.app.api.desktop import (
    DraggableToolButton,
    TreeHeaderWidget,
    TreeToolbarPersistence,
    current_theme,
    refresh_tool_icons,
    style_tool_button,
    svg_icon,
)

_GENERATED_DIRECTORIES = {"__pycache__", ".git", ".venv", "venv", "node_modules", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


class _ScopeFilter(QSortFilterProxyModel):
    """Filter cached Qt filesystem entries without traversing directories ourselves."""

    def __init__(self, source, parent=None):
        super().__init__(parent)
        self.setSourceModel(source)
        self.owned: tuple[Path, ...] | None = None
        self.anchor: Path | None = None
        self.refresh_icons()
        self.sort(0)

    def refresh_icons(self):
        colors = {(QIcon.Mode.Normal, QIcon.State.Off): "muted",
                  (QIcon.Mode.Selected, QIcon.State.Off): "text"}
        self._folder_icon = svg_icon("folder", colors, size=16)
        self._file_icon = svg_icon("file", colors, size=16)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 0:
            return self._folder_icon if self.sourceModel().isDir(self.mapToSource(index)) else self._file_icon
        return super().data(index, role)

    def lessThan(self, left, right):
        model = self.sourceModel()
        left_dir, right_dir = model.isDir(left), model.isDir(right)
        if left_dir != right_dir:
            return left_dir
        return model.fileName(left).casefold() < model.fileName(right).casefold()

    def set_owned_paths(self, paths):
        self.owned = tuple(paths)
        self.invalidateFilter()

    def filterAcceptsRow(self, row, parent):
        model = self.sourceModel()
        index = model.index(row, 0, parent)
        path = Path(model.filePath(index))
        if path.name in _GENERATED_DIRECTORIES and model.isDir(index):
            if self.owned is None and not (self.anchor and self.anchor.is_relative_to(path)):
                return False
        if self.owned is None:
            return True
        return any(path == owned or path.is_relative_to(owned) or owned.is_relative_to(path) for owned in self.owned)


class FileExplorer(TreeToolbarPersistence, QWidget):
    fileActivated = pyqtSignal(object)
    openFolderRequested = pyqtSignal()
    actionRequested = pyqtSignal(str)
    toggleRequested = pyqtSignal()

    def __init__(self, parent=None, settings_manager=None):
        super().__init__(parent)
        self.settings_manager = settings_manager
        self.mode = "text_editor"
        self._expanded_toolbar_groups = None
        self.setObjectName("editorFileExplorer")
        self.setMinimumWidth(218)
        self.root: Path | None = None
        self._related_paths: tuple[Path, ...] = ()
        self._owned_paths: tuple[Path, ...] = ()
        self._pending_path: Path | None = None
        self.file_model = QFileSystemModel(self)
        self.file_model.setReadOnly(True)
        self.file_model.setOption(QFileSystemModel.Option.DontUseCustomDirectoryIcons, True)
        self.file_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.Files | QDir.Filter.Hidden | QDir.Filter.NoDotAndDotDot)
        self.file_model.sort(0, Qt.SortOrder.AscendingOrder)
        self.file_model.directoryLoaded.connect(self._directory_loaded)
        self.source_filter = _ScopeFilter(self.file_model, self)
        self.related_filter = _ScopeFilter(self.file_model, self)
        self.related_filter.set_owned_paths(())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 6, 4)
        layout.setSpacing(6)
        self.action_buttons = {}
        for action, icon, title in (
            ("new", "new_file", "New document"),
            ("open", "file", "Open file…"),
            ("open_folder", "folder", "Open folder…"),
            ("up", "folder_up", "Go to parent folder"),
            ("save", "save", "Save"),
            ("save_as", "save_as", "Save as…"),
            ("toggle", "sidebar_collapse", "Collapse file tree"),
            ("undo", "undo", "Undo"),
            ("redo", "redo", "Redo"),
            ("find", "search", "Find / Replace"),
            ("wrap", "wrap", "Wrap lines"),
            ("collapse_all", "collapse_all", "Collapse folders"),
            ("recover", "reload", "Recover editor"),
        ):
            button = style_tool_button(DraggableToolButton(action), icon, title)
            self.action_buttons[action] = button
            if action not in ("wrap", "recover", "toggle", "collapse_all", "open_folder", "up"):
                button.clicked.connect(lambda _checked=False, a=action: self.actionRequested.emit(a))
        self.up_button = self.action_buttons["up"]
        self.up_button.setEnabled(False)
        self.up_button.clicked.connect(self.go_up)
        self.open_button = self.action_buttons["open_folder"]
        self.open_button.clicked.connect(self.openFolderRequested.emit)
        self.collapse_button = self.action_buttons["collapse_all"]
        self.collapse_button.clicked.connect(self.collapse_all)
        self.toggle_button = self.action_buttons["toggle"]
        self.toggle_button.clicked.connect(self.toggleRequested.emit)
        self.action_buttons["wrap"].setCheckable(True)
        self.header_widget = TreeHeaderWidget(self)
        self.header_widget.set_groups(self._load_toolbar_groups())
        self.header_widget.layout_changed.connect(self._save_toolbar_layout)
        self.action_buttons["recover"].hide()
        layout.addWidget(self.header_widget)
        title = QLabel("EXPLORER")
        title.setObjectName("explorerHeading")
        layout.addWidget(title)
        self.empty_label = QLabel("Open a folder to browse files.")
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.empty_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.empty_label)
        self.sections = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(self.sections, 1)
        source_section = QWidget()
        source_layout = QVBoxLayout(source_section)
        source_layout.setContentsMargins(0, 0, 0, 0)
        source_layout.setSpacing(4)
        self.root_label = QLabel()
        self.root_label.setObjectName("explorerRoot")
        self.root_label.setTextFormat(Qt.TextFormat.PlainText)
        source_layout.addWidget(self.root_label)
        self.source_tree = self._tree(self.source_filter)
        source_layout.addWidget(self.source_tree, 1)
        self.sections.addWidget(source_section)
        self.related_section = QWidget()
        related_layout = QVBoxLayout(self.related_section)
        related_layout.setContentsMargins(0, 6, 0, 0)
        related_layout.setSpacing(4)
        related_title = QLabel("RELATED FILES")
        related_title.setObjectName("explorerHeading")
        related_title.setToolTip("Owned profiles and assets outside this source folder; these open their actual installed files.")
        related_layout.addWidget(related_title)
        self.related_tree = self._tree(self.related_filter)
        related_layout.addWidget(self.related_tree, 1)
        self.sections.addWidget(self.related_section)
        self.sections.setSizes([400, 160])
        self.sections.hide()
        self.related_section.hide()
        self.apply_theme()

    def _default_toolbar_groups(self):
        buttons = self.action_buttons
        return {
            "row1_left": [buttons[action] for action in ("new", "open", "open_folder", "up", "save", "save_as")],
            "row1_right": [buttons["toggle"]],
            "row2_left": [buttons[action] for action in ("undo", "redo", "find", "wrap", "collapse_all")],
            "row2_right": [buttons["recover"]],
        }

    def detach_toggle_button(self):
        """Keep the customized position while the same control lives beside the tabs."""
        groups = self.header_widget.header_layout.groups
        self._expanded_toolbar_groups = {key: list(buttons) for key, buttons in groups.items()}
        self.header_widget.set_groups({
            key: [button for button in buttons if button is not self.toggle_button]
            for key, buttons in groups.items()
        })
        self.header_widget.header_layout.removeWidget(self.toggle_button)

    def restore_toggle_button(self):
        if self._expanded_toolbar_groups is not None:
            self.header_widget.set_groups(self._expanded_toolbar_groups)
            self._expanded_toolbar_groups = None

    def _tree(self, model):
        tree = QTreeView()
        tree.setModel(model)
        tree.setHeaderHidden(True)
        tree.setUniformRowHeights(True)
        tree.setIndentation(20)
        tree.setIconSize(QSize(16, 16))
        tree.setRootIsDecorated(True)
        tree.setItemsExpandable(True)
        tree.setExpandsOnDoubleClick(False)
        tree.setAnimated(False)
        tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        for column in (1, 2, 3):
            tree.hideColumn(column)
        tree.clicked.connect(lambda index: self._click(tree, index))
        tree.activated.connect(lambda index: self._activate(tree, index))
        tree.doubleClicked.connect(lambda index: self._enter_directory(tree, index))
        return tree

    def _click(self, tree, index):
        source = tree.model().mapToSource(index)
        if source.isValid() and self.file_model.isDir(source):
            # The disclosure arrow is handled by QTreeView itself; clicked is
            # emitted for the row's icon/name, so either target toggles once.
            tree.setExpanded(index, not tree.isExpanded(index))
        else:
            self._activate(tree, index)

    def _activate(self, tree, index):
        source = tree.model().mapToSource(index)
        if source.isValid() and not self.file_model.isDir(source):
            self.fileActivated.emit(Path(self.file_model.filePath(source)))

    def _enter_directory(self, tree, index):
        source = tree.model().mapToSource(index)
        if source.isValid() and self.file_model.isDir(source):
            self.open_root(self.file_model.filePath(source), self._owned_paths)

    def go_up(self):
        if self.root and self.root.parent != self.root:
            self.open_root(self.root.parent, self._owned_paths)

    def open_root(self, root, owned_paths=()):
        root = Path(root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"Folder does not exist: {root}")
        owned = tuple(sorted({Path(path).expanduser().resolve() for path in owned_paths}))
        related = tuple(path for path in owned if not path.is_relative_to(root))
        if self.root == root and self._related_paths == related:
            self._owned_paths = owned
            return
        self.root, self._related_paths, self._owned_paths = root, related, owned
        self.up_button.setEnabled(root.parent != root)
        self.empty_label.hide()
        self.sections.show()
        self.source_filter.anchor = root
        self.source_filter.invalidateFilter()
        self._pending_path = None
        self.root_label.setText(root.name)
        self.root_label.setToolTip(str(root))
        self.source_tree.setAccessibleName(f"Files in {root.name}")
        self.related_tree.setAccessibleName("Related component files")
        common = Path(os.path.commonpath([str(root), *(str(path.parent) for path in related)]))
        self.file_model.setRootPath(str(common))
        self.source_tree.setRootIndex(self.source_filter.mapFromSource(self.file_model.index(str(root))))
        self.source_tree.collapseAll()
        self.related_filter.set_owned_paths(related)
        self.related_tree.setRootIndex(self.related_filter.mapFromSource(self.file_model.index(str(common))))
        self.related_tree.collapseAll()
        self.related_section.setVisible(bool(related))
        self.setToolTip(str(root))

    def collapse_all(self):
        self.source_tree.collapseAll()
        self.related_tree.collapseAll()

    def reveal(self, path):
        """Follow document tabs; programmatic selection never opens another file."""
        self.source_tree.clearSelection()
        self.related_tree.clearSelection()
        self._pending_path = Path(path).resolve() if path else None
        self._reveal_pending()

    def _directory_loaded(self, _directory):
        self._reveal_pending()

    def _reveal_pending(self):
        path = self._pending_path
        if path is None or self.root is None:
            return
        if path.is_relative_to(self.root):
            tree = self.source_tree
        elif any(path == owned or path.is_relative_to(owned) for owned in self._related_paths):
            tree = self.related_tree
        else:
            self._pending_path = None
            return
        index = tree.model().mapFromSource(self.file_model.index(str(path)))
        if not index.isValid():
            return  # QFileSystemModel is still loading the directory.
        parent = index.parent()
        while parent.isValid() and parent != tree.rootIndex():
            tree.expand(parent)
            parent = parent.parent()
        tree.setCurrentIndex(index)
        tree.scrollTo(index)
        self._pending_path = None

    def apply_theme(self):
        colors = current_theme()["colors"]
        for proxy, tree in ((self.source_filter, self.source_tree), (self.related_filter, self.related_tree)):
            proxy.refresh_icons()
            tree.viewport().update()
        refresh_tool_icons(self)
        self.setStyleSheet(
            f"QWidget#editorFileExplorer {{ background: {colors['panel']}; border-radius: 6px; }}"
            f"QLabel#explorerHeading {{ color: {colors['muted']}; font-size: 10px; font-weight: 600; letter-spacing: 1px; }}"
            f"QLabel#explorerRoot {{ color: {colors['text']}; font-weight: 600; padding: 4px; }}"
            f"QTreeView {{ background: {colors['panel']}; color: {colors['text']}; border: none; outline: none; }}"
            "QTreeView::item { min-height: 24px; padding: 1px 3px; border-radius: 4px; }"
            f"QTreeView::item:hover {{ background: {colors['surface']}; }}"
            f"QTreeView::item:selected {{ background: {colors['border']}; color: {colors['text']}; }}"
            f"QSplitter::handle {{ background: {colors['panel']}; }}"
        )
