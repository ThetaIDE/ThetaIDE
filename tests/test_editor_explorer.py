"""Native filesystem exploration: hierarchy, ownership, live updates and selection."""

import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QTreeView

from frontend.plugins.core.text_editor.explorer import FileExplorer

_app = None


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    assert predicate()


@pytest.fixture
def explorer():
    global _app
    if QApplication.instance() is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    _app = QApplication.instance() or QApplication([])
    widget = FileExplorer()
    widget.resize(280, 620)
    widget.show()
    yield widget
    widget.close()
    widget.deleteLater()
    QApplication.processEvents()


def index_for(explorer, path, related=False):
    proxy = explorer.related_filter if related else explorer.source_filter
    return proxy.mapFromSource(explorer.file_model.index(str(path)))


def test_native_hierarchy_and_lazy_live_filesystem_updates(explorer, tmp_path, monkeypatch):
    root = tmp_path / "component"
    folder = root / "agents"
    folder.mkdir(parents=True)
    file = folder / "model.py"
    file.write_text("value = 1\n")
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "model.pyc").write_bytes(b"cache")
    metadata = root / ".theta_component.json"
    metadata.write_text("{}")
    def forbid_scan(*args, **kwargs):
        pytest.fail("Explorer must leave directory traversal to QFileSystemModel")
    monkeypatch.setattr(Path, "rglob", forbid_scan)
    explorer.open_root(root)
    assert isinstance(explorer.source_tree, QTreeView)
    explorer.reveal(file)
    wait_for(lambda: index_for(explorer, file).isValid())
    assert index_for(explorer, file).parent() == index_for(explorer, folder)
    assert explorer.source_tree.isExpanded(index_for(explorer, folder))
    assert not index_for(explorer, cache).isValid()
    wait_for(lambda: index_for(explorer, metadata).isValid())
    assert explorer.source_tree.isColumnHidden(1)
    assert explorer.file_model.isReadOnly()
    added = folder / "added.py"
    added.write_text("added = True")
    wait_for(lambda: index_for(explorer, added).isValid())
    added.unlink()
    wait_for(lambda: not index_for(explorer, added).isValid())


def test_related_paths_are_scoped_and_activate_actual_files(explorer, tmp_path):
    root = tmp_path / "src/usr/models/fork"
    root.mkdir(parents=True)
    source = root / "model.py"
    source.write_text("value = 1")
    config = tmp_path / "in/config/model/fork.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("architecture: fork")
    unrelated = config.with_name("unrelated.yaml")
    unrelated.write_text("architecture: other")
    asset = tmp_path / "plot/fork/routing.py"
    asset.parent.mkdir(parents=True)
    asset.write_text("# owned plot")
    explorer.open_root(root, [config, asset])
    assert explorer.root_label.text() == "fork"
    assert explorer.root_label.toolTip() == str(root)
    assert not explorer.related_section.isHidden()
    explorer.reveal(config)
    wait_for(lambda: index_for(explorer, config, True).isValid())
    assert not index_for(explorer, unrelated, True).isValid()
    selected = []
    explorer.fileActivated.connect(selected.append)
    explorer.related_tree.activated.emit(index_for(explorer, config, True))
    assert selected == [config]
    explorer.reveal(source)
    wait_for(lambda: explorer.source_tree.currentIndex() == index_for(explorer, source))
    assert not explorer.related_tree.selectionModel().hasSelection()
    explorer.open_root(root)
    assert explorer.related_section.isHidden()


def test_collapse_and_root_switch_do_not_open_files(explorer, tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    (first / "nested").mkdir(parents=True)
    second.mkdir()
    path = first / "nested/file.txt"
    path.write_text("text")
    opened = []
    explorer.fileActivated.connect(opened.append)
    explorer.open_root(first)
    explorer.reveal(path)
    wait_for(lambda: explorer.source_tree.currentIndex() == index_for(explorer, path))
    explorer.collapse_button.click()
    assert not explorer.source_tree.isExpanded(index_for(explorer, path.parent))
    explorer.open_root(second)
    assert explorer.file_model.filePath(explorer.source_filter.mapToSource(explorer.source_tree.rootIndex())) == str(second)
    assert opened == []
    assert explorer.source_tree.editTriggers() == QTreeView.EditTrigger.NoEditTriggers


def test_folder_rows_and_arrows_toggle_once_and_files_open(explorer, tmp_path):
    folder = tmp_path / "nested"
    folder.mkdir()
    file = folder / "model.py"
    file.write_text("value = 1")
    opened = []
    explorer.fileActivated.connect(opened.append)
    explorer.open_root(tmp_path)
    tree = explorer.source_tree
    wait_for(lambda: index_for(explorer, folder).isValid())
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton,
                     pos=tree.visualRect(index_for(explorer, folder)).center())
    wait_for(lambda: tree.isExpanded(index_for(explorer, folder)) and index_for(explorer, file).isValid())
    assert opened == []
    # The native disclosure arrow must not also run the row click toggle.
    rect = tree.visualRect(index_for(explorer, folder))
    arrow = QPoint(rect.left() - tree.indentation() // 2, rect.center().y())
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=arrow)
    assert not tree.isExpanded(index_for(explorer, folder))
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=arrow)
    assert tree.isExpanded(index_for(explorer, folder))
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, pos=tree.visualRect(index_for(explorer, file)).center())
    assert opened == [file]
    tree.setCurrentIndex(index_for(explorer, folder))
    QTest.keyClick(tree, Qt.Key.Key_Left)
    assert not tree.isExpanded(index_for(explorer, folder))
    QTest.keyClick(tree, Qt.Key.Key_Right)
    assert tree.isExpanded(index_for(explorer, folder))


def test_folders_sort_first_and_all_entries_have_icons(explorer, tmp_path):
    for name in ("z_folder", "B_folder"):
        (tmp_path / name).mkdir()
    for name in ("a.py", "Z.py", "B.py"):
        (tmp_path / name).write_text("# file")
    explorer.open_root(tmp_path)
    model = explorer.source_filter
    wait_for(lambda: model.rowCount(explorer.source_tree.rootIndex()) == 5)
    root = explorer.source_tree.rootIndex()
    entries = [model.index(row, 0, root) for row in range(5)]
    assert [index.data() for index in entries] == ["B_folder", "z_folder", "a.py", "B.py", "Z.py"]
    assert all(not index.data(Qt.ItemDataRole.DecorationRole).isNull() for index in entries)
