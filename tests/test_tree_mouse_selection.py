"""Exercise native mouse selection, including the editor tab's selection echo."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMenu
from frontend.config_tree import ConfigTreeWidget
from frontend.results.explorer import RunExplorer

app = QApplication.instance() or QApplication([])


def click(tree, item, modifiers=Qt.KeyboardModifier.NoModifier):
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, modifiers,
                     tree.visualItemRect(item).center())
    app.processEvents()


def test_config_command_toggle_shift_range_and_tab_echo(tmp_path):
    root = tmp_path / "experiment"
    (root / "bench").mkdir(parents=True)
    for name in ("a", "b", "c"):
        (root / "bench" / f"{name}.yaml").write_text("{}")
    widget = ConfigTreeWidget(root_dir=root, mode="experiments")
    widget.resize(400, 500)
    widget.show()
    tree = widget.tree
    tree.expandAll()
    app.processEvents()
    items = [widget.find_file_item(f"bench/{name}.yaml") for name in ("a", "b", "c")]
    opened = []
    def preview(path, rel):
        opened.append(rel)
        widget.select_file(rel, emit_signal=False)
    widget.file_selected.connect(preview)
    click(tree, items[0])
    assert tree.selectedItems() == [items[0]]
    click(tree, items[2], Qt.KeyboardModifier.ControlModifier)
    assert set(map(id, tree.selectedItems())) == {id(items[0]), id(items[2])}
    assert opened == ["bench/a.yaml"]
    # Even an asynchronous tab echo must preserve an existing multi-selection.
    widget.select_file("bench/c.yaml", emit_signal=False)
    assert len(tree.selectedItems()) == 2
    click(tree, items[0], Qt.KeyboardModifier.ControlModifier)
    assert tree.selectedItems() == [items[2]]
    assert opened == ["bench/a.yaml"]
    click(tree, items[0])
    click(tree, items[2], Qt.KeyboardModifier.ShiftModifier)
    assert len(tree.selectedItems()) == 3
    click(tree, items[1])
    assert tree.selectedItems() == [items[1]]
    assert opened[-1] == "bench/b.yaml"
    widget.close()


def test_results_command_selection_survives_right_click(monkeypatch):
    widget = RunExplorer()
    widget.set_runs([{"group": "g", "experiment_id": name, "agents": [{"name": "a"}]}
                     for name in ("a", "b", "c")])
    widget.resize(400, 500)
    widget.show()
    tree = widget.tree
    tree.expandAll()
    app.processEvents()
    group = tree.topLevelItem(0)
    items = [group.child(i) for i in range(3)]
    click(tree, items[0])
    click(tree, items[2], Qt.KeyboardModifier.ControlModifier)
    assert len(tree.selectedItems()) == 2
    monkeypatch.setattr(QMenu, "exec", lambda *args: None)
    position = tree.visualItemRect(items[0]).center()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.RightButton, pos=position)
    widget._context_menu(position)
    assert len(tree.selectedItems()) == 2
    assert widget.deletion_targets() == [("g", "a"), ("g", "c")]
    click(tree, items[0], Qt.KeyboardModifier.ControlModifier)
    assert tree.selectedItems() == [items[2]]
    widget.close()


def test_tree_double_click_and_open_new_tab_reuse_preview(tmp_path):
    from types import SimpleNamespace
    from frontend.app import Window
    from frontend.config_tab_manager import ConfigTabManager

    root = tmp_path / "experiment"
    (root / "bench").mkdir(parents=True)
    for name in ("a", "b"):
        (root / "bench" / f"{name}.yaml").write_text("seed: 42\n")
    widget = ConfigTreeWidget(root_dir=root, mode="experiments")
    manager = ConfigTabManager()
    window = SimpleNamespace(config_tab_manager=manager)
    widget.file_selected.connect(lambda path, rel: Window.on_config_file_selected(window, path, rel))
    widget.file_opened.connect(lambda path, rel: Window.on_config_file_opened(window, path, rel))
    manager.current_tab_changed.connect(lambda rel, viewer: widget.select_file(rel, emit_signal=False))
    widget.resize(400, 500)
    widget.show()
    widget.tree.expandAll()
    app.processEvents()
    item = widget.find_file_item("bench/a.yaml")
    click(widget.tree, item)
    preview = manager.current_viewer()
    position = widget.tree.visualItemRect(item).center()
    QTest.mouseDClick(widget.tree.viewport(), Qt.MouseButton.LeftButton, pos=position)
    app.processEvents()
    assert manager.count() == 1
    assert manager.current_viewer() is preview
    assert manager.tabBar().tabData(0) == "pinned"
    # The context-menu command emits the same file_opened signal.
    widget.file_opened.emit(root / "bench/a.yaml", "bench/a.yaml")
    assert manager.count() == 1
    click(widget.tree, widget.find_file_item("bench/b.yaml"))
    assert manager.count() == 2
    click(widget.tree, item)
    assert manager.count() == 2
    assert manager.current_viewer() is preview
    widget.close()
    manager.close()
