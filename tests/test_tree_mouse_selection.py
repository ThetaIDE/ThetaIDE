"""Exercise native mouse selection, including the editor tab's selection echo."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMenu
from frontend.results.explorer import RunExplorer

app = QApplication.instance() or QApplication([])


def click(tree, item, modifiers=Qt.KeyboardModifier.NoModifier):
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.LeftButton, modifiers,
                     tree.visualItemRect(item).center())
    app.processEvents()


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
    deleted = []
    widget.delete_selection_requested.connect(deleted.append)

    def delete_selection(menu, _position):
        next(action for action in menu.actions() if action.text() == "Delete…").trigger()

    monkeypatch.setattr(QMenu, "exec", delete_selection)
    position = tree.visualItemRect(items[0]).center()
    QTest.mouseClick(tree.viewport(), Qt.MouseButton.RightButton, pos=position)
    widget._context_menu(position)
    assert len(tree.selectedItems()) == 2
    assert deleted == [[("g", "a"), ("g", "c")]]
    click(tree, items[0], Qt.KeyboardModifier.ControlModifier)
    assert tree.selectedItems() == [items[2]]
    widget.close()


