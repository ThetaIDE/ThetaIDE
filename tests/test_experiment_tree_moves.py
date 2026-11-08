"""Multi-selection moves preserve experiment files and reject conflicting batches."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication
from frontend.config_tree import ConfigTreeWidget


def make_tree(tmp_path):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "experiment"
    (root / "bench" / "Cartpole").mkdir(parents=True)
    for name in ("one", "two"):
        (root / "bench" / f"{name}.yaml").write_text("# @package _global_\ndefaults: []\n")
    tree = ConfigTreeWidget(root_dir=root, mode="experiments")
    return app, tree, root


def test_move_selected_experiments(tmp_path):
    app, widget, root = make_tree(tmp_path)
    group = widget.tree.topLevelItem(0)
    files = [group.child(i) for i in range(group.childCount())
             if group.child(i).data(0, Qt.ItemDataRole.UserRole).get("type") == "file"]
    for item in files:
        item.setSelected(True)
    assert len(widget.tree._dragged_yamls()) == 2
    assert widget.tree.move_many_handler is not None
    paths, error = widget.move_experiment_yamls(["bench/one.yaml", "bench/two.yaml"], "bench/Cartpole")
    assert error is None
    assert len(paths) == 2
    for name in ("one", "two"):
        assert not (root / "bench" / f"{name}.yaml").exists()
        assert (root / "bench/Cartpole" / f"{name}.yaml").read_text() == "# @package _global_\ndefaults: []\n"
    widget.close()


def test_conflict_prevents_entire_batch_and_rejects_outside_group(tmp_path):
    app, widget, root = make_tree(tmp_path)
    (root / "bench/Cartpole/two.yaml").write_text("existing")
    paths, error = widget.move_experiment_yamls(["bench/one.yaml", "bench/two.yaml"], "bench/Cartpole")
    assert paths is None and error
    assert (root / "bench/one.yaml").exists()
    assert (root / "bench/two.yaml").exists()
    (root / "other").mkdir()
    assert widget.move_experiment_yamls(["bench/one.yaml"], "other")[1]
    assert widget.move_experiment_yamls(["bench/one.yaml"], "../../")[1]
    widget.close()
