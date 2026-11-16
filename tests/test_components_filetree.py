"""Tests for Components filetree layout, headers, renames, and System section."""
import pytest
from PyQt6.QtWidgets import QApplication

from frontend.config_tree import ConfigTreeWidget, find_config_root


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_components_panel_namespaced_subfolders(qapp):
    """Verify that Figures, Data Loaders, and Evaluators organize into namespaced subfolders."""
    config_root = find_config_root()
    test_subdirs = [
        config_root / "plot" / "mock_ns",
        config_root / "data" / "mock_ns",
        config_root / "eval" / "mock_ns",
    ]
    created_files = []
    try:
        for sdir in test_subdirs:
            sdir.mkdir(parents=True, exist_ok=True)
            dummy_file = sdir / "test_item.yaml"
            dummy_file.write_text("test_key: 123\n", encoding="utf-8")
            created_files.append(dummy_file)

        tree_widget = ConfigTreeWidget(mode="components")
        tree = tree_widget.tree

        headers = {}
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            headers[item.text(0)] = item

        # Check Figures subfolders
        figures_item = headers["Figures"]
        figures_children = [figures_item.child(j).text(0) for j in range(figures_item.childCount())]
        assert "mock_ns" in figures_children, f"Expected 'mock_ns' in Figures, got {figures_children}"
        if (config_root / "plot" / "thetalib").exists():
            assert "thetalib" in figures_children

        # Check Data Loaders subfolders
        data_item = headers["Data Loaders"]
        data_children = [data_item.child(j).text(0) for j in range(data_item.childCount())]
        assert "mock_ns" in data_children, f"Expected 'mock_ns' in Data Loaders, got {data_children}"

        # Check Evaluators subfolders
        eval_item = headers["Evaluators"]
        eval_children = [eval_item.child(j).text(0) for j in range(eval_item.childCount())]
        assert "mock_ns" in eval_children, f"Expected 'mock_ns' in Evaluators, got {eval_children}"

        # Verify selecting namespaced files
        selected = tree_widget.select_file("plot/mock_ns/test_item.yaml")
        assert selected is True
        curr = tree.currentItem()
        assert curr is not None
        assert curr.text(0) == "test_item"
        assert curr.parent().text(0) == "mock_ns"
        assert curr.parent().parent().text(0) == "Figures"
    finally:
        for f in created_files:
            if f.exists():
                f.unlink()
        for sdir in test_subdirs:
            if sdir.exists() and not any(sdir.iterdir()):
                sdir.rmdir()

