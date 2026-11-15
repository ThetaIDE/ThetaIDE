"""Tests for dynamic refresh of models and components in Experiment method dropdowns."""
import json

import pytest
import yaml
from PyQt6.QtWidgets import QApplication

from frontend.config_viewer import ConfigViewer, config_tree, invalidate_config_tree


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_invalidate_config_tree_reloads():
    """Verify that invalidate_config_tree forces config_tree to rediscover from disk."""
    tree1 = config_tree()
    assert tree1 is not None
    tree2 = config_tree()
    assert tree1 is tree2  # Cached

    invalidate_config_tree()
    tree3 = config_tree()
    assert tree3 is not None
    # After invalidation, a new ConfigTree object is instantiated
    assert tree3 is not tree1


def test_dynamic_model_dropdown_population(qapp, tmp_path, fixture_workspace, monkeypatch):
    """Verify that adding/removing a model YAML dynamically reflects in experiment method dropdowns."""
    model_cfg_dir = fixture_workspace / "src/usr/config" / "model"
    model_cfg_dir.mkdir(parents=True, exist_ok=True)
    test_model_yaml = model_cfg_dir / "testtempmodel.yaml"

    try:
        # 1. Initially testtempmodel does not exist
        if test_model_yaml.exists():
            test_model_yaml.unlink()

        invalidate_config_tree()
        tree_initial = config_tree()
        assert "testtempmodel" not in tree_initial.models

        # 2. Add a new model YAML
        test_model_yaml.write_text("type: testtempmodel\nhidden_sizes: [128, 128]\n", encoding="utf-8")

        # Invalidate cache (as done by scaffold_dialog, installer, or app.py on tab switch)
        invalidate_config_tree()
        tree_after_add = config_tree()
        assert "testtempmodel" in tree_after_add.models

        # 3. Load an experiment in ConfigViewer
        viewer = ConfigViewer()
        exp_path = tmp_path / "test_exp.yaml"
        exp_path.write_text(
            "experiment_id: test_exp\n"
            "paradigm: online_rl\n"
            "methods:\n"
            "  m1:\n"
            "    agent: ppo\n"
            "    model: mlp\n",
            encoding="utf-8",
        )
        viewer.load_file(exp_path, "experiment/test/test_exp.yaml")

        # Find any method card's model combo box
        found_combo = False
        for child in viewer.findChildren(object):
            # Check for ComboBox holding model names
            if hasattr(child, "currentText") and hasattr(child, "itemText") and hasattr(child, "count"):
                items = [child.itemText(i) for i in range(child.count())]
                if "testtempmodel" in items:
                    found_combo = True
                    break

        assert found_combo is True, "Newly added model must appear in method card model dropdown!"

        # 4. Now simulate deleting the model
        test_model_yaml.unlink()
        invalidate_config_tree()
        tree_after_delete = config_tree()
        assert "testtempmodel" not in tree_after_delete.models

        # Re-render viewer boxes (simulating reload_data on tab switch or after deletion)
        viewer.reload_data(force_tree_refresh=True)

        found_after_delete = False
        for child in viewer.findChildren(object):
            if hasattr(child, "currentText") and hasattr(child, "itemText") and hasattr(child, "count"):
                items = [child.itemText(i) for i in range(child.count())]
                if "testtempmodel" in items:
                    found_after_delete = True
                    break

        assert found_after_delete is False, "Deleted model must no longer appear in method card model dropdown!"

        viewer.deleteLater()

    finally:
        if test_model_yaml.exists():
            test_model_yaml.unlink()
        invalidate_config_tree()
