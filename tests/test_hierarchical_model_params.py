"""Tests for hierarchical model parameters in ConfigViewer and Component Plugin standardization."""
import json
from pathlib import Path
import pytest
import yaml
from PyQt6.QtWidgets import QApplication, QLineEdit

from frontend.config_viewer import ConfigViewer, IndentedHierarchyBlock, config_tree
from src.app.core.model_registry import MODEL_REGISTRY, auto_discover_models, get_model_class


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_nsfr_and_neumann_standardized_plugins():
    """Verify nsfr and neumann have valid component manifests, pyproject.toml, and register."""
    auto_discover_models()

    # Check nsfr
    nsfr_manifest = Path("src/usr/models/nsfr/.theta_component.json")
    if not nsfr_manifest.is_file():
        pytest.skip("nsfr/neumann plugins not installed in local workspace")
    assert nsfr_manifest.is_file()
    meta_nsfr = json.loads(nsfr_manifest.read_text(encoding="utf-8"))
    assert meta_nsfr["id"] == "nsfr"
    assert meta_nsfr["kind"] == "model"
    assert Path("src/usr/models/nsfr/pyproject.toml").is_file()
    assert "nsfr" in MODEL_REGISTRY

    # Check neumann
    neumann_manifest = Path("src/usr/models/neumann/.theta_component.json")
    assert neumann_manifest.is_file()
    meta_neumann = json.loads(neumann_manifest.read_text(encoding="utf-8"))
    assert meta_neumann["id"] == "neumann"
    assert meta_neumann["kind"] == "model"
    assert Path("src/usr/models/neumann/pyproject.toml").is_file()
    assert "neumann" in MODEL_REGISTRY


def test_generic_component_viewer_renders_lists_and_nested_dict(qapp, tmp_path):
    """Verify that model YAMLs with lists and nested dicts are rendered as editable fields."""
    model_yaml = tmp_path / "custom_hierarchical_model.yaml"
    model_yaml.write_text(
        "type: custom_hierarchical_model\n"
        "hidden_sizes: [64, 64]\n"
        "dropout: 0.1\n"
        "encoder:\n"
        "  layers: 3\n"
        "  activation: gelu\n",
        encoding="utf-8",
    )

    viewer = ConfigViewer()
    viewer.is_component_mode = True
    viewer.load_file(model_yaml, "model/custom_hierarchical_model.yaml")

    # In raw_data, hidden_sizes should be [64, 64]
    assert viewer.raw_data["hidden_sizes"] == [64, 64]
    assert viewer.raw_data["dropout"] == 0.1
    assert viewer.raw_data["encoder"]["layers"] == 3

    # Edit the list parameter; editors emit typed values.
    viewer._on_field_edited("hidden_sizes", [128, 128, 64])
    assert viewer.raw_data["hidden_sizes"] == [128, 128, 64]

    # Edit nested dictionary parameter
    viewer._on_generic_path_edited(("encoder", "layers"), 4)
    assert viewer.raw_data["encoder"]["layers"] == 4

    # Add a new nested parameter
    viewer._on_generic_path_edited(("encoder", "hidden_dim"), 256)
    assert viewer.raw_data["encoder"]["hidden_dim"] == 256

    # Remove a parameter
    viewer._remove_generic_path(("encoder", "activation"))
    assert "activation" not in viewer.raw_data["encoder"]

    # Save to disk and verify round-trip
    viewer.save_to_disk()
    saved = yaml.safe_load(model_yaml.read_text(encoding="utf-8"))
    assert saved["hidden_sizes"] == [128, 128, 64]
    assert saved["encoder"]["layers"] == 4
    assert saved["encoder"]["hidden_dim"] == 256
    assert "activation" not in saved["encoder"]


def test_method_card_renders_nested_model_parameters(qapp, tmp_path):
    """Verify that method cards render nested dictionary parameters as indented hierarchy blocks."""
    exp_yaml = tmp_path / "test_nested_exp.yaml"
    exp_yaml.write_text(
        "# @package _global_\n"
        "paradigm: online_rl\n"
        "env: cartpole\n"
        "methods:\n"
        "  my_method:\n"
        "    agent: ppo\n"
        "    model:\n"
        "      myfourthmlp:\n"
        "        hidden_sizes: [64, 64]\n"
        "        backbone:\n"
        "          dim: 128\n"
        "          heads: 4\n",
        encoding="utf-8",
    )

    viewer = ConfigViewer()
    viewer.load_file(exp_yaml, "experiment/test/test_nested_exp.yaml")

    # Check method spec in raw_data
    method = viewer.raw_data["methods"]["my_method"]
    model_cfg = method["model"]["myfourthmlp"]
    assert model_cfg["hidden_sizes"] == [64, 64]
    assert model_cfg["backbone"]["dim"] == 128

    # Verify editing nested param in method card
    viewer._on_method_nested_param_edited("my_method", "backbone", "dim", 256)
    assert viewer.raw_data["methods"]["my_method"]["model"]["myfourthmlp"]["backbone"]["dim"] == 256

    # Verify removing nested param
    viewer._remove_method_nested_param("my_method", "backbone", "heads")
    assert "heads" not in viewer.raw_data["methods"]["my_method"]["model"]["myfourthmlp"]["backbone"]

    # Save to disk and verify
    viewer.save_to_disk()
    saved = yaml.safe_load(exp_yaml.read_text(encoding="utf-8"))
    assert saved["methods"]["my_method"]["model"]["myfourthmlp"]["backbone"]["dim"] == 256
    assert "heads" not in saved["methods"]["my_method"]["model"]["myfourthmlp"]["backbone"]
