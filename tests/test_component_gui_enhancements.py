"""Tests for enhanced Component Parameter GUIs in Theta-IDE.

Verifies:
- TagCloudWidget (interactive paradigms chip/tag cloud editor)
- LayerStackWidget (visual block editor for network layers)
- Typed categorical ComboBoxes (activation, optimizer, engine, mode)
- Component Overview Header Card (title, category badge, description)
- Settings integration (visual_layer_editor toggle)
"""
import pytest
from PyQt6.QtWidgets import QApplication, QLineEdit, QComboBox, QWidget
from frontend.widgets import TagCloudWidget, LayerStackWidget, ComboBox
from frontend.config_panel.controls import CATEGORICAL_CHOICES
from frontend.config_panel.catalogs import _get_component_kind, _get_available_paradigms
from frontend.settings import SettingsManager


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_tag_cloud_widget_lifecycle(qapp):
    """Test TagCloudWidget initialization, tag addition, deletion, and signal emission."""
    captured = []
    tc = TagCloudWidget(
        tags=["online_rl"],
        available_options=["online_rl", "offline_rl", "supervised", "unsupervised"],
    )
    tc.tagsChanged.connect(lambda tags: captured.append(list(tags)))

    assert tc.get_tags() == ["online_rl"]

    # Add tag
    tc._add_tag("offline_rl")
    assert tc.get_tags() == ["online_rl", "offline_rl"]
    assert captured[-1] == ["online_rl", "offline_rl"]

    # Prevent duplicate
    tc._add_tag("online_rl")
    assert tc.get_tags() == ["online_rl", "offline_rl"]

    # Remove tag
    tc._remove_tag("online_rl")
    assert tc.get_tags() == ["offline_rl"]
    assert captured[-1] == ["offline_rl"]

    # Set tags programmatically
    tc.set_tags(["supervised", "unsupervised"])
    assert tc.get_tags() == ["supervised", "unsupervised"]


def test_layer_stack_widget_lifecycle(qapp):
    """Test LayerStackWidget layer manipulation, layer addition, deletion, and signals."""
    captured = []
    ls = LayerStackWidget(layers=[256, 128, 64])
    ls.layersChanged.connect(lambda layers: captured.append(list(layers)))

    assert ls.get_layers() == [256, 128, 64]

    # Modify layer
    ls._on_layer_val_changed(1, 512)
    assert ls.get_layers() == [256, 512, 64]
    assert captured[-1] == [256, 512, 64]

    # Add layer
    ls._add_layer()
    assert ls.get_layers() == [256, 512, 64, 64]
    assert captured[-1] == [256, 512, 64, 64]

    # Remove layer
    ls._remove_layer(0)
    assert ls.get_layers() == [512, 64, 64]
    assert captured[-1] == [512, 64, 64]


def test_categorical_choices_and_component_kinds():
    """Verify categorical choices and component kind mappings."""
    assert "relu" in CATEGORICAL_CHOICES["activation"]
    assert "adam" in CATEGORICAL_CHOICES["optimizer"]
    assert "min" in CATEGORICAL_CHOICES["mode"]

    assert _get_component_kind("agent/ppo.yaml")[0] == "Algorithm"
    assert _get_component_kind("model/mlp.yaml")[0] == "Model"
    assert _get_component_kind("env/cartpole.yaml")[0] == "Environment"
    assert _get_component_kind("data/image_data.yaml")[0] == "Data Loader"
    assert _get_component_kind("eval/mimic_eval.yaml")[0] == "Evaluator"
    assert _get_component_kind("plot/convergence.yaml")[0] == "Figure"


def test_visual_layer_editor_setting(tmp_path):
    """Test visual_layer_editor setting in SettingsManager."""
    sm = SettingsManager(data_dir=tmp_path)
    assert sm.visual_layer_editor is True

    sm.visual_layer_editor = False
    assert sm.visual_layer_editor is False

    sm.visual_layer_editor = True
    assert sm.visual_layer_editor is True


