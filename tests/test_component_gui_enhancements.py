"""Tests for enhanced Component Parameter GUIs in Theta-IDE.

Verifies:
- TagCloudWidget (interactive paradigms chip/tag cloud editor)
- LayerStackWidget (visual block editor for network layers)
- Typed categorical ComboBoxes (activation, optimizer, engine, mode)
- Component Overview Header Card (title, category badge, description)
- Settings integration (visual_layer_editor toggle)
"""
import pytest
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QLineEdit, QComboBox, QWidget
from frontend.widgets import TagCloudWidget, LayerStackWidget, ComboBox
from frontend.config_viewer import ConfigViewer
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


def test_config_viewer_renders_enhanced_widgets(qapp, tmp_path):
    """Test ConfigViewer rendering with TagCloudWidget, LayerStackWidget, ComboBox, and Overview Card."""
    cv = ConfigViewer(parent=None)

    yaml_file = tmp_path / "custom_model.yaml"
    yaml_file.write_text(
        """\
description: "A test model architecture"
compatible_paradigms:
  - online_rl
  - offline_rl
hidden_sizes:
  - 128
  - 64
activation: relu
optimizer: adam
learning_rate: 0.001
use_layernorm: true
""",
        encoding="utf-8",
    )

    cv.load_file(yaml_file, rel_path="model/custom_model.yaml")

    widgets = cv.findChildren(QWidget)

    # 1. Overview Card verification
    overview_texts = [w.text() for w in widgets if isinstance(w, QLineEdit)]
    assert any("A test model architecture" in t for t in overview_texts)

    # 2. TagCloudWidget verification
    tag_clouds = [w for w in widgets if isinstance(w, TagCloudWidget)]
    assert len(tag_clouds) == 1
    assert tag_clouds[0].get_tags() == ["online_rl", "offline_rl"]

    # 3. LayerStackWidget verification
    layer_stacks = [w for w in widgets if isinstance(w, LayerStackWidget)]
    assert len(layer_stacks) == 1
    assert layer_stacks[0].get_layers() == [128, 64]

    # 4. Typed ComboBox verification
    combos = [w for w in widgets if isinstance(w, QComboBox)]
    combo_texts = [c.currentText() for c in combos]
    assert "relu" in combo_texts
    assert "adam" in combo_texts

    # 5. Disable visual_layer_editor and verify fallback
    class MockSettings:
        visual_layer_editor = False
        show_resolved_defaults = False

    cv.settings_manager = MockSettings()
    cv.load_file(yaml_file, rel_path="model/custom_model.yaml")

    widgets_disabled = cv.findChildren(QWidget)
    assert len([w for w in widgets_disabled if isinstance(w, LayerStackWidget)]) == 0
    # TagCloud should still render
    assert len([w for w in widgets_disabled if isinstance(w, TagCloudWidget)]) == 1

