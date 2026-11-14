import sys

import pytest
import yaml

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox, QLineEdit, QToolButton
from frontend.config_viewer import ConfigViewer
from component_fixtures import optional_workspace

app = QApplication.instance() or QApplication(sys.argv[:1])


def test_sweeps_without_visible_parameter_rows_are_editable(tmp_path, optional_tuner_metadata):
    path = tmp_path / "experiment.yaml"
    data = {"tuning": {"component": "optuna", "metric": "eval/reward"}, "methods": {"m": {"agent": "ppo", "model": "dnn", "tune": {
        "agent.gae_lambda": {"type": "float", "low": 0.8, "high": 0.99},
        "model.hidden_sizes.0": {"type": "choice", "values": [32, 64]},
    }}}}
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    viewer = ConfigViewer(workspace=optional_workspace("optuna"))
    viewer.load_file(path, "experiment/group/experiment.yaml")
    for target in data["methods"]["m"]["tune"]:
        assert viewer.findChild(QComboBox, f"sweepType_m_{target}") is not None
    low = viewer.findChild(QAbstractSpinBox, "sweep_low_m_agent.gae_lambda")
    low.setValue(0.7)
    assert viewer.raw_data["methods"]["m"]["tune"]["agent.gae_lambda"]["low"] == 0.7
    viewer.findChild(QToolButton, "sweepRemove_m_model.hidden_sizes.0").click()
    assert "model.hidden_sizes.0" not in viewer.raw_data["methods"]["m"]["tune"]
    viewer.close()


def test_optional_sweep_controls_and_nested_overrides(tmp_path, optional_tuner_metadata):
    from PyQt6.QtGui import QAction
    from frontend.config_viewer import IndentedHierarchyBlock
    from frontend.widgets import SpinBox, ToggleSlider

    path = tmp_path / "experiment.yaml"
    path.write_text("tuning:\n  component: optuna\n  metric: val/loss\n  seed: 17\nmethods: {}\n")
    viewer = ConfigViewer(workspace=optional_workspace("optuna"))
    viewer.load_file(path, "experiment/group/experiment.yaml")
    try:
        assert viewer.findChild(ToggleSlider, "sweepEnabled") is None
        assert viewer.findChild(SpinBox, "sweep_n_jobs") is None
        assert viewer.findChild(SpinBox, "sweep_tpe_candidates") is None
        group = next(block for block in viewer.findChildren(IndentedHierarchyBlock) if block.property("sweepParent") == "sampler")
        assert group.findChild(SpinBox, "sweep_seed").value() == 17
        add = viewer.findChild(QToolButton, "sweepAddSubOption_sampler")
        add.menu().findChild(QAction, "sweepAdd_tpe_candidates").trigger()
        assert viewer.raw_data["tuning"]["options"]["optuna"]["tpe_candidates"] == 24
        group = next(block for block in viewer.findChildren(IndentedHierarchyBlock) if block.property("sweepParent") == "sampler")
        assert group.findChild(SpinBox, "sweep_tpe_candidates") is not None
        viewer.findChild(QToolButton, "sweepRemoveOption_seed").click()
        assert "seed" not in viewer.raw_data["tuning"]
        assert viewer.findChild(SpinBox, "sweep_seed") is None
        viewer.findChild(QComboBox, "sweep_sampler").setCurrentText("random")
        assert viewer.findChild(SpinBox, "sweep_tpe_candidates") is None
        viewer.findChild(QComboBox, "sweep_sampler").setCurrentText("tpe")
        assert viewer.findChild(SpinBox, "sweep_tpe_candidates").value() == 24
    finally:
        viewer.close()


def test_card_order_survives_hidden_cards_and_restart(tmp_path, optional_tuner_metadata):
    from frontend.settings import SettingsManager

    path = tmp_path / "experiment.yaml"
    path.write_text("tuning:\n  component: optuna\n  metric: val/loss\nmethods: {}\n")
    settings = SettingsManager(tmp_path / "workspace")
    viewer = ConfigViewer(settings_manager=settings)
    viewer.load_file(path, "experiment/group/experiment.yaml")
    original = viewer._visible_experiment_boxes()
    assert original[-1] == "Hyperparameter Sweep"
    viewer._move_experiment_box("Hyperparameter Sweep", -1)
    expected = viewer._visible_experiment_boxes()
    tuning = viewer.raw_data.pop("tuning")
    viewer._render_boxes()
    viewer.raw_data["tuning"] = tuning
    viewer._render_boxes()
    assert viewer._visible_experiment_boxes() == expected
    viewer.close()

    # A fresh settings manager reads the persisted TOML, not the first viewer's cache.
    restarted = ConfigViewer(settings_manager=SettingsManager(tmp_path / "workspace"))
    restarted.load_file(path, "experiment/group/experiment.yaml")
    try:
        assert restarted._visible_experiment_boxes() == expected
        restarted._move_experiment_box("Hyperparameter Sweep", 1)
        assert restarted._visible_experiment_boxes() == original
    finally:
        restarted.close()


def test_legacy_enabled_flag_does_not_offer_a_separate_activation_control(tmp_path, optional_tuner_metadata):
    from PyQt6.QtWidgets import QPushButton

    path = tmp_path / "experiment.yaml"
    path.write_text("tuning:\n  component: optuna\n  enabled: false\n  metric: val/loss\n  study_mode: resume\nmethods: {}\n")
    viewer = ConfigViewer(workspace=optional_workspace("optuna"))
    viewer.load_file(path, "experiment/group/experiment.yaml")
    try:
        assert viewer.findChild(QLineEdit, "sweep_study_name") is not None
        assert not any(button.text() == "Enable sweep" for button in viewer.findChildren(QPushButton))
    finally:
        viewer.close()
