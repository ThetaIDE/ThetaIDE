"""Unit tests for ExportExperimentDialog."""
import pytest
import yaml

from PyQt6.QtWidgets import QApplication, QMessageBox
from frontend.export_dialog import ExportExperimentDialog


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    return app


@pytest.fixture
def workspace(fixture_workspace):
    """The fixture workspace supplies tabular_classification/quick_test and its providers."""
    return fixture_workspace


def test_export_dialog_initialization(qapp, workspace):
    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
    )

    assert dialog.experiment_id == "tabular_classification/quick_test"
    assert dialog.radio_zip.isChecked()
    assert not dialog.options_card.isHidden()
    assert dialog.edit_dest.text().endswith(".zip")
    assert dialog.edit_version.text() == "1.0.0"
    dialog.deleteLater()


def test_export_dialog_format_toggle(qapp, workspace):
    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
    )

    # Switch to YAML format
    dialog.radio_yaml.setChecked(True)
    assert dialog.options_card.isHidden()
    assert dialog.edit_dest.text().endswith(".yaml")

    # Switch back to ZIP
    dialog.radio_zip.setChecked(True)
    assert not dialog.options_card.isHidden()
    assert dialog.edit_dest.text().endswith(".zip")
    dialog.deleteLater()


def test_export_dialog_export_zip(qapp, workspace, tmp_path_factory, monkeypatch):
    out_zip = tmp_path_factory.mktemp("export") / "exported_cartpole.zip"

    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
    )
    dialog.edit_dest.setText(str(out_zip))
    dialog.edit_version.setText("1.0.5")
    dialog.chk_checkpoints.setChecked(False)
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: warnings.append(args[2:]))

    dialog._on_export_clicked()

    assert not warnings, warnings
    assert dialog.export_result is not None
    assert dialog.export_result.success is True
    assert out_zip.is_file()
    dialog.deleteLater()


def test_export_dialog_export_yaml(qapp, workspace, tmp_path_factory):
    out_yaml = tmp_path_factory.mktemp("export") / "exported_cartpole.yaml"

    sample_yaml = "total_timesteps: 5000\nseed: 99\n"
    dialog = ExportExperimentDialog(
        experiment_id="tabular_classification/quick_test",
        workspace_root=workspace,
        raw_yaml_text=sample_yaml,
    )
    dialog.radio_yaml.setChecked(True)
    dialog.edit_dest.setText(str(out_yaml))

    dialog._on_export_clicked()

    assert out_yaml.is_file()
    data = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert data["total_timesteps"] == 5000
    assert data["seed"] == 99
    dialog.deleteLater()
