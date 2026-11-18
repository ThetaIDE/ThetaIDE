import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QLineEdit, QToolButton
from frontend.config_viewer import ConfigViewer

app = QApplication.instance() or QApplication([])


def test_variability_editor_preserves_evaluation_default(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text("paradigm: online_rl\nseed: 7\nmethods:\n  m: {agent: ppo, model: dnn}\n")
    viewer = ConfigViewer()
    viewer.load_file(path, "experiment/sample.yaml")
    viewer.add_block("variability")
    assert "eval_episodes" not in viewer.raw_data
    add = next(button for button in viewer.findChildren(QToolButton) if button.toolTip() == "Add variability parameter")
    next(action for action in add.menu().actions() if action.text() == "training_seeds").trigger()
    app.processEvents()
    seeds = viewer.findChild(QLineEdit, "variabilityTrainingSeeds")
    seeds.setText("4, 5")
    seeds.editingFinished.emit()
    assert viewer.raw_data["variability"]["training_seeds"] == [4, 5]
    viewer.remove_block("variability")
    assert "variability" not in viewer.raw_data
    viewer.close()

