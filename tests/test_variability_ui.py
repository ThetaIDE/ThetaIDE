import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication, QLineEdit
from frontend.config_viewer import ConfigViewer
from frontend.results.executions import ExecutionsView

app = QApplication.instance() or QApplication([])


def test_variability_editor_preserves_evaluation_default(tmp_path):
    path = tmp_path / "sample.yaml"
    path.write_text("paradigm: online_rl\nseed: 7\nmethods:\n  m: {agent: ppo, model: dnn}\n")
    viewer = ConfigViewer()
    viewer.load_file(path, "experiment/sample.yaml")
    viewer.add_block("variability")
    assert "eval_episodes" not in viewer.raw_data
    seeds = viewer.findChild(QLineEdit, "variabilityTrainingSeeds")
    seeds.setText("4, 5")
    seeds.editingFinished.emit()
    assert viewer.raw_data["variability"]["training_seeds"] == [4, 5]
    viewer.remove_block("variability")
    assert "variability" not in viewer.raw_data
    viewer.close()


def test_execution_view_exposes_runs_and_episodes():
    view = ExecutionsView(object(), lambda *args: None)
    run = {"run_id": "r", "method": "m", "training_seed": 1, "status": "completed", "experiment_id": "e"}
    event = {"training_step": 10, "episode_count": 2, "episode_outcomes": [1, 3]}
    view._loaded({"manifest": {"group": "g", "runs": [run]}, "events": {"r": [event]},
                  "curves": {"m": [{"step": 10, "mean": 2, "n": 1, "std": None}]}, "errors": []}, None)
    assert "1 of 1" in view.summary.text()
    assert view.event.count() == 1
    assert len(view.episode_chart.series[0][1]) == 2
    assert view.chart.band is None
    view.close()
