"""The experiment pane's Recorded Metrics box (telemetry.metrics)."""

import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt6.QtWidgets")
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])

from frontend.metric_picker import MetricPicker, logged_metrics  # noqa: E402


def write_run(logs, group, experiment, header):
    version = logs / group / experiment / "ppo" / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").write_text(",".join(header) + "\n", encoding="utf-8")


def test_known_metrics_come_from_the_newest_earlier_run(tmp_path):
    write_run(tmp_path, "csc510", "cartpole_demo_20261001-100000", ["step", "eval/reward", "losses/total_loss"])
    newer = "cartpole_demo_20261002-100000"
    write_run(tmp_path, "csc510", newer, ["epoch", "step", "eval/reward", "losses/value_loss", "transitions"])
    os.utime(tmp_path / "csc510" / newer, (2e9, 2e9))
    write_run(tmp_path, "csc510", "other_experiment", ["losses/other"])
    assert logged_metrics(tmp_path, "csc510", "cartpole_demo") == ["eval/reward", "losses/value_loss"]
    assert logged_metrics(tmp_path, "csc510", "never_ran") == []


def test_a_section_records_all_of_it_and_unticking_it_starts_from_nothing():
    from frontend.widgets import ToggleSlider
    picker = MetricPicker(["eval/", "losses/total_loss"], ["eval/reward", "eval/reward_std", "losses/total_loss",
                                                          "losses/value_loss"])
    emitted = []
    assert not picker.findChildren(QtWidgets.QCheckBox)
    assert all(isinstance(toggle, ToggleSlider) for toggle in picker.section_boxes.values())
    picker.changed.connect(emitted.append)
    assert picker.selection() == ["eval/", "losses/total_loss"]
    assert not picker.metric_boxes["eval/reward"].isEnabled()  # covered by eval/ (all)
    picker.section_boxes["eval"].setChecked(False)
    assert picker.selection() == ["losses/total_loss"] and not picker.metric_boxes["eval/reward"].isChecked()
    picker.metric_boxes["eval/reward"].setChecked(True)
    assert emitted[-1] == ["eval/reward", "losses/total_loss"]
    picker.metric_boxes["eval/reward"].setChecked(False)
    picker.metric_boxes["losses/total_loss"].setChecked(False)
    assert emitted[-1] == [] and picker.empty_warning.isVisibleTo(picker)


def test_the_box_edits_telemetry_metrics_and_trash_records_everything_again(tmp_path):
    from frontend.config_viewer import ConfigViewer

    path = tmp_path / "exp.yaml"
    path.write_text("paradigm: online_rl\ntotal_timesteps: 1000\ntelemetry:\n  sinks: [csv, stream]\n"
                    "methods:\n  ppo:\n    agent: ppo\n    model: dnn\n", encoding="utf-8")
    viewer = ConfigViewer()
    viewer.auto_save = False
    viewer.load_file(path, "experiment/test_group/exp.yaml")
    assert "telemetry.metrics" not in viewer.field_widgets  # no box until it is added
    viewer.add_block("metrics")
    picker = viewer.field_widgets["telemetry.metrics"]
    assert set(viewer.raw_data["telemetry"]["metrics"]) == {f"{s}/" for s in picker.section_boxes}  # all, at first
    for section in ("train", "losses", "diagnostics", "val", "test", "time", "system"):
        picker.section_boxes[section].setChecked(False)
    assert viewer.raw_data["telemetry"] == {"sinks": ["csv", "stream"], "metrics": ["eval/"]}
    viewer.remove_block("metrics")
    assert viewer.raw_data["telemetry"] == {"sinks": ["csv", "stream"]}  # other telemetry settings are kept
    viewer.close()
