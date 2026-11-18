"""The shared sweep importance chart renders whatever a tuner's study snapshot reports."""
import copy
import os
import sys

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PyQt6.QtWidgets import QApplication, QProgressBar

from frontend.monitor.sweep_view import SweepView

app = QApplication.instance() or QApplication(sys.argv[:1])


def test_importance_bars_rank_parameters_update_and_clear_when_switching_studies():
    view = SweepView()
    study = {"study": "live", "direction": "maximize", "trials": [],
             "parameter_importance": {"small": 0.2, "large": 0.8},
             "parameter_importance_info": {"completed_trials": 5, "evaluator": "fANOVA", "status": "ready"}}
    view.set_studies([("method", study)])
    chart = view.importance_view
    assert chart.scores == [("large", 0.8), ("small", 0.2)]
    assert [bar.value() for bar in chart.scroll.widget().findChildren(QProgressBar)] == [8000, 2000]
    assert "5 completed" in chart.status.text()
    updated = copy.deepcopy(study)
    updated["parameter_importance"] = {"small": 0.6, "large": 0.4}
    updated["parameter_importance_info"]["completed_trials"] = 6
    view.set_studies([("method", updated), ("other", {"study": "starting", "trials": []})])
    assert chart.scores == [("small", 0.6), ("large", 0.4)]
    assert "6 completed" in chart.status.text()
    view.selector.setCurrentIndex(1)
    assert chart.scores == []
    assert chart.scroll.isHidden()
    assert "0/5" in chart.status.text()
    view.set_studies([])
    assert "Select a study" in chart.status.text()
    view.close()


def test_saved_importance_is_visible_without_importance_metadata():
    view = SweepView()
    view.set_studies([("method", {"study": "plain", "trials": [], "parameter_importance": {"lr": 1.0}})])
    assert view.importance_view.scores == [("lr", 1.0)]
    view.close()
