"""The results tree displays the selected scope rather than accumulating switches."""
import os
import sys

import pytest

pytest.importorskip("PyQt6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from frontend.results.data import Manifest
from frontend.results.explorer import RunExplorer

app = QApplication.instance() or QApplication(sys.argv[:1])


def test_selection_replaces_previous_scope_and_restores_version():
    explorer = RunExplorer()
    entries = [
        {"group": "g", "experiment_id": name, "agents": [{"name": "a", "started": date}, {"name": "b"}]}
        for name, date in [("z", "2026-01-01T00:00:00"), ("a", "2026-02-01T00:00:00")]
    ]
    explorer.set_runs(entries)
    group = explorer.tree.topLevelItem(0)
    explorer.tree.setCurrentItem(group)
    assert {s.experiment_id for s in explorer.checked_sources()} == {"a"}
    experiment = group.child(0)
    run = experiment.child(0)
    explorer.add_manifest(Manifest("g", "a", {"a": ["version_0", "version_1"], "b": ["version_0", "version_1"]}, []))
    explorer.tree.setCurrentItem(run)
    assert {(s.agent, s.version) for s in explorer.checked_sources()} == {("a", "version_1"), ("b", "version_1")}
    agent = run.child(0)
    explorer.tree.setCurrentItem(agent.child(0))
    assert [(s.agent, s.version) for s in explorer.checked_sources()] == [("a", "version_0")]
    assert not run.flags() & Qt.ItemFlag.ItemIsUserCheckable
    explorer.set_runs(entries)
    explorer.add_manifest(Manifest("g", "a", {"a": ["version_0", "version_1"], "b": ["version_0", "version_1"]}, []))
    assert [(s.agent, s.version) for s in explorer.checked_sources()] == [("a", "version_0")]
    explorer.tree.setCurrentItem(explorer.tree.topLevelItem(0).child(1).child(0))
    assert {s.experiment_id for s in explorer.checked_sources()} == {"z"}
    explorer.deleteLater()


def test_delete_selection_expands_experiments_and_deduplicates_groups():
    explorer = RunExplorer()
    explorer.set_runs([
        {"group": "g", "experiment_id": name, "agents": [{"name": "a"}]}
        for name in ("x_20260101-000000", "x_20260102-000000", "y")
    ])
    group = explorer.tree.topLevelItem(0)
    group.child(0).setSelected(True)
    group.child(0).child(0).setSelected(True)
    group.child(1).setSelected(True)
    assert explorer.deletion_targets() == [
        ("g", "x_20260101-000000"), ("g", "x_20260102-000000"), ("g", "y")]
    received = []
    explorer.delete_selection_requested.connect(received.append)
    explorer.btn_delete.click()
    assert received == [explorer.deletion_targets()]
    group.setSelected(True)
    assert explorer.deletion_targets() == [("g", None)]
    explorer.deleteLater()


def test_batch_delete_refreshes_once_after_all_callbacks(monkeypatch):
    from types import SimpleNamespace
    from frontend.results.panel import ResultsPanel
    from PyQt6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
    calls, callbacks, completed = [], [], []

    def delete(*args):
        calls.append(args[:-1])
        callbacks.append(args[-1])

    panel = SimpleNamespace(
        unfinished=set(), status=SimpleNamespace(setText=lambda text: None),
        loader=SimpleNamespace(backend=SimpleNamespace(delete_run=delete, delete_group=delete)),
        _run_action_done=lambda *args: completed.append(args),
    )
    ResultsPanel._delete_selection(panel, [("g", "a"), ("h", None)])
    assert calls == [("g", "a"), ("h",)]
    callbacks[0](None, "failed")
    assert completed == []
    callbacks[1](None, None)
    assert completed == [("Delete", "failed")]
