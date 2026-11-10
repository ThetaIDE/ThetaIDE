"""Saved diagnostics share the native viewer and Results' backend/version selection."""
from io import BytesIO
import json

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from frontend.results.data import Source
from frontend.results.panel import ResultsPanel


def output():
    return {"name": "interp/features/heatmap", "label": "Activation heatmap", "kind": "matrix",
            "visual_schema": "theta.visual-output/v1", "artifact_schema": "theta.activation-matrix/v1",
            "axes": ["probe", "feature"], "value_semantics": "Activation, model units",
            "retention": "snapshots", "requires": ["probe_forward.v1"], "data": {
                "schema": "theta.activation-matrix/v1", "sample_ids": "sample_ids",
                "layers": [{"id": "network.1", "array": "activation", "feature_ids": [0, 1],
                            "coverage": "2/2 features"}]}}


class Backend:
    def __init__(self):
        self.files = {}
        self.requests = []
        self.deferred = []
        self.defer_arrays = False
        for version in ("version_0", "version_1"):
            for index in range(2):
                path = f"logs/g/run/ppo/{version}/monitors/features/snapshot_{index:05d}"
                meta = {"schema": "theta.monitor-snapshot/v1", "instance": "features", "target": "network",
                        "optimizer_updates": (index+1)*10, "transitions": (index+1)*100,
                        "arrays": f"snapshot_{index:05d}.npz", "lineage": version,
                        "probe_set_sha256": "fixed", "visuals": {"heatmap": output()}}
                self.files[path+'.json'] = json.dumps(meta).encode()
                archive = BytesIO()
                np.savez_compressed(archive, activation=np.array([[index, 1.], [2., 3.]]), sample_ids=[11, 22])
                self.files[path+'.npz'] = archive.getvalue()

    def list_runs(self, callback, *, completed_only=False):
        callback({"runs": [{"group": "g", "experiment_id": "run", "finished": True,
                            "agents": [{"name": "ppo"}]}]}, None)

    def run_manifest(self, group, experiment, callback, *, completed_only=False):
        callback({"group": group, "experiment_id": experiment, "agents": {"ppo": ["version_0", "version_1"]},
                  "artifacts": [{"path": path, "kind": "metadata", "size": len(data)}
                                for path, data in self.files.items()]}, None)

    def run_metrics(self, group, experiment, agent, callback, **kwargs):
        callback({"version": kwargs.get("version") or "version_1",
                  "metrics": [{"step": 1, "loss": 2}], "columns": ["step", "loss"]}, None)

    def run_file(self, group, experiment, path, callback):
        self.requests.append(path)
        if self.defer_arrays and path.endswith('.npz'):
            self.deferred.append((path, callback))
        else:
            callback(self.files.get(path), None if path in self.files else "Missing file")

    def finish_arrays(self):
        pending, self.deferred = self.deferred, []
        for path, callback in pending:
            callback(self.files[path], None)


@pytest.fixture
def panel():
    app = QApplication.instance() or QApplication([])
    backend = Backend()
    widget = ResultsPanel(backend)
    widget.refresh()
    widget.explorer.select_run('g', 'run')
    yield widget, backend, app
    widget.close()


def drain(app):
    for _ in range(5):
        app.processEvents()


def open_diagnostics(panel, app):
    panel.tabs.setCurrentWidget(panel.diagnostics_page)
    drain(app)
    return panel.diagnostics_view


def test_results_tab_opens_saved_visuals_and_loads_only_selected_arrays(panel):
    widget, backend, app = panel
    assert widget.diagnostics_view is None
    assert not any(path.endswith('.npz') for path in backend.requests)
    diagnostics = open_diagnostics(widget, app)
    view = diagnostics.view
    assert 'version_1' in view.run.currentText()
    assert view.step.count() == 2
    assert 'optimizer step 20' in view.caption.text()
    assert view.canvas.scene().items()
    arrays = [path for path in backend.requests if path.endswith('.npz')]
    assert arrays == ['logs/g/run/ppo/version_1/monitors/features/snapshot_00001.npz']
    assert all('/version_1/' in path for path in backend.requests)
    view.follow.setChecked(False)
    view.step.setCurrentIndex(0)
    drain(app)
    assert 'optimizer step 10' in view.caption.text()
    assert backend.requests[-1].endswith('snapshot_00000.npz')
    view.compare.setCurrentIndex(2)
    assert 'unavailable' not in view.caption.text().lower()


def test_selected_version_and_hidden_tab_do_not_mix_snapshots(panel):
    widget, backend, app = panel
    diagnostics = open_diagnostics(widget, app)
    widget.tabs.setCurrentWidget(widget.curves_page)
    backend.requests.clear()
    widget._selection_changed([Source('g', 'run', 'ppo', 'version_0')])
    drain(app)
    assert not backend.requests
    widget.tabs.setCurrentWidget(widget.diagnostics_page)
    drain(app)
    assert 'version_0' in diagnostics.view.run.currentText()
    assert all('/version_0/' in path for path in backend.requests)


def test_deferred_arrays_refresh_the_open_view_and_refresh_reloads(panel):
    widget, backend, app = panel
    backend.defer_arrays = True
    diagnostics = open_diagnostics(widget, app)
    assert 'Loading snapshot arrays' in diagnostics.view.caption.text()
    backend.finish_arrays()
    drain(app)
    assert 'optimizer step 20' in diagnostics.view.caption.text()
    backend.defer_arrays = False
    count = len(backend.requests)
    widget.refresh()
    drain(app)
    assert len(backend.requests) > count
    assert diagnostics.view.canvas.scene().items()


def test_incomplete_snapshots_are_omitted_and_path_traversal_is_rejected(panel):
    widget, backend, app = panel
    base = 'logs/g/run/ppo/version_1/monitors/features/'
    del backend.files[base+'snapshot_00000.npz']
    metadata = json.loads(backend.files[base+'snapshot_00001.json'])
    metadata['arrays'] = '../outside.npz'
    backend.files[base+'snapshot_00001.json'] = json.dumps(metadata).encode()
    widget.refresh()
    diagnostics = open_diagnostics(widget, app)
    assert 'incomplete' in diagnostics.note.text()
    assert 'beside their metadata' in diagnostics.note.text()
    assert diagnostics.view.step.count() == 0
    assert not any(path.endswith('.npz') for path in backend.requests)


def test_old_runs_have_a_clear_empty_state_and_keep_files_available(panel):
    widget, backend, app = panel
    backend.files.clear()
    widget.refresh()
    diagnostics = open_diagnostics(widget, app)
    assert 'No saved rich diagnostics' in diagnostics.note.text()
    assert 'Files' in diagnostics.note.text()
    assert widget.tabs.indexOf(widget._pages['Files']) >= 0


def test_opening_diagnostics_keeps_files_and_explorer_resizable(panel):
    widget, _, app = panel
    widget.resize(1800, 900)
    widget.show()
    open_diagnostics(widget, app)
    widget.tabs.setCurrentWidget(widget._pages['Files'])
    drain(app)
    splitter = widget.explorer.parentWidget()
    splitter.moveSplitter(240, 1)
    drain(app)
    assert abs(widget.explorer.width()-240) <= 2
