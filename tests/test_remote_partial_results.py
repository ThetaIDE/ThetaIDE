"""Terminated allocations expose partial results without claiming publication succeeded."""
import json
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication

from frontend.results.panel import ResultsPanel
from frontend.model import remote_run_is_active
from src.app.services import catalog
from src.app.sites.result_state import results_complete

app = QApplication.instance() or QApplication([])


def test_results_tree_updates_when_partial_publication_is_later_sealed():
    panel = ResultsPanel(SimpleNamespace())
    entry = dict(group='group', experiment_id='run', agents=[], finished=True,
                 has_run_logs=True, incomplete_results=True)
    panel._runs_listed([entry])
    run = panel.explorer.tree.topLevelItem(0).child(0).child(0)
    assert run.text(0) == 'run · incomplete results'
    assert 'Job ended' in run.toolTip(0)
    entry['incomplete_results'] = False
    panel._runs_listed([entry])
    run = panel.explorer.tree.topLevelItem(0).child(0).child(0)
    assert run.text(0) == 'run'
    assert 'Job ended' not in run.toolTip(0)
    panel.close()


@pytest.mark.parametrize('scheduler', ['completed', 'failed', 'cancelled', 'timeout', 'node_fail'])
def test_terminal_snapshot_without_worker_end_marker_is_readable(tmp_path, monkeypatch, scheduler):
    logs = tmp_path / 'results/logs'
    monkeypatch.setattr(catalog, 'LOGS_DIR', logs)
    directory = logs / 'group/run/agent/version_0'
    directory.mkdir(parents=True)
    (directory / 'metrics.csv').write_text('step,loss\n1,0.5\n')
    remote = dict(group='group', result_experiment_id='run', execution_id='execution',
                  state='running', scheduler_state=scheduler, publication_manifest={'sealed': False})
    path = tmp_path / '.thetaide/site_sync.json'
    path.parent.mkdir()
    path.write_text(json.dumps({'cluster': {'executions': [remote]}}))
    entries = catalog.list_runs(completed_only=True)
    assert entries[0]['finished'] and entries[0]['incomplete_results']
    assert not entries[0]['in_progress']
    assert catalog.run_metrics('group', 'run', 'agent', completed_only=True)['total_rows'] == 1
    assert catalog.run_manifest('group', 'run', completed_only=True)['agents'] == {'agent': ['version_0']}
    assert not remote_run_is_active({'remote': remote})
    assert not results_complete(remote)


def test_terminal_scheduler_evidence_wins_over_old_status_error():
    remote = dict(state='running', scheduler_state='failed', status_error='old query failure',
                  publication_manifest={'sealed': False})
    assert not remote_run_is_active({'remote': remote})
    assert not results_complete(remote)
