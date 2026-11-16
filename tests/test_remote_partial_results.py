"""Terminated allocations expose partial results without claiming publication succeeded."""
import json

import pytest
from PyQt6.QtWidgets import QApplication

from frontend.model import remote_run_is_active
from src.app.services import catalog
from src.app.sites.result_state import results_complete

app = QApplication.instance() or QApplication([])


@pytest.mark.parametrize('scheduler', ['completed', 'failed', 'cancelled', 'timeout', 'node_fail'])
def test_terminal_snapshot_without_worker_end_marker_is_readable(tmp_path, monkeypatch, scheduler):
    logs = tmp_path / 'src/usr/results/logs'
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
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
