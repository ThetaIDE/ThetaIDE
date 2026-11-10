"""Monitor full sync is pull-only and scoped to the displayed execution."""
import pytest
from PyQt6.QtWidgets import QApplication

from frontend.site_sync import SiteSyncControl
from src.app.sites import transfer
from src.app.sites.storage import read_json, write_json, workspace_id
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication([])


def test_sync_experiment_downloads_all_outputs_for_displayed_run(monkeypatch, tmp_path):
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    monkeypatch.setattr(transfer, 'connections', lambda *_: {})
    monkeypatch.setattr(transfer, 'push', lambda *_a, **_k: pytest.fail('Experiment sync uploaded code'))
    run = {'remote': {'site': 'cluster', 'execution_id': 'selected-execution'}}
    calls = []
    def sync(root, site, mode, **kwargs):
        calls.append((site, mode, kwargs['execution_id']))
        return {'last_sync': '2026-11-10T12:00:00Z', 'transferred': 1}
    monkeypatch.setattr(transfer, 'sync', sync)
    control = SiteSyncControl('metrics', run_provider=lambda: run)
    control.backend._submit = lambda fn, *, callback: callback(fn(), None)
    completed = []
    control.synced.connect(completed.append)
    control.populate()
    action = next(action for action in control.menu.actions() if action.text() == 'Sync experiment')
    assert action.isEnabled()
    action.trigger()
    assert calls == [('cluster', 'results', 'selected-execution')]
    assert completed == ['cluster']
    assert control.mode == 'metrics'
    control.close()


@pytest.mark.parametrize('run', [None, {'simulated': True}, {'remote': {'site': 'cluster'}}])
def test_sync_experiment_disabled_without_remote_execution(run, monkeypatch, tmp_path):
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    monkeypatch.setattr(transfer, 'connections', lambda *_: {})
    control = SiteSyncControl('metrics', run_provider=lambda: run)
    control.populate()
    action = next(action for action in control.menu.actions() if action.text() == 'Sync experiment')
    assert not action.isEnabled()
    control.close()


def test_scoped_transfer_filters_remote_and_keeps_other_experiment_status(monkeypatch, tmp_path):
    identity = workspace_id(tmp_path)
    old = {'execution_id': 'other-execution', 'state': 'running'}
    selected = {'execution_id': 'selected-execution', 'state': 'completed'}
    write_json(tmp_path / '.thetaide/site_sync.json', {'cluster': {'executions': [old]}})
    monkeypatch.setattr(transfer, 'connection', lambda *_: {'host': 'cluster', 'remote_root': '/project'})
    calls = []
    def remote(root, site, command, *args):
        calls.append((command, args))
        return {'project_id': identity, 'files': [], 'executions': [selected]}
    monkeypatch.setattr(transfer, 'remote', remote)
    result = transfer.sync(tmp_path, 'cluster', 'results', execution_id='selected-execution')
    assert calls == [('_index', ('--mode', 'results', '--execution', 'selected-execution'))]
    assert result['executions'] == [selected]
    assert read_json(tmp_path / '.thetaide/site_sync.json')['cluster']['executions'] == [old, selected]
