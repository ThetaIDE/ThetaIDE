"""Run uploads code; monitor/results controls only download data."""
import pytest
from PyQt6.QtWidgets import QApplication

from frontend.site_sync import SiteSyncControl, ExperimentSiteSyncControl
from src.app.sites import transfer
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication([])


@pytest.mark.parametrize('mode', ['metrics', 'results', 'experiment'])
def test_every_sync_control_downloads_without_uploading_or_launching(mode, monkeypatch, tmp_path):
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    monkeypatch.setattr(transfer, 'connections', lambda *_: {'cluster': {'host': 'alias', 'remote_root': '/project'}})
    monkeypatch.setattr(transfer, 'push', lambda *_a, **_k: pytest.fail('Sync attempted a code upload'))
    from frontend import site_launch
    monkeypatch.setattr(site_launch, 'launch', lambda *_: pytest.fail('Sync attempted to launch an experiment'))
    calls = []
    def pull(root, site, mode, **options):
        calls.append((site, mode, options.get('diagnostics', False)))
        return {'last_sync': '2026-11-10T12:00:00Z', 'transferred': 0, 'skipped': 1}
    monkeypatch.setattr(transfer, 'sync', pull)
    control = ExperimentSiteSyncControl() if mode == 'experiment' else SiteSyncControl(mode)
    # Run the transfer deterministically while still exercising the UI action wiring.
    control.backend._submit = lambda fn, *, callback: callback(fn(), None)
    if mode == 'experiment':
        control.set_config({'site': {'name': 'cluster'}})
        control.button.click()
        assert calls == [('cluster', 'results', False)]
    else:
        control.populate()
        actions = [action for action in control.menu.actions() if action.isEnabled() and not action.isSeparator()]
        assert all(action.text().startswith('Pull ') for action in actions)
        for action in actions:
            action.trigger()
        expected = [('cluster', mode, False)]
        if mode == 'metrics': expected.append(('cluster', mode, True))
        assert calls == expected
    control.close()


@pytest.mark.parametrize('action', ['push', 'launch', 'check', 'typo'])
def test_sync_control_rejects_actions_other_than_downloads(action, monkeypatch, tmp_path):
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    control = SiteSyncControl('results')
    control.backend._submit = lambda *_a, **_k: pytest.fail('Invalid action scheduled a worker')
    with pytest.raises(ValueError, match='only download'):
        control.start('cluster', action)
    assert control.button.isEnabled()
    control.close()
