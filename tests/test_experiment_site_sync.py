from types import SimpleNamespace
from unittest.mock import Mock

from PyQt6.QtWidgets import QApplication
from frontend.site_sync import ExperimentSiteSyncControl
from frontend.app import Window

app = QApplication.instance() or QApplication([])


def test_selected_site_button_pulls_results_and_hides_for_local():
    control = ExperimentSiteSyncControl()
    control.start = Mock()
    assert control.isHidden()
    control.set_config({'site': {'name': 'ncshare/common', 'scheduler': 'slurm'}}, experiment='group/recipe')
    assert not control.isHidden()
    control.button.click()
    control.start.assert_called_once_with('ncshare/common', 'experiment', experiment='group/recipe')
    assert control.mode == 'results'
    control.set_config({'site': {'name': 'local', 'scheduler': 'local'}})
    assert control.isHidden()
    control.set_config({'hydra': {'runtime': {'choices': {'site': 'another/site'}}}})
    assert control.site == 'another/site' and not control.isHidden()
    control.set_config({'site': {'name': 'local/default', 'scheduler': 'local'}}, experiment='group/recipe')
    assert control.isHidden()
    control.set_config(None)
    assert control.isHidden()
    control.close()


def test_remote_work_keeps_monitor_visible_without_local_jobs():
    host = SimpleNamespace(runs=[{'remote': {'site': 'cluster', 'scheduler_state': 'running'}, 'status': 'running'}],
                           tabs=SimpleNamespace(set_tab_visible=Mock()), pane_sliders={},
                           _queue_has_active_jobs=False)
    Window._set_monitor_active(host, False)
    host.tabs.set_tab_visible.assert_called_with('monitor', True)
    host.runs[0]['status'] = 'completed'
    host.runs[0]['remote']['scheduler_state'] = 'completed'
    Window._set_monitor_active(host, False)
    host.tabs.set_tab_visible.assert_called_with('monitor', False)



def test_remote_queued_run_does_not_need_local_job_id():
    host = SimpleNamespace(runs=[{'remote': {'site': 'cluster'}, 'status': 'queued',
                                 'backend': {'experiment_id': 'experiment'}}],
                           active_runs=[], queue_running=False,
                           _set_monitor_active=Mock())
    Window.on_queue_changed(host, {'running': False, 'active': [], 'queued': []})
    host._set_monitor_active.assert_called_once_with(False)


def test_idle_monitor_is_hidden_even_if_it_was_opened_while_already_idle():
    host = SimpleNamespace(runs=[], tabs=SimpleNamespace(set_tab_visible=Mock()),
                           pane_sliders={}, _queue_has_active_jobs=False)
    Window._set_monitor_active(host, False)
    host.tabs.set_tab_visible.assert_called_once_with('monitor', False)


def test_unconfirmed_remote_status_does_not_keep_monitor_visible():
    host = SimpleNamespace(runs=[{'remote': {'site': 'cluster', 'scheduler_state': 'unknown',
                                           'status_error': 'accounting unavailable'}, 'status': 'running'}],
                           tabs=SimpleNamespace(set_tab_visible=Mock()), pane_sliders={},
                           _queue_has_active_jobs=True)
    Window._set_monitor_active(host, False)
    host.tabs.set_tab_visible.assert_called_once_with('monitor', False)


def test_remote_display_uses_scheduler_status_for_unfinished_work():
    from frontend.model import _remote_status
    assert _remote_status({'remote': {'state': 'running', 'scheduler_state': 'unknown',
                                     'status_error': 'accounting unavailable'}}) == 'incomplete'
    assert _remote_status({'remote': {'state': 'prepared', 'scheduler_state': 'running'}}) == 'running'
    assert _remote_status({'remote': {'state': 'running', 'scheduler_state': 'completed'}}) == 'completed'
    assert _remote_status({'remote': {'state': 'failed', 'status_error': 'query failed'}}) == 'failed'


def test_replaced_default_button_never_strands_the_menu_and_nothing_polls():
    """The experiment control deletes its default button; the menu must outlive it.

    A 1 s availability poll once closed a menu owned by that deleted button,
    raising inside a Qt slot and aborting the desktop shortly after launch.
    """
    from PyQt6.QtCore import QEvent, QTimer
    control = ExperimentSiteSyncControl()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert control.menu.parent() is control
    control.set_config({'site': {'name': 'cluster', 'scheduler': 'not_installed'}}, experiment='g/e')
    assert control.refresh_availability() is False
    assert not [timer for timer in control.findChildren(QTimer) if timer.isActive()]
    control.close()


def test_site_listing_builds_one_config_resolver(tmp_path, monkeypatch):
    """Each resolver scans every installed manifest; listing N sites must not build N of them."""
    from src.app.config_resolution import ConfigResolver
    from src.app.core.provider_discovery import provider_discovery_snapshot
    from src.app.sites import transfer
    from src.app.workspace import WorkspacePaths
    for name in ('alpha', 'beta', 'gamma'):
        transfer.configure(tmp_path, name, 'h-' + name, '/scratch/' + name)
    built = []
    original = ConfigResolver.__init__
    def counted(self, *args, **kwargs):
        built.append(1)
        original(self, *args, **kwargs)
    monkeypatch.setattr(ConfigResolver, '__init__', counted)
    found = transfer.connections(tmp_path)
    assert {'alpha/default', 'beta/default', 'gamma/default'} <= set(found)
    assert found['beta/default']['host'] == 'h-beta'
    assert len(built) == 1
    # A read operation shares one resolver across every resolve_config call.
    built.clear()
    paths = WorkspacePaths(tmp_path)
    with provider_discovery_snapshot():
        for name in ('alpha', 'beta', 'gamma'):
            paths.resolve_config(f'site/{name}/default.yaml')
        transfer.connections(tmp_path)
    assert len(built) == 1
