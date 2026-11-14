"""Independent tuner monitors reconcile workspace, readiness and lifecycle changes."""
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QApplication

from frontend.monitor.panel import MonitorPanel
from src.app.core.contributions import ContributionRegistry
from src.app.workspace import WorkspacePaths


@pytest.fixture
def application():
    app = QApplication.instance() or QApplication([])
    yield app


def fixture(tmp_path):
    root = tmp_path / 'src/usr/packages/lattice'
    shutil.copytree(Path(__file__).parent / 'fixtures/contribution_tuner', root)
    (root / 'provider.py').write_text('raise AssertionError("monitor discovery imported tuner")')
    paths = WorkspacePaths(tmp_path, application=tmp_path / 'empty-app')
    config = {'tuning': {'component': 'lattice', 'options': {'lattice': {'token': 'configured'}}}}
    run = {'id': 'run', 'status': 'running', 'config': config,
           'backend': {'group': 'g', 'experiment_id': 'e'}}
    summary = tmp_path / 'results/checkpoints/g/e/m/tuning_summary.json'
    summary.parent.mkdir(parents=True)
    summary.write_text(json.dumps({'component': 'lattice', 'study': 'search', 'stop_reason': 'running',
        'trials': [{'id': 0, 'state': 'running', 'execution_started': 1}]}))
    host = SimpleNamespace(workspace_paths=paths, selected=run, runs=[run],
        jobs=SimpleNamespace(project_root=tmp_path, queue_changed=Mock()), live_for_run=lambda *a, **k: {})
    return host, root, summary


def test_monitor_has_no_unconditional_tuning_widget(application, tmp_path):
    paths = WorkspacePaths(tmp_path, application=tmp_path / 'empty-app')
    host = SimpleNamespace(workspace_paths=paths, selected={'id': 'run', 'status': 'running'},
                           live_for_run=lambda *a, **k: {})
    panel = MonitorPanel(host)
    panel.refresh_live()
    assert panel.sweep_view is None and 'Sweep' not in panel._views
    assert panel.show_view('Sweep') is False
    assert panel.tuning_notice.isHidden()
    panel.close()


def test_monitor_mount_disable_enable_and_close(application, tmp_path):
    host, root, summary = fixture(tmp_path)
    before = summary.read_bytes()
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    original = panel.sweep_view
    assert panel.views.currentWidget() is original
    assert original.studies[0][1]['component'] == 'lattice'
    panel.refresh_live()
    assert panel.sweep_view is original
    registry = ContributionRegistry(host.workspace_paths)
    registry.set_enabled('tuner', 'lattice', False)
    panel.refresh_live()
    assert panel.sweep_view is None and panel.views.currentWidget() is panel.charts_view
    assert panel._focused_sweep is None and original.parent() is None
    assert original.studies == [] and original.lives == {}
    assert 'disabled' in panel.tuning_notice.text()
    registry.set_enabled('tuner', 'lattice', True)
    panel.focus_sweep('m:search')
    assert panel.sweep_view is not None and panel.sweep_view is not original
    assert panel.tuning_notice.isHidden()
    panel.close()
    assert panel.sweep_view is None and not panel._tuning_monitors.scope.active
    panel.refresh_live()  # Queued host notifications cannot remount a closed panel.
    assert panel.sweep_view is None
    assert before == summary.read_bytes()


def test_readiness_removal_and_saved_recipe_are_preserved(application, tmp_path):
    host, root, summary = fixture(tmp_path)
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    tuning = host.selected['config']['tuning']
    tuning['options']['lattice']['token'] = ''
    panel.refresh_live()
    assert panel.sweep_view is None and 'configure tuning.options.lattice.token' in panel.tuning_notice.text()
    tuning['options']['lattice']['token'] = 'restored'
    panel.focus_sweep('m:search')
    assert panel.sweep_view is not None
    (root / '.theta_component.json').unlink()
    panel.refresh_live()
    assert panel.sweep_view is None and 'not installed' in panel.tuning_notice.text()
    assert tuning['component'] == 'lattice' and summary.is_file()
    panel.close()


def test_workspace_switch_releases_same_provider_mount(application, tmp_path):
    host, root, _ = fixture(tmp_path / 'one')
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    previous = panel.sweep_view
    other, _, _ = fixture(tmp_path / 'two')
    host.workspace_paths = other.workspace_paths
    host.selected, host.runs, host.jobs = other.selected, other.runs, other.jobs
    panel.refresh_live()
    assert panel.sweep_view is not None and panel.sweep_view is not previous
    assert previous.parent() is None and previous.studies == []
    panel.close()


def test_changed_package_replaces_monitor_and_alias_uses_same_owner(application, tmp_path):
    host, root, summary = fixture(tmp_path)
    manifest = root / '.theta_component.json'
    data = json.loads(manifest.read_text())
    data['capabilities'][0]['aliases'] = ['lattice-alias']
    manifest.write_text(json.dumps(data))
    host.selected['config']['tuning']['component'] = 'lattice-alias'
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    previous = panel.sweep_view
    assert previous is not None and len(panel._tuning_monitors.scope.active) == 1
    data['version'] = '1.1.0'
    manifest.write_text(json.dumps(data))
    panel.refresh_live()
    assert panel.sweep_view is not previous and previous.parent() is None
    panel.close()


def test_saved_provider_without_monitor_slot_is_diagnostic(application, tmp_path):
    host, root, _ = fixture(tmp_path)
    manifest = root / '.theta_component.json'
    data = json.loads(manifest.read_text())
    data['capabilities'][0]['contributions'] = []
    manifest.write_text(json.dumps(data))
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    assert panel.sweep_view is None and 'does not contribute' in panel.tuning_notice.text()
    panel.close()


def test_queue_tracks_independent_exports_even_without_installed_provider(application, tmp_path):
    from frontend.runner_manager import JobManager
    host, root, summary = fixture(tmp_path)
    data = json.loads(summary.read_text())
    data.update(component='another-tuner', requested_trials=2, initial_trials=0)
    data['trials'][0]['log_version'] = 'version_trial'
    summary.write_text(json.dumps(data))
    manager = SimpleNamespace(project_root=tmp_path, logs_dir=tmp_path / 'results/logs')
    for name in ('_sweep_summaries', '_run_dir', '_replicate_state'):
        setattr(manager, name, getattr(JobManager, name).__get__(manager))
    job = {'job_id': 'job', 'status': 'running', 'group': 'g', 'experiment_id': 'e'}
    items = JobManager._execution_queue_items(manager, job)
    assert any(item.get('sweep_parent') for item in items)
    assert any(item.get('sweep_trial') and item['agents'] == ['m · search · trial 0'] for item in items)


def test_unconfigured_enabled_monitor_still_requires_readiness(application, tmp_path):
    host, root, _ = fixture(tmp_path)
    manifest = root / '.theta_component.json'
    data = json.loads(manifest.read_text())
    data['capabilities'][0]['contributions'][-1]['availability'] = 'enabled'
    manifest.write_text(json.dumps(data))
    host.selected['config']['tuning']['options']['lattice']['token'] = ''
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    assert panel.sweep_view is None and 'configure' in panel.tuning_notice.text()
    panel.close()


@pytest.mark.parametrize('source', ['telemetry', 'saved-file'])
def test_readiness_uses_monitored_launch_instead_of_flat_desktop_config(application, tmp_path, source):
    from frontend.live_run import LiveRun
    import yaml
    host, _, _ = fixture(tmp_path)
    config = host.selected['config']
    host.selected['config'] = {'name': 'desktop-display-only'}
    host.composed_config = {'tuning': {'component': 'unrelated-editor-provider'}}
    if source == 'telemetry':
        live = LiveRun()
        live.config = config
        live.start = {'type': 'start'}
        host.live_for_run = lambda *a, **k: {'m · search · trial 0': live}
    else:
        folder = tmp_path / 'results/logs/g/e'
        folder.mkdir(parents=True)
        (folder / 'config.yaml').write_text(yaml.safe_dump(config))
        host.run_folder = lambda run: folder
    panel = MonitorPanel(host)
    panel.focus_sweep('m:search')
    assert panel.sweep_view is not None and panel.tuning_notice.isHidden()
    assert panel.sweep_view.studies[0][1]['study'] == 'search'
    panel.close()
