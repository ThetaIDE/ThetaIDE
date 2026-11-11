"""Scheduler termination retires jobs; artifact sealing tracks completeness."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PyQt6.QtWidgets import QApplication

from frontend.app import Window
from frontend.model import disk_run, remote_run_is_active
from frontend.monitor.panel import MonitorPanel
from src.app.services import catalog
from src.app.sites.result_state import results_complete

app = QApplication.instance() or QApplication([])


def save_remote(tmp_path, remote):
    path = tmp_path / '.thetaide/site_sync.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'cluster': {'executions': [remote]}}))


@pytest.mark.parametrize('scheduler', ['running', 'completed', 'failed', 'cancelled', 'timeout'])
@pytest.mark.parametrize('marker', ['sealed', 'legacy'])
def test_completion_comes_from_artifacts_not_scheduler(tmp_path, monkeypatch, scheduler, marker):
    logs = tmp_path / 'results/logs'
    monkeypatch.setattr(catalog, 'LOGS_DIR', logs)
    run = 'nested/run__execution'
    directory = logs / 'group' / run / 'agent/version_0'
    directory.mkdir(parents=True)
    (directory / 'metrics.csv').write_text('transitions,eval/reward\n10,12\n')
    # Even completed individual versions do not finalize a remote publication.
    (directory / 'runtime.json').write_text(json.dumps({'completed_at': '2026-11-10T12:00:00Z'}))
    remote = dict(site='cluster', group='group', result_experiment_id=run, execution_id='execution',
                  state=scheduler, scheduler_state=scheduler)
    if marker == 'sealed':
        remote['publication_manifest'] = {'schema': 'theta.publication/v1', 'sealed': False}
    save_remote(tmp_path, remote)
    entry = catalog.list_runs()[0]
    monitor_run = disk_run(entry)
    active = scheduler == 'running'
    assert remote_run_is_active(monitor_run) == active
    assert MonitorPanel._belongs_in_monitor(monitor_run) == active
    assert not results_complete(remote)
    if active:
        assert catalog.list_runs(completed_only=True) == []
        with pytest.raises(FileNotFoundError):
            catalog.run_manifest('group', run, completed_only=True)
        with pytest.raises(FileNotFoundError):
            catalog.run_metrics('group', run, 'agent', completed_only=True)
    else:
        partial = catalog.list_runs(completed_only=True)[0]
        assert partial['finished'] and not partial['in_progress']
        assert partial['incomplete_results']
        assert catalog.run_manifest('group', run, completed_only=True)['incomplete_results']
        assert catalog.run_metrics('group', run, 'agent', completed_only=True)['total_rows'] == 1
    if marker == 'sealed':
        remote['publication_manifest']['sealed'] = True
    else:
        remote['artifacts_finalized'] = True
    save_remote(tmp_path, remote)
    entries = catalog.list_runs(completed_only=True)
    assert len(entries) == 1 and entries[0]['finished'] and not entries[0]['in_progress']
    assert not entries[0]['incomplete_results']
    assert not remote_run_is_active(disk_run(entries[0]))
    assert not MonitorPanel._belongs_in_monitor(disk_run(entries[0]))
    assert catalog.run_metrics('group', run, 'agent', completed_only=True)['total_rows'] == 1
    assert catalog.run_manifest('group', run, completed_only=True)['agents'] == {'agent': ['version_0']}


def test_sealed_failed_run_without_runtime_exposes_saved_metrics(tmp_path, monkeypatch):
    logs = tmp_path / 'results/logs'
    monkeypatch.setattr(catalog, 'LOGS_DIR', logs)
    directory = logs / 'group/run/agent/version_0'
    directory.mkdir(parents=True)
    (directory / 'metrics.csv').write_text('step,loss\n1,0.5\n')
    save_remote(tmp_path, dict(group='group', result_experiment_id='run', execution_id='execution',
                              state='failed', publication_manifest={'sealed': True}))
    assert catalog.list_runs(completed_only=True)[0]['finished']
    assert catalog.run_metrics('group', 'run', 'agent', completed_only=True)['total_rows'] == 1


def test_sealed_failure_before_metrics_still_has_readable_result_files(tmp_path, monkeypatch):
    logs = tmp_path / 'results/logs'
    monkeypatch.setattr(catalog, 'LOGS_DIR', logs)
    directory = tmp_path / 'results/slurm/execution'
    directory.mkdir(parents=True)
    (directory / 'pipeline.log').write_text('Configuration failed')
    save_remote(tmp_path, dict(group='group', result_experiment_id='nested/run', execution_id='execution',
                              state='failed', publication_manifest={'sealed': True}))
    entries = catalog.list_runs(completed_only=True)
    assert len(entries) == 1 and entries[0]['finished'] and entries[0]['has_run_logs']
    manifest = catalog.run_manifest('group', 'nested/run', completed_only=True)
    assert manifest['artifacts'][0]['path'] == 'slurm/execution/pipeline.log'
    assert catalog.read_run_file('group', 'nested/run', 'slurm/execution/pipeline.log') == b'Configuration failed'
    with pytest.raises(PermissionError):
        catalog.read_run_file('group', 'nested/run', 'slurm/other/pipeline.log')


@pytest.mark.parametrize('persisted', [False, True])
@pytest.mark.parametrize('ending', ['sealed', 'completed', 'failed', 'cancelled', 'timeout'])
def test_remote_sync_hands_last_run_to_results(monkeypatch, persisted, ending):
    entry = dict(group='group', experiment_id='nested/run', agents=[],
                 remote=dict(site='cluster', execution_id='execution', state='running',
                             scheduler_state='running', publication_manifest={'sealed': False}))
    run = disk_run(entry)
    run['transient'] = not persisted
    host = SimpleNamespace(runs=[run], selected=run, active_runs=[], _quitting=False,
        jobs=SimpleNamespace(is_busy=lambda: False), refresh_runs=Mock(),
        _set_monitor_active=Mock(), select_run=Mock(), results_view=Mock(),
        results_panel=object(), tabs=Mock())
    host._open_last_run_results = lambda item: Window._open_last_run_results(host, item)
    monkeypatch.setattr('frontend.app.QTimer.singleShot', lambda delay, callback: callback())
    Window.disk_runs_received(host, {'runs': [entry]}, None)
    host.results_view.open_run.assert_not_called()
    if ending == 'sealed':
        entry['remote'] = {**entry['remote'], 'publication_manifest': {'sealed': True}}
    else:
        entry['remote'] = {**entry['remote'], 'state': ending, 'scheduler_state': ending}
    Window.disk_runs_received(host, {'runs': [entry]}, None)
    host.results_view.open_run.assert_called_once_with('group', 'nested/run', failure=ending in ('failed', 'timeout'), job_id=None)
    host.tabs.setCurrentWidget.assert_called_once_with(host.results_panel)
    Window.disk_runs_received(host, {'runs': [entry]}, None)
    assert host.results_view.open_run.call_count == 1


def test_remote_completion_waits_for_other_running_allocations():
    run = dict(status='running', remote=dict(site='cluster', scheduler_state='running',
                                           publication_manifest={'sealed': False}))
    host = SimpleNamespace(runs=[run], active_runs=[], _quitting=False,
        jobs=SimpleNamespace(is_busy=lambda: False), _set_monitor_active=Mock(), results_view=Mock())
    Window._open_last_run_results(host, {'backend': {'group': 'g', 'experiment_id': 'e'}})
    host.results_view.open_run.assert_not_called()
    host._set_monitor_active.assert_not_called()


def test_managed_publication_must_be_sealed_even_with_old_finalized_receipt():
    assert not results_complete({'artifacts_finalized': True, 'publication_manifest': {'sealed': False}})
    assert results_complete({'publication_state': 'sealed', 'status_error': 'accounting unavailable'})


@pytest.mark.parametrize('state', ['failed', 'cancelled', 'timeout', 'out_of_memory', 'node_fail', 'submission_failed'])
def test_terminated_unsealed_run_without_metrics_exposes_failure_logs(tmp_path, monkeypatch, state):
    monkeypatch.setattr(catalog, 'LOGS_DIR', tmp_path / 'results/logs')
    directory = tmp_path / 'results/slurm/execution'
    directory.mkdir(parents=True)
    (directory / 'pipeline.err').write_text('Worker stopped before final publication')
    record = dict(group='group', result_experiment_id='nested/run', execution_id='execution',
                  state=state, publication_manifest={'sealed': False})
    if state != 'submission_failed':
        record['scheduler_state'] = state
    save_remote(tmp_path, record)
    entry = catalog.list_runs(completed_only=True)[0]
    assert entry['finished'] and entry['incomplete_results'] and not entry['in_progress']
    assert entry['has_run_logs']
    assert not remote_run_is_active(disk_run(entry))
    manifest = catalog.run_manifest('group', 'nested/run', completed_only=True)
    assert manifest['incomplete_results']
    assert manifest['artifacts'][0]['path'] == 'slurm/execution/pipeline.err'
    assert not results_complete(record)


def test_finished_worker_does_not_retire_still_running_allocation():
    assert remote_run_is_active(dict(status='failed', remote=dict(
        state='failed', scheduler_state='completing', publication_manifest={'sealed': False})))
