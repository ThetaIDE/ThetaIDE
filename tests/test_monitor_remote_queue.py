"""Synced allocations share Monitor's queue display, not local job ownership."""
from types import SimpleNamespace
from unittest.mock import Mock

from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from frontend.live_run import LiveRun
from frontend.monitor.panel import MonitorPanel
from frontend.monitor.remote_queue import queue_id, queue_jobs

app = QApplication.instance() or QApplication([])


def remote_run(identity='execution', status='running', site='cluster', agents=None):
    return {'id': identity, 'status': status, 'simulated': False,
            'remote': {'site': site, 'execution_id': identity, 'job_id': '123',
                       'scheduler_state': 'pending' if status == 'queued' else status,
                       'artifacts_finalized': status == 'completed',
                       'last_sync': '2026-11-10T12:00:00Z', 'experiment': 'group/experiment'},
            'backend': {'group': 'group', 'experiment_id': 'experiment__' + identity,
                        'agents': agents or []}, 'config': {}, 'metrics': []}


def test_remote_queue_groups_replicates_and_keeps_sites_distinct():
    first, replicate = remote_run(agents=['ppo']), remote_run(agents=['dqn'])
    replicate['id'] = 'replicate'
    other = remote_run(site='other-cluster')
    completed = remote_run('completed', status='completed')
    unknown = remote_run('unknown')
    unknown['remote']['status_error'] = 'accounting unavailable'
    failed = remote_run('failed', status='failed')
    jobs = queue_jobs([first, replicate, other, completed, unknown, failed])
    assert len(jobs) == 2
    assert queue_id(unknown) not in {job["job_id"] for job in jobs}
    assert jobs[0]['agents'] == ['ppo', 'dqn']
    assert jobs[0]['job_id'] != jobs[1]['job_id']
    assert 'scheduler job 123' in jobs[0]['description']
    assert 'last synced 2026-11-10T12:00:00Z' in jobs[0]['description']


def test_remote_queue_uses_receipt_methods_before_any_logs_exist():
    run = remote_run()
    run['remote']['config'] = {'methods': {'ppo': {}, 'dqn': {}}}
    assert queue_jobs([run])[0]['agents'] == ['ppo', 'dqn']


def test_overlay_shows_local_and_remote_jobs_refreshes_and_opens_remote_run():
    class Jobs(QObject):
        queue_changed = pyqtSignal(dict)
        def queue_state(self):
            return {'active': [{'job_id': '123', 'experiment_id': 'local', 'status': 'running',
                                'agents': ['ppo']}], 'queued': [], 'running': True}
        cancel = Mock()
    jobs = Jobs()
    running = remote_run(agents=['ppo'])
    queued = remote_run('pending', status='queued')
    host = SimpleNamespace(jobs=jobs, runs=[running, queued], selected=running,
                           run_for_job=lambda job_id: {'description': 'Local run'} if job_id == '123' else None,
                           live_for_run=Mock(return_value={'ppo': LiveRun()}),
                           open_job=Mock(), select_run=Mock())
    monitor = MonitorPanel(host)
    monitor.focus_agent = Mock()
    monitor.resize(1000, 700)
    monitor.show()
    QTest.mouseClick(monitor.queue_button, Qt.MouseButton.LeftButton)
    active_key, pending_key = queue_id(running), queue_id(queued)
    cards = monitor.queue_preview.cards
    assert '123:ppo' in cards
    assert active_key + ':experiment' in cards
    assert active_key + ':ppo' in cards
    assert pending_key in cards
    assert not monitor.queue_preview.empty.isVisible()
    assert not cards[active_key + ':ppo']._can_cancel
    assert cards['123:ppo']._can_cancel
    # Even a programmatically requested remote cancellation cannot hit local job 123.
    monitor._cancel_queue_job(active_key)
    jobs.cancel.assert_not_called()
    QTest.mouseClick(cards[active_key + ':ppo'], Qt.MouseButton.LeftButton)
    host.select_run.assert_called_once_with(running)
    host.open_job.assert_not_called()
    monitor.focus_agent.assert_called_once_with('ppo')
    assert not monitor.queue_overlay.isVisible()
    # A sync can finish the remote job without any local queue_changed event.
    QTest.mouseClick(monitor.queue_button, Qt.MouseButton.LeftButton)
    running['status'] = 'completed'
    running['remote']['scheduler_state'] = 'completed'
    running['remote']['artifacts_finalized'] = True
    host.selected = queued
    monitor.set_runs(host.runs)
    assert active_key + ':ppo' not in monitor.queue_preview.cards
    assert pending_key in monitor.queue_preview.cards
    assert '123:ppo' in monitor.queue_preview.cards
    monitor.close()
