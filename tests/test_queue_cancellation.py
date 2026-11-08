"""Queue cancellation stays scoped to the selected worker."""
import os
import subprocess
import sys
import threading

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def test_cancel_silent_worker_without_stopping_sibling(tmp_path):
    from src.app.services.cancellation import request_path, TrainingCancelled
    from src.app.pipeline.datasets import _execute_training
    import pytest

    target = 'method:selected'
    env = dict(os.environ, THETA_CANCEL_DIR=str(tmp_path), THETA_CANCEL_TARGETS='["method:selected"]')
    sibling = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
    timer = threading.Timer(.3, lambda: request_path(tmp_path, target).touch())
    timer.start()
    try:
        with pytest.raises(TrainingCancelled):
            _execute_training([sys.executable, '-c', 'import time; time.sleep(30)'], env, tmp_path / 'worker.log')
        assert sibling.poll() is None
    finally:
        timer.join()
        sibling.terminate()
        sibling.wait()


def test_card_hover_and_cancel_does_not_open_monitor():
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from frontend.queue_panel import QueuePanel

    app = QApplication.instance() or QApplication([])
    opened, cancelled = [], []
    panel = QueuePanel(lambda *args: opened.append(args), split_methods=True,
                       on_cancel=lambda *args: cancelled.append(args))
    data = {'active': [{'job_id': 'j', 'agents': ['m'], 'status': 'running'}]}
    panel.resize(600, 400)
    panel.set_data(data)
    panel.show()
    app.processEvents()
    card = panel.cards['j:m']
    try:
        QTest.mouseMove(panel, QPoint(2, 2))
        app.processEvents()
        assert card.cancel_button.isHidden()
        before = card.status.geometry()
        QTest.mouseMove(card, card.rect().center())
        app.processEvents()
        assert not card.cancel_button.isHidden()
        assert card.status.geometry() == before
        QTest.mouseMove(card.cancel_button, card.cancel_button.rect().center())
        app.processEvents()
        assert not card.cancel_button.isHidden()
        QTest.mousePress(card.cancel_button, Qt.MouseButton.LeftButton)
        # A queue poll between press and release must not swallow the click.
        panel.set_data(data)
        app.processEvents()
        assert panel.cards['j:m'] is card
        assert not card.cancel_button.isHidden()
        QTest.mouseRelease(card.cancel_button, Qt.MouseButton.LeftButton)
        assert cancelled == [('j', 'method:m')]
        assert opened == []
        QTest.mouseMove(panel, QPoint(2, 2))
        app.processEvents()
        assert card.cancel_button.isHidden()
        panel.set_data(data)
        app.processEvents()
        assert card.cancel_button.isHidden()
        assert not card.cancel_button.isEnabled()
    finally:
        panel.close()


def test_refresh_reuses_reorders_and_removes_cards():
    from PyQt6.QtWidgets import QApplication
    from frontend.queue_panel import QueuePanel

    app = QApplication.instance() or QApplication([])
    panel = QueuePanel(lambda *args: None, split_methods=True)
    first = {'job_id': 'first', 'agents': ['m'], 'status': 'running'}
    second = {'job_id': 'second', 'agents': ['m'], 'status': 'running'}
    panel.set_data({'active': [first, second]})
    card = panel.cards['first:m']
    other = panel.cards['second:m']
    panel.set_data({'active': [second, dict(first, status='pending')]})
    assert panel.cards_layout.itemAt(0).widget() is other
    assert panel.cards_layout.itemAt(1).widget() is card
    assert card.status.text() == 'Pending'
    panel.set_data({'active': [first]})
    assert panel.cards_layout.itemAt(0).widget() is card
    assert other.isHidden()
    assert card.cancel_button.isHidden()
    panel.close()


def test_runner_writes_scoped_request_without_cancelling_job(tmp_path):
    from PyQt6.QtWidgets import QApplication
    from frontend.runner_manager import JobManager
    from src.app.services.cancellation import requested

    app = QApplication.instance() or QApplication([])
    manager = JobManager(store_path=tmp_path / 'jobs.db', project_root=tmp_path)
    manager.jobs['j'] = {'job_id': 'j', 'status': 'running'}
    manager.cancel('j', 'trial:v1')
    directory = tmp_path / 'cancellations/j'
    assert requested(['trial:v1'], directory)
    assert not requested(['trial:v2'], directory)
    assert manager.jobs['j']['status'] == 'running'


def test_cancel_pending_trial_keeps_next_trial(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from src.app.services.cancellation import request_path
    from src.app.tuners.optuna.parallel import launch_trials
    import src.app.tuners.optuna.parallel as parallel

    monkeypatch.setenv('THETA_CANCEL_DIR', str(tmp_path))
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    request_path(tmp_path, 'trial:selected').touch()
    started = []

    class Pool:
        path = tmp_path / 'pool.db'
        def __init__(self, *args): pass
        def sample(self): pass
        def enqueue(self, count=1, **kwargs): return ['ticket'] * count
        def dequeue(self, ticket): pass
        def acquire(self, *args, **kwargs): return 'token', None
        def attach(self, *args): pass
        def release(self, *args): pass

    class Process:
        pid = 123
        returncode = 0
        def __init__(self, args):
            started.append(args[-2])
            from pathlib import Path
            Path(args[-1]).write_text('{"value": 42}')
        def poll(self): return 0

    monkeypatch.setattr(parallel, 'ResourcePool', Pool)
    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    def load(config, overrides):
        return OmegaConf.create({'hydra': {'job': {}, 'sweep': {'dir': str(tmp_path)}}, 'theta_tuning': {'log_version': overrides[0]}})
    impl = SimpleNamespace(task_function=load, config=OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}}), job_idx=0, n_jobs=2,
                           hydra_context=SimpleNamespace(config_loader=SimpleNamespace(load_sweep_config=load)))
    results = launch_trials(impl, [('selected',), ('sibling',)])
    assert results[0].return_value == {'theta_trial_status': 'pruned'}
    assert results[1].return_value == 42
    assert len(started) == 1


def test_sweep_card_labels_preserve_queue_identity_and_cancel_scope(tmp_path):
    import json
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication
    from frontend.runner_manager import JobManager
    from frontend.queue_panel import QueuePanel

    app = QApplication.instance() or QApplication([])
    for method in ('ppo', 'dqn'):
        root = tmp_path / 'results/checkpoints/g/experiment' / method
        root.mkdir(parents=True)
        (root / 'tuning_summary.json').write_text(json.dumps({
            'component': 'optuna', 'study': 'experiment-search', 'stop_reason': 'running',
            'trials': [{'id': 1, 'state': 'running', 'log_version': f'{method}-v1'}],
        }))
    manager = JobManager(store_path=tmp_path / 'jobs.db', project_root=tmp_path)
    job = {'job_id': 'j', 'group': 'g', 'experiment_id': 'experiment',
           'status': 'running', 'agents': ['ppo', 'dqn']}
    items = manager._execution_queue_items(job)
    opened, cancelled = [], []
    panel = QueuePanel(lambda *args: opened.append(args),
                       on_cancel=lambda *args: cancelled.append(args))
    panel.set_data({'active': items})
    identity = 'ppo · experiment-search · trial 1'
    card = panel.cards[f'j:{identity}']
    assert card.experiment_id.text() == 'experiment · ppo / Trial 1'
    panel.set_data({'active': manager._execution_queue_items(job)})
    assert panel.cards[f'j:{identity}'] is card
    QTest.mouseClick(card, Qt.MouseButton.LeftButton)
    assert opened == [('j', identity)]
    card.cancel_button.click()
    assert cancelled == [('j', 'trial:ppo-v1')]
    assert 'j:dqn · experiment-search · trial 1' in panel.cards
    panel.close()
    assert app is not None
