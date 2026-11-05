"""Ordering, hierarchy and bounded work for the monitor queue."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def test_nested_fifo_reserves_earlier_trial_before_coordinator_imports(tmp_path, monkeypatch):
    from src.app.services import resource_budget as resources
    monkeypatch.setattr(resources.psutil, 'cpu_percent', lambda *args: 0)
    monkeypatch.setattr(resources.psutil.Process, 'children', lambda *args, **kwargs: [])
    pool = resources.ResourcePool(tmp_path / 'resources.db')
    spec = {'key': 'small', 'memory': .1 * resources.GIB, 'minimum_memory': .1 * resources.GIB, 'cpu': 1}
    capacity = (8, 32 * resources.GIB, 30 * resources.GIB)
    first, second = pool.enqueue(2)
    # Later coordinator finishes importing first. Its seed must still wait.
    later_seed = pool.enqueue(parent=second)[0]
    token, reason = pool.acquire(spec, ticket=later_seed, capacity=capacity)
    assert token is None and 'earlier queued' in reason
    earlier_seeds = pool.enqueue(2, parent=first)
    assert pool.acquire(spec, ticket=later_seed, capacity=capacity)[0] is None
    assert pool.acquire(spec, ticket=earlier_seeds[1], capacity=capacity)[0] is None
    token, reason = pool.acquire(spec, ticket=earlier_seeds[0], capacity=capacity)
    assert token and not reason
    pool.release(token)
    # Cancelling the head unblocks its successor.
    pool.dequeue(earlier_seeds[1])
    token, reason = pool.acquire(spec, ticket=later_seed, capacity=capacity)
    assert token and not reason
    pool.release(token)
    with pool.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM waiters').fetchone()[0] == 0


def test_hierarchy_progress_numeric_order_and_lazy_expansion():
    from PyQt6.QtWidgets import QApplication
    from frontend.queue_panel import QueuePanel
    app = QApplication.instance() or QApplication([])
    opened = []
    reads = []
    panel = QueuePanel(lambda *args: opened.append(args), lives_for_job=lambda jid: reads.append(jid) or {})
    base = dict(job_id='j', experiment_id='experiment', status='running', queue_method='m', queue_study='study')
    jobs = [dict(base, sweep_parent=True, sweep_key='m:study', trials_finished=3, trials_total=20)]
    # Deliberately unordered snapshots must never put trial 10 ahead of trial 2.
    for trial in [10, 2, 1, *range(11, 35)]:
        jobs.append(dict(base, queue_trial=trial, queue_seed=42, queue_envs=4, sweep_trial=True,
                         agents=[f'm · study · trial {trial} · seed 42']))
    panel.set_data({'active': jobs})
    assert len(panel.cards) == 2  # experiment + collapsed method
    method = panel.cards['j:method:m']
    assert method.progress_fraction == .15
    assert '3/20 trials finished' in method.progress_text.text()
    assert panel.cards['j:experiment'].progress_fraction is None
    assert panel.cards['j:experiment'].expand_button.isChecked()
    assert not reads
    method.expand_button.click()
    keys = list(panel.cards)
    assert keys.index('j:method:m:study:trial:1') < keys.index('j:method:m:study:trial:2') < keys.index('j:method:m:study:trial:10')
    trial = panel.cards['j:method:m:study:trial:2']
    trial.expand_button.click()
    seed = panel.cards['j:m · study · trial 2 · seed 42']
    assert seed.experiment_id.text() == 'Training seed 42'
    assert seed.expand_button.isHidden()
    assert not any(key.endswith(':environments') for key in panel._nodes)
    method.expand_button.click()
    assert len(panel.cards) == 2
    # Polling preserves both widget identity and the collapsed state.
    panel.set_data({'active': jobs})
    assert panel.cards['j:method:m'] is method
    assert len(panel.cards) == 2
    assert not method.expand_button.isChecked()
    panel.close()
    assert app is not None


def test_flat_queue_preserves_fifo_across_running_and_pending_states():
    from PyQt6.QtWidgets import QApplication
    from frontend.queue_panel import QueuePanel
    app = QApplication.instance() or QApplication([])
    panel = QueuePanel(lambda *args: None, split_methods=True)
    panel.set_data({'active': [dict(job_id='first', status='pending', agents=['m']),
                               dict(job_id='second', status='running', agents=['m'])]})
    assert list(panel.cards) == ['first:m', 'second:m']
    panel.close()
    assert app is not None


def test_capacity_retry_keeps_its_place_ahead_of_undispatched_seeds(tmp_path, monkeypatch):
    import pytest
    from src.app.services.resource_budget import ResourcePool, ResourceCapacityRetry
    pool = ResourcePool(tmp_path / 'resources.db')
    first, later = pool.enqueue(2)
    starts = []
    def acquire(spec, *, ticket, **kwargs):
        with pool.connect() as db:
            head = db.execute('SELECT id FROM waiters ORDER BY position LIMIT 1').fetchone()[0]
            assert ticket == head
            starts.append(ticket)
            db.execute('DELETE FROM waiters WHERE id=?', (ticket,))
        return 'lease', ''
    monkeypatch.setattr(pool, 'acquire', acquire)
    monkeypatch.setattr(pool, 'release', lambda token: None)
    with pytest.raises(ResourceCapacityRetry) as caught:
        with pool.slot({}, ticket=first):
            raise ResourceCapacityRetry()
    assert caught.value.resource_ticket == first
    with pool.slot({}, ticket=caught.value.resource_ticket):
        pass
    assert starts == [first, first]
    pool.dequeue(later)


def test_trial_progress_keeps_finished_seeds_in_denominator_and_updates_collapsed():
    from types import SimpleNamespace
    from frontend.runner_manager import JobManager
    from frontend.queue_panel import QueuePanel
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    runs = [dict(run_id=str(i), method='m', training_seed=i, status=state)
            for i, state in enumerate(['completed', 'failed', 'running', 'pending'])]
    execution = {'runs': runs, 'parent': {'kind': 'trial', 'study': 's', 'trial_number': 1}}
    summary = {'study': 's', 'stop_reason': 'running', 'requested_trials': 4,
               'trials': [{'id': 0, 'state': 'complete'},
                          {'id': 1, 'state': 'running', 'log_version': 'trial-1', 'execution': execution}]}
    manager = SimpleNamespace(_sweep_summaries=lambda _: [('m', summary)],
                              _replicate_state=lambda r: r['status'])
    base = dict(job_id='j', experiment_id='e', status='running')
    panel = QueuePanel(lambda *_: None)
    panel.set_data({'active': JobManager._execution_queue_items(manager, base)})
    assert panel.cards['j:experiment'].progress_fraction is None
    assert panel.cards['j:method:m'].progress_fraction == .25
    trial = panel.cards['j:method:m:s:trial:1']
    assert trial.progress_fraction == .5
    assert trial.progress_text.text() == '2/4 training runs finished · 50%'
    assert len([key for key in panel.cards if key.startswith('j:m · s · trial 1 · seed')]) == 2
    trial.expand_button.click()
    runs[2]['status'] = 'completed'
    panel.set_data({'active': JobManager._execution_queue_items(manager, base)})
    assert panel.cards['j:method:m:s:trial:1'] is trial
    assert trial.progress_fraction == .75
    assert not trial.expand_button.isChecked()
    # All seeds can finish before objective scoring closes the trial.
    runs[3]['status'] = 'completed'
    panel.set_data({'active': JobManager._execution_queue_items(manager, base)})
    assert trial.progress_fraction == 1
    assert '4/4 training runs finished' in trial.progress_text.text()
    panel.close()
    assert app is not None


def test_ordinary_method_counts_all_runs_and_seed_shows_training_steps():
    from types import SimpleNamespace
    from frontend.runner_manager import JobManager
    from frontend.queue_panel import QueuePanel
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    runs = [dict(run_id=str(i), method='m', training_seed=i, status=state)
            for i, state in enumerate(['completed', 'running', 'queued'])]
    manager = SimpleNamespace(_sweep_summaries=lambda _: [],
                              _execution_manifests=lambda _: [{'runs': runs}],
                              _replicate_state=lambda r: r['status'])
    live = SimpleNamespace(start=None, events=[], rows=[{}], finish=None, budget=100,
                           progress=lambda: 40, rate=lambda: 2, artifacts=[])
    panel = QueuePanel(lambda *_: None, lives_for_job=lambda _: {'m · seed 1': live})
    panel.set_data({'active': JobManager._execution_queue_items(manager,
                    dict(job_id='j', status='running', experiment_id='e'))})
    method = panel.cards['j:method:m']
    assert method.progress_fraction == 1/3
    assert method.progress_text.text() == '1/3 training runs finished · 33%'
    seed = panel.cards['j:m · seed 1']
    assert seed.progress_fraction == .4
    assert seed.progress_text.text() == '40/100 steps · 40%'
    assert panel.cards['j:experiment'].progress_fraction is None
    panel.close()
    assert app is not None
