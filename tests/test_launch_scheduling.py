"""Training limits, ready-worker ordering and honest live phases."""
import json
import os
import threading
import time

import pytest
from omegaconf import OmegaConf

from src.app.services.resource_budget import GIB, ResourceBudget, ResourcePool
from src.app.telemetry.run import Run
from src.app.telemetry.sinks import MemorySink


def test_host_load_and_cold_start_do_not_override_theta_limits(tmp_path, monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, 'cpu_percent', lambda *_: 100)
    monkeypatch.setattr(psutil.Process, 'children', lambda *a, **kw: [])
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\ncpu_percent=100\nmemory_percent=25\nthreads_per_job=1\nmax_parallel=2\n')
    pool = ResourcePool(tmp_path / 'pool.db', settings)
    spec = {'key': 'cold', 'memory': .2 * GIB, 'cpu': 1}
    tokens = []
    # Both cold workers fit Theta's allocation even if other apps use the host.
    for _ in range(2):
        token, reason = pool.acquire(spec, capacity=(8, 32 * GIB, .01 * GIB))
        assert token and not reason
        tokens.append(token)
    token, reason = pool.acquire(spec, capacity=(8, 32 * GIB, 32 * GIB))
    assert token is None and 'worker limit (2/2' in reason
    pool.release(tokens.pop())
    token, reason = pool.acquire(spec, capacity=(8, 32 * GIB, .01 * GIB))
    assert token and not reason
    for token in [token, *tokens]:
        pool.release(token)


def test_cpu_reservations_do_not_shrink_during_quiet_worker_phases(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.app.services import resource_budget
    class Process:
        pid = 123
        def __init__(self, pid): self.pid = pid
        def create_time(self): return 1
        def status(self): return 'running'
        def children(self, recursive): return []
        def memory_info(self): return SimpleNamespace(rss=.1 * GIB)
        def cpu_times(self): return SimpleNamespace(user=0, system=0)
    monkeypatch.setattr(resource_budget.psutil, 'Process', Process)
    monkeypatch.setattr(resource_budget, 'load_training_settings', lambda _: {'cpu_percent': 25, 'threads_per_job': 2})
    pool = ResourcePool(tmp_path / 'pool.db')
    spec = {'key': 'method', 'memory': .2 * GIB, 'cpu': .25}
    token, _ = pool.acquire(spec, capacity=(8, 32 * GIB, 30 * GIB))
    assert token
    with pool.connect() as db:
        db.execute('UPDATE tasks SET last_sample=?, last_cpu=0', (time.time() - 10,))
    pool.sample()
    with pool.connect() as db:
        assert db.execute('SELECT cpu FROM tasks').fetchone()[0] >= 2
    token2, reason = pool.acquire(spec, capacity=(8, 32 * GIB, 30 * GIB))
    assert token2 is None and 'CPU' in reason


def test_shared_worker_cap_is_atomic_and_changes_apply_to_new_starts(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import psutil
    monkeypatch.setattr(psutil.Process, 'children', lambda *a, **kw: [])
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\nmax_parallel=2\ncpu_percent=100\nthreads_per_job=1\n')
    path = tmp_path / 'pool.db'
    pool = ResourcePool(path, settings)
    spec = {'key': 'small', 'memory': .1 * GIB, 'cpu': 1}
    def acquire(_):
        return ResourcePool(path, settings).acquire(spec, capacity=(8, 32*GIB, 32*GIB))
    with ThreadPoolExecutor(max_workers=6) as executor:
        tokens = [token for token, _ in executor.map(acquire, range(6)) if token]
    assert len(tokens) == 2
    settings.write_text('[training]\nmax_parallel=1\ncpu_percent=100\nthreads_per_job=1\n')
    assert acquire(None)[0] is None
    pool.release(tokens.pop())
    assert acquire(None)[0] is None  # the remaining admitted worker continues
    pool.release(tokens.pop())
    token, reason = acquire(None)
    assert token and not reason
    pool.release(token)


@pytest.mark.parametrize('parallel', [1, 2])
def test_managed_replicates_respect_limit_and_only_queue_ready_seeds(tmp_path, monkeypatch, parallel):
    from src.app.pipeline.variability import run_execution
    monkeypatch.chdir(tmp_path)
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\ncpu_percent=100\nthreads_per_job=1\n')
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    monkeypatch.setenv('THETAIDE_SETTINGS_FILE', str(settings))
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'e', 'plotting': {'enabled': False}})
    plan = {'training_seeds': [1, 2, 3, 4], 'evaluation_seed': None, 'parallel_runs': parallel}
    pool = ResourcePool(tmp_path / 'pool.db', settings)
    lock, ready = threading.Lock(), threading.Event()
    active, peak = 0, 0
    def dispatch(cfg, run, started, waiting):
        nonlocal active, peak
        with lock:
            with pool.connect() as db:
                assert db.execute('SELECT COUNT(*) FROM waiters').fetchone()[0] <= parallel
            active += 1
            peak = max(peak, active)
            if active == parallel:
                ready.set()
        assert ready.wait(3)
        time.sleep(.01)
        with lock:
            active -= 1
    path = run_execution(cfg, {'methods': {'m': {}}}, plan, dispatch=dispatch)
    assert peak == parallel


def test_sequential_replicates_do_not_block_another_ready_experiment(tmp_path, monkeypatch):
    from src.app.pipeline.variability import run_execution
    monkeypatch.chdir(tmp_path)
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\ncpu_percent=100\nthreads_per_job=1\n')
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    monkeypatch.setenv('THETAIDE_SETTINGS_FILE', str(settings))
    pool = ResourcePool(tmp_path / 'pool.db', settings)
    spec = {'key': 'small', 'memory': .1 * GIB, 'cpu': 1}
    def dispatch(cfg, run, started, waiting):
        token, reason = pool.acquire(spec, ticket=run['resource_ticket'], capacity=(8, 32*GIB, 30*GIB))
        assert token and not reason
        other = pool.enqueue()[0]
        other_token, reason = pool.acquire(spec, ticket=other, capacity=(8, 32*GIB, 30*GIB))
        assert other_token and not reason
        pool.release(other_token)
        pool.release(token)
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'e', 'plotting': {'enabled': False}})
    run_execution(cfg, {'methods': {'m': {}}},
                  {'training_seeds': [1, 2], 'evaluation_seed': None, 'parallel_runs': 1}, dispatch=dispatch)


def test_testing_and_diagnostics_preserve_progress_and_clear_stale_training_speed(tmp_path):
    from frontend.live_run import LiveRun
    from src.app.telemetry.records import record
    live = LiveRun(budget=1000)
    def timed(record_type, t, **fields):
        return dict(record(record_type, **fields), t=t)
    live.add([timed('metrics', 10, step=100, data={'transitions': 100}),
              timed('metrics', 11, step=400, data={'transitions': 400})])
    assert live.rate() == 300
    for stage in ['testing', 'diagnostics']:
        live.add([timed('event', 12, kind='stage', data={'stage': stage})])
        assert live.progress() == 400 and live.rate() is None and live.eta_seconds() is None
        live.add([timed('event', 50, kind='stage', data={'stage': 'training'})])
        assert live.progress() == 400 and live.rate() is None
    live.add([timed('metrics', 51, step=450, data={'transitions': 450}),
              timed('metrics', 52, step=500, data={'transitions': 500})])
    assert live.rate() == 50
    sink = MemorySink()
    with Run(tmp_path / 'run', [sink], system_interval=0) as run:
        run.set_stage('training')
        with run.phase('testing'):
            run.log({'eval/reward': 1}, step=500)
        assert [r['data']['stage'] for r in sink.records if r.get('kind') == 'stage'] == ['training', 'testing', 'training']


def test_queue_shows_testing_and_suppresses_training_speed():
    from PyQt6.QtWidgets import QApplication
    from frontend.live_run import LiveRun
    from frontend.queue_panel import ProcessCard
    app = QApplication.instance() or QApplication([])
    live = LiveRun(1000)
    live.add([{'type': 'start', 'run': {'started': 1}},
              {'type': 'metrics', 't': 10, 'step': 400, 'data': {'transitions': 400}},
              {'type': 'event', 'kind': 'stage', 'data': {'stage': 'testing'}}])
    card = ProcessCard({'job_id': 'j', 'status': 'running', 'execution_started': True}, 'm', show_progress=True)
    card.update_progress({'m': live})
    assert card.status.text() == 'Testing'
    assert card.speed.text() == 'Speed: —' and card.eta.text() == 'Time left: —'
    assert '400/1,000 steps' in card.progress_text.text()
    card.close()


def test_new_worker_refreshes_thread_limit_after_coordinator_was_started(tmp_path, monkeypatch):
    from src.app.services.resource_budget import configure_worker_threads, managed_environment
    settings = tmp_path / 'settings.toml'
    settings.write_text('[training]\nthreads_per_job=1\n')
    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    monkeypatch.setenv('THETAIDE_SETTINGS_FILE', str(settings))
    for key, value in managed_environment(8).items():
        monkeypatch.setenv(key, value)
    configure_worker_threads()
    assert os.environ['OMP_NUM_THREADS'] == '1' and os.environ['THETA_MANAGED_THREADS'] == '1'


def test_collapsed_queue_parents_follow_testing_without_loading_hidden_lives():
    from PyQt6.QtWidgets import QApplication
    from frontend.queue_panel import QueuePanel
    app = QApplication.instance() or QApplication([])
    stages = {'m · seed 1': 'testing'}
    reads = []
    panel = QueuePanel(lambda *_: None, lives_for_job=lambda jid: reads.append(jid) or {},
                       phases_for_job=lambda _: stages)
    panel.set_data({'active': [dict(job_id='j', experiment_id='e', status='running', queue_method='m',
                                    queue_seed=1, agents=['m · seed 1'])]})
    method = panel.cards['j:method:m']
    method.expand_button.click()
    reads.clear()
    panel._update_times(force=True)
    assert method.status.text() == 'Testing'
    assert panel.cards['j:experiment'].status.text() == 'Testing'
    assert not reads
    stages['m · seed 1'] = 'training'
    panel._update_times(force=True)
    assert method.status.text() == 'Training' and not reads
    panel.close()


def test_sb3_initial_and_final_evaluations_are_testing_before_metrics(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from hydra.core.hydra_config import HydraConfig
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_module
    import torch
    pytest.importorskip('stable_baselines3')
    torch.set_num_threads(1)
    catalog = get_catalog()
    runner = load_module(catalog.package('sb3-dev'), 'sb3_runner.py')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(HydraConfig, 'get', lambda: SimpleNamespace(mode=SimpleNamespace(name='RUN')))
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'e', 'total_timesteps': 4, 'seed': 73,
                            'agent': {'name': 'dqn', 'algorithm': 'sb3-dev/dqn', 'buffer_size': 32,
                                      'learning_starts': 100, 'batch_size': 4},
                            'model': {'sb3_native_policy': True}, 'env': {'env_id': 'CartPole-v1'},
                            'eval_episodes': 1, 'eval_freq_steps': 100,
                            'variability_run_id': 'r', 'variability_method': 'dqn', 'variability_version': 'version_0',
                            'telemetry': {'system_interval': 0}})
    from src.app.core.execution import RunContext
    context = RunContext('r', catalog.paths, 73, tmp_path / 'checkpoints')
    try:
        runner.execute(cfg, context)
    finally:
        context.close()
    rows = [json.loads(line) for line in next(tmp_path.glob('results/logs/g/e/dqn/version_0/events.jsonl')).read_text().splitlines()]
    stage = None
    evaluations = []
    for row in rows:
        if row.get('kind') == 'stage':
            stage = row['data']['stage']
        if 'eval/reward' in row.get('data', {}):
            assert stage == 'testing'
            evaluations.append(row['step'])
    assert evaluations == [0, 4]
    assert any(r.get('data', {}).get('stage') == 'training' for r in rows)


@pytest.mark.skipif(os.sys.platform != 'darwin', reason='macOS scheduling API')
def test_macos_research_activity_requests_qos_and_records_applied_state():
    from src.app.services.worker_priority import research_activity, scheduling_metadata
    with research_activity():
        details = scheduling_metadata()
        assert details['requested'] == 'user_initiated'
        assert details['thread_qos_applied'], details
        # App Nap protection is available with the desktop app's Cocoa extra.
        pytest.importorskip('Foundation')
        assert details['app_nap_disabled'], details
    assert scheduling_metadata() == {}
