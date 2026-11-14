"""Capacity governs new starts; admitted training runs are never restarted."""
import json
from contextlib import contextmanager

import pytest

from src.app.services.resource_budget import GIB, ResourceBudget, ResourcePool, uses_accelerator, workload_estimate


def test_cpu_adapter_is_detected_automatically_for_method_config():
    config = {"methods": {"PPO": {"agent": "sb3-dev/ppo", "model": "basics/mlp"},
                          "rules": {"agent": "sb3-dev/ppo", "model": "blendrl"}}}
    assert not uses_accelerator(config)
    assert not uses_accelerator({"agent": {"algorithm": "sb3-dev/ppo"}, "model": {}})


def test_same_method_seeds_share_learned_resource_profile():
    first = {"agent": {"algorithm": "sb3-dev/ppo", "seed": 1}, "model": {"hidden_sizes": [64, 64]}}
    second = {**first, "agent": {**first["agent"], "seed": 200}}
    assert workload_estimate(first)["key"] == workload_estimate(second)["key"]


def test_short_worker_keeps_observed_size_for_next_seed(tmp_path, monkeypatch):
    import psutil

    pool = ResourcePool(tmp_path / "pool.db")
    def gone(pid):
        raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil, "Process", gone)
    with pool.connect() as db:
        db.execute('INSERT INTO tasks(id,key,pid,peak,cpu) VALUES(?,?,?,?,?)',
                   ('ended', 'method', 999, 2 * GIB, .7))
        pool._sample(db)
    memory, _, samples = pool.estimate({"key": "method", "memory": GIB})
    assert memory == pytest.approx(2.3 * GIB) and samples == 1


def test_growth_learns_usage_and_blocks_new_starts_without_evicting_workers(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import time
    from src.app.services import resource_budget

    pool = ResourcePool(tmp_path / "pool.db")
    class Process:
        def __init__(self, pid): self.pid = pid
        def create_time(self): return 1
        def status(self): return 'running'
        def children(self, recursive): return []
        def memory_info(self): return SimpleNamespace(rss=(2 if self.pid == 1 else 4) * GIB)
        def cpu_times(self): return SimpleNamespace(user=1, system=0)

    monkeypatch.setattr(resource_budget.psutil, 'Process', Process)
    monkeypatch.setattr(resource_budget.psutil, 'cpu_percent', lambda _: 0)
    monkeypatch.setattr(resource_budget, 'load_training_settings', lambda _: {'memory_percent': 25})
    with pool.connect() as db:
        for name, pid, reserved in [('old', 1, 3), ('new', 2, 1)]:
            db.execute('INSERT INTO tasks(id,key,pid,birth,memory,admitted_memory,cpu,created) VALUES(?,?,?,?,?,?,?,?)',
                       (name, name + '-method', pid, 1, reserved * GIB, reserved * GIB, 1, time.time() - 10))
    pool.sample()
    assert not pool.should_retry("old")
    assert not pool.should_retry("new")
    memory, _, _ = pool.estimate({"key": "new-method", "memory": GIB})
    assert memory == pytest.approx(4.6 * GIB)
    token, reason = pool.acquire({'key': 'next', 'memory': GIB, 'cpu': 1},
                                capacity=(8, 16 * GIB, 8 * GIB))
    assert token is None
    assert "RAM" in reason
    with pool.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0] == 2
        # Sampling preserved the first worker's unused admission headroom.
        assert db.execute('SELECT memory FROM tasks WHERE id="old"').fetchone()[0] == 3 * GIB
        assert not db.execute('SELECT 1 FROM tasks WHERE retry=1').fetchone()


def test_legacy_retry_flags_do_not_restart_workers_or_block_new_starts(tmp_path, monkeypatch):
    from src.app.services import resource_budget
    pool = ResourcePool(tmp_path / "pool.db")
    with pool.connect() as db:
        db.execute('INSERT INTO tasks(id,key,retry) VALUES(?,?,1)', ('legacy', 'legacy'))
    monkeypatch.setattr(pool, '_sample', lambda db: [])
    monkeypatch.setattr(resource_budget.psutil, 'cpu_percent', lambda _: 0)
    monkeypatch.setattr(resource_budget, 'load_training_settings', lambda _: {})
    assert not pool.should_retry('legacy')
    token, reason = pool.acquire({'key': 'next', 'memory': GIB, 'cpu': 1},
                                capacity=(8, 32 * GIB, 24 * GIB))
    assert token and not reason


def test_training_holds_one_slot_until_completion_without_capacity_rechecks(monkeypatch):
    from src.app.pipeline import datasets, runtime
    from src.app.services import resource_budget

    events = []
    attempts = []

    class Pool:
        def __init__(self, *args):
            pass

        @contextmanager
        def slot(self, *args, **kwargs):
            events.append("acquire")
            try:
                yield "token"
            finally:
                events.append("release")

        def attach(self, *args):
            pass

        def should_retry(self, token):
            raise AssertionError('Running training must not be evicted for capacity')

    monkeypatch.setattr(runtime, "get_subprocess_env", lambda _, **__: {
        "THETA_RESOURCE_POOL": "unused.db", "THETA_RESOURCE_SPEC": json.dumps({"memory": 1})})
    monkeypatch.setattr(resource_budget, "ResourcePool", Pool)

    def execute(cmd, env, output_log, on_start=None):
        attempts.append(1)
        on_start(42)

    monkeypatch.setattr(datasets, "_execute_training", execute)
    datasets.run_experiment([], python_executable="python",
        on_start=lambda pid: events.append("running"), on_wait=lambda: events.append("pending"))
    assert events == ["acquire", "running", "release"]
    assert len(attempts) == 1


def test_trial_runs_once_without_capacity_restarts(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from omegaconf import OmegaConf
    from component_fixtures import optional_module
    parallel = optional_module('optuna', 'parallel.py')

    monkeypatch.setenv('THETA_RESOURCE_POOL', str(tmp_path / 'pool.db'))
    processes, waiting, completed = [], [], []

    class Pool:
        path = tmp_path / 'pool.db'
        def __init__(self, *args): pass
        def sample(self): pass
        def enqueue(self, count=1, **kwargs): return ['ticket'] * count
        def dequeue(self, ticket): pass
        def acquire(self, *args, **kwargs): return 'token', ''
        def attach(self, *args): pass
        def release(self, *args): pass
        def should_retry(self, token): raise AssertionError('Running trial must not be evicted')

    class Process:
        pid = 123
        returncode = None
        def __init__(self, args):
            processes.append(self)
            Path(args[-1]).write_text('{"value": 42}')
            self.polls = 0
        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else 0
        def wait(self): self.returncode = -15

    monkeypatch.setattr(parallel, 'ResourcePool', Pool)
    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    monkeypatch.setattr(parallel, 'kill_process_tree', lambda pid: None)
    monkeypatch.setattr(parallel.time, 'sleep', lambda _: None)
    def load(config, overrides):
        return OmegaConf.create({'hydra': {'job': {}, 'sweep': {'dir': str(tmp_path)}},
                                 'theta_tuning': {'log_version': 'same-trial'}})
    impl = SimpleNamespace(task_function=load, config=OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}}),
        job_idx=0, n_jobs=2, hydra_context=SimpleNamespace(config_loader=SimpleNamespace(load_sweep_config=load)))
    results = parallel.launch_trials(impl, [('x=1',)], on_wait=waiting.append,
                                    on_result=lambda i, result: completed.append(i))
    assert len(processes) == 1 and waiting == [] and completed == [0]
    assert len(results) == 1 and results[0].return_value == 42
