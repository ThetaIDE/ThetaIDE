"""Predicted worker sizes govern concurrency, with retries for underestimated growth."""
import json
from contextlib import contextmanager

import pytest

from src.app.services.resource_budget import GIB, ResourceBudget, ResourcePool, uses_accelerator, workload_estimate


def test_cpu_adapter_is_detected_automatically_for_method_config():
    config = {"methods": {"PPO": {"agent": "sb3/ppo", "model": "basics/mlp"},
                          "rules": {"agent": "sb3/ppo", "model": "blendrl"}}}
    assert not uses_accelerator(config)
    assert not uses_accelerator({"agent": {"algorithm": "sb3/ppo"}, "model": {}})


def test_same_method_seeds_share_learned_resource_profile():
    first = {"agent": {"algorithm": "sb3/ppo", "seed": 1}, "model": {"hidden_sizes": [64, 64]}}
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


def test_capacity_recovery_targets_growing_worker_and_learns_its_size(tmp_path):
    pool = ResourcePool(tmp_path / "pool.db")
    active = [dict(id="old", key="old-method", rss=2 * GIB, admitted_memory=3 * GIB,
                   memory=3 * GIB, cpu=1, retry=0, created=1),
              dict(id="new", key="new-method", rss=4 * GIB, admitted_memory=GIB,
                   memory=4 * GIB, cpu=1, retry=0, created=2)]
    with pool.connect() as db:
        for task in active:
            db.execute('INSERT INTO tasks(id,key) VALUES(?,?)', (task['id'], task['key']))
        pool._rebalance(db, active, ResourceBudget(memory_percent=25), (8, 16 * GIB, 8 * GIB))
    assert not pool.should_retry("old")
    assert pool.should_retry("new")
    memory, _, _ = pool.estimate({"key": "new-method", "memory": GIB})
    assert memory == pytest.approx(4.6 * GIB)
    # The retry must wait until its siblings free enough room.
    reason = ResourceBudget(memory_percent=25).admission_reason(
        cores=8, total=16 * GIB, available=8 * GIB,
        active_memory=[2 * GIB], reservations=[3 * GIB], active_threads=[1], next_memory=memory)
    assert "RAM" in reason


def test_soft_target_change_alone_does_not_restart_correctly_estimated_workers(tmp_path):
    pool = ResourcePool(tmp_path / "pool.db")
    active = [dict(id=str(i), rss=3 * GIB, admitted_memory=3 * GIB, memory=3 * GIB,
                   retry=0, created=i) for i in range(2)]
    with pool.connect() as db:
        pool._rebalance(db, active, ResourceBudget(memory_percent=25), (8, 16 * GIB, 8 * GIB))
        assert not db.execute('SELECT 1 FROM tasks WHERE retry=1').fetchone()


def test_training_retry_releases_slot_and_reports_pending_before_restart(monkeypatch):
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
            return len(attempts) == 1

    monkeypatch.setattr(runtime, "get_subprocess_env", lambda _: {
        "THETA_RESOURCE_POOL": "unused.db", "THETA_RESOURCE_SPEC": json.dumps({"memory": 1})})
    monkeypatch.setattr(resource_budget, "ResourcePool", Pool)

    def execute(cmd, env, output_log, on_start=None, should_retry=None):
        attempts.append(1)
        on_start(42)
        if should_retry():
            raise resource_budget.ResourceCapacityRetry()

    monkeypatch.setattr(datasets, "_execute_training", execute)
    datasets.run_experiment([], python_executable="python",
        on_start=lambda pid: events.append("running"), on_wait=lambda: events.append("pending"))
    assert events == ["acquire", "running", "release", "pending", "acquire", "running", "release"]


def test_retry_watchdog_stops_only_its_worker(tmp_path):
    import sys
    from src.app.pipeline.datasets import _execute_training
    from src.app.services.resource_budget import ResourceCapacityRetry

    with pytest.raises(ResourceCapacityRetry):
        _execute_training([sys.executable, '-c', 'import time; time.sleep(10)'],
                          {"THETA_CANCEL_TARGETS": "[]"}, tmp_path / 'worker.log', should_retry=lambda: True)


def test_trial_retry_preserves_trial_identity_and_does_not_report_failure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from omegaconf import OmegaConf
    from src.app.tuners.optuna import parallel

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
        def should_retry(self, token): return len(processes) == 1

    class Process:
        pid = 123
        returncode = None
        def __init__(self, args):
            processes.append(self)
            if len(processes) > 1:
                Path(args[-1]).write_text('{"value": 42}')
                self.returncode = 0
        def poll(self): return self.returncode
        def wait(self): self.returncode = -15

    monkeypatch.setattr(parallel, 'ResourcePool', Pool)
    monkeypatch.setattr(parallel.subprocess, 'Popen', Process)
    monkeypatch.setattr(parallel, 'kill_process_tree', lambda pid: None)
    monkeypatch.setattr(parallel.time, 'sleep', lambda _: None)
    def load(config, overrides):
        return OmegaConf.create({'hydra': {'job': {}, 'sweep': {'dir': str(tmp_path)}},
                                 'theta_tuning': {'log_version': 'same-trial'}})
    impl = SimpleNamespace(task_function=load, config=OmegaConf.create({'hydra': {'sweep': {'dir': str(tmp_path)}}}),
        job_idx=0, hydra_context=SimpleNamespace(config_loader=SimpleNamespace(load_sweep_config=load)))
    results = parallel.launch_trials(impl, [('x=1',)], on_wait=waiting.append,
                                    on_result=lambda i, result: completed.append(i))
    assert len(processes) == 2 and waiting == [0] and completed == [0]
    assert len(results) == 1 and results[0].return_value == 42
