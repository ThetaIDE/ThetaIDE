"""Capacity governs new starts; admitted training runs are never restarted."""
import json
from contextlib import contextmanager

import pytest

from src.app.services.resource_budget import GIB, ResourcePool, uses_accelerator, workload_estimate


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


