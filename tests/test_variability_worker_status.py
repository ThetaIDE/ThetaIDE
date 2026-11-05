"""Resource waiters become running only after subprocess startup."""
import json
from contextlib import contextmanager

from omegaconf import OmegaConf


def test_variance_waits_for_worker_start_before_marking_running(tmp_path, monkeypatch):
    from src.app.pipeline import local_runner
    from src.app.pipeline.variability import run_execution

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "plotting": {"enabled": False}})
    ctx = {"methods": {"m": {"agent": "ppo"}}, "sanitized_extra_args": []}
    plan = {"training_seeds": [1], "evaluation_seed": None, "parallel_runs": 1}
    seen = []

    def dispatch(cfg, context):
        path = next(tmp_path.glob("results/executions/g/e/*/manifest.json"))
        run = json.loads(path.read_text())["runs"][0]
        assert run["status"] == "pending"
        assert not run.get("execution_started")
        context["on_worker_start"](12345)
        run = json.loads(path.read_text())["runs"][0]
        assert run["status"] == "running"
        assert run["worker_pid"] == 12345
        assert run["execution_started"]
        seen.append(run)

    monkeypatch.setattr(local_runner, "_run_methods_sequential", dispatch)
    monkeypatch.setattr(local_runner, "run_plotting_phase", lambda *_: None)
    run_execution(cfg, ctx, plan)
    assert len(seen) == 1
    path = next(tmp_path.glob("results/executions/g/e/*/manifest.json"))
    assert json.loads(path.read_text())["runs"][0]["status"] == "completed"


def test_worker_start_notification_happens_inside_resource_slot(monkeypatch):
    from src.app.pipeline import datasets, runtime
    from src.app.services import resource_budget

    events = []

    class Pool:
        def __init__(self, *args):
            pass

        @contextmanager
        def slot(self, *args, **kwargs):
            assert events == []
            events.append("admitted")
            yield "lease"
            events.append("released")

        def attach(self, token, pid):
            assert (token, pid) == ("lease", 42)
            events.append("attached")

    monkeypatch.setattr(runtime, "get_subprocess_env", lambda _: {
        "THETA_RESOURCE_POOL": "unused.db", "THETA_RESOURCE_SPEC": json.dumps({"memory": 1})})
    monkeypatch.setattr(resource_budget, "ResourcePool", Pool)
    monkeypatch.setattr(resource_budget, "workload_estimate", lambda _: {"memory": 1, "minimum_memory": 0})

    def execute(cmd, env, output_log, on_start=None, should_retry=None):
        assert events == ["admitted"]
        on_start(42)

    monkeypatch.setattr(datasets, "_execute_training", execute)
    datasets.run_experiment([], python_executable="python", on_start=lambda pid: events.append(("started", pid)))
    assert events == ["admitted", "attached", ("started", 42), "released"]
