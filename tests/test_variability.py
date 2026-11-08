import json

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.variability import make_plan, run_execution


def config():
    return OmegaConf.create({"group": "g", "experiment_id": "e", "paradigm": "online_rl",
                             "eval_episodes": 3, "variability": {"training_seeds": [1, 2],
                             "evaluation_seed": 100, "parallel_runs": 2}})


def context():
    return {"methods": {"first": {"agent": "ppo", "model": "dnn"},
                        "second": {"agent": "ppo", "model": "dnn"}},
            "is_interactive": True, "sanitized_extra_args": []}


def test_plan_is_opt_in_and_validates():
    cfg = config()
    assert make_plan(cfg, context())["training_seeds"] == [1, 2]
    cfg.variability.training_seeds = [1, 1]
    with pytest.raises(ConfigurationError, match="unique"):
        make_plan(cfg, context())
    cfg.variability = None
    assert make_plan(cfg, context()) is None


def test_plan_rejects_unsupported_combinations():
    cfg = config()
    with pytest.raises(ConfigurationError, match="Slurm"):
        make_plan(cfg, {**context(), "is_interactive": False})
    cfg.tuning = {"enabled": True}
    assert make_plan(cfg, context())["training_seeds"] == [1, 2]


def test_execution_isolated_and_failures_retained(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calls = []
    def run(cfg, ctx):
        calls.append(ctx["sanitized_extra_args"])
        if "++seed=2" in ctx["sanitized_extra_args"]:
            raise RuntimeError("failed training")
    monkeypatch.setattr("src.app.pipeline.local_runner._run_methods_sequential", run)
    cfg, ctx = config(), context()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="replicates failed"):
            run_execution(cfg, ctx, make_plan(cfg, ctx))
    manifests = list(tmp_path.glob("results/executions/g/e/*/manifest.json"))
    assert len(manifests) == 2
    runs = [run for path in manifests for run in json.loads(path.read_text())["runs"]]
    assert len({run["run_id"] for run in runs}) == 8
    assert len({(run["logs"], run["agent"], run["version"]) for run in runs}) == 8
    assert sum(run["status"] == "failed" for run in runs) == 4
    assert all(any(arg.startswith("++experiment_id=") for arg in call) for call in calls)


def test_evaluation_event_records_episode_outcomes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from src.app.core.callbacks import EnvironmentEvaluatorCallback

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "variability_run_id": "r",
                            "variability_method": "m"})
    callback = EnvironmentEvaluatorCallback(cfg)
    def evaluate(trainer, module):
        callback._episode_rewards = [1.0, 3.0]
        callback._evaluation_seed = 100
        return 2.0, 1.0
    monkeypatch.setattr(callback, "evaluate", evaluate)
    trainer = SimpleNamespace(loggers=[], global_step=10)
    module = SimpleNamespace(cfg={"paradigm": "online_rl"}, log=lambda *a, **kw: None)
    callback.evaluate_and_log(trainer, module, 10)
    event = json.loads((tmp_path / "results/logs/g/e/m/version_0/evaluation_events.jsonl").read_text())
    assert event["episode_outcomes"] == [1.0, 3.0]
    assert event["training_step"] == 10
    assert event["deterministic_actions"] is None


def test_repetition_is_independent_of_paradigm_and_runner():
    cfg = config()
    cfg.paradigm = "supervised"
    cfg.eval_episodes = 0
    del cfg.variability.evaluation_seed
    ctx = {"methods": {"custom": {"model": "dnn", "entrypoint": "custom.py"}},
           "is_interactive": True, "sanitized_extra_args": []}
    plan = make_plan(cfg, ctx)
    assert plan["capabilities"]["custom"]["episode_events"] is False
    assert plan["training_seeds"] == [1, 2]
    cfg.variability.evaluation_seed = 100
    with pytest.raises(ConfigurationError, match="does not support.*evaluation_seed"):
        make_plan(cfg, ctx)


def test_default_seed_and_disabled_tuning():
    cfg = config()
    del cfg.variability.training_seeds
    cfg.seed = 17
    cfg.tuning = {"enabled": False}
    assert make_plan(cfg, context())["training_seeds"] == [17]


def test_aggregate_uses_run_means_without_extrapolation(tmp_path, monkeypatch):
    from src.app.pipeline.execution_data import ExecutionData
    monkeypatch.chdir(tmp_path)
    runs = []
    for run_id, status, series in [("a", "completed", [(0, [0, 2]), (10, [10, 12])]),
                                   ("b", "completed", [(5, [5, 7]), (15, [15, 17])]),
                                   ("c", "failed", [(0, [100]), (15, [100])])]:
        log = tmp_path / run_id
        log.mkdir()
        runs.append({"run_id": run_id, "method": "m", "status": status, "logs": str(log)})
        events = [{"schema_version": 1, "run_id": run_id, "metric": "eval/reward", "training_step": step,
                   "episode_outcomes": outcomes} for step, outcomes in series]
        (log / "evaluation_events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"schema_version": 1, "runs": runs}))
    points = ExecutionData(path).aggregate("m")
    assert [(p["step"], p["mean"], p["n"]) for p in points] == [(0, 1, 1), (5, 6, 2), (10, 11, 2), (15, 16, 1)]
    assert points[0]["std"] is None
    assert points[1]["std"] == 0
    assert ExecutionData(path).aggregate("m", [-1, 20]) == []


def test_cancel_only_matching_execution(tmp_path, monkeypatch):
    from src.app.pipeline.variability import mark_cancelled
    monkeypatch.chdir(tmp_path)
    for name, pid in [("first", 123), ("second", 456)]:
        path = tmp_path / "results/executions/g/e" / name / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"owner_pid": pid, "runs": [{"status": "completed"},
                                                                 {"status": "running"}, {"status": "queued"}]}))
    mark_cancelled("g", "e", 123)
    first = json.loads((tmp_path / "results/executions/g/e/first/manifest.json").read_text())
    second = json.loads((tmp_path / "results/executions/g/e/second/manifest.json").read_text())
    assert [r["status"] for r in first["runs"]] == ["completed", "cancelled", "cancelled"]
    assert second["runs"][1]["status"] == "running"
