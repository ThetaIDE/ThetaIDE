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
    with pytest.raises(ConfigurationError, match="tuning"):
        make_plan(cfg, context())


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
    assert len({run["logs"] for run in runs}) == 8
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
    event = json.loads((tmp_path / "results/logs/g/e/evaluation_events.jsonl").read_text())
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


def test_real_sb3_replicates_record_episodes(tmp_path, monkeypatch):
    import importlib.util
    import os
    import shutil
    import sys
    if importlib.util.find_spec("stable_baselines3") is None:
        pytest.skip("stable-baselines3 is optional")
    project = __import__("pathlib").Path(__file__).resolve().parents[1]
    config_dir = tmp_path / "in/config"
    shutil.copytree(project / "in/config", config_dir)
    (config_dir / "agent/ppo.yaml").write_text("algorithm: ppo\nn_steps: 8\nbatch_size: 4\n")
    (config_dir / "model/dnn.yaml").write_text("architecture: dnn\n")
    (config_dir / "experiment/smoke.yaml").write_text(
        "# @package _global_\ngroup: g\nexperiment_id: e\nparadigm: online_rl\n"
        "total_timesteps: 8\neval_episodes: 2\neval_freq_steps: 8\n"
        "env: {name: cartpole, env_id: CartPole-v1}\n")
    runner = tmp_path / "src/usr/algorithms/sb3-suite/sb3_runner.py"
    runner.parent.mkdir(parents=True)
    shutil.copy2(project / "src/usr/algorithms/sb3-suite/sb3_runner.py", runner)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "e", "experiment_name": "smoke",
                            "paradigm": "online_rl", "eval_episodes": 2,
                            "variability": {"training_seeds": [1, 2], "evaluation_seed": 100, "parallel_runs": 1}})
    ctx = {"methods": {"m": {"agent": "ppo", "model": "dnn", "entrypoint": str(runner), "python": sys.executable}},
           "sanitized_extra_args": [], "storage_url": None, "is_sweep": False, "is_interactive": True}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(project), os.environ.get("PYTHONPATH", "")]))
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    run_execution(cfg, ctx, make_plan(cfg, ctx))
    from src.app.pipeline.execution_data import ExecutionData
    execution = ExecutionData(next(tmp_path.glob("results/executions/g/e/*/manifest.json")))
    assert len(execution.runs()) == 2
    for run in execution.runs():
        assert run["status"] == "completed"
        assert (tmp_path / run["checkpoints"] / "m" / "latest_model.zip").exists()
        events = execution.events(run)
        assert [event["training_step"] for event in events] == [0, 8]
        assert all(len(event["episode_outcomes"]) == 2 and event["deterministic_actions"] for event in events)
        assert [event["evaluation_seed"] for event in events] == [100, 101]
    assert all(point["n"] == 2 for point in execution.aggregate("m"))


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
