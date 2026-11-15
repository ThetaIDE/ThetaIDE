"""Results browser data logic (frontend/model.py): rows, sort keys, filters and command-line run records."""
from frontend.model import (RESULT_COLUMNS, Config, disk_run, disk_run_id, format_duration,
                            matches_filters, metric_points_from_api, new_run, result_row, run_source)


def entry(**overrides):
    """One GET /api/runs item."""
    data = {
        "group": "sweeps", "experiment_id": "cli_run", "metadata": {}, "finished": True,
        "config": {"env": "cartpole", "seed": 7, "total_timesteps": 10000,
                   "methods": {"ppo": {"agent": "ppo", "model": "dnn", "lr": 0.001, "batch_size": 128, "gamma": 0.95}}},
        "agents": [{"name": "ppo", "started": "2026-10-01T10:00:00+00:00", "finished": "2026-10-01T10:01:15+00:00",
                    "training_time_seconds": 75.0, "timesteps": 10240, "best_reward": 120.0, "latest_reward": 110.0}],
    }
    data.update(overrides)
    return data


def cells(run, disk=None):
    return dict(zip(RESULT_COLUMNS, result_row(run, disk)))


def test_disk_run_record():
    run = disk_run(entry())
    assert run["id"] == "cli-sweeps-cli_run" and run["origin"] == "cli" and run["transient"]
    assert run["status"] == "completed" and run["metrics"] == [] and not run["simulated"]
    assert run["config"] == {"name": "cli_run", "env": "cartpole", "method_name": "ppo", "agent": "ppo", "model": "dnn",
                             "seed": 7, "total_timesteps": 10000, "lr": 0.001, "batch_size": 128,
                             "gamma": 0.95}
    assert Config(**run["config"])  # loadable by the run store and the experiment builder
    assert run["backend"] == {"group": "sweeps", "experiment_id": "cli_run", "agents": ["ppo"], "total_timesteps": 10000}
    assert run["builder_compatible"]


def test_disk_run_unfinished_unusual_and_unsafe_names():
    run = disk_run(entry(finished=False, experiment_id="a b/c:d",
                         config={"env": "minigrid", "seed": "${oc.env:SEED}", "methods": {}}))
    assert run["status"] == "incomplete"
    assert run["id"] == "cli-sweeps-a_b_c_d"  # usable as a folder name
    assert run["config"]["seed"] == Config().seed  # unresolved interpolation falls back to the default
    assert not run["builder_compatible"]
    assert disk_run_id("g", "x") == "cli-g-x"


def test_row_for_command_line_run_before_and_after_metrics_load():
    disk = entry()
    run = disk_run(disk)
    before = cells(run, disk)
    assert before["Method"][0] == "ppo / dnn" and before["Environment"][0] == "cartpole" and before["Seed"] == ("7", 7)
    assert before["Timesteps"] == ("10,240", 10240) and before["Duration"] == ("1 min 15 s", 75.0)
    assert before["Best reward"] == ("120.0", 120.0) and before["Latest reward"] == ("110.0", 110.0)
    assert before["Source"][0] == "Command line" and before["Started"][1] is not None
    run["metrics"] = [{"step": 5120, "reward": 120.0}, {"step": 10240, "reward": 110.0}]
    assert cells(run, disk) == before  # loading the curve does not change the row


def test_row_for_app_and_queued_runs():
    trained = new_run(Config(name="mine"), simulated=False)
    trained.update(status="completed", backend={"group": "thetaide", "experiment_id": "mine_20261002-104250"},
                   metrics=[{"step": 100, "reward": 5.0}, {"step": 200, "reward": 9.0}, {"step": 300, "reward": 7.0}])
    row = cells(trained)
    assert row["Experiment"][0] == "mine_20261002-104250" and row["Source"][0] == "Trained (app)"
    assert (row["Best reward"][0], row["Latest reward"][0], row["Timesteps"][0]) == ("9.0", "7.0", "300")
    assert row["Duration"] == ("—", None)  # known only once the backend lists its folder
    assert cells(trained, entry())["Duration"][0] == "1 min 15 s"
    queued = new_run(Config(), simulated=False)
    queued.update(status="queued", backend={"group": "g", "experiment_id": "q"})
    assert cells(queued)["Best reward"] == ("—", None) and cells(queued)["Timesteps"] == ("—", None)


def test_format_duration():
    assert [format_duration(s) for s in (None, 14.76, 75, 3725)] == ["—", "14.8 s", "1 min 15 s", "1 h 02 min"]


def test_filters():
    cli = disk_run(entry())
    unfinished = disk_run(entry(finished=False))
    app_run = new_run(Config(), simulated=False)
    app_run.update(status="running", backend={"group": "g", "experiment_id": "e"})
    assert [matches_filters(r, source="real") for r in (cli, app_run)] == [True, True]
    assert [matches_filters(r, source="cli") for r in (cli, app_run)] == [True, False]
    assert not matches_filters(cli, source="simulated")
    assert [matches_filters(r, status="stopped") for r in (cli, unfinished)] == [False, True]
    assert matches_filters(app_run, status="active") and not matches_filters(app_run, status="completed")


def test_metric_points_from_api_reads_text_values():
    rows = [{"eval/reward": "12.5", "transitions": "1000.0", "epoch": ""},
            {"eval/reward": "13.0", "transitions": "", "epoch": "4"}]  # no step count: not plottable
    assert metric_points_from_api(rows) == [
        {"step": 1000, "reward": 12.5, "reward_std": None, "loss": None, "policy_loss": None, "value_loss": None,
         "entropy": None, "approx_kl": None}]
