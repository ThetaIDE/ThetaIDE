"""In-process services the desktop app calls directly: the run catalog, config composition and metrics readers.

These cover what the FastAPI endpoints used to (GET /api/runs, POST /api/config/compose, ...), now as plain functions.
"""

import json

import pytest

from frontend.model import BASE_EXPERIMENT, METRIC_COLUMNS, Config, metric_points
from src.app.pipeline.compose import comparable_config, compose_experiment
from src.app.services import catalog, config_service
from src.app.services.metrics import MetricsTail, latest_metrics_csv, read_metrics_rows


@pytest.fixture(autouse=True)
def _register_test_agents():
    from src.app.core.agent_registry import AGENT_REGISTRY, register_agent
    from src.app.core.base_agent import OfflineAgentBase

    @register_agent("cql")
    class MockCQL(OfflineAgentBase):
        pass

    yield

    AGENT_REGISTRY.pop("cql", None)


# ── catalog ──────────────────────────────────────────────────────────────────


def test_environments_exclude_base_templates():
    names = [e["name"] for e in catalog.list_environments()]
    assert not any(name.startswith("_") for name in names)


def test_methods_list_registered_agents():
    methods = catalog.list_methods()
    assert isinstance(methods["registered_agents"], list)
    assert "method_styles" in methods


def test_experiments_exclude_base_templates():
    for name in (e["name"] for e in catalog.list_experiments()):
        assert not name.endswith("_base") and "/_" not in name


def write_run(root, group, experiment_id, *, env="cartpole", seed=7, versions=None):
    """A results/logs/<group>/<experiment_id> folder like the pipeline writes."""
    run_dir = root / group / experiment_id
    agent_dir = run_dir / "ppo"
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(
        f"env:\n  name: {env}\n  env_id: CartPole-v1\nseed: {seed}\ntotal_timesteps: 10000\nparadigm: online_rl\n"
        "methods:\n  ppo:\n    agent: ppo\n    model: dnn\n    lr: 0.0003\n    batch_size: 64\n    gamma: 0.99\n",
        encoding="utf-8")
    for number, (rewards, finished) in (versions or {0: ([10.0, 30.0, 20.0], True)}).items():
        version = agent_dir / f"version_{number}"
        version.mkdir(parents=True)
        runtime = {"start_time_iso": "2026-10-01T10:00:00+00:00", "training_time_seconds": 75.0}
        if finished:
            runtime["end_time_iso"] = "2026-10-01T10:01:15+00:00"
        (version / "runtime.json").write_text(json.dumps(runtime), encoding="utf-8")
        lines = ["epoch,eval/reward,transitions"]
        for i, reward in enumerate(rewards, start=1):
            lines.append(f",{reward},{i * 1000}.0")
            lines.append(f"{i},{reward + 1000},")  # Lightning's duplicate epoch row, without transitions
        (version / "metrics.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return run_dir


@pytest.fixture
def logs(tmp_path, monkeypatch):
    root = tmp_path / "src/usr/results" / "logs"
    root.mkdir(parents=True)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    return root


def test_list_runs_summarizes_each_run(logs):
    write_run(logs, "sweeps", "cli_run")
    (data,) = catalog.list_runs()
    assert (data["group"], data["experiment_id"], data["metadata"]) == ("sweeps", "cli_run", {})
    assert data["config"]["env"] == "cartpole" and data["config"]["seed"] == 7
    assert data["config"]["methods"] == {"ppo": {"agent": "ppo", "model": "dnn", "lr": 0.0003, "batch_size": 64,
                                                 "gamma": 0.99}}
    (agent,) = data["agents"]
    assert agent["name"] == "ppo" and agent["training_time_seconds"] == 75.0
    # Rows without a step count (Lightning's duplicate epoch rows) are ignored, as in the GUI's charts.
    assert (agent["best_reward"], agent["latest_reward"], agent["timesteps"]) == (30.0, 20.0, 3000)
    assert data["finished"] is True


def test_list_runs_unfinished_and_unreadable(logs):
    write_run(logs, "g", "unfinished", versions={0: ([5.0], False)})
    empty = logs / "g" / "empty"
    empty.mkdir(parents=True)
    (empty / "config.yaml").write_text(": not yaml : [", encoding="utf-8")
    runs = {r["experiment_id"]: r for r in catalog.list_runs()}
    assert runs["unfinished"]["finished"] is False and runs["unfinished"]["agents"][0]["finished"] is None
    assert runs["empty"] == {"group": "g", "experiment_id": "empty", "metadata": {}, "config": {}, "agents": [],
                             "finished": False}


def test_newest_version_is_chosen_by_number(logs):
    write_run(logs, "g", "multi", versions={9: ([1.0], True), 10: ([99.0], True)})
    (data,) = catalog.list_runs()
    assert data["agents"][0]["best_reward"] == 99.0  # version_10, not version_9
    metrics = catalog.run_metrics("g", "multi", "ppo")
    assert metrics["source"].replace("\\", "/").endswith("version_10/metrics.csv")


def test_run_metrics_missing_raises(logs):
    with pytest.raises(FileNotFoundError):
        catalog.run_metrics("nonexistent_group", "nonexistent_exp", "nonexistent_agent")


def test_no_logs_folder_means_no_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    assert catalog.list_runs() == []


# ── config composition ───────────────────────────────────────────────────────


def gui_compose(config):
    return compose_experiment(BASE_EXPERIMENT, config.overrides())


def test_compose_report_matches_direct_composition(fixture_workspace):
    config = Config(name="seed_study-7", method_name="ppo", seed=7, total_timesteps=25000, lr=1e-6, batch_size=256, gamma=0.9)
    body = config_service.compose_report(BASE_EXPERIMENT, config.overrides())
    assert body["valid"], body["errors"]
    assert body["argv"] == ["python", "run_pipeline.py", BASE_EXPERIMENT, *config.overrides()]
    expected = comparable_config(gui_compose(config).cfg)
    body["config"].pop("experiment_name")
    # The report names the selected site for the panel; composition leaves that to the selector.
    assert body["config"]["site"].pop("name") == "local/default"
    assert body["config"] == expected
    train_overrides = body["methods"]["ppo"]["train_overrides"]
    assert f"++agent.lr={config.lr}" in train_overrides
    assert f"++agent.batch_size={config.batch_size}" in train_overrides


def test_compose_report_flags_an_invalid_override():
    body = config_service.compose_report(BASE_EXPERIMENT, ["seed=="])
    assert not body["valid"] and body["errors"][0]["stage"] == "compose"


def test_compose_report_flags_a_paradigm_violation(fixture_workspace):
    body = config_service.compose_report(BASE_EXPERIMENT, Config().overrides() + ["++methods.ppo.agent=d3rlpy-dev/cql"])
    assert not body["valid"] and body["errors"][0]["stage"] == "validation"


def test_compose_report_flags_an_unknown_experiment():
    assert not config_service.compose_report("thetaide/does_not_exist")["valid"]


def test_schema_health_check_composes_a_real_workspace_experiment(fixture_workspace):
    schema = config_service.config_schema()
    assert schema["base_experiment"]
    assert schema["fixed_overrides"] == []
    assert schema["fields"] == []
    assert config_service.compose_report(schema["base_experiment"])["valid"]


def test_compose_report_does_not_infer_an_algorithm_specific_budget():
    def notices(total):
        body = config_service.compose_report(BASE_EXPERIMENT, Config(total_timesteps=total).overrides())
        return [n for n in body["notices"] if "rollouts" in n]

    assert notices(10000) == []
    assert notices(10240) == []


def test_prepare_launch_describes_the_job(fixture_workspace):
    plan = config_service.prepare_launch(BASE_EXPERIMENT, Config(name="pytest_plan_only", method_name="ppo", total_timesteps=3000).overrides())
    assert (plan["group"], plan["experiment_id"], plan["agents"]) == ("thetaide", "pytest_plan_only", ["ppo"])
    assert plan["effective_timesteps"] == 3000


def test_prepare_launch_refuses_a_clash_with_an_active_job(fixture_workspace):
    active = [{"job_id": "j1", "status": "running", "group": "thetaide", "experiment_id": "pytest_clash"}]
    with pytest.raises(config_service.LaunchError) as error:
        config_service.prepare_launch(BASE_EXPERIMENT, Config(name="pytest_clash", method_name="ppo").overrides(), active_jobs=active)
    assert error.value.conflict


# ── metrics readers ──────────────────────────────────────────────────────────


def test_metrics_reader_skips_partial_line_and_picks_newest_version(tmp_path):
    for version in (2, 10, 9):
        (tmp_path / f"version_{version}").mkdir()
        (tmp_path / f"version_{version}" / "metrics.csv").write_text("step\n", encoding="utf-8")
    newest = latest_metrics_csv(tmp_path)
    assert newest.parent.name == "version_10"
    newest.write_text("eval/reward,step,transitions\n20.5,0,0.0\n,1,512.0\n31.2,2,10", encoding="utf-8")
    rows = read_metrics_rows(newest)
    assert rows == [{"eval/reward": 20.5, "step": 0.0, "transitions": 0.0}, {"step": 1.0, "transitions": 512.0}]


def test_metrics_tail_reads_only_appended_complete_rows(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("step,eval/reward,loss\n0,10.0,0.5\n50,15.0,0.4\n", encoding="utf-8")
    tail = MetricsTail(csv_file)
    rows, reset = tail.read()
    assert rows == [{"step": 0.0, "eval/reward": 10.0, "loss": 0.5}, {"step": 50.0, "eval/reward": 15.0, "loss": 0.4}]
    assert not reset

    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("100,22.0,0.3\n")
    assert tail.read() == ([{"step": 100.0, "eval/reward": 22.0, "loss": 0.3}], False)

    with open(csv_file, "a", encoding="utf-8") as f:
        f.write("150,25.0")  # still being flushed
    assert tail.read() == ([], False)
    with open(csv_file, "a", encoding="utf-8") as f:
        f.write(",0.2\n")
    assert tail.read() == ([{"step": 150.0, "eval/reward": 25.0, "loss": 0.2}], False)


def test_metrics_tail_restarts_when_lightning_rewrites_the_file(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("step,loss\n0,0.5\n1,0.4\n2,0.3\n3,0.2\n", encoding="utf-8")
    tail = MetricsTail(csv_file)
    tail.read()
    csv_file.write_text("step,loss,entropy\n0,0.5,0.6\n", encoding="utf-8")  # new column: rewritten, shorter
    rows, reset = tail.read()
    assert reset and rows == [{"step": 0.0, "loss": 0.5, "entropy": 0.6}]


def test_metrics_tail_restarts_when_the_rewrite_is_longer(tmp_path):
    """The common case, and the one a size check misses.

    A supervised run logs step,val/loss during the validation sanity check and then gains
    train/loss once training starts, so the rewritten file is bigger than what it replaced.
    Keeping the old header zipped 4 columns onto 3 names: train/loss was filed as val/loss,
    the real val/loss was dropped, and the monitor displayed one under the other's label.
    """
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("epoch,step,val/loss\n0,12,0.43\n", encoding="utf-8")
    tail = MetricsTail(csv_file)
    assert tail.read() == ([{"epoch": 0.0, "step": 12.0, "val/loss": 0.43}], False)

    csv_file.write_text(
        "epoch,step,train/loss,val/loss\n0,12,0.57,0.43\n0,13,0.55,0.41\n", encoding="utf-8"
    )
    rows, reset = tail.read()
    assert reset, "a wider header is a rewrite even though the file grew"
    assert rows == [
        {"epoch": 0.0, "step": 12.0, "train/loss": 0.57, "val/loss": 0.43},
        {"epoch": 0.0, "step": 13.0, "train/loss": 0.55, "val/loss": 0.41},
    ]


def test_metrics_tail_keeps_streaming_after_a_rewrite(tmp_path):
    """A reset must not turn every later read into one, or the monitor would redraw the
    whole run on each append."""
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("epoch,step,val/loss\n0,12,0.43\n", encoding="utf-8")
    tail = MetricsTail(csv_file)
    tail.read()
    csv_file.write_text("epoch,step,train/loss,val/loss\n0,12,0.57,0.43\n", encoding="utf-8")
    tail.read()

    with open(csv_file, "a", encoding="utf-8") as handle:
        handle.write("0,13,0.55,0.41\n")
    rows, reset = tail.read()
    assert not reset
    assert rows == [{"epoch": 0.0, "step": 13.0, "train/loss": 0.55, "val/loss": 0.41}]


def test_metrics_tail_does_not_call_the_first_read_a_reset(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("step,loss\n0,0.5\n", encoding="utf-8")
    assert MetricsTail(csv_file).read() == ([{"step": 0.0, "loss": 0.5}], False)


def test_metrics_tail_waits_for_a_complete_header(tmp_path):
    csv_file = tmp_path / "metrics.csv"
    csv_file.write_text("step,lo", encoding="utf-8")  # still being written
    tail = MetricsTail(csv_file)
    assert tail.read() == ([], False)
    csv_file.write_text("step,loss\n0,0.5\n", encoding="utf-8")
    assert tail.read() == ([{"step": 0.0, "loss": 0.5}], False)


def test_metric_points_keep_evaluation_and_training_rows_separate():
    rows = [{"eval/reward": 20.0, "eval/reward_std": 8.5, "transitions": 0.0},
            {"losses/total_loss": 1.5, "losses/entropy": 0.69, "losses/approx_kl": 0.001, "transitions": 512.0},
            {"eval/reward": 20.0, "step": 0.0},  # epoch-level duplicate without transitions
            {"time/train": 0.3, "transitions": 600.0}]
    evaluation, training, timing = metric_points(rows)
    assert (evaluation["step"], evaluation["reward"], evaluation["reward_std"], evaluation["entropy"]) == (0, 20.0, 8.5, None)
    assert (training["step"], training["reward"], training["loss"], training["entropy"], training["approx_kl"]) == \
        (512, None, 1.5, 0.69, 0.001)
    assert set(evaluation) == set(training) == {"step", *METRIC_COLUMNS}
    # Columns outside METRIC_COLUMNS are kept under their own names, not dropped.
    assert (timing["step"], timing["time/train"], timing["reward"]) == (600, 0.3, None)


def test_metric_points_keep_metrics_from_any_engine():
    """SB3 (through ContractLogger) logs loss names the Lightning agents never use."""
    rows = [{"step": 1000.0, "transitions": 1000.0, "eval/reward": 15.0, "losses/train/value_loss": 3.5},
            {"transitions": 2000.0, "custom/success_rate": 0.4, "epoch": 3.0}]
    first, second = metric_points(rows)
    assert (first["reward"], first["losses/train/value_loss"]) == (15.0, 3.5)
    assert "step" in first and first["step"] == 1000  # the counter becomes the position, not a metric
    assert second["custom/success_rate"] == 0.4 and "epoch" not in second


# ── Results panel: versions, incremental metrics, manifests and run files ─────
# (ported from the FastAPI endpoints results-explorer added: GET .../metrics?version=&since=, .../manifest,
# .../files/{path})


def test_run_metrics_returns_every_column_and_supports_versions_and_since(logs):
    write_run(logs, "g", "multi", versions={0: ([1.0, 2.0], True), 1: ([5.0, 6.0, 7.0], True)})
    newest = catalog.run_metrics("g", "multi", "ppo")
    assert newest["version"] == "version_1" and newest["versions"] == ["version_0", "version_1"]
    assert newest["columns"] == ["epoch", "eval/reward", "transitions"] and newest["total_rows"] == 6
    older = catalog.run_metrics("g", "multi", "ppo", version="0")
    assert older["version"] == "version_0" and older["total_rows"] == 4
    # `since` returns only rows after the ones the caller already has.
    tail = catalog.run_metrics("g", "multi", "ppo", since=4)
    assert len(tail["metrics"]) == 2 and tail["metrics"][0]["eval/reward"] == "7.0"
    with pytest.raises(FileNotFoundError):
        catalog.run_metrics("g", "multi", "ppo", version="9")
    with pytest.raises(ValueError):
        catalog.run_metrics("g", "..", "ppo")


def test_following_a_run_reads_only_what_was_appended(logs, monkeypatch):
    """Polls (since=N-1, as the GUI's) continue from where the last read stopped; a half-written row waits."""
    run = write_run(logs, "g", "live", versions={0: ([1.0, 2.0], False)})
    metrics_csv = run / "ppo" / "version_0" / "metrics.csv"
    first = catalog.run_metrics("g", "live", "ppo")
    assert first["total_rows"] == 4

    parsed = []
    real_parse = catalog._parse_rows
    monkeypatch.setattr(catalog, "_parse_rows", lambda header, body: parsed.append(body) or real_parse(header, body))
    with open(metrics_csv, "a", encoding="utf-8") as f:
        f.write(",3.0,3000.0\n3,1003.0,\n,4.0,40")  # two complete rows, then one still being written
    poll = catalog.run_metrics("g", "live", "ppo", since=first["total_rows"] - 1)
    assert poll["total_rows"] == 6
    assert [row["eval/reward"] for row in poll["metrics"]] == ["1002.0", "3.0", "1003.0"]  # last held row, then new
    assert b"1.0" not in parsed[-1] and b"40" not in parsed[-1]  # neither old rows nor the partial one were parsed

    with open(metrics_csv, "a", encoding="utf-8") as f:
        f.write("00.0\n")  # the half-written row completes
    poll = catalog.run_metrics("g", "live", "ppo", since=poll["total_rows"] - 1)
    assert poll["total_rows"] == 7 and poll["metrics"][-1]["transitions"] == "4000.0"


def test_a_rewritten_metrics_file_is_read_whole(logs):
    """New columns make the logger rewrite metrics.csv with a new header: continuing the old read would
    file values under the wrong names, so that read starts over."""
    run = write_run(logs, "g", "grow", versions={0: ([1.0, 2.0], False)})
    metrics_csv = run / "ppo" / "version_0" / "metrics.csv"
    first = catalog.run_metrics("g", "grow", "ppo")
    metrics_csv.write_text("epoch,eval/reward,transitions,train/loss\n" + "".join(
        f",{r},{i}000.0,0.{i}\n" for i, r in enumerate([1.0, 2.0, 3.0, 4.0, 5.0], start=1)), encoding="utf-8")
    poll = catalog.run_metrics("g", "grow", "ppo", since=first["total_rows"] - 1)
    assert poll["columns"] == ["epoch", "eval/reward", "transitions", "train/loss"]
    assert poll["total_rows"] == 5
    assert [row["train/loss"] for row in poll["metrics"]] == ["0.4", "0.5"]  # rows 3 and 4 of the new file


def test_list_runs_rereads_a_changed_config_and_hands_out_copies(logs):
    run = write_run(logs, "g", "cfg")
    assert catalog.list_runs()[0]["config"]["seed"] == 7
    config = run / "config.yaml"
    config.write_text(config.read_text(encoding="utf-8").replace("seed: 7", "seed: 1234"), encoding="utf-8")
    listed = catalog.list_runs()[0]
    assert listed["config"]["seed"] == 1234
    listed["config"]["seed"] = -1
    listed["metadata"]["tampered"] = True
    again = catalog.list_runs()[0]
    assert again["config"]["seed"] == 1234 and "tampered" not in again["metadata"]


def new_session():
    """Forget the catalog's in-memory caches, as a freshly started app would have."""
    for cache in catalog._PERSISTED_CACHES.values():
        cache.clear()
    catalog._persisted_loaded_for = None


def test_summaries_are_kept_between_sessions(logs, monkeypatch):
    write_run(logs, "g", "a")
    write_run(logs, "g", "b")
    first = catalog.list_runs()
    assert catalog._summary_cache_file().is_file()

    new_session()
    monkeypatch.setattr(catalog, "_read_agent_summary", lambda *a: pytest.fail("re-parsed an unchanged metrics.csv"))
    monkeypatch.setattr(catalog, "_read_config_summary", lambda *a: pytest.fail("re-parsed an unchanged config.yaml"))
    key = lambda run: (run["group"], run["experiment_id"])
    assert sorted(catalog.list_runs(), key=key) == sorted(first, key=key)


def test_a_run_changed_or_deleted_between_sessions_is_noticed(logs):
    import shutil

    changed = write_run(logs, "g", "changed", versions={0: ([1.0, 2.0], True)})
    gone = write_run(logs, "g", "gone")
    catalog.list_runs()
    new_session()
    with open(changed / "ppo" / "version_0" / "metrics.csv", "a", encoding="utf-8") as f:
        f.write(",500.0,9000.0\n")
    shutil.rmtree(gone)
    runs = {run["experiment_id"]: run for run in catalog.list_runs()}
    assert runs["changed"]["agents"][0]["best_reward"] == 500.0
    assert "gone" not in runs
    saved = json.loads(catalog._summary_cache_file().read_text(encoding="utf-8"))
    assert not any("gone" in path for path in saved["agents"])


def test_an_unusable_cache_file_never_breaks_listing(logs, monkeypatch, tmp_path):
    write_run(logs, "g", "a")
    catalog._summary_cache_file().parent.mkdir(parents=True)
    catalog._summary_cache_file().write_text("{not json", encoding="utf-8")
    new_session()
    assert [run["experiment_id"] for run in catalog.list_runs()] == ["a"]

    blocker = tmp_path / "not_a_folder"
    blocker.write_text("", encoding="utf-8")
    monkeypatch.setattr(catalog, "_summary_cache_file", lambda **kwargs: blocker / "cache.json")  # can't be written
    new_session()
    assert [run["experiment_id"] for run in catalog.list_runs()] == ["a"]


def _write_artifacts(logs):
    """A run with plots, checkpoints and TensorBoard files next to its logs, plus another run's file."""
    results = logs.parent
    run = write_run(logs, "g", "exp")
    plots = results / "plots" / "g" / "exp"
    (plots / "losses" / "ppo").mkdir(parents=True)
    (plots / "losses" / "ppo" / "losses_value_loss.png").write_bytes(b"\x89PNG fake")
    (plots / "time_report.csv").write_bytes(b"Method,Avg\nppo,1.0\n")
    (plots / "hyperparameters_report.md").write_text("# Report\n", encoding="utf-8")
    ckpt = results / "checkpoints" / "g" / "exp" / "ppo" / "0"
    ckpt.mkdir(parents=True)
    (ckpt / "best_model.ckpt").write_bytes(b"weights")
    tb = results / "tensorboard" / "g" / "exp" / "ppo" / "version_0"
    tb.mkdir(parents=True)
    (tb / "events.out.tfevents.1.host.0").write_bytes(b"tb")
    other = results / "plots" / "g" / "other"
    other.mkdir(parents=True)
    (other / "secret.csv").write_text("x\n1\n", encoding="utf-8")
    return run


def test_manifest_lists_every_artifact_of_a_run(logs):
    _write_artifacts(logs)
    data = catalog.run_manifest("g", "exp")
    kinds = {a["path"]: a["kind"] for a in data["artifacts"]}
    assert kinds == {
        "logs/g/exp/config.yaml": "config",
        "logs/g/exp/ppo/version_0/metrics.csv": "metrics",
        "logs/g/exp/ppo/version_0/runtime.json": "metadata",
        "plots/g/exp/hyperparameters_report.md": "report",
        "plots/g/exp/losses/ppo/losses_value_loss.png": "figure",  # nested plots are found too
        "plots/g/exp/time_report.csv": "table",
        "checkpoints/g/exp/ppo/0/best_model.ckpt": "checkpoint",
        "tensorboard/g/exp/ppo/version_0/events.out.tfevents.1.host.0": "tensorboard",
    }
    assert data["agents"] == {"ppo": ["version_0"]} and data["truncated"] is False
    metrics = next(a for a in data["artifacts"] if a["kind"] == "metrics")
    assert (metrics["root"], metrics["agent"], metrics["version"]) == ("logs", "ppo", "version_0")
    assert metrics["size"] > 0 and metrics["modified"] > 0
    with pytest.raises(FileNotFoundError):
        catalog.run_manifest("g", "missing")


def test_run_files_are_read_only_from_that_run(logs):
    _write_artifacts(logs)
    results = logs.parent
    assert catalog.read_run_file("g", "exp", "plots/g/exp/time_report.csv") == b"Method,Avg\nppo,1.0\n"
    with pytest.raises(FileNotFoundError):
        catalog.read_run_file("g", "exp", "plots/g/exp/nope.csv")
    # Another run's file, paths climbing out of results/, absolute paths and bad segments are all refused.
    (results.parent / "secret.txt").write_text("top secret", encoding="utf-8")
    for path in ("plots/g/other/secret.csv", "logs/g/exp/../../../../secret.txt",
                 "plots/g/exp/../../../../secret.txt", str(results.parent / "secret.txt")):
        with pytest.raises(PermissionError):
            catalog.read_run_file("g", "exp", path)
    with pytest.raises(ValueError):
        catalog.run_manifest("g", "..\\exp")


def test_agent_summaries_are_cached_until_their_files_change(logs, monkeypatch):
    run = write_run(logs, "g", "cached")
    reads = []
    real = catalog._read_agent_summary
    monkeypatch.setattr(catalog, "_read_agent_summary", lambda *a: reads.append(a) or real(*a))
    catalog.list_runs()
    catalog.list_runs()
    assert len(reads) == 1  # the second listing came from the cache
    metrics = run / "ppo" / "version_0" / "metrics.csv"
    metrics.write_text(metrics.read_text(encoding="utf-8") + ",99.0,9000.0\n", encoding="utf-8")
    assert catalog.list_runs()[0]["agents"][0]["best_reward"] == 99.0 and len(reads) == 2

