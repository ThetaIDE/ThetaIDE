"""In-app job manager: run_pipeline.py as a QProcess, with output, metrics and status delivered as signals.

Most tests run a fake run_pipeline.py from a temporary project folder: it prints, writes a metrics row,
then waits for a release file so the test decides when each job finishes. Two slow tests train for real.
"""

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from frontend.job_manager import JobManager  # noqa: E402
from frontend.model import BASE_EXPERIMENT, Config, metric_points  # noqa: E402
from src.app.services import LOGS_DIR  # noqa: E402
from src.app.services.config_service import LaunchError  # noqa: E402
from src.app.services.job_store import JobStore  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture(autouse=True)
def isolated_job_config(fixture_workspace, monkeypatch):
    import src.app.pipeline.compose as compose_module
    monkeypatch.chdir(fixture_workspace)
    monkeypatch.setattr(compose_module, "CONFIG_DIR", fixture_workspace / "in" / "config")


@pytest.fixture(autouse=True)
def _register_test_agents():
    from src.usr.algorithms.agent_registry import AGENT_REGISTRY, register_agent
    from src.usr.algorithms.base_agent import OfflineAgentBase

    @register_agent("cql")
    class MockCQL(OfflineAgentBase):
        pass

    yield

    AGENT_REGISTRY.pop("cql", None)

FAKE_PIPELINE = '''
import os, sys, time
from pathlib import Path
exp_id = next(a.split("=", 1)[1].strip("'") for a in sys.argv[1:] if a.startswith("++experiment_id="))
version = Path(os.environ["FAKE_LOGS"]) / "thetaide" / exp_id / "ppo" / "version_0"
version.mkdir(parents=True, exist_ok=True)
print("started " + exp_id, flush=True)
(version / "metrics.csv").write_text("eval/reward,transitions\\n10.0,512\\n", encoding="utf-8")
print("Evaluation at 512", flush=True)
release = Path(os.environ["FAKE_CONTROL"]) / exp_id
while not release.exists():
    time.sleep(0.05)
code = release.read_text().strip()
print("done", flush=True)
sys.exit(int(code or 0))
'''


def pump(predicate, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met")


def settle(seconds=0.3):
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.02)


class Harness:
    """A JobManager running the fake pipeline, with the signals it emitted recorded."""

    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path / "project"
        self.root.mkdir()
        (self.root / "run_pipeline.py").write_text(FAKE_PIPELINE, encoding="utf-8")
        self.logs = tmp_path / "logs"
        self.control = tmp_path / "control"
        self.control.mkdir()
        monkeypatch.setenv("FAKE_LOGS", str(self.logs))
        monkeypatch.setenv("FAKE_CONTROL", str(self.control))
        # Allow tiny test scripts on a busy machine. The old count setting is
        # retained here to exercise compatibility; it no longer limits width.
        settings = tmp_path / 'settings.toml'
        settings.write_text('[training]\nmemory_percent = 95\nmax_parallel = 1\ncpu_percent=100\n')
        self.manager = JobManager(store_path=tmp_path / "jobs" / "jobs.db", project_root=self.root,
                                  logs_dir=self.logs, settings_path=settings)
        self.started, self.output, self.metrics, self.statuses = [], {}, {}, {}
        self.manager.job_changed.connect(self._changed)
        self.manager.job_output.connect(lambda job_id, lines: self.output.setdefault(job_id, []).extend(lines))
        self.manager.job_metrics.connect(lambda job_id, agent, rows, reset: self.metrics.setdefault(job_id, []).extend(rows))

    def _changed(self, job):
        if job["status"] == "running" and job["experiment_id"] not in self.started:
            self.started.append(job["experiment_id"])
        self.statuses[job["job_id"]] = job["status"]

    def launch(self, name, queue=False):
        # The workspace fixture supplies these profiles; the test no longer
        # relies on locally installed component configs being present.
        config = Config(name=name, method_name="ppo", agent="ppo", model="dnn")
        return self.manager.launch(BASE_EXPERIMENT, config.overrides(), queue=queue)

    def release(self, job, code=0):
        (self.control / job["experiment_id"]).write_text(str(code), encoding="utf-8")

    def status(self, job):
        return self.manager.job(job["job_id"])["status"]

    def close(self):
        for job in self.manager.active_jobs():
            self.release(job)
        self.manager.shutdown()


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    yield harness
    harness.close()


def tag():
    return uuid.uuid4().hex[:6]


# ── one job ──────────────────────────────────────────────────────────────────

def test_launch_streams_output_and_metrics_then_completes(jobs):
    job = jobs.launch(f"stream_{tag()}")
    assert (job["group"], job["agents"]) == ("thetaide", ["ppo"])
    pump(lambda: any("Evaluation at" in line for line in jobs.output.get(job["job_id"], [])))
    pump(lambda: jobs.metrics.get(job["job_id"]))
    assert jobs.metrics[job["job_id"]][0] == {"eval/reward": 10.0, "transitions": 512.0}
    assert jobs.status(job) == "running"
    jobs.release(job)
    pump(lambda: jobs.status(job) == "completed")
    final = jobs.manager.job(job["job_id"])
    assert final["returncode"] == 0 and final["finished"]
    assert "done" in jobs.output[job["job_id"]]


def test_output_is_also_written_to_the_job_log_file(jobs):
    job = jobs.launch(f"logfile_{tag()}")
    jobs.release(job)
    pump(lambda: jobs.status(job) == "completed")
    log = (jobs.manager.jobs_dir / f"{job['job_id']}.log").read_text(encoding="utf-8")
    assert "Evaluation at 512" in log


STREAM_PIPELINE = '''
import json, os, sys, time
from pathlib import Path
exp_id = next(a.split("=", 1)[1].strip("'") for a in sys.argv[1:] if a.startswith("++experiment_id="))
version = Path(os.environ["FAKE_LOGS"]) / "thetaide" / exp_id / "ppo" / "version_0"
version.mkdir(parents=True, exist_ok=True)
records = [
    {"v": 1, "t": time.time(), "type": "start", "run": {}, "config": {}},
    {"v": 1, "t": time.time(), "type": "metrics", "step": 1, "data": {"eval/reward": 10.0, "transitions": 512}},
    {"v": 1, "t": time.time(), "type": "system", "data": {"cpu_percent": 150.0, "rss_mb": 300.0, "children": 2}},
    {"v": 1, "t": time.time(), "type": "event", "kind": "checkpoint", "level": "info", "message": "saved", "data": {}},
]
with open(version / "events.jsonl", "w", encoding="utf-8") as f:
    f.write("".join(json.dumps(r) + "\\n" for r in records))
print("streamed", flush=True)
release = Path(os.environ["FAKE_CONTROL"]) / exp_id
while not release.exists():
    time.sleep(0.05)
sys.exit(0)
'''


def test_runs_with_a_telemetry_stream_report_every_record_and_their_own_resource_use(jobs):
    (jobs.root / "run_pipeline.py").write_text(STREAM_PIPELINE, encoding="utf-8")
    records, hardware = [], []
    jobs.manager.job_records.connect(lambda job_id, agent, new, reset: records.extend(new))
    jobs.manager.job_hardware.connect(lambda job_id, sample: hardware.append(sample))
    job = jobs.launch(f"streamed_{tag()}")
    pump(lambda: len(records) >= 4)
    assert [r["type"] for r in records] == ["start", "metrics", "system", "event"]
    assert jobs.metrics[job["job_id"]] == [{"eval/reward": 10.0, "transitions": 512.0, "step": 1.0}]
    # The training process measured itself (with its children): no outside sampling replaces it.
    assert hardware and hardware[-1]["cpu_percent"] == 150.0 and hardware[-1]["memory_mb"] == 300.0
    jobs.release(job)
    pump(lambda: jobs.status(job) == "completed")
    assert all(sample.get("children") == 2 for sample in hardware)


def test_nonzero_exit_marks_the_job_failed(jobs):
    job = jobs.launch(f"fails_{tag()}")
    jobs.release(job, code=3)
    pump(lambda: jobs.status(job) == "failed")
    assert jobs.manager.job(job["job_id"])["returncode"] == 3


def test_cancel_stops_a_running_job(jobs):
    job = jobs.launch(f"stopme_{tag()}")
    pump(lambda: jobs.status(job) == "running" and jobs.output.get(job["job_id"]))
    pid = jobs.manager.job(job["job_id"])["pid"]
    assert jobs.manager.cancel(job["job_id"])["status"] == "cancelled"
    pump(lambda: jobs.manager.job(job["job_id"]).get("finished"))
    assert jobs.status(job) == "cancelled"
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        assert str(pid) not in listing.stdout


def test_launch_rejects_an_invalid_config(jobs):
    with pytest.raises(LaunchError) as error:
        jobs.manager.launch(BASE_EXPERIMENT, Config().overrides() + ["++methods.ppo.agent=cql"])
    assert "forbidden" in str(error.value) or "not allowed" in str(error.value)
    assert not error.value.conflict


def test_launch_refuses_to_overwrite_existing_results(jobs):
    experiment_id = f"pytest_existing_{tag()}"
    log_dir = LOGS_DIR / "thetaide" / experiment_id
    log_dir.mkdir(parents=True)
    try:
        (log_dir / "config.yaml").write_text("previous run", encoding="utf-8")
        with pytest.raises(LaunchError) as error:
            jobs.launch(experiment_id)
        assert error.value.conflict
        assert (log_dir / "config.yaml").exists()
    finally:
        shutil.rmtree(log_dir, ignore_errors=True)


def test_jobs_left_running_by_a_previous_session_are_marked_interrupted(tmp_path):
    store = JobStore(tmp_path / "jobs.db")
    store.save_job({"job_id": "old", "status": "running", "pid": 999999, "group": "thetaide", "experiment_id": "x"})
    manager = JobManager(store_path=tmp_path / "jobs.db")
    assert manager.job("old")["status"] in ("failed", "interrupted")
    assert not manager.is_busy()


# ── queue ────────────────────────────────────────────────────────────────────

def test_queue_admits_jobs_in_order_with_moves_and_removals(jobs):
    t = tag()
    a, b, c = (jobs.launch(f"{n}_{t}", queue=True) for n in "abc")
    assert all(job["status"] == "queued" for job in (a, b, c))
    snapshot = jobs.manager.queue_state()
    assert not snapshot['active']
    assert [j['experiment_id'] for j in snapshot['queued']] == [f'{n}_{t}' for n in 'abc']

    with pytest.raises(LaunchError) as clash:  # same experiment ID as a queued job would overwrite its results
        jobs.launch(f"c_{t}", queue=True)
    assert clash.value.conflict

    assert jobs.manager.move(c["job_id"], 1) == 1
    assert jobs.manager.cancel(b["job_id"])["status"] == "cancelled"
    assert [j["experiment_id"] for j in jobs.manager.queue_state()["queued"]] == [f"a_{t}", f"c_{t}"]
    assert jobs.manager.start_queue()['running']
    pump(lambda: jobs.started == [f"a_{t}", f"c_{t}"])
    jobs.release(a)
    jobs.release(c)
    pump(lambda: not jobs.manager.queue_state()["active"])

    final = {j["experiment_id"]: j["status"] for j in jobs.manager.queue_state()["finished"]}
    assert final == {f"a_{t}": "completed", f"c_{t}": "completed", f"b_{t}": "cancelled"}
    assert jobs.started == [f"a_{t}", f"c_{t}"]  # the removed job never ran


def test_queued_job_runs_alongside_direct_job_when_resources_fit(jobs):
    t = tag()
    direct = jobs.launch(f"direct_{t}")
    pump(lambda: jobs.started == [f"direct_{t}"])
    queued = jobs.launch(f"queued_{t}", queue=True)
    jobs.manager.start_queue()
    pump(lambda: jobs.started == [f"direct_{t}", f"queued_{t}"])
    jobs.release(direct)
    jobs.release(queued)


def test_only_queued_jobs_can_be_moved(jobs):
    job = jobs.launch(f"solo_{tag()}", queue=True)
    jobs.manager.start_queue()
    pump(lambda: jobs.manager.queue_state()["active"])
    with pytest.raises(LaunchError):
        jobs.manager.move(job["job_id"], 0)
    jobs.release(job)


def test_queue_waits_for_start_pauses_on_request_and_pauses_itself_when_drained(jobs):
    t = tag()
    first, second = (jobs.launch(f"{n}_{t}", queue=True) for n in ("first", "second"))
    assert first["queue_running"] is False
    settle()
    assert jobs.started == [] and jobs.manager.queue_state()["running"] is False  # adding does not start

    jobs.manager.start_queue()
    pump(lambda: jobs.started == [f"first_{t}"])
    assert jobs.manager.pause_queue()["running"] is False
    jobs.release(first)
    pump(lambda: not jobs.manager.queue_state()["active"])
    settle()
    assert jobs.started == [f"first_{t}"]  # paused: the current job finished, the next did not start

    jobs.manager.start_queue()
    pump(lambda: jobs.started == [f"first_{t}", f"second_{t}"])
    jobs.release(second)
    pump(lambda: jobs.manager.queue_state()["running"] is False)  # drained, so it paused itself

    late = jobs.launch(f"late_{t}", queue=True)
    settle()
    assert jobs.started == [f"first_{t}", f"second_{t}"]  # a job added after draining waits for Start
    assert jobs.manager.start_queue()["running"]
    pump(lambda: jobs.started[-1] == f"late_{t}")
    jobs.release(late)


def test_starting_an_empty_queue_leaves_it_paused(jobs):
    assert jobs.manager.start_queue() == {"running": False, "queued": 0}


def test_queue_changes_are_signalled(jobs):
    seen = []
    jobs.manager.queue_changed.connect(seen.append)
    job = jobs.launch(f"signal_{tag()}", queue=True)
    assert seen and seen[-1]["queued"][0]["job_id"] == job["job_id"]


# ── real training (slow) ─────────────────────────────────────────────────────

@pytest.fixture
def experiment_ids(project_root):
    """Unique experiment IDs whose real results are removed after the test."""
    created = []

    def make(prefix="pytest_job"):
        experiment_id = f"{prefix}_{uuid.uuid4().hex[:8]}"
        created.append(experiment_id)
        return experiment_id

    yield make
    for experiment_id in created:
        for kind in ("logs", "checkpoints", "plots", "tensorboard"):
            shutil.rmtree(project_root / "results" / kind / "thetaide" / experiment_id, ignore_errors=True)


def real_overrides(experiment_id, total_timesteps):
    return Config(name=experiment_id, total_timesteps=total_timesteps).overrides() + ["no_plot=true"]


def _has_real_training_plugins():
    has_cartpole = Path("in/config/env/cartpole.yaml").exists() or Path("in/envs/cartpole").exists()
    has_ppo = Path("in/config/agent/ppo.yaml").exists() or Path("in/config/agent/sb3/ppo.yaml").exists()
    return has_cartpole and has_ppo


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("THETA_RUN_REAL_TRAINING_TESTS") != "1",
    reason="real training is an opt-in integration test; CI uses deterministic fake-pipeline coverage",
)
def test_real_training_streams_logs_and_metrics_until_completed(tmp_path, experiment_ids):
    if not _has_real_training_plugins():
        pytest.skip("Real training tests require cartpole and ppo component plugins installed")
    manager = JobManager(store_path=tmp_path / "jobs.db")
    output, rows, records = [], [], []
    manager.job_output.connect(lambda job_id, lines: output.extend(lines))
    manager.job_metrics.connect(lambda job_id, agent, new, reset: rows.extend(new))
    manager.job_records.connect(lambda job_id, agent, new, reset: records.extend(new))
    experiment_id = experiment_ids()
    job = manager.launch(BASE_EXPERIMENT, real_overrides(experiment_id, 3000))
    assert (job["group"], job["experiment_id"], job["agents"]) == ("thetaide", experiment_id, ["ppo"])
    assert job["effective_timesteps"] == 3000

    pump(lambda: manager.job(job["job_id"])["status"] not in ("pending", "running"), timeout=240)
    final = manager.job(job["job_id"])
    assert final["status"] == "completed", "\n".join(output[-40:])
    assert any("Evaluation at" in line for line in output)

    points = metric_points(rows)
    assert [p["step"] for p in points if p["reward"] is not None][-1] >= 3000
    assert any(p["loss"] is not None for p in points)
    assert max(p["step"] for p in points) >= job["effective_timesteps"]
    # The live stream: started, sampled its own resource use, and finished with the run.
    kinds = [r["type"] for r in records]
    assert kinds[0] == "start" and "system" in kinds and records[-1]["type"] == "finish"
    assert records[-1]["status"] == "completed"


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("THETA_RUN_REAL_TRAINING_TESTS") != "1",
    reason="real training is an opt-in integration test; CI uses deterministic fake-pipeline coverage",
)
def test_real_cancel_stops_pipeline_and_training_subprocesses(tmp_path, experiment_ids):
    if not _has_real_training_plugins():
        pytest.skip("Real training tests require cartpole and ppo component plugins installed")
    manager = JobManager(store_path=tmp_path / "jobs.db")
    output, rows = [], []
    manager.job_output.connect(lambda job_id, lines: output.extend(lines))
    manager.job_metrics.connect(lambda job_id, agent, new, reset: rows.extend(new))
    experiment_id = experiment_ids()
    job = manager.launch(BASE_EXPERIMENT, real_overrides(experiment_id, 2000000))
    # Wait for the first metrics (from the live stream, which can arrive before Lightning flushes metrics.csv)
    pump(lambda: rows, timeout=240)

    assert manager.cancel(job["job_id"])["status"] == "cancelled"
    pump(lambda: manager.job(job["job_id"]).get("finished"), timeout=30)
    assert manager.job(job["job_id"])["status"] == "cancelled"

    # train.py is a grandchild of the job; if it survived, its telemetry stream would keep growing
    # (its system sampler alone appends a record every two seconds).
    stream = next((LOGS_DIR / "thetaide" / experiment_id / "ppo").glob("version_*/events.jsonl"))
    size = stream.stat().st_size
    time.sleep(4)
    assert stream.stat().st_size == size
    pid = manager.job(job["job_id"])["pid"]
    if os.name == "nt":
        listing = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True)
        assert str(pid) not in listing.stdout
    assert Path(stream).exists()


def test_failed_run_saves_separate_logs_in_results(jobs, monkeypatch):
    monkeypatch.setattr("frontend.job_manager.prepare_launch", lambda experiment, overrides, **kwargs: {
        "experiment": experiment, "group": "thetaide", "experiment_id": "early_failure",
        "agents": ["ppo"], "total_timesteps": 1, "effective_timesteps": 1, "run_in_parallel": False,
    })
    pipeline = jobs.root / "run_pipeline.py"
    pipeline.write_text("import sys\nprint('output text', flush=True)\nprint('error text', file=sys.stderr, flush=True)\nsys.exit(2)\n", encoding="utf-8")
    job = jobs.launch(f"failure_{tag()}")
    pump(lambda: jobs.status(job) == "failed")
    run_dir = jobs.logs / job["group"] / job["experiment_id"]
    assert (run_dir / f"{job['job_id']}.stdout.log").read_text() == "output text\n"
    assert (run_dir / f"{job['job_id']}.stderr.log").read_text() == "error text\n"
    pump(lambda: {"output text", "error text"} <= set(jobs.output.get(job["job_id"], [])))


def test_detached_runner_continues_queue_and_replays_metrics_after_reconnect(jobs):
    first = jobs.launch(f'detached_a_{tag()}', queue=True)
    second = jobs.launch(f'detached_b_{tag()}', queue=True)
    jobs.manager.start_queue()
    pump(lambda: jobs.output.get(first['job_id']))
    old = jobs.manager
    old.shutdown()
    jobs.release(first)
    # No Qt event processing: the desktop client is disconnected.
    deadline = time.time() + 20
    while time.time() < deadline:
        record = JobStore(old.jobs_dir / 'jobs.db').get_job(second['job_id'])
        if record['status'] == 'running':
            break
        time.sleep(.05)
    else:
        pytest.fail('The detached queue did not start its next job')
    manager = JobManager(store_path=old.jobs_dir / 'jobs.db', project_root=jobs.root, logs_dir=jobs.logs)
    jobs.manager = manager
    rows = []
    manager.job_metrics.connect(lambda job_id, agent, new, reset: rows.extend(new) if job_id == second['job_id'] else None)
    pump(lambda: rows)
    assert manager.job(first['job_id'])['status'] == 'completed'
    assert rows == [{'eval/reward': 10.0, 'transitions': 512.0}]
    jobs.release(second)
    pump(lambda: manager.job(second['job_id'])['status'] == 'completed')


def test_completion_while_disconnected_preserves_exit_code_and_output(jobs):
    job = jobs.launch(f'offline_failure_{tag()}')
    pump(lambda: jobs.output.get(job['job_id']))
    old = jobs.manager
    old.shutdown()
    jobs.release(job, code=7)
    deadline = time.time() + 20
    store = JobStore(old.jobs_dir / 'jobs.db')
    while time.time() < deadline and store.get_job(job['job_id'])['status'] == 'running':
        time.sleep(.05)
    manager = JobManager(store_path=old.jobs_dir / 'jobs.db', project_root=jobs.root, logs_dir=jobs.logs)
    jobs.manager = manager
    assert manager.job(job['job_id'])['returncode'] == 7
    assert manager.job(job['job_id'])['status'] == 'failed'
    pump(lambda: 'done' in manager.log_lines(job['job_id']))


def test_cancel_after_reconnect_stops_the_original_process(jobs):
    job = jobs.launch(f'reconnected_cancel_{tag()}')
    pump(lambda: jobs.output.get(job['job_id']))
    old = jobs.manager
    old.shutdown()
    manager = JobManager(store_path=old.jobs_dir / 'jobs.db', project_root=jobs.root, logs_dir=jobs.logs)
    jobs.manager = manager
    assert manager.job(job['job_id'])['pid'] == job['pid']
    assert manager.cancel(job['job_id'])['status'] == 'cancelled'
    pump(lambda: manager.job(job['job_id']).get('finished'))
