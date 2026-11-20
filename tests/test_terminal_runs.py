"""Terminal pipelines announce themselves and an open desktop's runner follows them as jobs."""
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

import src.app.pipeline.compose as compose  # noqa: E402
import src.app.services.config_service as config_service  # noqa: E402
from frontend.runner_manager import JobManager  # noqa: E402
from src.app.services import terminal_runs  # noqa: E402
from src.app.services.terminal_runs import announced, announcement_dir  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

DESCRIPTION = {"experiment": "Group/Exp", "group": "Group", "experiment_id": "exp", "agents": ["ppo"],
               "total_timesteps": 512, "effective_timesteps": 512, "run_in_parallel": False,
               "local": True, "accelerator": False, "resource_estimate": None}


@pytest.fixture
def composed(tmp_path, monkeypatch):
    monkeypatch.delenv(terminal_runs.RUNNER_JOB_VARIABLE, raising=False)
    monkeypatch.delenv("THETA_CANCEL_DIR", raising=False)
    monkeypatch.setattr(compose, "method_plans", lambda composed: {})
    monkeypatch.setattr(config_service, "describe_launch", lambda composed, plans: dict(DESCRIPTION))
    return SimpleNamespace(workspace=SimpleNamespace(jobs=tmp_path / "data/runs/jobs"), extra_args=["seed=1"])


def only_announcement(composed):
    paths = list(announcement_dir(composed.workspace.jobs).glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


def test_announcement_records_the_run_its_output_and_its_exit(composed):
    with announced(composed, enabled=True) as job_id:
        record = only_announcement(composed)
        assert record["job_id"] == job_id and record["pid"] == os.getpid()
        assert record["overrides"] == ["seed=1"] and "returncode" not in record
        assert os.environ["THETA_CANCEL_DIR"].endswith(f"cancellations/{job_id}")
        print("training step 1")
    assert only_announcement(composed)["returncode"] == 0
    log = composed.workspace.jobs / f"{job_id}.log"
    assert "training step 1" in log.read_text()


@pytest.mark.parametrize("raised, code, interrupted", [
    (SystemExit(3), 3, False), (SystemExit(None), 0, False), (KeyboardInterrupt(), 130, True),
    (RuntimeError("boom"), 1, False)])
def test_announcement_records_how_the_run_ended(composed, raised, code, interrupted):
    with pytest.raises(type(raised)):
        with announced(composed, enabled=True):
            raise raised
    record = only_announcement(composed)
    assert record["returncode"] == code
    assert bool(record.get("interrupted")) is interrupted
    assert sys.stdout is sys.__stdout__ or not isinstance(sys.stdout, terminal_runs._Tee)


def test_desktop_launches_nested_pipelines_and_allocations_do_not_announce(composed, monkeypatch):
    with announced(composed, enabled=False) as job_id:
        assert job_id is None
    monkeypatch.setenv(terminal_runs.RUNNER_JOB_VARIABLE, "desktop-job")
    with announced(composed, enabled=True) as job_id:
        assert job_id is None
    assert not announcement_dir(composed.workspace.jobs).exists()


# ── The runner follows announcements ─────────────────────────────────────────

class Terminal:
    """A stand-in terminal pipeline: a live process, its announcement and its metrics."""

    def __init__(self, tmp_path, name="terminal-a"):
        import psutil
        self.process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.jobs_dir = tmp_path / "jobs"
        self.path = announcement_dir(self.jobs_dir) / f"{name}.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.record = {**DESCRIPTION, "job_id": name, "overrides": [], "pid": self.process.pid,
                       "process_started": psutil.Process(self.process.pid).create_time(), "started": time.time()}
        self.write()
        version = tmp_path / "logs/Group/exp/ppo/version_0"
        version.mkdir(parents=True)
        (version / "metrics.csv").write_text("eval/reward,transitions\n10.0,512\n", encoding="utf-8")

    def write(self, **outcome):
        self.path.write_text(json.dumps({**self.record, **outcome}), encoding="utf-8")

    def exit(self, **outcome):
        if outcome:
            self.write(**outcome)
        self.process.kill()
        self.process.wait()


def runner(tmp_path):
    return JobManager(store_path=tmp_path / "jobs/jobs.db", project_root=tmp_path, logs_dir=tmp_path / "logs")


def settle(manager, predicate, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        manager.adopt_terminal_runs()
        manager._on_path_changed("")
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met")


def test_runner_adopts_a_terminal_run_streams_its_metrics_and_settles_its_exit(tmp_path):
    terminal = Terminal(tmp_path)
    manager = runner(tmp_path)
    rows = []
    manager.job_metrics.connect(lambda job_id, agent, new, reset: rows.extend(new))
    try:
        settle(manager, lambda: rows)
        job = manager.job("terminal-a")
        assert job["status"] == "running" and job["request"]["source"] == "terminal"
        assert rows == [{"eval/reward": 10.0, "transitions": 512.0}]
        assert manager.running_count() == 1
        terminal.exit(returncode=0, finished=time.time())
        settle(manager, lambda: manager.job("terminal-a")["status"] == "completed")
        assert manager.job("terminal-a")["returncode"] == 0
        assert not terminal.path.exists()
    finally:
        terminal.exit()


def test_a_terminal_run_that_vanishes_without_an_exit_is_an_error(tmp_path):
    terminal = Terminal(tmp_path)
    manager = runner(tmp_path)
    try:
        settle(manager, lambda: manager.job("terminal-a"))
        terminal.exit()
        settle(manager, lambda: manager.job("terminal-a")["status"] == "error")
        assert "without recording an exit code" in manager.job("terminal-a")["error"]
    finally:
        terminal.exit()


def test_ctrl_c_in_the_terminal_is_a_cancellation(tmp_path):
    terminal = Terminal(tmp_path)
    manager = runner(tmp_path)
    try:
        settle(manager, lambda: manager.job("terminal-a"))
        terminal.exit(returncode=130, interrupted=True)
        settle(manager, lambda: manager.job("terminal-a")["status"] == "cancelled")
    finally:
        terminal.exit()


def test_desktop_cancel_stops_the_terminal_process(tmp_path):
    terminal = Terminal(tmp_path)
    manager = runner(tmp_path)
    try:
        settle(manager, lambda: manager.job("terminal-a"))
        assert manager.cancel("terminal-a")["status"] == "cancelled"
        terminal.process.wait(timeout=10)
        settle(manager, lambda: manager.job("terminal-a").get("finished"))
        assert manager.job("terminal-a")["status"] == "cancelled"
        assert not terminal.path.exists()
    finally:
        terminal.exit()


def test_a_restarted_runner_keeps_following_an_unsettled_terminal_run(tmp_path):
    terminal = Terminal(tmp_path)
    try:
        first = runner(tmp_path)
        settle(first, lambda: first.job("terminal-a"))
        del first
        second = runner(tmp_path)
        assert second.job("terminal-a")["status"] == "running"
        terminal.exit(returncode=7, finished=time.time())
        settle(second, lambda: second.job("terminal-a")["status"] == "failed")
        assert second.job("terminal-a")["returncode"] == 7
    finally:
        terminal.exit()


def test_a_run_that_finished_while_no_desktop_was_open_is_adopted_as_finished(tmp_path):
    terminal = Terminal(tmp_path)
    terminal.exit(returncode=0, finished=time.time())
    manager = runner(tmp_path)
    manager.adopt_terminal_runs()
    assert manager.job("terminal-a")["status"] == "completed"
    assert not terminal.path.exists()


def test_desktop_launches_tell_their_pipeline_not_to_announce():
    import inspect
    assert "RUNNER_JOB_VARIABLE" in inspect.getsource(JobManager._start)


def test_desktop_runner_and_terminal_share_one_job_folder(tmp_path):
    from src.app.workspace import WorkspacePaths
    assert WorkspacePaths(tmp_path).jobs == (tmp_path / ".thetaide/runs/jobs").resolve()
    assert WorkspacePaths(tmp_path, tmp_path / "state").jobs == (tmp_path / "state/runs/jobs").resolve()


def test_macos_process_identity_survives_clock_adjustment(monkeypatch):
    import psutil
    kernel = SimpleNamespace(create_time=lambda *, monotonic: 100.0 if monotonic else 500.0)
    process = SimpleNamespace(_proc=kernel, status=lambda: "running", create_time=lambda: 500.0)
    monkeypatch.setattr(terminal_runs.sys, "platform", "darwin")
    monkeypatch.setattr(psutil, "Process", lambda pid: process)
    assert terminal_runs._process_started(123) == 100.0
    assert terminal_runs.process_alive(123, 100.0) is True
    assert terminal_runs.process_alive(123, 99.0) is False  # reused PID


def test_denied_process_inspection_does_not_settle_live_terminal_job(tmp_path, monkeypatch):
    import psutil
    terminal = Terminal(tmp_path)
    manager = runner(tmp_path)
    try:
        settle(manager, lambda: manager.job("terminal-a"))
        def denied(pid):
            raise psutil.AccessDenied(pid)
        monkeypatch.setattr(psutil, "Process", denied)
        assert terminal_runs.process_alive(terminal.process.pid, terminal.record["process_started"]) is None
        manager.adopt_terminal_runs()
        assert manager.job("terminal-a")["status"] == "running"
        assert terminal.path.exists()
    finally:
        terminal.exit()
