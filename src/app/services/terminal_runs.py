"""Pipelines started from a terminal announce themselves so an open ThetaIDE follows them.

``run_pipeline.py`` writes ``<jobs>/terminal/<job_id>.json`` when a local run starts,
tees its console output to the job log, and records its exit in the same file. The desktop's
training runner adopts each announcement as a job, so monitors, queue cards, logs and
cancellation work as they do for runs the desktop launched. Announcements the runner has not
settled yet stay on disk, so a desktop opened later still adopts runs that started or finished
while it was closed.
"""
from __future__ import annotations

import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

RUNNER_JOB_VARIABLE = "THETA_RUNNER_JOB"  # set by the training runner on the pipelines it starts
SOURCE = "terminal"


def announcement_dir(directory: Path) -> Path:
    return Path(directory) / "terminal"


def is_terminal(job) -> bool:
    return (job.get("request") or {}).get("source") == SOURCE


def _write(path: Path, record: dict):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record), encoding="utf-8")
    temporary.replace(path)


def _process_birth_time(process):
    # psutil on macOS adjusts public create_time() after clock changes, relative
    # to when each interpreter imported psutil. A long-lived tracker and a new
    # pipeline can therefore disagree about the same PID after sleep/wake.
    # Use the kernel birth time, retaining PID-reuse protection without that
    # interpreter-local adjustment. Other platforms use the public API.
    if sys.platform == "darwin":
        return process._proc.create_time(monotonic=True)
    return process.create_time()


def _process_started(pid: int):
    import psutil
    try:
        return _process_birth_time(psutil.Process(pid))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def process_alive(pid, started) -> bool | None:
    """Process identity, or None if inspection is denied (not evidence of exit)."""
    import psutil
    try:
        process = psutil.Process(int(pid))
        return (process.status() != psutil.STATUS_ZOMBIE
                and (started is None or abs(_process_birth_time(process) - float(started)) < 1e-2))
    except psutil.AccessDenied:
        return None
    except (psutil.NoSuchProcess, TypeError, ValueError):
        return False


def announcements(directory: Path):
    """``(path, record)`` for each announcement in a runner job directory."""
    for path in sorted(announcement_dir(directory).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # being replaced; read on the next pass
        if isinstance(record, dict) and record.get("job_id") == path.stem:
            yield path, record


def read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class _Tee:
    """A console stream that also appends to the job log the desktop reads."""

    def __init__(self, stream, log):
        self._stream, self._log = stream, log

    def write(self, text):
        written = self._stream.write(text)
        try:
            self._log.write(text)
            self._log.flush()
        except (OSError, ValueError):
            pass  # the console stays authoritative
        return written

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _exit_code(code) -> int:
    if code is None:
        return 0
    return code if isinstance(code, int) else 1


@contextmanager
def announced(composed, *, enabled: bool):
    """Announce this pipeline for the duration of the block and record how it ended.

    Runs the desktop launched are already its jobs; ``enabled`` is false for nested pipelines
    and scheduler allocations.
    """
    if not enabled or os.environ.get(RUNNER_JOB_VARIABLE):
        yield None
        return
    from src.app.pipeline.compose import method_plans
    from src.app.services.config_service import describe_launch

    description = describe_launch(composed, method_plans(composed))
    root = composed.workspace.jobs
    directory = announcement_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    job_id = f"{SOURCE}-{uuid.uuid4().hex}"
    path = directory / f"{job_id}.json"
    record = {**description, "job_id": job_id, "overrides": list(composed.extra_args),
              "pid": os.getpid(), "process_started": _process_started(os.getpid()), "started": time.time()}
    _write(path, record)
    # Method and replicate cancellation from the desktop reaches this run as it does a desktop launch.
    os.environ["THETA_CANCEL_DIR"] = str((root / "cancellations" / job_id).resolve())
    log = open(root / f"{job_id}.log", "w", encoding="utf-8")
    stdout, stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = _Tee(stdout, log), _Tee(stderr, log)
    outcome = {"returncode": 1}
    try:
        yield job_id
        outcome = {"returncode": 0}
    except SystemExit as exc:
        outcome = {"returncode": _exit_code(exc.code)}
        raise
    except KeyboardInterrupt:
        outcome = {"returncode": 130, "interrupted": True}
        raise
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        log.close()
        _write(path, {**record, **outcome, "finished": time.time()})
