"""Detached runners must retire even when their workspace or IPC writes fail."""
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip('PyQt6')
from frontend.job_runner import RunnerLifecycle


@pytest.fixture
def lifecycle(tmp_path):
    return RunnerLifecycle(tmp_path, SimpleNamespace(is_busy=lambda: False), Mock())


def test_deleted_workspace_exits_without_trying_to_update(lifecycle):
    shutil.rmtree(lifecycle.directory)
    tick = Mock()
    lifecycle.update(tick)
    lifecycle.app.quit.assert_called_once()
    tick.assert_not_called()


def test_idle_shutdown_survives_repeated_update_failures(lifecycle, monkeypatch, caplog):
    clock = [lifecycle.last_request]
    monkeypatch.setattr('frontend.job_runner.time.monotonic', lambda: clock[0])
    tick = Mock(side_effect=OSError('snapshot write failed'))
    with caplog.at_level(logging.ERROR):
        for offset in range(61):
            clock[0] = lifecycle.last_request + offset
            lifecycle.update(tick)
        assert len(caplog.records) == 2
    lifecycle.app.quit.assert_not_called()
    clock[0] += 1
    lifecycle.update(tick)
    lifecycle.app.quit.assert_called_once()
    assert tick.call_count == 61


@pytest.mark.parametrize('busy, connected', [(True, False), (False, True)])
def test_running_jobs_and_connected_clients_keep_runner_alive(lifecycle, busy, connected):
    lifecycle.manager.is_busy = lambda: busy
    lifecycle.last_request -= 120
    if connected:
        (lifecycle.directory / 'desktop.client').touch()
    tick = Mock()
    lifecycle.update(tick)
    lifecycle.app.quit.assert_not_called()
    tick.assert_called_once()


def test_stale_client_does_not_prevent_idle_shutdown(lifecycle):
    heartbeat = lifecycle.directory / 'desktop.client'
    heartbeat.touch()
    stale = time.time() - 120
    os.utime(heartbeat, (stale, stale))
    lifecycle.last_request -= 120
    lifecycle.update(Mock())
    lifecycle.app.quit.assert_called_once()


def test_real_runner_exits_when_temporary_workspace_is_removed(tmp_path):
    directory = tmp_path / 'jobs' / 'runner'
    directory.mkdir(parents=True)
    config = directory / 'startup.config'
    config.write_text(json.dumps({
        'store_path': str(directory.parent / 'jobs.db'),
        'project_root': str(tmp_path),
        'logs_dir': str(tmp_path / 'logs'),
        'ipc_dir': str(directory),
    }))
    source_root = Path(__file__).resolve().parents[1]
    with (tmp_path / 'runner.log').open('wb') as output:
        process = subprocess.Popen(
            [sys.executable, '-m', 'frontend.job_runner', str(config)],
            cwd=source_root, stdout=output, stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 20
            while not (directory / 'state.json').exists():
                assert process.poll() is None, (tmp_path / 'runner.log').read_text()
                assert time.monotonic() < deadline, 'Runner never became ready'
                time.sleep(.05)
            shutil.rmtree(directory)
            assert process.wait(timeout=5) == 0
            assert (tmp_path / 'runner.log').stat().st_size < 100_000
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
