"""Live monitor streams every started method and keeps the selected curve in front."""
import pytest

pytest.importorskip("PyQt6")
from frontend.runner_manager import JobManager
from frontend.monitor.panel import MonitorPanel


def test_scan_reads_all_started_sources(monkeypatch):
    class Feed:
        def __init__(self, path, version=None):
            self.name = path.name

        def read(self):
            return [{"type": "metrics", "data": {}}], False

    class Signal:
        def __init__(self):
            self.calls = []

        def emit(self, *args):
            self.calls.append(args)

    from pathlib import Path
    import frontend.runner_manager as module
    monkeypatch.setattr(module, "AgentFeed", Feed)
    monkeypatch.setattr(module, "metrics_row", lambda record: {"step": 1})
    class Manager:
        jobs = {"job": {"status": "running"}}
        _feeds = {}
        _streamed_hardware = set()
        job_records = Signal()
        job_metrics = Signal()
        job_hardware = Signal()

        def _run_sources(self, job):
            return [(name, Path("/tmp"), name, "", status) for name, status in
                    [("a", "running"), ("b", "running"), ("c", "completed"), ("d", "queued")]]

    manager = Manager()
    JobManager._scan_metrics(manager, "job")
    assert [call[1] for call in manager.job_records.calls] == ["a", "b", "c"]


def test_focused_method_is_last_without_hiding_siblings():
    class Panel:
        focused_agent = "a"

    lives = {"a": object(), "b": object()}
    assert list(MonitorPanel.shown_lives(Panel(), lives)) == ["b", "a"]


def test_run_owns_hardware_policy_even_without_resource_samples(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock

    manager = SimpleNamespace(jobs={"job": {"status": "running"}},
        _feeds={("job", "m"): SimpleNamespace(read=lambda: ([{"type": "start", "config": {
            "telemetry": {"monitors": {"disabled": ["system/"]}}}}], False))},
        _run_sources=lambda job: [("m", tmp_path, "m", "", "running")],
        _streamed_hardware=set(), job_records=Mock(), job_metrics=Mock(), job_hardware=Mock())
    JobManager._scan_metrics(manager, "job")
    assert "job" in manager._streamed_hardware
    # Returns before accessing any of the external probe's state.
    JobManager._sample_hardware(manager, "job")
    manager.job_hardware.emit.assert_not_called()
