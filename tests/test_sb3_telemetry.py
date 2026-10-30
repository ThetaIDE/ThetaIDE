import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "sb3_runner_telemetry", Path(__file__).resolve().parents[1] / "src/usr/algorithms/sb3-suite/sb3_runner.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class Capture:
    def __init__(self):
        self.rows = []

    def log(self, metrics, **position):
        self.rows.append((metrics, position))


def test_completed_episodes_are_logged_between_reporting_intervals():
    capture = Capture()
    episodes = runner.EpisodeMetrics(capture, window=2)
    episodes.record([{}, {"episode": {"r": 10, "l": 7}}], 17)
    episodes.record([{"episode": {"r": 30, "l": 9}}], 42)
    episodes.record([{"episode": {"r": 50, "l": 11}}], 81)
    assert [row[0]["train/episode_reward"] for row in capture.rows] == [10, 30, 50]
    assert capture.rows[-1] == ({"train/episode_reward": 50, "train/episode_length": 11,
                                "train/reward": 40, "train/length": 10},
                               {"step": 81, "transitions": 81})


def test_sb3_logger_dump_forwards_diagnostics_and_losses():
    logger_module = pytest.importorskip("stable_baselines3.common.logger")
    capture = Capture()
    callback = SimpleNamespace(num_timesteps=2048)

    class Writer(runner.ThetaSB3Writer, logger_module.KVWriter):
        pass

    logger = logger_module.Logger(folder=None, output_formats=[Writer(capture, callback)])
    logger.record("train/approx_kl", 0.02)
    logger.record("train/value_loss", 1.5)
    logger.record("time/fps", 120)
    logger.record("train/invalid", float("nan"))
    logger.dump(step=9)
    assert capture.rows == [({"diagnostics/approx_kl": 0.02, "losses/train/value_loss": 1.5,
                              "time/fps": 120.0}, {"step": 2048, "transitions": 2048})]


def test_episode_metrics_reach_live_monitor_points(tmp_path):
    import csv
    from frontend.model import metric_points_from_api
    from src.app.telemetry.run import Run
    from src.app.telemetry.sinks import CsvSink

    sink = CsvSink(tmp_path / "version_0")
    run = Run(tmp_path / "version_0", sinks=[sink], system_interval=0)
    runner.EpisodeMetrics(run).record([{"episode": {"r": 31, "l": 31}}], 31)
    run.finish()
    with (tmp_path / "version_0" / "metrics.csv").open() as f:
        points = metric_points_from_api(list(csv.DictReader(f)))
    assert points[0]["train/episode_reward"] == 31
    assert points[0]["train/reward"] == 31
