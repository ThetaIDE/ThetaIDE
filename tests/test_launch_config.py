import os

from src.app.pipeline.launch_config import frozen_config
from src.app.workspace import CONFIG_SNAPSHOT_ENV, WorkspacePaths


def workspace(tmp_path, monkeypatch):
    live = tmp_path / "src/usr/config"
    (live / "experiment/G").mkdir(parents=True)
    (live / "config.yaml").write_text("seed: null\n")
    (live / "experiment/G/sweep.yaml").write_text("total_timesteps: 120000\n")
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv(CONFIG_SNAPSHOT_ENV, raising=False)
    return WorkspacePaths(tmp_path, data_dir=tmp_path / ".thetaide"), live


def test_a_launch_composes_from_its_snapshot_while_the_live_config_changes(tmp_path, monkeypatch):
    paths, live = workspace(tmp_path, monkeypatch)
    with frozen_config(paths) as snapshot:
        assert os.environ[CONFIG_SNAPSHOT_ENV] == str(snapshot)
        assert paths.config == snapshot and paths.primary_config == snapshot
        (live / "experiment/G/sweep.yaml").write_text("total_timesteps: 300000\n")
        selected = paths.config_file("src/usr/config/experiment/G/sweep.yaml")
        assert selected == (snapshot / "experiment/G/sweep.yaml").resolve()
        assert selected.read_text() == "total_timesteps: 120000\n"
        assert WorkspacePaths.from_config_root(snapshot).root == paths.root
        with frozen_config(paths) as nested:  # a launch inside a launch keeps the outer snapshot
            assert nested == snapshot
    assert CONFIG_SNAPSHOT_ENV not in os.environ
    assert not snapshot.exists()
    assert paths.config == live


def test_other_workspaces_ignore_a_launch_snapshot(tmp_path, monkeypatch):
    paths, _ = workspace(tmp_path, monkeypatch)
    other = WorkspacePaths(tmp_path / "other")
    with frozen_config(paths):
        assert other.config == (tmp_path / "other/src/usr/config").resolve()
