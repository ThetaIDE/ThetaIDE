"""The sweep worker must compose configs owned by workspace components."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize("explicit_searchpath", [False, True])
def test_worker_composes_component_site_config(tmp_path, explicit_searchpath):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text("defaults:\n  - site: ncshare/common\n  - _self_\n")
    command = [sys.executable, "-m", "src.app.pipeline.provider_worker",
               "src/app/pipeline/repeated_task.py"]
    if explicit_searchpath:
        from src.app.workspace import WorkspacePaths
        roots = [p.as_uri() for p in WorkspacePaths(root).config_roots() if p.is_dir()]
        command.append("hydra.searchpath=[" + ",".join(roots) + "]")
    command.extend(["--config-path", str(config), "--cfg", "job"])
    environment = {**os.environ, "THETA_WORKSPACE": str(root),
                   "THETA_DATA_DIR": str(tmp_path / "data")}
    environment.pop("THETA_PROVIDER_SNAPSHOT", None)
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.site.scheduler == "slurm"
    assert cfg.site.resources.partition == "common"


def test_multirun_flag_before_overrides_launches_trials(tmp_path):
    root = Path(__file__).resolve().parents[1]
    config = tmp_path / "config"
    config.mkdir()
    (config / "config.yaml").write_text(
        "defaults:\n  - site: ncshare/common\n  - _self_\nprobe: 0\n")
    # Exercise Hydra's real parser, sweeper and launcher without starting training.
    script = (
        "import sys; from src.app.pipeline import repeated_task as worker; "
        "worker.run_objective = lambda cfg: print('TRIAL_PROBE', cfg.probe); "
        "sys.argv = ['repeated_task.py', *sys.argv[1:]]; worker.launch()"
    )
    command = [sys.executable, "-c", script, "--config-path", str(config),
               "--multirun", "probe=1,2", "hydra/sweeper=basic",
               "hydra.sweep.dir=" + (tmp_path / "sweep").as_posix()]
    environment = {**os.environ, "THETA_WORKSPACE": str(root),
                   "THETA_DATA_DIR": str(tmp_path / "data")}
    result = subprocess.run(command, cwd=root, env=environment, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "TRIAL_PROBE 1" in result.stdout
    assert "TRIAL_PROBE 2" in result.stdout
