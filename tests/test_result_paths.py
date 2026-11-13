"""Nested experiment recipes keep their hierarchy across launches and result readers."""

import json

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.compose import compose_experiment, method_plans
from src.app.pipeline.result_paths import check_result_path


@pytest.fixture(autouse=True)
def isolate_provider_snapshot(monkeypatch):
    # Saving a launch config exports this path for workers. Do not leave a
    # temporary test workspace's snapshot in the environment of later tests.
    monkeypatch.setenv("THETA_PROVIDER_SNAPSHOT", "")


@pytest.mark.parametrize("override,expected", [
    (None, "path/dir/run"),
    ("renamed", "path/dir/renamed"),
    ("path/dir/renamed", "path/dir/renamed"),
])
def test_nested_recipe_identity_reaches_every_provider(fixture_workspace, monkeypatch, override, expected):
    from hydra.core.override_parser.overrides_parser import OverridesParser
    import src.app.pipeline.compose as composition

    monkeypatch.chdir(fixture_workspace)
    root = fixture_workspace / "in/config"
    monkeypatch.setattr(composition, "CONFIG_DIR", root)
    recipe = root / "experiment/thetaide/path/dir/run.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("# @package _global_\ndefaults:\n  - /experiment/thetaide/_base\n  - _self_\n")
    overrides = [f"++experiment_id={override}"] if override else []
    composed = compose_experiment("thetaide/path/dir/run", overrides)
    assert (composed.cfg.group, composed.cfg.experiment_id) == ("thetaide", expected)
    parser = OverridesParser.create()
    for plan in method_plans(composed).values():
        identity = {item.key_or_group: item.value() for item in parser.parse_overrides(plan["train_overrides"])
                    if item.key_or_group in ("group", "experiment_id")}
        assert identity == {"group": "thetaide", "experiment_id": expected}
    # Resolved overrides must remain stable when preflight composes them again.
    again = compose_experiment(composed.experiment, composed.launch_args)
    assert (again.cfg.group, again.cfg.experiment_id) == ("thetaide", expected)


def test_nested_results_can_be_discovered_read_and_plotted(tmp_path, monkeypatch):
    from frontend.figures.model import available_experiments
    from plot.base import BasePlotter
    from src.app import telemetry
    from src.app.pipeline.experiment_snapshot import save_experiment_snapshot, save_run_config
    from src.app.pipeline.local_runner import _setup_output_dirs
    from src.app.services import catalog, executions

    monkeypatch.chdir(tmp_path)
    logs = tmp_path / "results/logs"
    monkeypatch.setattr(catalog, "LOGS_DIR", logs)
    cfg = OmegaConf.create({"group": "group", "experiment_id": "path/dir/run", "seed": 73,
                            "plotting": {"enabled": True}, "plots": ["losses"]})
    _setup_output_dirs(cfg)
    save_experiment_snapshot(cfg, {"experiment_yaml": b"seed: 73\n"})
    save_run_config(cfg)
    with telemetry.init(cfg.group, cfg.experiment_id, "ppo", system_interval=0) as run:
        run.log({"eval/reward": 42}, step=1, transitions=1)
    (entry,) = catalog.list_runs()
    assert (entry["group"], entry["experiment_id"]) == ("group", "path/dir/run")
    assert entry["agents"][0]["best_reward"] == 42
    assert catalog.run_metrics("group", "path/dir/run", "ppo")["total_rows"] == 1
    assert catalog.run_manifest("group", "path/dir/run")["agents"] == {"ppo": ["version_0"]}
    assert available_experiments(logs) == [("group", "path/dir/run")]
    plotter = BasePlotter("test")
    assert plotter.get_experiment_config("group/path/dir/run")["seed"] == 73
    assert "ppo" in plotter.load_metrics("group", "group/path/dir/run")
    assert plotter.get_effective_config("group/path/dir/run")[2].as_posix() == "results/plots/group/path/dir/run"
    manifest = tmp_path / "results/executions/group/path/dir/run/execution/manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"runs": [{"status": "completed", "experiment_id": cfg.experiment_id,
                                             "agent": "ppo", "version": "version_0", "training_seed": 73}]}))
    assert executions.list_executions(completed_only=True) == [str(manifest.relative_to(tmp_path))]
    details = catalog.run_manifest("group", "path/dir/run")["version_details"]
    assert details["ppo"]["version_0"] == {"seed": 73, "status": "completed"}


@pytest.mark.parametrize("path", ["../run", "dir/../run", "/run", "dir//run", "dir/./run", "dir\\run", "C:/run"])
def test_nested_result_paths_reject_traversal(path):
    with pytest.raises(ValueError):
        check_result_path(path)


@pytest.mark.parametrize("consolidate", [False, True])
def test_slurm_nested_results_use_valid_script_names(tmp_path, monkeypatch, consolidate):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    from src.app.sites.execution import submit

    monkeypatch.chdir(tmp_path)
    (tmp_path / "run_pipeline.py").write_text("# pipeline")
    submitted = []
    def sbatch(arguments, **kwargs):
        assert arguments[0] == "sbatch"
        submitted.append(Path(arguments[-1]).read_text())
        return SimpleNamespace(returncode=0, stdout="123\n", stderr="")
    monkeypatch.setattr("subprocess.run", sbatch)
    cfg = OmegaConf.create({"group": "group", "experiment_id": "path/dir/run",
                            "site": {"name": "cluster", "python_executable": sys.executable},
                            "paradigm": "online_rl", "consolidate": consolidate, "plots": ["losses"]})
    record = submit(tmp_path, "group/path/dir/run", [], cfg)
    directory = tmp_path / "results/slurm" / record["execution_id"]
    assert (directory / "job.sbatch").is_file()
    assert len(submitted) == 1
    assert record["result_experiment_id"] == "path/dir/run/" + record["execution_id"]
    assert "#SBATCH --job-name=theta_" in submitted[0]
    assert not (tmp_path / "results/plots/group/path/dir/run").exists()
