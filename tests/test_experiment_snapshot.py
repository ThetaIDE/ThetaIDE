from types import SimpleNamespace

from src.app.pipeline.experiment_snapshot import save_experiment_snapshot


def test_snapshots_preserve_original_recipe_across_recoveries(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = SimpleNamespace(group="group", experiment_id="run")
    save_experiment_snapshot(cfg, {"experiment_yaml": b"seed: 1\n"})
    save_experiment_snapshot(cfg, {"experiment_yaml": b"seed: 2\n"})
    directory = tmp_path / "results/logs/group/run"
    assert (directory / "experiment.yaml").read_bytes() == b"seed: 1\n"
    assert [p.read_bytes() for p in directory.glob("experiment_*.yaml")] == [b"seed: 2\n"]


def test_run_snapshot_handles_hydra_runtime_interpolation(tmp_path, monkeypatch):
    import yaml
    from omegaconf import OmegaConf
    from src.app.pipeline.experiment_snapshot import save_run_config

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({
        "group": "group", "experiment_id": "run", "seed": 42,
        "env": {"seed": "${seed}"},
        "tuning": {"n_trials": 10},
        "hydra": {"job": {"num": "???"}, "sweep": {"subdir": "${hydra.job.num}"}},
    })
    save_run_config(cfg)
    saved = yaml.safe_load((tmp_path / "results/logs/group/run/config.yaml").read_text())
    assert saved["env"]["seed"] == 42
    assert saved["tuning"] == {"n_trials": 10}
    assert "hydra" not in saved
    assert cfg.hydra.job._get_node("num")._is_missing()


def test_run_snapshot_with_actual_composed_hydra_config(tmp_path, monkeypatch):
    import yaml
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from src.app.pipeline.compose import CONFIG_DIR
    from src.app.pipeline.experiment_snapshot import save_run_config

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", return_hydra_config=True)
    finally:
        GlobalHydra.instance().clear()
    cfg.group = "group"
    cfg.experiment_id = "run"
    monkeypatch.chdir(tmp_path)
    save_run_config(cfg)
    saved = yaml.safe_load((tmp_path / "results/logs/group/run/config.yaml").read_text())
    assert "hydra" not in saved
    assert saved["group"] == "group"


def test_local_launch_reaches_methods_with_hydra_runtime_placeholder(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from src.app.pipeline import local_runner

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"group": "group", "experiment_id": "run", "no_plot": True,
                            "hydra": {"job": {"num": "???"},
                                      "sweep": {"subdir": "${hydra.job.num}"}}})
    calls = []
    monkeypatch.setattr(local_runner, "_setup_output_dirs", lambda cfg, **kwargs: None)
    monkeypatch.setattr(local_runner, "run_methods", lambda cfg, context: calls.append("methods"))
    local_runner.run_local_training(cfg, {})
    assert calls == ["methods"]
    assert (tmp_path / "results/logs/group/run/config.yaml").is_file()
