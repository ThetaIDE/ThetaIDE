import json


from src.app.pipeline.trial_artifacts import trial_checkpoint_dir


def test_study_names_cannot_escape_the_trial_directory(tmp_path):
    path = trial_checkpoint_dir(tmp_path, "../other", 0)
    assert path.is_relative_to(tmp_path / "sweeps")
    assert path.parent.name == "%2E%2E%2Fother"


def test_plotman_reads_generic_tuning_summary_and_writes_table_and_figure(tmp_path, monkeypatch):
    from src.plugins.components.bundled.standard_figures.tuning import TuningPlotter
    from src.app.core.plotter_registry import auto_discover_plotters, get_plotter_class
    from src.app.workspace import WorkspacePaths

    results = WorkspacePaths(tmp_path).results
    root = results / "checkpoints/g/e/m"
    root.mkdir(parents=True)
    (root / "tuning_summary.json").write_text(json.dumps({
        "schema_version": 1, "component": "another-tuner", "direction": "minimize",
        "trials": [
            {"id": 0, "state": "complete", "value": 4, "parameters": {"lr": 0.1},
             "intermediate_values": {"0": 5, "1": 4}},
            {"id": 1, "state": "fail", "value": None, "parameters": {"lr": 0.2}},
            {"id": 2, "state": "complete", "value": 2, "parameters": {"lr": 0.3}},
        ],
    }))
    study_root = root / "sweeps/older"
    study_root.mkdir(parents=True)
    (study_root / "tuning_summary.json").write_text((root / "tuning_summary.json").read_text())
    pruned_root = root.parent / "only_pruned"
    pruned_root.mkdir()
    (pruned_root / "tuning_summary.json").write_text(json.dumps({
        "schema_version": 1, "component": "example", "direction": "minimize",
        "trials": [{"id": 0, "state": "pruned", "value": None, "parameters": {},
                    "intermediate_values": {"0": 2}}],
    }))
    output = tmp_path / "plots"
    plotter = TuningPlotter(paths=WorkspacePaths(tmp_path))
    monkeypatch.setattr(plotter, "get_effective_config", lambda *args: ({}, "g", output))
    plotter.run("e")
    assert "param.lr" in (output / "tuning/m/trials.csv").read_text()
    assert (output / "tuning/m/objective.png").is_file()
    assert (output / "tuning/m/intermediate.png").is_file()
    auto_discover_plotters()
    assert get_plotter_class("tuning") is TuningPlotter
    from src.app.core.plotting.manager import run_experiment_plots

    (root.parent / "config.yaml").write_text("group: g\nplots:\n  tuning: {}\n")
    # Plotting resolves configs and outputs in the selected workspace, never the CWD.
    run_experiment_plots("g/e", paths=WorkspacePaths(tmp_path))
    assert (results / "plots/g/e/tuning/m/trials.csv").is_file()
    assert (results / "plots/g/e/tuning/m/objective.png").is_file()
    assert (results / "plots/g/e/tuning/m/intermediate.png").is_file()
    assert (results / "plots/g/e/tuning/m/studies/older/trials.csv").is_file()
    assert (results / "plots/g/e/tuning/only_pruned/intermediate.png").is_file()


