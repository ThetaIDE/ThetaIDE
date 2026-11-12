"""Component selections and parameters survive the panel, Hydra, and runtime."""

from types import SimpleNamespace

import pytest
import yaml
from omegaconf import OmegaConf

from src.app.pipeline.compose import compose_experiment
from src.app.services.experiment_exporter import _component_names, _plot_names


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def isolated_component_workspace(fixture_workspace, monkeypatch):
    import src.app.pipeline.compose as compose_module
    from frontend.config_model import ConfigTree
    import frontend.config_viewer as viewer_module

    monkeypatch.chdir(fixture_workspace)
    config_root = fixture_workspace / "in" / "config"
    monkeypatch.setattr(compose_module, "CONFIG_DIR", config_root)
    monkeypatch.setattr(viewer_module, "config_tree", lambda *args, **kwargs: ConfigTree(config_root))
    return fixture_workspace


def test_hydra_composes_component_overrides(isolated_component_workspace):
    path = isolated_component_workspace / "in/config/experiment/thetaide/_pytest_components.yaml"
    path.write_text(
        "# @package _global_\n"
        "defaults:\n  - thetaide/_base\n"
        "data_module:\n  name: tabular_data\n  params:\n    batch_size: 32\n"
        "evaluator:\n  name: EpisodicRewardEvalProtocol\n  params:\n    threshold: 0.8\n"
        "plot_overrides:\n  losses:\n    smoothing_window: 7\n",
        encoding="utf-8",
    )
    try:
        cfg = compose_experiment("thetaide/_pytest_components").cfg
        assert cfg.data_module.name == "tabular_data"
        assert cfg.data_module.params.batch_size == 32
        assert cfg.evaluator.params.threshold == 0.8
        assert cfg.plot_overrides.losses.smoothing_window == 7
    finally:
        path.unlink(missing_ok=True)


def test_export_traces_mapping_component_names():
    assert _component_names({"name": "my_evaluator", "params": {"threshold": 0.8}}) == {"my_evaluator"}
    assert _component_names([{"name": "one"}, "two"]) == {"one", "two"}
    assert _plot_names({"losses": {"dpi": 300}, "style": {"grid": True}}) == {"losses"}


def test_registered_evaluator_receives_recipe_parameters():
    from src.app.core.eval_registry import register_evaluator
    from src.app.core.lightning_builder import ConfiguredEvaluatorCallback, configured_evaluator_callbacks

    @register_evaluator("component_override_probe")
    class Probe:
        def __init__(self, threshold):
            self.threshold = threshold

        def evaluate(self, model, data_source, cfg):
            return {"eval/probe": self.threshold}

    cfg = OmegaConf.create({"evaluator": {"name": "component_override_probe", "params": {"threshold": 0.8}}})
    callback, = configured_evaluator_callbacks(cfg)
    assert isinstance(callback, ConfiguredEvaluatorCallback)
    assert callback.evaluator.threshold == 0.8
    recorded = []
    trainer = SimpleNamespace(
        logger=SimpleNamespace(log_metrics=lambda metrics, step: recorded.append((metrics, step))),
        global_step=4,
        datamodule=SimpleNamespace(val_dataloader=lambda: [1, 2]),
    )
    callback.on_fit_end(trainer, object())
    assert recorded == [({"eval/probe": 0.8}, 4)]
    with pytest.raises(ValueError, match="not registered"):
        configured_evaluator_callbacks(OmegaConf.create({"evaluator": "missing_probe"}))

    @register_evaluator("online_only_probe")
    class OnlineOnlyProbe(Probe):
        @classmethod
        def get_supported_paradigms(cls):
            return ["online_rl"]

    with pytest.raises(ValueError, match="does not support paradigm"):
        configured_evaluator_callbacks(OmegaConf.create({
            "paradigm": "supervised", "evaluator": {"name": "online_only_probe", "params": {"threshold": 0.8}}
        }))

    assert configured_evaluator_callbacks(OmegaConf.create({
        "evaluator": {"name": "component_override_probe", "params": {"enabled": False, "threshold": 0.8}}
    })) == []


def test_evaluation_hook_honors_frequency():
    from src.app.core.eval_registry import register_evaluator
    from src.app.core.lightning_builder import configured_evaluator_callbacks

    @register_evaluator("frequency_probe")
    class Hook:
        def __init__(self):
            self.events = []

        def on_eval_start(self, agent, env, stage="val"):
            self.events.append("start")

        def evaluate_step(self, agent, batch_or_transition):
            self.events.append("step")
            return {}

        def on_eval_end(self):
            self.events.append("end")
            return {"eval/hook": 1.0}

    callback, = configured_evaluator_callbacks(OmegaConf.create({
        "evaluator": {"name": "frequency_probe", "params": {"frequency": 2}}
    }))
    recorded = []
    trainer = SimpleNamespace(
        current_epoch=0, datamodule=object(), global_step=2,
        logger=SimpleNamespace(log_metrics=lambda metrics, step: recorded.append(metrics)),
    )
    callback.on_validation_start(trainer, object())
    callback.on_validation_batch_end(trainer, object(), None, [1], 0)
    callback.on_validation_end(trainer, object())
    assert callback.evaluator.events == []
    trainer.current_epoch = 1
    callback.on_validation_start(trainer, object())
    callback.on_validation_batch_end(trainer, object(), None, [1], 0)
    callback.on_validation_end(trainer, object())
    assert callback.evaluator.events == ["start", "step", "end"]
    assert recorded == [{"eval/hook": 1.0}]


def test_figure_options_do_not_become_figure_names():
    from plot.manager import get_requested_plots

    assert get_requested_plots({"plots": {
        "losses": {"smoothing_window": 7}, "title": "Trial", "style": {"grid": True},
        "convergence": None,
    }}, "example") == {"losses": {"smoothing_window": 7}}


def test_plotter_applies_separate_figure_override(monkeypatch, tmp_path):
    from plot.base import BasePlotter

    plotter = BasePlotter("losses")
    plotter.default_cfg = {"smoothing_window": 10}
    monkeypatch.setattr(plotter, "get_experiment_config", lambda *args, **kwargs: {
        "group": "probe", "plots": ["losses"],
        "plot_overrides": {"losses": {"smoothing_window": 7}},
        "output_dir": str(tmp_path),
    })
    settings, _, _ = plotter.get_effective_config("example")
    assert settings["smoothing_window"] == 7


def test_panel_edits_component_parameters(qapp, tmp_path):
    from frontend.config_viewer import ConfigViewer

    path = tmp_path / "example.yaml"
    path.write_text("# @package _global_\ndefaults:\n  - thetaide/_base\n", encoding="utf-8")
    viewer = ConfigViewer()
    viewer.load_file(path, "experiment/thetaide/example.yaml")
    viewer._set_component_choice("data_module", "fixture_data_module")
    viewer._add_component_parameter("data_module", "batch_size", 32)
    viewer._set_component_choice("evaluator", "fixture_evaluator")
    viewer._add_plot_parameter("losses", "smoothing_window", 7)
    assert viewer.save_to_disk()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["data_module"] == {"name": "fixture_data_module", "params": {"batch_size": 32}}
    assert saved["evaluator"] == {"name": "fixture_evaluator", "params": {}}
    assert saved["plot_overrides"]["losses"]["smoothing_window"] == 7
    assert path.read_text(encoding="utf-8").startswith("# @package _global_")


def test_panel_filters_components_by_declared_paradigm(qapp, tmp_path):
    from unittest.mock import patch

    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer

    root = tmp_path / "config"
    (root / "paradigms").mkdir(parents=True)
    (root / "paradigms" / "unsupervised.yaml").write_text(
        "environment_source: static_dataset\nmethod: {algorithm: forbidden, model: required}\n",
        encoding="utf-8",
    )
    (root / "data").mkdir()
    (root / "data" / "table.yaml").write_text(
        "data_module: fixture_table\ncompatible_paradigms: [unsupervised]\n", encoding="utf-8"
    )
    (root / "data" / "simulator.yaml").write_text(
        "data_module: fixture_simulator\ncompatible_paradigms: [online_rl]\n", encoding="utf-8"
    )
    viewer = ConfigViewer()
    viewer.raw_data["paradigm"] = "unsupervised"
    with patch("frontend.config_viewer.config_tree", return_value=ConfigTree(root)):
        choices = viewer._component_choices("data_module")
    assert "fixture_table" in choices
    assert "fixture_simulator" not in choices


def test_inherited_figure_list_accepts_parameter_override(qapp, isolated_component_workspace):
    from frontend.config_viewer import ConfigViewer

    group = isolated_component_workspace / "in/config/experiment/_pytest_plot_overrides"
    group.mkdir()
    base = group / "_base.yaml"
    recipe = group / "example.yaml"
    base.write_text("# @package _global_\nparadigm: online_rl\nplots: [losses, reports]\n", encoding="utf-8")
    recipe.write_text(
        "# @package _global_\ndefaults:\n  - _pytest_plot_overrides/_base\n",
        encoding="utf-8",
    )
    try:
        viewer = ConfigViewer()
        viewer.load_file(recipe, "experiment/_pytest_plot_overrides/example.yaml")
        viewer._add_plot_parameter("losses", "smoothing_window", 7)
        assert viewer.save_to_disk()
        saved = yaml.safe_load(recipe.read_text(encoding="utf-8"))
        assert "plots" not in saved
        assert saved["plot_overrides"]["losses"]["smoothing_window"] == 7
        cfg = compose_experiment("_pytest_plot_overrides/example").cfg
        assert list(cfg.plots) == ["losses", "reports"]
        assert cfg.plot_overrides.losses.smoothing_window == 7
    finally:
        recipe.unlink(missing_ok=True)
        base.unlink(missing_ok=True)
        group.rmdir()


def test_inherited_figure_mapping_can_remove_one_figure(qapp, isolated_component_workspace):
    from frontend.config_viewer import ConfigViewer
    from plot.manager import get_requested_plots

    group = isolated_component_workspace / "in/config/experiment/_pytest_plot_removal"
    group.mkdir()
    base = group / "_base.yaml"
    recipe = group / "example.yaml"
    base.write_text(
        "# @package _global_\nparadigm: online_rl\nplots:\n  losses: {}\n  reports: {}\n",
        encoding="utf-8",
    )
    recipe.write_text("# @package _global_\ndefaults:\n  - _pytest_plot_removal/_base\n", encoding="utf-8")
    try:
        viewer = ConfigViewer()
        viewer.load_file(recipe, "experiment/_pytest_plot_removal/example.yaml")
        viewer._set_plot_override("losses", False)
        assert viewer.save_to_disk()
        cfg = compose_experiment("_pytest_plot_removal/example").cfg
        assert get_requested_plots({"plots": OmegaConf.to_container(cfg.plots, resolve=True)}, "example") == {"reports": {}}
    finally:
        recipe.unlink(missing_ok=True)
        base.unlink(missing_ok=True)
        group.rmdir()


def test_all_inherited_figures_can_be_removed():
    from plot.manager import get_requested_plots

    assert get_requested_plots({"plots": {"losses": None, "reports": None}}, "example") == {}


def test_figures_box_adds_and_removes_cards(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFrame, QLabel, QToolButton
    from frontend.config_viewer import ConfigViewer

    path = tmp_path / "example.yaml"
    path.write_text("experiment_id: example\nplots: []\n", encoding="utf-8")
    viewer = ConfigViewer()
    monkeypatch.setattr(viewer, "_component_choices", lambda kind: {
        "sample": {"plotter": "sample", "dpi": 120}
    } if kind == "plots" else {})
    viewer.load_file(path, "experiment/example.yaml")
    buttons = viewer.findChildren(QToolButton)
    add = next(button for button in buttons if button.toolTip() == "Add figure")
    add.menu().actions()[0].trigger()
    qapp.processEvents()
    assert viewer.raw_data["plots"] == ["sample"]
    assert viewer.raw_data["plot_overrides"]["sample"] == {"dpi": 120}
    assert any(any(label.text() == "sample" for label in card.findChildren(QLabel))
               for card in viewer.findChildren(QFrame, "card"))
    remove = next(button for button in viewer.findChildren(QToolButton)
                  if button.toolTip() == "Remove sample figure")
    remove.click()
    assert viewer.raw_data["plots"] == []
    viewer.close()


def test_figures_box_is_optional_and_uses_two_column_cards(qapp, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QGridLayout, QLabel, QToolButton
    from frontend.config_viewer import ConfigViewer, IndentedHierarchyBlock

    path = tmp_path / "example.yaml"
    path.write_text("experiment_id: example\n", encoding="utf-8")
    viewer = ConfigViewer()
    monkeypatch.setattr(viewer, "_component_choices", lambda kind: {
        "one": {"plotter": "one", "style": {"color": "red"}},
        "two": {"plotter": "two", "dpi": 120},
    } if kind == "plots" else {})
    viewer.load_file(path, "experiment/example.yaml")
    assert not any(w.text() == "Figures and Tables" for w in viewer.findChildren(QLabel))
    viewer.add_component_override("plots")
    for name in ("one", "two"):
        viewer._set_plot_override(name, True, {})
    qapp.processEvents()
    one = next(w for w in viewer.findChildren(QToolButton) if w.toolTip() == "Remove one figure")
    two = next(w for w in viewer.findChildren(QToolButton) if w.toolTip() == "Remove two figure")
    box_layout = one.parentWidget().parentWidget().layout
    grid = next(box_layout.itemAt(i).layout() for i in range(box_layout.count())
                if isinstance(box_layout.itemAt(i).layout(), QGridLayout))
    assert grid.getItemPosition(grid.indexOf(one.parentWidget()))[:2] == (0, 0)
    assert grid.getItemPosition(grid.indexOf(two.parentWidget()))[:2] == (0, 1)
    assert one.parentWidget().findChild(IndentedHierarchyBlock) is not None
    viewer._remove_figures_box()
    assert "plots" not in viewer.raw_data
    assert not any(w.text() == "Figures and Tables" for w in viewer.findChildren(QLabel) if w.isVisible())
    viewer.close()


@pytest.mark.parametrize("config_path", ["in/config/config.yaml", "src/app/config_defaults/config.yaml"])
def test_root_config_disables_plotting_by_default(config_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert yaml.safe_load((root / config_path).read_text())["no_plot"] is True


def test_no_plot_selection_skips_phase_and_plot_directory(qapp, tmp_path, monkeypatch):
    from src.app.pipeline import local_runner
    from src.app.pipeline.plot_config import should_run_plotting

    monkeypatch.chdir(tmp_path)
    cfg = OmegaConf.create({"group": "g", "experiment_id": "run"})
    assert not should_run_plotting(cfg)
    assert not should_run_plotting(OmegaConf.create({"plots": [], "env": {"default_plots": ["losses"]}}))
    assert not should_run_plotting(OmegaConf.create({"plots": ["losses"]}))
    assert should_run_plotting(OmegaConf.create({"plots": ["losses"], "no_plot": False}))
    assert not should_run_plotting(OmegaConf.create({"plots": [], "no_plot": False}))
    assert not should_run_plotting(OmegaConf.create({"plots": ["losses"], "no_plot": True}))
    local_runner._setup_output_dirs(cfg)
    assert not (tmp_path / "results/plots").exists()
    from src.app.pipeline import datasets
    monkeypatch.setattr(datasets, "run_plotting", lambda *args, **kwargs: pytest.fail("plotting ran"))
    local_runner.run_plotting_phase(cfg, {})


def test_slurm_uses_one_allocation_for_the_whole_pipeline(tmp_path, monkeypatch):
    from src.app.pipeline import slurm_runner

    monkeypatch.chdir(tmp_path)
    submitted = []
    monkeypatch.setattr(slurm_runner, "submit", lambda root, experiment, overrides, cfg:
                        submitted.append((experiment, overrides, cfg)) or {"job_id": "1"})
    cfg = OmegaConf.create({"group": "g", "experiment_id": "run", "experiment_name": "g/run"})
    assert slurm_runner.run_slurm_training(cfg, {"sanitized_extra_args": ["seed=3"]}) == (["1"], [], True)
    assert len(submitted) == 1 and submitted[0][:2] == ("g/run", ["seed=3"])
    assert not (tmp_path / "results/plots").exists()


def test_remove_inherited_figures_masks_group_selection(qapp, tmp_path, monkeypatch):
    from frontend.config_viewer import ConfigViewer
    from src.app.pipeline.plot_config import should_run_plotting

    path = tmp_path / "example.yaml"
    path.write_text("experiment_id: example\n", encoding="utf-8")
    viewer = ConfigViewer()
    monkeypatch.setattr(viewer, "_get_resolved_defaults", lambda: {"plots": ["losses"]})
    viewer.load_file(path, "experiment/group/example.yaml")
    viewer._remove_figures_box()
    assert "plots" in viewer.raw_data and viewer.raw_data["plots"] is None
    assert not should_run_plotting(OmegaConf.create(viewer.raw_data))
    viewer.close()


def test_plot_manager_without_selection_writes_no_output(tmp_path, monkeypatch):
    from plot import manager

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(manager.BasePlotter, "get_experiment_config", lambda *args, **kwargs: {})
    manager.run_experiment_plots("run")
    assert not (tmp_path / "results/plots").exists()


def test_plot_manager_preserves_parameter_override_precedence(tmp_path, monkeypatch):
    from plot import manager

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(manager.BasePlotter, "get_experiment_config", lambda *args, **kwargs: {
        "plots": {"sample": {"dpi": 100}}, "plot_overrides": {"sample": {"dpi": 200}},
    })
    monkeypatch.setattr(manager.BasePlotter, "get_group", lambda *args: "g")
    received = []
    class Sample:
        def run(self, exp_id, cli_overrides=None):
            received.append(cli_overrides)
    monkeypatch.setattr(manager, "discover_plotters", lambda: {"sample": Sample})
    manager.run_experiment_plots("run")
    assert received == [{"dpi": 200}]
