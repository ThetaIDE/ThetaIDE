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
