"""Unit tests for PluginScaffolder service and ScaffoldPluginDialog GUI."""
from pathlib import Path
import pytest

from PyQt6.QtWidgets import QApplication
from frontend.scaffold_dialog import ScaffoldPluginDialog
from src.app.services.plugin_scaffolder import PluginScaffolder, ScaffoldOptions, ScaffoldResult


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(["--platform", "offscreen"])
    return app


def test_scaffold_model_standard(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(
        name="simple_mlp",
        kind="model",
        author="Unit Tester",
    )
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "simple_mlp_model.py").is_file()
    assert (res.target_dir / "__init__.py").is_file()
    assert (res.target_dir / ".theta_component.json").is_file()
    assert (res.target_dir / "pyproject.toml").is_file()
    assert (tmp_path / "in" / "config" / "model" / "simple_mlp.yaml").is_file()


def test_scaffold_agent(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="soft_actor_critic", kind="agent")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "soft_actor_critic_agent.py").is_file()
    assert (tmp_path / "in" / "config" / "agent" / "soft_actor_critic.yaml").is_file()


def test_scaffold_environment(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="quadrotor_sim", kind="env")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "env.py").is_file()
    assert (tmp_path / "in" / "config" / "env" / "quadrotor_sim.yaml").is_file()
    assert (tmp_path / "src" / "usr" / "rules" / "quadrotor_sim" / "default.rules").is_file()


def test_scaffold_datamodule(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="trajectory_streamer", kind="datamodule")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "trajectory_streamer_data_module.py").is_file()
    assert (tmp_path / "in" / "config" / "data" / "trajectory_streamer.yaml").is_file()


def test_scaffold_evaluator(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="reachability_checker", kind="evaluator")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "reachability_checker_evaluator.py").is_file()
    assert (tmp_path / "in" / "config" / "eval" / "reachability_checker.yaml").is_file()


def test_scaffold_plotter(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="pareto_frontier", kind="plotter")
    res = scaffolder.scaffold(opts)
    assert res.success is True
    assert (res.target_dir / "pareto_frontier_plotter.py").is_file()
    assert (res.target_dir / "__init__.py").is_file()
    assert (res.target_dir / ".theta_component.json").is_file()
    assert (tmp_path / "in" / "config" / "plot" / "pareto_frontier.yaml").is_file()


def test_scaffold_invalid_kind(tmp_path):
    scaffolder = PluginScaffolder(workspace_root=tmp_path)
    opts = ScaffoldOptions(name="invalid_comp", kind="unsupported_kind")
    res = scaffolder.scaffold(opts)
    assert res.success is False
    assert "Invalid component kind" in res.error


def test_scaffold_dialog_gui(qapp, tmp_path):
    dialog = ScaffoldPluginDialog(workspace_root=tmp_path)
    dialog.edit_name.setText("test_dialog_model")

    # Trigger scaffolding
    dialog._on_scaffold_clicked()

    assert dialog.scaffold_result is not None
    assert dialog.scaffold_result.success is True
    assert (dialog.scaffold_result.target_dir / "test_dialog_model_model.py").is_file()
    dialog.deleteLater()


def test_components_panel_plus_button(qapp, tmp_path):
    from frontend.components_panel import ComponentsPanel
    panel = ComponentsPanel()
    panel.components_tree.reset_toolbar_layout()
    assert hasattr(panel, "btn_plus")
    assert panel.btn_plus is not None
    assert panel.btn_plus.property("svg_icon") == "plus"
    assert panel.btn_hub is not None

    header_widgets = [
        panel.components_tree.header_layout.itemAt(i).widget()
        for i in range(panel.components_tree.header_layout.count())
        if panel.components_tree.header_layout.itemAt(i).widget() is not None
    ]
    assert panel.btn_plus in header_widgets
    assert panel.btn_hub in header_widgets
    plus_idx = header_widgets.index(panel.btn_plus)
    hub_idx = header_widgets.index(panel.btn_hub)
    assert abs(plus_idx - hub_idx) == 1
    panel.deleteLater()


