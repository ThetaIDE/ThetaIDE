"""Unit tests for component actions: opening code in editor and uninstalling/deleting components."""
from pathlib import Path
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from frontend.component_actions import (
    find_component_assets,
    DeleteComponentDialog,
)
from frontend.config_viewer import ConfigViewer, SUBTLE_ICON_COLORS
from frontend.components_panel import ComponentsPanel
from frontend.sidetabs import svg_icon


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_svg_icons_exist_and_render(qapp):
    """Test that pencil.svg and trash.svg icons exist and load properly."""
    pencil = svg_icon("pencil", SUBTLE_ICON_COLORS)
    assert not pencil.isNull()

    trash = svg_icon("trash", SUBTLE_ICON_COLORS)
    assert not trash.isNull()


def test_find_component_assets_multi_config_suite(tmp_path):
    """Test resolving a multi-config component suite like Atari."""
    # Setup mock workspace
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "in" / "config" / "env" / "atari").mkdir(parents=True)
    (ws / "in" / "envs" / "atari").mkdir(parents=True)
    (ws / "src").mkdir()

    # Create dummy configs
    c1 = ws / "in" / "config" / "env" / "atari" / "pong.yaml"
    c2 = ws / "in" / "config" / "env" / "atari" / "breakout.yaml"
    c1.write_text("env:\n  name: pong\n")
    c2.write_text("env:\n  name: breakout\n")

    # Create dummy env code
    env_py = ws / "in" / "envs" / "atari" / "env_vectorized.py"
    env_py.write_text("# atari code")

    # Create manifest with installed_from to simulate a Hub-installed package
    meta_file = ws / "in" / "envs" / "atari" / ".theta_component.json"
    import json
    meta_file.write_text(json.dumps({
        "id": "atari-env",
        "name": "Atari 2600 Environment Suite",
        "kind": "env",
        "version": "1.0.0",
        "installed_from": "https://raw.githubusercontent.com/theta-hub/packages/atari.zip",
        "target_path": "in/envs/atari",
        "config_paths": [
            "in/config/env/atari/pong.yaml",
            "in/config/env/atari/breakout.yaml",
        ]
    }))

    assets = find_component_assets(c1, "env/atari/pong.yaml", workspace_dir=ws)
    assert assets["is_component"] is True
    assert assets["is_hub_installed"] is True
    assert assets["action_type"] == "uninstall"
    assert assets["comp_name"] == "Atari 2600 Environment Suite"
    assert assets["primary_code_file"] == env_py
    assert ws / "in" / "envs" / "atari" in assets["files_to_delete"]
    assert c1 in assets["files_to_delete"]
    assert c2 in assets["files_to_delete"]


def test_find_component_assets_local_scaffolded_component(tmp_path):
    """Test resolving a locally scaffolded component (not installed from Hub)."""
    ws = tmp_path / "workspace_local"
    ws.mkdir()
    (ws / "in" / "config" / "model").mkdir(parents=True)
    (ws / "src" / "usr" / "models" / "myfirstmlp").mkdir(parents=True)

    cfg = ws / "in" / "config" / "model" / "myfirstmlp.yaml"
    cfg.write_text("type: myfirstmlp\n")

    code_py = ws / "src" / "usr" / "models" / "myfirstmlp" / "myfirstmlp_model.py"
    code_py.write_text("# myfirstmlp code")

    # Local manifest without installed_from
    meta_file = ws / "src" / "usr" / "models" / "myfirstmlp" / ".theta_component.json"
    import json
    meta_file.write_text(json.dumps({
        "id": "myfirstmlp",
        "name": "MyfirstmlpModel",
        "kind": "model",
        "version": "0.1.0",
        "target_path": "src/usr/models/myfirstmlp",
        "config_path": "in/config/model/myfirstmlp.yaml",
    }))

    assets = find_component_assets(cfg, "model/myfirstmlp.yaml", workspace_dir=ws)
    assert assets["is_component"] is True
    assert assets["is_hub_installed"] is False
    assert assets["action_type"] == "delete_component"
    assert assets["primary_code_file"] == code_py
    assert ws / "src" / "usr" / "models" / "myfirstmlp" in assets["files_to_delete"]
    assert cfg in assets["files_to_delete"]


def test_find_component_assets_loose_config(tmp_path):
    """Test resolving a standalone config not tied to any component code."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    (ws / "in" / "config" / "agent").mkdir(parents=True)
    (ws / "src").mkdir()

    cfg = ws / "in" / "config" / "agent" / "my_custom_loose_agent.yaml"
    cfg.write_text("agent: custom\n")

    assets = find_component_assets(cfg, "agent/my_custom_loose_agent.yaml", workspace_dir=ws)
    assert assets["is_component"] is False
    assert assets["is_hub_installed"] is False
    assert assets["action_type"] == "delete_config"
    assert assets["files_to_delete"] == [cfg]


def test_delete_component_dialog(qapp, tmp_path):
    """Test DeleteComponentDialog interactions."""
    files = [tmp_path / "foo.yaml", tmp_path / "src" / "foo"]

    # Local component: should say "Delete Component"
    dlg_local = DeleteComponentDialog("myfirstmlp", files, is_component=True, is_hub_installed=False)
    assert "Delete Component" in dlg_local.windowTitle()
    assert dlg_local.dont_ask_again is False
    dlg_local.chk_dont_ask.setChecked(True)
    dlg_local._on_confirm()
    assert dlg_local.dont_ask_again is True
    dlg_local.deleteLater()

    # Hub component: should say "Uninstall Component"
    dlg_hub = DeleteComponentDialog("Atari", files, is_component=True, is_hub_installed=True)
    assert "Uninstall Component" in dlg_hub.windowTitle()
    dlg_hub.deleteLater()


def test_config_viewer_component_buttons(qapp, tmp_path):
    """Test btn_edit_code, btn_export_component, and btn_delete_component layout and visibility."""
    viewer = ConfigViewer()
    assert hasattr(viewer, "btn_edit_code")
    assert hasattr(viewer, "btn_export_component")
    assert hasattr(viewer, "btn_delete_component")

    # In default mode (not component mode), buttons are hidden
    cfg = tmp_path / "test.yaml"
    cfg.write_text("key: value\n")
    viewer.load_file(cfg, "test.yaml")
    assert viewer.btn_edit_code.isHidden()
    assert viewer.btn_export_component.isHidden()
    assert viewer.btn_delete_component.isHidden()

    # In component mode, buttons are shown when a file is loaded
    viewer.is_component_mode = True
    viewer.load_file(cfg, "test.yaml")
    assert not viewer.btn_edit_code.isHidden()
    assert not viewer.btn_export_component.isHidden()
    assert not viewer.btn_delete_component.isHidden()

    # Verify relative layout: pencil -> export -> trash
    hb_layout = viewer.header_bar.layout()
    idx_pencil = -1
    idx_export = -1
    idx_trash = -1
    for i in range(hb_layout.count()):
        w = hb_layout.itemAt(i).widget()
        if w is viewer.btn_edit_code:
            idx_pencil = i
        elif w is viewer.btn_export_component:
            idx_export = i
        elif w is viewer.btn_delete_component:
            idx_trash = i

    assert idx_pencil != -1
    assert idx_export != -1
    assert idx_trash != -1
    assert idx_pencil < idx_export < idx_trash
    assert idx_export == idx_pencil + 1
    assert idx_trash == idx_export + 1

    viewer.deleteLater()


def test_components_panel_integration(qapp, tmp_path, monkeypatch):
    """Test ComponentsPanel wiring for editor opening, auto-save on navigation, and component export/deletion."""
    # Setup mock workspace
    ws = tmp_path / "ws"
    ws.mkdir()
    cfg_dir = ws / "in" / "config" / "model"
    cfg_dir.mkdir(parents=True)
    code_dir = ws / "src" / "usr" / "models" / "sample_mlp"
    code_dir.mkdir(parents=True)

    cfg = cfg_dir / "sample_mlp.yaml"
    cfg.write_text("type: sample_mlp\nhidden_sizes: [64, 64]\n")

    cfg2 = cfg_dir / "another_model.yaml"
    cfg2.write_text("type: another_model\n")

    code_file = code_dir / "sample_mlp_model.py"
    code_file.write_text("# sample model")

    meta_file = code_dir / ".theta_component.json"
    import json
    meta_file.write_text(json.dumps({
        "id": "sample_mlp",
        "name": "SampleMLP",
        "kind": "model",
        "version": "0.1.0",
        "target_path": "src/usr/models/sample_mlp",
        "config_path": "in/config/model/sample_mlp.yaml"
    }))

    panel = ComponentsPanel()
    # Point tree to test root
    panel.components_tree.root_dir = ws / "in" / "config"
    panel.on_component_selected(cfg, "model/sample_mlp.yaml")

    # Assert viewer is in component mode and buttons are visible
    assert panel.viewer.is_component_mode is True
    assert not panel.viewer.btn_edit_code.isHidden()
    assert not panel.viewer.btn_export_component.isHidden()
    assert not panel.viewer.btn_delete_component.isHidden()

    # Test auto-save when switching components
    panel.viewer.raw_data["hidden_sizes"] = [128, 128]
    panel.viewer.is_dirty = True
    panel.on_component_selected(cfg2, "model/another_model.yaml")
    # Verify cfg was automatically saved with the modified hidden_sizes
    assert "128" in cfg.read_text()

    # Switch back to first component
    panel.on_component_selected(cfg, "model/sample_mlp.yaml")
    # Test auto-save when switching to Hub
    panel.viewer.raw_data["hidden_sizes"] = [256, 256]
    panel.viewer.is_dirty = True
    panel.show_hub()
    assert "256" in cfg.read_text()

    # Test open in editor
    opened_targets = []
    monkeypatch.setattr("frontend.component_actions.open_in_editor", lambda p: opened_targets.append(p) or True)
    panel.open_active_component_in_editor()
    assert len(opened_targets) == 1
    assert opened_targets[0] == code_file.parent
    assert (ws / ".thetaide/components/context/sample_mlp.md").is_file()

    # Test export component dialog wiring
    from frontend.export_dialog import ExportComponentDialog
    from src.app.services.component_exporter import ComponentExportResult
    mock_res = ComponentExportResult(success=True, export_path=tmp_path / "export.zip", files_included=["file1"])
    monkeypatch.setattr(ExportComponentDialog, "exec", lambda self: 1)
    monkeypatch.setattr(ExportComponentDialog, "__init__", lambda self, *args, **kwargs: setattr(self, "export_result", mock_res) or setattr(self, "comp_name", "SampleMLP") or None)
    monkeypatch.setattr("PyQt6.QtWidgets.QMessageBox.information", lambda *args, **kwargs: None)
    panel.prompt_export_active_component()

    # Test delete component (mock dialog acceptance)
    from PyQt6.QtWidgets import QDialog
    monkeypatch.setattr(DeleteComponentDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    panel.prompt_delete_active_component()

    # Verify both config and source files were deleted
    assert not cfg.exists()
    assert not code_dir.exists()

    panel.deleteLater()


def test_standard_model_actor_critic_wrapper():
    """Test that StandardModelActorCriticWrapper adapts a pure standard model (only forward()) to RL protocols."""
    import torch
    import torch.nn as nn
    from src.app.core.protocols import StandardModelActorCriticWrapper

    class PureForwardModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, x, **kwargs):
            return self.linear(x)

    model = PureForwardModel()
    wrapper = StandardModelActorCriticWrapper(model, obs_dim=4, n_actions=2)

    obs = torch.randn(5, 4)
    res = wrapper.get_action_and_value(obs)
    assert res.action.shape == (5,)
    assert res.logprob.shape == (5,)
    assert res.value.shape == (5, 1)

    probs = wrapper.get_action_probs(obs)
    assert probs.shape == (5, 2)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(5))


def test_hub_view_modes_include_standalone_plugin_packages(qapp):
    """Verify Component Hub can manage standalone packages and Plugin Hub stays plugin-only."""
    from frontend.hub.client import HubClient
    from frontend.hub.dialog import HubView
    from frontend.hub.models import HubComponent

    client = HubClient(workspace_dir=Path.cwd(), data_dir=Path.home() / ".thetaide")
    # Mock components across all kinds
    client.components = [
        HubComponent(id="obsidian-notes", name="Obsidian", kind="plugin", version="0.1.0"),
        HubComponent(id="antigravity", name="Antigravity", kind="plugin", version="0.1.5"),
        HubComponent(id="dnn", name="DNN", kind="model", version="1.0.0"),
        HubComponent(id="sb3-suite", name="SB3 Suite", kind="method", version="1.0.0"),
        HubComponent(id="atari-env", name="Atari", kind="env", version="1.0.0"),
        HubComponent(id="basic_suite", name="Basic Suite", kind="suite", version="1.0.0"),
    ]

    # 1. Component Hub (mode="components")
    view_comp = HubView(client=client, mode="components")
    view_comp._populate_cards(client.components)
    comp_kinds = {c.component.kind for c in view_comp.cards.values()}
    assert "plugin" in comp_kinds
    assert "suite" in comp_kinds
    assert "model" in comp_kinds
    assert "method" in comp_kinds
    assert "env" in comp_kinds

    pills_comp = list(view_comp.pills.keys())
    assert "experiment" in pills_comp
    assert "suite" in pills_comp
    assert "plugin" in pills_comp
    assert pills_comp == ["plugin", "experiment", "suite", "method", "model", "env", "data", "eval", "plot", "all"]

    # 2. Community Plugins Hub (mode="plugins")
    view_plug = HubView(client=client, mode="plugins")
    view_plug._populate_cards(client.components)
    plug_kinds = {c.component.kind for c in view_plug.cards.values()}
    assert plug_kinds == {"plugin"}

    pills_plug = list(view_plug.pills.keys())
    assert "plugin" in pills_plug
    assert "suite" not in pills_plug
    assert "method" not in pills_plug
    assert "model" not in pills_plug
    assert "env" not in pills_plug

    view_comp.deleteLater()
    view_plug.deleteLater()


