"""Native package ownership, config retention and Hub file-tree integration."""
from types import SimpleNamespace

import pytest
from PyQt6.QtWidgets import QApplication, QLabel

from frontend.config_tree import ComponentDeleteDialog
from frontend.hub.dialog import UninstallComponentDialog
from src.app.components.configuration import ConfigurationFiles
from src.app.components.installer import HubInstaller

from hub_test_support import native_component

app = QApplication.instance() or QApplication([])


@pytest.fixture
def environment(tmp_path):
    workspace, data = tmp_path / "workspace", tmp_path / "state"
    component = native_component(tmp_path, "mock-game", kind="env", configs={
        "in/config/env/mock_game/level1.yaml": "name: mock_game/level1\n",
        "in/config/env/mock_game/level2.yaml": "name: mock_game/level2\n",
    })
    installer = HubInstaller(workspace, data)
    installer.install(component, install_dependencies=False)
    source = installer.resolve_target_dir(component)
    config_dir = installer.paths.config / "env/mock_game"
    return SimpleNamespace(installer=installer, component=component, source=source,
                           config_dir=config_dir, first=config_dir / "level1.yaml",
                           second=config_dir / "level2.yaml")


def test_has_configs_detection(environment):
    e = environment
    assert e.installer.has_configs(e.component)
    e.first.unlink()
    e.second.unlink()
    assert not e.installer.has_configs(e.component)


def test_find_component_for_source_path(environment):
    e = environment
    meta = e.installer.find_component_for_path(e.source)
    assert meta["id"] == e.component.id
    assert meta["name"] == e.component.name


def test_find_component_for_config_dir_path(environment):
    e = environment
    meta = e.installer.find_component_for_path(e.config_dir)
    assert meta is not None
    assert meta["id"] == e.component.id


def test_find_component_for_config_file_path(environment):
    e = environment
    assert e.installer.find_component_for_path(e.first)["id"] == e.component.id


def test_find_component_returns_none_for_unrelated_path(environment):
    e = environment
    other = e.installer.paths.config / "experiment/my_exp"
    other.mkdir(parents=True)
    assert e.installer.find_component_for_path(other) is None


def test_shared_config_folder_reports_all_owners(environment, tmp_path):
    e = environment
    other = native_component(tmp_path, "other-game", kind="env", configs={
        "in/config/env/mock_game/other.yaml": "name: other-game\n",
    })
    e.installer.install(other, install_dependencies=False)
    meta = e.installer.find_component_for_path(e.config_dir)
    assert set(meta["shared_owners"]) == {e.component.id, other.id}
    assert e.installer.find_component_for_path(e.installer.paths.root) is None


def test_installed_files_are_distinct_from_user_copies(environment):
    e = environment
    configs = ConfigurationFiles(e.installer.paths)
    user_copy = configs.copy(e.first, "env/mock_game/custom")
    assert e.installer.is_installed_file(e.source / "provider.py")
    assert e.installer.is_installed_file(e.first)
    assert e.installer.is_installed_file(e.config_dir)
    assert not e.installer.is_installed_file(user_copy)
    e.installer.protect_installed_components()
    assert (e.source / "provider.py").stat().st_mode & 0o222 == 0
    with pytest.raises(PermissionError, match="read-only"):
        configs.configure(e.first, ["name=changed"])
    configs.configure(user_copy, ["name=custom"])
    assert user_copy.stat().st_mode & 0o200


def test_uninstall_retains_edited_configs_and_reinstall_restores_them(environment):
    e = environment
    e.first.write_text("# personal\nname: custom_level\n")
    assert e.installer.uninstall(e.component)
    assert not e.source.exists()
    assert not e.config_dir.exists()
    retained = e.installer.paths.data_dir / "retained_configs/mock-game/in/config/env/mock_game"
    assert (retained / "level1.yaml").read_text() == "# personal\nname: custom_level\n"
    # Unchanged shipped defaults are downloaded again rather than retained.
    assert not (retained / "level2.yaml").exists()
    assert e.installer.install(e.component, install_dependencies=False)
    assert e.first.read_text() == "# personal\nname: custom_level\n"
    assert e.second.read_text() == "name: mock_game/level2\n"


def test_uninstall_removes_unchanged_configs(environment):
    e = environment
    assert e.installer.uninstall(e.component)
    assert not e.source.exists()
    assert not e.first.exists()
    assert not e.second.exists()
    assert not e.config_dir.exists()


def test_uninstall_by_metadata(environment):
    e = environment
    meta = e.installer.find_component_for_path(e.config_dir)
    assert meta is not None
    assert e.installer.uninstall_by_metadata(meta)
    assert not e.source.exists()
    assert not e.config_dir.exists()


def test_dialog_instantiation():
    delete = ComponentDeleteDialog("in/config/env/mock_game", "Mock Game")
    assert delete.choice == "cancel"
    uninstall = UninstallComponentDialog("Mock Game")
    assert "restored" in " ".join(label.text() for label in uninstall.findChildren(QLabel))


def test_reinstall_merges_new_defaults_without_losing_user_values_or_comments(tmp_path):
    from frontend.hub.installer import _add_missing_yaml_defaults
    import yaml
    retained = "# user settings\nmodel:\n  width: 64 # chosen\n  nested:\n    keep: true\n"
    packaged = "model:\n  width: 32\n  nested:\n    keep: false\n    new: 4\n  logic_lr: null\nnew_top: 3\n"
    result = _add_missing_yaml_defaults(retained, packaged)
    assert "# user settings" in result and "# chosen" in result
    assert yaml.safe_load(result) == {"model": {"width": 64, "nested": {"keep": True, "new": 4}, "logic_lr": None}, "new_top": 3}


def test_update_and_reinstall_preserve_user_config_and_add_defaults(tmp_path):
    import yaml

    workspace, data = tmp_path / "workspace", tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    installer = HubInstaller(workspace, data)
    component = native_component(tmp_path, configs={"in/config/model/example.yaml": "width: 32\n"})
    for version, config in (("1.1.0", "width: 32\nlogic_lr: null\n"),
                            ("1.2.0", "width: 32\nlogic_lr: null\nblender_lr: null\n")):
        component.releases.update(native_component(tmp_path, version=version,
            configs={"in/config/model/example.yaml": config}).releases)
    path = workspace / "in/config/model/example.yaml"
    assert installer.install(component, "1.0.0")
    # Simulate a profile edited outside the IDE write guard.
    path.chmod(path.stat().st_mode | 0o200)
    path.write_text("# personal\nwidth: 64\n")
    assert installer.install(component, "1.1.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None}
    assert "# personal" in path.read_text()
    assert installer.uninstall(component)
    assert installer.install(component, "1.2.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None, "blender_lr": None}
    assert installer.uninstall(component)
    assert installer.install(component, "1.2.0")
    assert yaml.safe_load(path.read_text()) == {"width": 64, "logic_lr": None, "blender_lr": None}
