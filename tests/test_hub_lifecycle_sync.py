"""Native package ownership, package-owned profiles and Hub file-tree integration."""
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
        "env/mock_game/level1.yaml": "name: mock_game/level1\n",
        "env/mock_game/level2.yaml": "name: mock_game/level2\n",
    })
    installer = HubInstaller(workspace, data)
    installer.install(component, install_dependencies=False)
    source = installer.resolve_target_dir(component)
    return SimpleNamespace(installer=installer, component=component, source=source,
                           first=installer.paths.resolve_config("env/mock_game/level1.yaml"),
                           second=installer.paths.resolve_config("env/mock_game/level2.yaml"))


def test_install_resolves_profiles_from_the_package(environment):
    e = environment
    assert e.first == e.source / "configs/env/mock_game/level1.yaml"
    assert e.first.read_text() == "name: mock_game/level1\n"
    assert not (e.installer.paths.config / "env/mock_game").exists()


def test_find_component_for_source_path(environment):
    e = environment
    meta = e.installer.find_component_for_path(e.source)
    assert meta["id"] == e.component.id
    assert meta["name"] == e.component.name
    assert e.installer.find_component_for_path(e.first)["id"] == e.component.id


def test_find_component_returns_none_for_user_paths(environment):
    e = environment
    other = e.installer.paths.config / "experiment/my_exp"
    other.mkdir(parents=True)
    assert e.installer.find_component_for_path(other) is None
    assert e.installer.find_component_for_path(e.installer.paths.root) is None


def test_installed_files_are_distinct_from_user_copies(environment):
    e = environment
    configs = ConfigurationFiles(e.installer.paths)
    user_copy = configs.copy(e.first, "env/mock_game/custom")
    assert user_copy == e.installer.paths.config / "env/mock_game/custom.yaml"
    assert e.installer.is_installed_file(e.source / "provider.py")
    assert e.installer.is_installed_file(e.first)
    assert not e.installer.is_installed_file(user_copy)
    assert e.installer.is_installed_file(e.source.parent, include_descendants=True)
    assert not e.installer.is_installed_file(user_copy.parent, include_descendants=True)
    e.installer.protect_installed_components()
    assert (e.source / "provider.py").stat().st_mode & 0o222 == 0
    with pytest.raises(PermissionError, match="read-only"):
        configs.configure(e.first, ["name=changed"])
    configs.configure(user_copy, ["name=custom"])
    assert user_copy.stat().st_mode & 0o200


def test_uninstall_keeps_user_overrides_and_removes_package_profiles(environment):
    e = environment
    override = ConfigurationFiles(e.installer.paths).create_override("env/mock_game/level1.yaml")
    override.write_text("# personal\nname: custom_level\n")
    assert e.installer.uninstall(e.component)
    assert not e.source.exists()
    assert override.read_text() == "# personal\nname: custom_level\n"
    assert e.installer.paths.config_resolver().resolve("env/mock_game/level2.yaml") is None
    assert e.installer.install(e.component, install_dependencies=False)
    assert e.installer.paths.resolve_config("env/mock_game/level1.yaml") == override
    assert e.installer.paths.resolve_config("env/mock_game/level2.yaml").read_text() == "name: mock_game/level2\n"


def test_uninstall_by_metadata(environment):
    e = environment
    meta = e.installer.find_component_for_path(e.first)
    assert meta is not None
    assert e.installer.uninstall_by_metadata(meta)
    assert not e.source.exists()


def test_dialog_instantiation():
    delete = ComponentDeleteDialog("env/mock_game", "Mock Game")
    assert delete.choice == "cancel"
    uninstall = UninstallComponentDialog("Mock Game")
    assert "kept" in " ".join(label.text() for label in uninstall.findChildren(QLabel))


def test_update_keeps_user_override_and_serves_new_package_defaults(tmp_path):
    import yaml

    workspace, data = tmp_path / "workspace", tmp_path / "data"
    workspace.mkdir()
    data.mkdir()
    installer = HubInstaller(workspace, data)
    component = native_component(tmp_path, configs={"model/example.yaml": "width: 32\n"})
    for version, config in (("1.1.0", "width: 32\nlogic_lr: null\n"),
                            ("1.2.0", "width: 32\nlogic_lr: null\nblender_lr: null\n")):
        component.releases.update(native_component(tmp_path, version=version,
            configs={"model/example.yaml": config}).releases)
    assert installer.install(component, "1.0.0")
    assert installer.install(component, "1.1.0")
    assert yaml.safe_load(installer.paths.resolve_config("model/example.yaml").read_text()) == {"width": 32, "logic_lr": None}
    override = ConfigurationFiles(installer.paths).create_override("model/example.yaml")
    override.write_text("# personal\nwidth: 64\n")
    assert installer.uninstall(component)
    assert installer.install(component, "1.2.0")
    # Overrides are complete user profiles; package updates never rewrite them.
    assert installer.paths.resolve_config("model/example.yaml") == override
    assert override.read_text() == "# personal\nwidth: 64\n"
