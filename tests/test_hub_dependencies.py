"""Native dependency packages install and retain independent ownership."""

import pytest

from src.app.components.installer import HubInstaller

from hub_test_support import native_component


def test_dependency_lookup_uses_the_requested_package(tmp_path):
    suite = native_component(tmp_path, "suite", kind="suite", dependencies=["child"])
    child = native_component(tmp_path, "child")
    catalog = {suite.id: suite, child.id: child}
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state", lookup_component_fn=catalog.get)
    assert installer.lookup_component("child") is child
    assert installer.lookup_component("suite") is suite
    assert installer.lookup_component("missing") is None


def test_recursive_cross_kind_dependencies_have_independent_ownership(tmp_path):
    data = native_component(tmp_path, "image-data", kind="datamodule", configs={
        "in/config/data/image_data.yaml": "data_module: image-data\n",
    })
    environment = native_component(tmp_path, "example-env", kind="env", dependencies=[data.id], configs={
        "in/config/env/example.yaml": "name: example-env\n",
    })
    panel = native_component(tmp_path, "example-panel", kind="plugin", dependencies=[data.id], files={
        "provider.py": "class Provider: pass\n",
        "plugin.json": '{"id": "example-panel", "entry_point": "Provider"}',
    })
    suite = native_component(tmp_path, "example-suite", kind="suite", dependencies=[environment.id, panel.id])
    catalog = {component.id: component for component in (suite, environment, panel, data)}
    installed = []
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state",
        lookup_component_fn=catalog.get, on_change_callback=lambda package_id, action: installed.append(package_id))
    assert installer.install(suite)
    assert set(installed) == set(catalog)
    assert installed.count(data.id) == 1
    for component in catalog.values():
        assert installer.check_installed(component) == (True, component.version)
        assert installer.resolve_target_dir(component).is_dir()
    assert installer.resolve_target_dir(panel).parent == installer.paths.installed_app
    assert not installer.paths.research.exists()
    assert not (installer.paths.root / "in/envs").exists()
    assert installer.uninstall(suite)
    for component in (environment, panel, data):
        assert installer.resolve_target_dir(component).is_dir()
    with pytest.raises(ValueError, match="required by"):
        installer.uninstall(data)
    assert installer.uninstall(panel)
    assert installer.uninstall(environment)
    assert installer.uninstall(data)
    assert not (installer.paths.config / "data/image_data.yaml").exists()
    assert not (installer.paths.config / "env/example.yaml").exists()


def test_missing_dependency_fails_before_installation(tmp_path):
    component = native_component(tmp_path, dependencies=["missing"])
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state", lookup_component_fn=lambda _: None)
    with pytest.raises(ValueError, match="Missing dependency"):
        installer.install(component)
    assert not installer.paths.installed_components.exists()


def test_dependency_version_mismatch_fails_before_installation(tmp_path):
    child = native_component(tmp_path, "child", version="1.0.0")
    parent = native_component(tmp_path, dependencies=[{"id": child.id, "version": ">=2.0.0"}])
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state", lookup_component_fn=lambda _: child)
    with pytest.raises(ValueError, match="does not satisfy"):
        installer.install(parent)
    assert not installer.paths.installed_components.exists()


def test_dependency_cycle_fails_before_installation(tmp_path):
    first = native_component(tmp_path, "first", dependencies=["second"])
    second = native_component(tmp_path, "second", dependencies=["first"])
    catalog = {first.id: first, second.id: second}
    installer = HubInstaller(tmp_path / "workspace", tmp_path / "state", lookup_component_fn=catalog.get)
    with pytest.raises(ValueError, match="dependency cycle"):
        installer.install(first)
    assert not installer.paths.installed_components.exists()
