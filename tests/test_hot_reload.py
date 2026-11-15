"""Hot reload invalidates authored packages through their provider module prefix."""
import json

from src.app.core.hot_reload import reload_component_modules
from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_loader import module_prefix
from src.app.workspace import WorkspacePaths


def authored_model(workspace, name):
    root = workspace / "src/usr/models" / name
    root.mkdir(parents=True)
    (root / "model.py").write_text("class Model: pass\n")
    (root / ".theta_component.json").write_text(json.dumps({
        "id": name, "name": name, "kind": "model", "version": "1.0.0", "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": name, "kind": "model", "entry_point": "model.py:Model"}],
        "payload": {"implementations": ["model.py"], "resources": [], "configs": {}}}))


def test_unmatched_prefix_reloads_nothing(tmp_path, monkeypatch):
    authored_model(tmp_path, "lattice")
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    assert reload_component_modules(target_prefix="nonexistent.prefix.test.") == []


def test_authored_packages_reload_by_provider_prefix(tmp_path, monkeypatch):
    authored_model(tmp_path, "lattice")
    authored_model(tmp_path, "prism")
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    paths = WorkspacePaths.for_process()
    prefixes = {p.id: module_prefix(p, paths) for p in get_catalog(paths).packages if p.origin == "user"}
    assert set(prefixes) == {"lattice", "prism"}

    assert sorted(reload_component_modules()) == sorted(prefixes.values())
    assert reload_component_modules(target_prefix=prefixes["prism"]) == [prefixes["prism"]]


def test_directly_imported_usr_modules_are_not_reload_targets(tmp_path, monkeypatch):
    authored_model(tmp_path, "lattice")
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    assert reload_component_modules(target_prefix="src.usr.models.lattice") == []
