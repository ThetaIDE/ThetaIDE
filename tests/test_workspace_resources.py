"""Packages declare the workspace resource directories they read; core treats entries as opaque."""
import json

import pytest

from src.app.core.workspace_resources import entries, entry, roots, workspace_directories
from src.app.workspace import WorkspacePaths


def logic_package(paths, package_id="prover", name="lemmas", depth=2, packaged=("arith/basic",)):
    root = paths.user("package") / package_id
    root.mkdir(parents=True)
    (root / "prover.py").write_text("def load(**params):\n    return params\n")
    for item in packaged:
        (root / "resources" / name / item).mkdir(parents=True)
        (root / "resources" / name / item / "facts.txt").write_text(f"{item}\n")
    manifest = {"id": package_id, "name": package_id, "kind": "model", "version": "1.0.0", "storage_version": 3,
                "api_version": 1, "workspace_resources": {name: {"depth": depth, "path": f"resources/{name}"}},
                "capabilities": [{"id": package_id, "kind": "model", "entry_point": "prover.py:load",
                                  "parameter_schema": {"type": "object", "properties": {
                                      "facts": {"type": "string", "x-theta-resource": name,
                                                "x-theta-actions": [{"type": "open-in-editor", "label": "Open facts"}]}}}}],
                "payload": {"implementations": ["prover.py"], "resources": [f"resources/{name}"], "configs": {}}}
    (root / ".theta_component.json").write_text(json.dumps(manifest))
    return root


def authored(paths, name, item):
    directory = paths.resource(name) / item
    directory.mkdir(parents=True)
    (directory / "facts.txt").write_text(f"{item}\n")
    return directory


def test_entries_combine_the_users_directory_and_the_declaring_package(tmp_path):
    paths = WorkspacePaths(tmp_path)
    package = logic_package(paths)
    mine = authored(paths, "lemmas", "arith/mine")
    assert roots("lemmas", paths) == [paths.resource("lemmas"), package / "resources/lemmas"]
    assert entries("lemmas", paths) == {"arith/basic": (package / "resources/lemmas/arith/basic").resolve(),
                                         "arith/mine": mine.resolve()}
    assert entry("lemmas", "arith/mine", paths) == mine.resolve()
    assert workspace_directories(paths) == [paths.resource("lemmas")]
    # The resource name means nothing to core: an undeclared one is simply unknown.
    with pytest.raises(ValueError, match="No installed package declares"):
        entries("rules", paths)


def test_conflicting_entries_fail_and_identical_copies_do_not(tmp_path):
    paths = WorkspacePaths(tmp_path)
    logic_package(paths)
    copy = authored(paths, "lemmas", "arith/basic")
    assert entries("lemmas", paths)["arith/basic"]
    (copy / "facts.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="Duplicate lemmas entry 'arith/basic'"):
        entries("lemmas", paths)


@pytest.mark.parametrize("entry_id", ["../basic", "/tmp/evil", "arith/../../evil", "arith/", "", "arith\\basic"])
def test_entry_ids_cannot_escape(tmp_path, entry_id):
    paths = WorkspacePaths(tmp_path)
    logic_package(paths)
    with pytest.raises(ValueError):
        entry("lemmas", entry_id, paths)


def test_packages_must_agree_on_depth_and_cannot_claim_reserved_directories(tmp_path):
    from src.app.components.archive import validate_native_package
    paths = WorkspacePaths(tmp_path)
    logic_package(paths)
    logic_package(paths, "other", depth=1, packaged=("flat",))
    with pytest.raises(ValueError, match="different depths"):
        entries("lemmas", paths)
    root = paths.user("package") / "other"
    manifest = json.loads((root / ".theta_component.json").read_text())
    manifest["workspace_resources"] = {"config": {"depth": 1}}
    with pytest.raises(ValueError, match="config"):
        validate_native_package(manifest, root)


def test_editor_action_opens_the_selected_entry_or_the_users_directory(tmp_path):
    from frontend.parameter_actions import opening_targets
    paths = WorkspacePaths(tmp_path)
    package = logic_package(paths)
    paths.resource("lemmas").mkdir(parents=True)
    action = {"type": "open-in-editor", "label": "Open facts", "resource": "lemmas"}
    assert opening_targets(action, value="arith/basic", context={}, workspace=tmp_path) == [
        ("Open facts", (package / "resources/lemmas/arith/basic").resolve())]
    assert opening_targets(action, value="unlisted", context={}, workspace=tmp_path) == [
        ("Open facts", paths.resource("lemmas").resolve())]


def test_form_picker_lists_entries_and_keeps_an_unavailable_value(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer
    import frontend.config_viewer as viewer_module
    app = QApplication.instance() or QApplication([])  # noqa: F841 (kept alive for the widgets)
    paths = WorkspacePaths(tmp_path)
    logic_package(paths)
    authored(paths, "lemmas", "arith/mine")
    paths.config.mkdir(parents=True)
    tree = ConfigTree(paths.config)
    assert tree.resource_entries("lemmas") == ["arith/basic", "arith/mine"]
    monkeypatch.setattr(viewer_module, "config_tree", lambda *a, **k: tree)
    widget = ConfigViewer()
    try:
        written = []
        combo = widget._schema_value_widget("arith/mine", {"type": "string", "x-theta-resource": "lemmas"}, written.append)
        assert [combo.itemData(i) for i in range(combo.count())] == [None, "arith/basic", "arith/mine"]
        assert combo.currentData() == "arith/mine"
        combo.setCurrentIndex(1)
        assert written == ["arith/basic"]
        combo = widget._schema_value_widget("gone", {"type": "string", "x-theta-resource": "lemmas"}, written.append)
        assert combo.currentData() == "gone"
    finally:
        widget.close()


def test_bundles_carry_the_selected_packages_workspace_resources(tmp_path):
    from src.app.core.provider_catalog import get_catalog
    paths = WorkspacePaths(tmp_path)
    logic_package(paths)
    package = get_catalog(paths).package("prover")
    assert workspace_directories(paths, [package]) == [paths.resource("lemmas")]
    assert workspace_directories(paths, []) == []
