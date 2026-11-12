"""Form-scoped reuse must never hide changes from the next operation."""
import json
from pathlib import Path

import pytest

from src.app.core.provider_catalog import get_catalog
from src.app.core.provider_discovery import current_snapshot, provider_discovery_snapshot
from src.app.core.provider_loader import invalidate
from src.app.workspace import WorkspacePaths


def package(root, name="toy"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.py").write_text("raise AssertionError('metadata discovery executed code')\n")
    manifest = root / ".theta_component.json"
    manifest.write_text(json.dumps({
        "id": name, "name": name, "kind": "model", "version": "1.0.0",
        "storage_version": 2, "api_version": 1,
        "payload": {"implementations": ["model.py"], "resources": [], "configs": {}},
        "capabilities": [{"id": name, "kind": "model", "entry_point": "model.py:Model"}],
    }))
    return manifest


@pytest.fixture
def paths(tmp_path):
    return WorkspacePaths(tmp_path / "workspace", application=tmp_path / "application")


def test_one_walk_and_catalog_check_per_operation(paths, monkeypatch):
    manifest = package(paths.user("model") / "toy")
    original = Path.rglob
    walks = []

    def walk(root, pattern):
        if pattern == ".theta_component.json":
            walks.append(root)
        return original(root, pattern)

    monkeypatch.setattr(Path, "rglob", walk)
    with provider_discovery_snapshot():
        catalog = get_catalog(paths)
        assert catalog.package("toy") is not None
        for _ in range(217):
            with provider_discovery_snapshot():
                assert get_catalog(WorkspacePaths(paths.root, application=paths.application)) is catalog
                discovered = paths.manifests()
                assert discovered == [manifest]
                discovered.clear()  # callers cannot mutate the cached inventory
    assert walks == [paths.user("model")]
    with provider_discovery_snapshot():
        assert get_catalog(paths) is catalog
    assert walks == [paths.user("model"), paths.user("model")]
    assert current_snapshot() is None


def test_external_edits_additions_removals_and_atomic_replacement(paths):
    manifest = package(paths.user("model") / "toy")
    with provider_discovery_snapshot():
        first = get_catalog(paths)
    meta = json.loads(manifest.read_text())
    meta["version"] = "2.0.0"
    replacement = manifest.with_suffix(".tmp")
    replacement.write_text(json.dumps(meta))
    replacement.replace(manifest)
    with provider_discovery_snapshot():
        assert get_catalog(paths).package("toy").version == "2.0.0"
        assert get_catalog(paths) is not first
    added = package(paths.user("model") / "nested/new", "new")
    with provider_discovery_snapshot():
        assert get_catalog(paths).package("new") is not None
    added.unlink()
    manifest.unlink()
    with provider_discovery_snapshot():
        assert get_catalog(paths).package("new") is None
        assert get_catalog(paths).package("toy") is None
    # Unscoped callers retain immediate freshness checks too.
    package(paths.user("model") / "toy")
    assert get_catalog(paths).package("toy") is not None


def test_workspace_state_application_and_extra_roots_are_isolated(paths, tmp_path):
    package(paths.user("model") / "toy")
    other = WorkspacePaths(tmp_path / "other", application=paths.application)
    package(other.user("model") / "other", "other")
    extra = tmp_path / "extra"
    package(extra / "addon", "addon")
    alternate_state = WorkspacePaths(paths.root, tmp_path / "state", paths.application)
    alternate_state.data_dir.mkdir()
    (alternate_state.data_dir / "provider_overrides.json").write_text('{"model:toy": "other"}')
    alternate_app = WorkspacePaths(paths.root, application=tmp_path / "other-app")
    package(alternate_app.bundled_components / "bundled", "bundled")
    with provider_discovery_snapshot():
        first = get_catalog(paths)
        assert get_catalog(other).package("other") is not None
        assert first.package("other") is None
        assert get_catalog(paths, [extra]).package("addon") is not None
        assert first.package("addon") is None
        assert get_catalog(alternate_state).overrides != first.overrides
        assert get_catalog(alternate_app).package("bundled") is not None
        assert first.package("bundled") is None
    addon_manifest = extra / "addon/.theta_component.json"
    addon_manifest.unlink()
    with provider_discovery_snapshot():
        assert get_catalog(paths, [extra]).package("addon") is None


def test_overrides_inventory_and_plugin_metadata_refresh(paths):
    manifest = package(paths.user("model") / "toy")
    meta = json.loads(manifest.read_text())
    meta["capabilities"] = [{"id": "panel", "kind": "plugin", "entry_point": "model.py:Model"}]
    manifest.write_text(json.dumps(meta))
    plugin = manifest.parent / "plugin.json"
    plugin.write_text('{"label": "Before"}')
    with provider_discovery_snapshot():
        before = get_catalog(paths)
        assert before.resolve("plugin", "panel")[1].metadata["label"] == "Before"
    plugin.write_text('{"label": "After change"}')
    with provider_discovery_snapshot():
        assert get_catalog(paths).resolve("plugin", "panel")[1].metadata["label"] == "After change"
    paths.data_dir.mkdir()
    (paths.data_dir / "provider_overrides.json").write_text('{"plugin:panel": "toy"}')
    with provider_discovery_snapshot():
        assert get_catalog(paths).overrides == {"plugin:panel": "toy"}
    inventory = paths.application / "src/plugins/inventory.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text('{"packages": [{"id": "test", "root": "unused"}]}')
    with provider_discovery_snapshot():
        after = get_catalog(paths)
        assert after.resolve("plugin", "panel")[1].metadata["label"] == "After change"
        assert after.overrides == {"plugin:panel": "toy"}
        assert after.distribution_inventory == [{"id": "test", "root": "unused"}]


def test_explicit_lifecycle_invalidation_inside_nested_scope(paths):
    manifest = package(paths.user("model") / "toy")
    with provider_discovery_snapshot():
        first = get_catalog(paths)
        with provider_discovery_snapshot():
            manifest.unlink()
            invalidate(paths, manifest.parent)
            assert get_catalog(paths).package("toy") is None
            assert get_catalog(paths) is not first
        assert get_catalog(paths).package("toy") is None


def test_exception_discards_snapshot(paths):
    with pytest.raises(RuntimeError, match="failed build"):
        with provider_discovery_snapshot():
            assert get_catalog(paths).package("toy") is None
            raise RuntimeError("failed build")
    assert current_snapshot() is None
    package(paths.user("model") / "toy")
    with provider_discovery_snapshot():
        assert get_catalog(paths).package("toy") is not None
