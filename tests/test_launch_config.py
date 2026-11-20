import os
import json
import shutil
import subprocess
import sys
import pytest

from src.app.pipeline.launch_config import frozen_config
from src.app.workspace import CONFIG_SNAPSHOT_ENV, WorkspacePaths
from src.app.pipeline.launch_inputs import INPUTS_SNAPSHOT_ENV


def workspace(tmp_path, monkeypatch):
    live = tmp_path / "src/usr/config"
    (live / "experiment/G").mkdir(parents=True)
    (live / "config.yaml").write_text("seed: null\n")
    (live / "experiment/G/sweep.yaml").write_text("total_timesteps: 120000\n")
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv(CONFIG_SNAPSHOT_ENV, raising=False)
    monkeypatch.delenv(INPUTS_SNAPSHOT_ENV, raising=False)
    return WorkspacePaths(tmp_path, data_dir=tmp_path / ".thetaide"), live


def test_a_launch_composes_from_its_snapshot_while_the_live_config_changes(tmp_path, monkeypatch):
    paths, live = workspace(tmp_path, monkeypatch)
    with frozen_config(paths) as snapshot:
        assert os.environ[CONFIG_SNAPSHOT_ENV] == str(snapshot)
        assert paths.config == snapshot and paths.primary_config == snapshot
        (live / "experiment/G/sweep.yaml").write_text("total_timesteps: 300000\n")
        selected = paths.config_file("src/usr/config/experiment/G/sweep.yaml")
        assert selected == (snapshot / "experiment/G/sweep.yaml").resolve()
        assert selected.read_text() == "total_timesteps: 120000\n"
        assert WorkspacePaths.from_config_root(snapshot).root == paths.root
        assert WorkspacePaths.from_config_root(paths.framework_config).root == paths.root
        with frozen_config(paths) as nested:  # a launch inside a launch keeps the outer snapshot
            assert nested == snapshot
    assert CONFIG_SNAPSHOT_ENV not in os.environ
    assert not snapshot.exists()
    assert paths.config == live


def test_other_workspaces_ignore_a_launch_snapshot(tmp_path, monkeypatch):
    paths, _ = workspace(tmp_path, monkeypatch)
    other = WorkspacePaths(tmp_path / "other")
    with frozen_config(paths):
        assert other.config == (tmp_path / "other/src/usr/config").resolve()
        assert other.launch_inputs_snapshot() is None


def test_stale_cleanup_preserves_active_and_foreign_launches(tmp_path, monkeypatch):
    from src.app.pipeline.launch_config import _remove_stale
    from src.app.components import transactions
    monkeypatch.setattr(transactions, "machine_id", lambda: "local")
    monkeypatch.setattr(transactions, "alive", lambda pid: pid == 42)
    for name, pid, machine in [("active", 42, "local"), ("dead", 43, "local"), ("foreign", 43, "remote")]:
        launch = tmp_path / name
        launch.mkdir()
        (launch / "owner.json").write_text(json.dumps({"pid": pid, "machine": machine}))
        os.utime(launch, (0, 0))
    _remove_stale(tmp_path)
    assert (tmp_path / "active").is_dir()
    assert (tmp_path / "foreign").is_dir()
    assert not (tmp_path / "dead").exists()


def test_workers_use_captured_packages_profiles_rules_and_overrides(tmp_path, monkeypatch):
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.workspace_resources import entry
    from src.app.pipeline.provider_worker import provider_snapshot, verify_snapshot

    paths, live = workspace(tmp_path, monkeypatch)
    package = paths.research / "models/toy"
    (package / "configs/model").mkdir(parents=True)
    (package / "model.py").write_text("from .helper import VALUE\nclass Model:\n    value = VALUE\n")
    (package / "helper.py").write_text("VALUE = 7\n")
    (package / "configs/model/toy.yaml").write_text("name: toy\nvalue: 7\n")
    (package / ".theta_component.json").write_text(json.dumps({
        "id": "toy", "name": "Toy", "kind": "model", "version": "1.0.0", "storage_version": 3, "api_version": 1,
        "capabilities": [{"id": "toy", "kind": "model", "entry_point": "model.py:Model"}],
        "workspace_resources": {"lemmas": {"depth": 2}},
        "payload": {"implementations": ["model.py", "helper.py"], "resources": [],
                    "configs": {"model/toy.yaml": "configs/model/toy.yaml"}}}))
    rule = paths.resource("lemmas") / "env/default/clauses.txt"
    rule.parent.mkdir(parents=True)
    rule.write_text("original rule\n")
    # External config symlinks must also become private file contents.
    external = tmp_path / "external.yaml"
    external.write_text("value: original\n")
    (live / "linked.yaml").symlink_to(external)
    with frozen_config(paths) as frozen:
        from src.app.components.transactions import assert_idle, provider_lease
        with provider_lease(paths):
            assert_idle(paths)  # Private-input workers do not lock live component edits.
        recorded = provider_snapshot(paths)
        frozen_package = get_catalog(paths).package("toy")
        assert frozen_package.origin == "user" and frozen_package.root != package
        assert (frozen / "linked.yaml").read_text() == "value: original\n"
        external.write_text("value: changed\n")
        shutil.rmtree(package)
        added = package.parent / "unrelated"
        shutil.copytree(frozen_package.root, added)
        meta = json.loads((added / ".theta_component.json").read_text())
        meta["id"] = "unrelated"
        meta["capabilities"][0]["id"] = "unrelated"
        (added / ".theta_component.json").write_text(json.dumps(meta))
        rule.write_text("edited rule\n")
        extra = rule.parent.parent / "unrelated/clauses.txt"
        extra.parent.mkdir()
        extra.write_text("new rule\n")
        (paths.data_dir / "provider_overrides.json").write_text('{"model:toy":"missing"}')
        (paths.data_dir / "extensions.json").write_text(json.dumps({"schema": "theta.extension-state/v1", "disabled": ["user:toy:model:toy"]}))
        (live / "experiment/G/sweep.yaml").write_text("total_timesteps: 999\n")
        assert entry("lemmas", "env/default", paths).joinpath("clauses.txt").read_text() == "original rule\n"
        verify_snapshot(recorded, paths)
        assert get_catalog(paths).package("unrelated") is None
        code = """
from src.app.workspace import WorkspacePaths
from src.app.core.provider_loader import resolve_provider
from src.app.core.provider_catalog import get_catalog
from src.app.core.workspace_resources import entry
p = WorkspacePaths.for_process()
assert resolve_provider('model', 'toy', p).value == 7
assert 'value: 7' in p.resolve_config('model/toy.yaml').read_text()
assert p.config.joinpath('experiment/G/sweep.yaml').read_text() == 'total_timesteps: 120000\\n'
assert p.config.joinpath('linked.yaml').read_text() == 'value: original\\n'
assert entry('lemmas', 'env/default', p).joinpath('clauses.txt').read_text() == 'original rule\\n'
assert get_catalog(p).overrides == {}
"""
        child = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        assert child.returncode == 0, child.stderr
        captured_rule = entry("lemmas", "env/default", paths) / "clauses.txt"
        captured_rule.write_text("corrupted private copy\n")
        with pytest.raises(RuntimeError, match="Workspace resources"):
            verify_snapshot(recorded, paths)
        with frozen_config(paths) as nested:
            assert nested == frozen
    assert INPUTS_SNAPSHOT_ENV not in os.environ
    assert get_catalog(paths).package("toy") is None
    assert rule.read_text() == "edited rule\n"
    # The next launch gets the changed workspace, not an old cached inventory.
    with frozen_config(paths):
        assert get_catalog(paths).package("toy") is None
        assert get_catalog(paths).package("unrelated") is not None
        assert entry("lemmas", "env/unrelated", paths).joinpath("clauses.txt").read_text() == "new rule\n"
        assert paths.config.joinpath("linked.yaml").read_text() == "value: changed\n"


def test_archive_only_workspace_can_snapshot_without_a_live_config_directory(tmp_path, monkeypatch):
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv(CONFIG_SNAPSHOT_ENV, raising=False)
    monkeypatch.delenv(INPUTS_SNAPSHOT_ENV, raising=False)
    paths = WorkspacePaths(tmp_path)
    live = paths.config

    with frozen_config(paths) as snapshot:
        assert snapshot.is_dir()
        assert paths.config == snapshot
        assert paths.primary_config == paths.framework_config

    assert not live.exists()
