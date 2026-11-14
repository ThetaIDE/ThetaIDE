import json

import pytest

from src.app.workspace import WorkspacePaths
from src.app.pipeline.datasets import resolve_dataset_path, resolve_dataset_for_method, ensure_online_dataset_path
from src.app.workspace_inventory import inventory


def test_dataset_discovery_is_workspace_scoped_and_does_not_cache_absence(tmp_path, monkeypatch):
    first, second = (WorkspacePaths(tmp_path / name, application=tmp_path / "app") for name in ("one", "two"))
    unrelated = tmp_path / "cwd"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    for paths in (first, second):
        target = paths.datasets / "iris.npz"
        assert resolve_dataset_path("iris", paths=paths) == paths.datasets / "iris"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"dataset sentinel")
        assert resolve_dataset_path("iris", paths=paths) == target
        assert resolve_dataset_for_method("method", {}, {"env": {"dataset_name": "iris"}}, paths=paths) == target
        generated, exists = ensure_online_dataset_path("g", "e", "m", paths=paths)
        assert generated == str(paths.datasets / "g/e/m") and not exists


def test_explicit_external_input_is_preserved(tmp_path):
    paths = WorkspacePaths(tmp_path / "workspace")
    external = tmp_path / "external.npz"
    external.write_bytes(b"external sentinel")
    assert resolve_dataset_path("iris", yaml_ds_path=str(external), paths=paths) == external
    assert resolve_dataset_for_method("method", {}, {"dataset_path": str(external)}, paths=paths) == external
    with pytest.raises(ValueError, match="dataset_path"):
        resolve_dataset_path("../external", paths=paths)


def test_inventory_preserves_ignored_bytes_modes_and_external_references(tmp_path):
    paths = WorkspacePaths(tmp_path / "workspace", application=tmp_path / "application")
    dataset = paths.datasets / "ignored.npz"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(b"ignored local input")
    dataset.chmod(0o640)
    recipe = paths.config / "experiment/ignored.yaml"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("dataset_path: /external/data.npz\n")
    first = inventory(paths)
    assert inventory(paths) == first
    record = next(item for item in first["files"] if item["path"] == "in/datasets/ignored.npz")
    assert record["mode"] == 0o640
    assert record["intended_destination"] == "src/usr/datasets/ignored.npz"
    assert record["tracked"] is False
    assert first["dataset_references"] == [{"file": "in/config/experiment/ignored.yaml", "field": "dataset_path",
                                               "value": "/external/data.npz", "external": True}]
    assert dataset.read_bytes() == b"ignored local input"
    assert not paths.user_inputs.exists()


def test_plot_and_export_consume_effective_profiles_from_two_workspaces(tmp_path, monkeypatch):
    from src.app.core.plotting.base import BasePlotter
    from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions
    from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
    from src.app.components.configuration import ConfigurationFiles
    from test_config_provenance import provider
    from zipfile import ZipFile
    unrelated = tmp_path / 'launch'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    for width, name in enumerate(('one', 'two'), 17):
        paths = WorkspacePaths(tmp_path / name)
        provider(paths, 'independent', width)
        (paths.user_config / 'experiment/probe').mkdir(parents=True)
        (paths.user_config / 'experiment/probe/run.yaml').write_text(f'group: probe\nseed: {width}\n')
        override = ConfigurationFiles(paths).create_override('model/fixture/net')
        override.write_text(f'architecture: net\nwidth: {width + 10}\n')
        result = ComponentExporter(paths).export(override, ComponentExportOptions(output_path=paths.root / 'model.zip'))
        assert result.success, result.error
        with ZipFile(result.export_path) as archive:
            assert f'width: {width + 10}' in archive.read('configs/model/fixture/net.yaml').decode()
        exported = ExperimentExporter(paths).export('probe/run', ExportOptions(output_path=paths.root / 'recipe.zip'))
        assert exported.success, exported.error
        with ZipFile(exported.archive_path) as archive:
            assert f'seed: {width}' in archive.read('configs/experiment/probe/run.yaml').decode()
        plotter = BasePlotter('manager', paths=paths)
        assert plotter.get_experiment_config('probe/run')['seed'] == width
        saved = paths.results / 'logs/probe/run/config.yaml'
        saved.parent.mkdir(parents=True)
        saved.write_text(f'group: probe\nseed: {width + 100}\n')
        assert plotter.get_experiment_config('probe/run')['seed'] == width + 100
    assert not list(unrelated.iterdir())


def test_site_template_configuration_creates_user_override(tmp_path):
    from src.app.sites.catalog import configure, load, defaults_path
    from component_fixtures import install_optional
    paths = WorkspacePaths(tmp_path)
    install_optional(paths, "slurm", tmp_path / "archives")
    resolved = paths.config_resolver().resolve('site/ncshare/_defaults.yaml', required=True)
    assert resolved.package.id == 'slurm'
    original = resolved.file.read_bytes()
    configure(paths, 'ncshare/gpu', {'host': 'test-alias', 'remote_root': '/scratch/test'})
    assert resolved.file.read_bytes() == original
    assert defaults_path(paths, 'ncshare/gpu') == paths.user_config / 'site/ncshare/_defaults.yaml'
    assert load(paths, 'ncshare/gpu')['connection']['host'] == 'test-alias'
    assert not paths.user('site').exists()


def test_api1_plotter_adapter_scopes_and_releases_workspace(tmp_path):
    from src.app.core.plotting.base import BasePlotter, create_plotter
    class OldPlotter(BasePlotter):
        def __init__(self): super().__init__('manager')
    paths = WorkspacePaths(tmp_path)
    assert create_plotter(OldPlotter, paths).workspace == paths
    from src.app.core.plotting.base import _PLOTTER_WORKSPACE
    assert _PLOTTER_WORKSPACE.get() is None
    class BrokenPlotter:
        def __init__(self): raise RuntimeError('constructor failure')
    with pytest.raises(RuntimeError, match='constructor failure'):
        create_plotter(BrokenPlotter, paths)
    assert _PLOTTER_WORKSPACE.get() is None


def test_plot_worker_environment_uses_explicit_workspace(tmp_path, monkeypatch):
    from src.app.pipeline.runtime import get_subprocess_env, get_pythonpath_entries
    first, second = WorkspacePaths(tmp_path / 'one'), WorkspacePaths(tmp_path / 'two')
    monkeypatch.setenv('THETA_WORKSPACE', str(first.root))
    env = get_subprocess_env(paths=second)
    assert env['THETA_WORKSPACE'] == str(second.root)
    assert env['THETA_DATA_DIR'] == str(second.data_dir)
    assert get_pythonpath_entries(paths=second)[0] == str(second.application)


def test_editor_group_metadata_uses_usr_profile(tmp_path):
    from frontend.config_model import ConfigTree
    paths = WorkspacePaths(tmp_path)
    base = paths.user_config / 'experiment/example/_base.yaml'
    base.parent.mkdir(parents=True)
    base.write_text('paradigm: supervised\nenv: example\nmethods: {first: {model: example}}\n')
    group = ConfigTree(paths.config, workspace=paths).experiment_groups['example']
    assert group.has_base and group.paradigm == 'supervised'
    assert group.env == 'example' and group.base_methods['first']['model'] == 'example'
    assert not paths.config.exists()
