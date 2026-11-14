"""Independent GUI fixture: edit, save, compose, export and plot outside the checkout."""
import pytest
from pathlib import Path
from zipfile import ZipFile

pytest.importorskip('PyQt6')
from PyQt6.QtWidgets import QApplication
from src.app.workspace import WorkspacePaths
from src.app.components.configuration import ConfigurationFiles
from src.app.pipeline.compose import compose_experiment
from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
from frontend.config_viewer import ConfigViewer
from frontend.config_model import ConfigTree
from frontend.figures.view import FigureWorkspace
from frontend.local_backend import LocalBackend
from test_config_provenance import provider, recipe


def test_two_desktops_keep_editor_backend_and_figures_in_their_workspace(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    unrelated = tmp_path / 'launch'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    viewers, figures, backends = [], [], []
    workspaces = [WorkspacePaths(tmp_path / name) for name in ('one', 'two')]
    try:
        for width, paths in enumerate(workspaces, 17):
            provider(paths, 'independent', width)
            recipe(paths)
            override = ConfigurationFiles(paths).create_override('model/fixture/net')
            viewer = ConfigViewer(workspace=paths)
            viewers.append(viewer)
            viewer.is_component_mode = True
            viewer.load_file(override, 'model/fixture/net.yaml')
            assert viewer.tree.workspace == paths
            assert viewer.raw_data['width'] == width
            viewer.document.set(('width',), width - 1)
            assert viewer.save_to_disk()
            assert compose_experiment('probe', workspace=paths).cfg.model.width == width - 1
            result = ComponentExporter(paths).export(override, ComponentExportOptions(output_path=paths.root / 'model.zip'))
            assert result.success, result.error
            with ZipFile(result.export_path) as archive:
                assert f'width: {width - 1}' in archive.read('configs/model/fixture/net.yaml').decode()
            fig = FigureWorkspace(paths=paths)
            figures.append(fig)
            assert fig.source.paths == paths
            backend = LocalBackend(workspace=paths, isolate_validation=True)
            backends.append(backend)
        # A second window must not redirect the first viewer's schema, save or validation worker.
        for width, paths, viewer, backend in zip((17, 18), workspaces, viewers, backends):
            assert viewer.tree.models['fixture/net'].parameters['width'] == width - 1
            report = backend._validator.call('compose', 'probe', [], None)
            assert report['config']['model']['width'] == width - 1
        assert not list(unrelated.iterdir())
    finally:
        for backend in backends: backend.shutdown()
        for widget in [*viewers, *figures]: widget.close(); widget.deleteLater()
        app.processEvents()


def test_optional_settings_lifecycle_and_clean_host(tmp_path):
    from frontend.extension_settings import ExtensionSettings
    from frontend.site_sync import SiteSyncControl
    from component_fixtures import install_optional
    from src.app.core.contributions import ContributionRegistry
    app = QApplication.instance() or QApplication([])
    paths = WorkspacePaths(tmp_path / 'workspace')
    paths.root.mkdir()
    host = ExtensionSettings('settings.site', workspace=paths)
    host.show()
    sync = SiteSyncControl('results', workspace=paths)
    sync.show()
    app.processEvents()
    assert not host.scope.active and host.maximumHeight() == 0
    assert not sync.button.isVisible()
    install_optional(paths, 'slurm', tmp_path / 'archives')
    host.reconcile()
    assert len(host.scope.active) == 1
    cfg = {'site': {'scheduler': 'slurm', 'connection': {'host': 'fixture', 'remote_root': '/scratch/test'}}}
    sync._config = cfg
    assert sync._refresh_availability()
    ContributionRegistry(paths).set_enabled('scheduler', 'slurm', False)
    host.reconcile()
    sync._refresh_availability()
    assert not host.scope.active and not sync.button.isVisible()
    ContributionRegistry(paths).set_enabled('scheduler', 'slurm', True)
    host.reconcile()
    assert host.scope.active
    host.close()
    host.reconcile()  # queued notifications cannot remount a closed host
    assert not host.scope.active and not host.timer.isActive()
    sync.close()
    host.deleteLater(); sync.deleteLater()
    app.processEvents()
