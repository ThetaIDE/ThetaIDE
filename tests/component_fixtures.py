"""Explicit native optional-package setup for integration tests."""
import json
from pathlib import Path
from zipfile import ZipFile
from scripts.package_optional_components import build
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.core.provider_catalog import get_catalog
from src.app.workspace import WorkspacePaths


def install_optional(paths, name, output):
    paths = paths if isinstance(paths, WorkspacePaths) else WorkspacePaths(paths)
    if get_catalog(paths).package(name) is not None: return
    source = Path(__file__).resolve().parents[1] / 'optional_components' / name
    archive = build(source, output)
    with ZipFile(archive) as bundle: metadata = json.loads(bundle.read('.theta_component.json'))
    component = HubComponent.from_dict(metadata)
    component.releases = {component.version: ReleaseInfo(component.version, archive.as_uri(), '')}
    HubInstaller(paths.root, paths.data_dir).install(component, install_dependencies=False)


_SESSION_WORKSPACES = {}


def optional_workspace(name):
    """A private workspace with one optional component installed once per test session."""
    if name not in _SESSION_WORKSPACES:
        import tempfile
        root = Path(tempfile.mkdtemp(prefix=f'theta-optional-{name}-'))
        paths = WorkspacePaths(root / 'workspace')
        install_optional(paths, name, root / 'archives')
        _SESSION_WORKSPACES[name] = paths
    return _SESSION_WORKSPACES[name]


def optional_module(name, relative):
    """Load `relative` from an installed optional component, never from a checkout path."""
    from src.app.core.provider_loader import load_module
    paths = optional_workspace(name)
    return load_module(get_catalog(paths).package(name), relative, paths=paths)


def optuna_plan(execution_id):
    """The serialized managed-SQLite plan the Optuna component records for a study allocation."""
    from src.app.pipeline.tuner_execution import TunerArtifact, TunerExecutionPlan
    return TunerExecutionPlan('optuna', execution_id,
        artifacts=(TunerArtifact('study', f'results/optuna/{execution_id}/study.sqlite', True),),
        trials=40, storage_kind='managed', distributions=('optuna', 'sqlalchemy'),
        continuation_notes='reinitialized from configured seed; suggestion sequence can differ').record()


class LocalHub:
    """Current storage-v3 releases of a Theta Hub checkout, installed from its packages/ directory.

    The checkout is THETA_HUB_ROOT, else ../theta-hub beside the IDE. Tests that need a release it
    lacks skip; they never download or pip-install anything.
    """

    def __init__(self, root=None):
        import os
        self.root = Path(root or os.environ.get('THETA_HUB_ROOT') or Path(__file__).resolve().parents[2] / 'theta-hub')
        self.entries = {}
        for path in sorted((self.root / 'registry').rglob('*.json')) if (self.root / 'registry').is_dir() else ():
            entry = json.loads(path.read_text())
            if entry.get('storage_version') == 3:
                self.entries[entry['id']] = entry

    def component(self, component_id):
        import pytest
        entry = self.entries.get(component_id)
        archive = self.root / 'packages' / Path(entry['releases'][entry['version']]['url']).name if entry else None
        if archive is None or not archive.is_file():
            pytest.skip(f'No storage-v3 release of {component_id} in {self.root}')
        release = entry['releases'][entry['version']]
        component = HubComponent.from_dict(entry)
        component.releases = {entry['version']: ReleaseInfo(release['tag'], archive.as_uri(), release['sha256'])}
        return component

    def install(self, installer, component_id):
        """Install a release after its component dependencies, without Python dependencies."""
        from src.app.components.lifecycle import dependencies
        component = self.component(component_id)
        for dependency, _ in dependencies(self.entries[component_id]):
            if get_catalog(installer.paths).package(dependency) is None:
                self.install(installer, dependency)
        assert installer.install(component, install_dependencies=False)
        return component
