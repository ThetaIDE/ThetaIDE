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
