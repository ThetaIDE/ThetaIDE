"""Test tuner and scheduler packages (``tests/fixtures``) installed through the native lifecycle."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.core.provider_catalog import get_catalog
from src.app.workspace import WorkspacePaths

FIXTURES = Path(__file__).resolve().parent / 'fixtures'
#: Installable fixture packages by ID: a tuner, a scheduler and a packaged site, without third-party libraries.
PACKAGES = {'search-tuner': 'search_tuner', 'batch-scheduler': 'batch_scheduler', 'template-site': 'template_site'}


def build_fixture(package_id, output):
    """A native archive of the fixture package ``package_id``."""
    source = FIXTURES / PACKAGES[package_id]
    metadata = json.loads((source / '.theta_component.json').read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = output / f"{metadata['id']}-{metadata['version']}.zip"
    with ZipFile(target, 'w', ZIP_DEFLATED) as archive:
        for path in sorted(source.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                info = ZipInfo(path.relative_to(source).as_posix(), date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())
    return target


def install_fixture(paths, package_id, output):
    """Install the fixture package ``package_id`` into the workspace ``paths`` like a Hub release."""
    paths = paths if isinstance(paths, WorkspacePaths) else WorkspacePaths(paths)
    if get_catalog(paths).package(package_id) is not None: return
    archive = build_fixture(package_id, output)
    with ZipFile(archive) as bundle: metadata = json.loads(bundle.read('.theta_component.json'))
    component = HubComponent.from_dict(metadata)
    component.releases = {component.version: ReleaseInfo(component.version, archive.as_uri(), '')}
    HubInstaller(paths.root, paths.data_dir).install(component, install_dependencies=False)


def cluster_site(root, *, execution=None):
    """Author a two-partition batch site, `cluster`, in the workspace at `root`.

    `cluster/cpu` is a CPU partition without --time or --gres, `cluster/gpu` a consolidated
    GPU partition, and `cluster/default` inherits the shared defaults. `execution` adds a managed
    execution policy to `cluster/cpu`. Tests use it instead of any institution's real site package.
    """
    from src.app.sites.catalog import create
    root = root.root if isinstance(root, WorkspacePaths) else Path(root)
    defaults = {'scheduler': 'batch', 'python_executable': 'auto', 'venv_dir': 'venv',
                'mail_user': None, 'mail_type': 'END,FAIL', 'consolidate': False,
                'resources': {'time': None, 'nodes': 1, 'gpu_type': None, 'gres': None},
                'module_loads': [], 'extra_pythonpath': [],
                'extra_env_vars': {'OMP_NUM_THREADS': '1'}, 'extra_sbatch_flags': [],
                'dataset_search_dirs': []}
    variants = {
        'cpu': {'scheduler': 'batch', 'resources': {'partition': 'cpu', 'cores': 4,
                   'memory': '8G', 'gpus': 0, 'no_gres': True}},
        'gpu': {'scheduler': 'batch', 'consolidate': True, 'gres_format': 'gpu:{gpu_type}:{gpus}',
                'resources': {'partition': 'gpu', 'gpus': 1, 'cores': 16, 'memory': '32G',
                              'no_gres': False}, 'module_loads': ['cuda']},
    }
    if execution is not None:
        variants['cpu']['execution'] = dict(execution)
    return create(root, 'cluster', defaults, variants)


_SESSION_WORKSPACES = {}


def fixture_workspace_with(package_id):
    """A private workspace with one fixture package installed once per test session. Do not change it."""
    if package_id not in _SESSION_WORKSPACES:
        import tempfile
        root = Path(tempfile.mkdtemp(prefix=f'theta-{package_id}-'))
        paths = WorkspacePaths(root / 'workspace')
        install_fixture(paths, package_id, root / 'archives')
        _SESSION_WORKSPACES[package_id] = paths
    return _SESSION_WORKSPACES[package_id]


def search_plan(execution_id):
    """The serialized plan the search tuner fixture records for a study allocation."""
    from src.app.pipeline.tuner_execution import TunerArtifact, TunerExecutionPlan
    return TunerExecutionPlan('search', execution_id,
        artifacts=(TunerArtifact('study', f'search/{execution_id}/study.sqlite', True),),
        trials=40, storage_kind='managed').record()


class LocalHub:
    """Current storage-v3 releases of a Theta Hub checkout, installed from its packages/ directory.

    The checkout is the explicit THETA_HUB_ROOT; nothing beside the IDE checkout is read. Without
    it, or when it lacks a release, tests skip; they never download or pip-install anything.
    """

    def __init__(self, root=None):
        import os
        root = root or os.environ.get('THETA_HUB_ROOT')
        self.root = Path(root) if root else None
        self.entries = {}
        if self.root is None:
            return
        for path in sorted((self.root / 'registry').rglob('*.json')) if (self.root / 'registry').is_dir() else ():
            entry = json.loads(path.read_text())
            if entry.get('storage_version') == 3:
                self.entries[entry['id']] = entry

    def component(self, component_id):
        import pytest
        entry = self.entries.get(component_id)
        archive = self.root / 'packages' / Path(entry['releases'][entry['version']]['url']).name if entry else None
        if self.root is None:
            pytest.skip('Set THETA_HUB_ROOT to a Theta Hub checkout to run real-package tests')
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
