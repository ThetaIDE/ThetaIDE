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


def cluster_site(root, *, execution=None):
    """Author a two-partition Slurm site, `cluster`, in the workspace at `root`.

    `cluster/common` is a CPU partition without --time or --gres, `cluster/gpu` a consolidated
    GPU partition, and `cluster/default` inherits the shared defaults. `execution` adds a managed
    execution policy to `cluster/common`. Tests use it instead of any institution's real site package.
    """
    from src.app.sites.catalog import create
    root = root.root if isinstance(root, WorkspacePaths) else Path(root)
    defaults = {'scheduler': 'slurm', 'python_executable': 'auto', 'venv_dir': 'venv',
                'mail_user': None, 'mail_type': 'END,FAIL', 'consolidate': False,
                'resources': {'time': None, 'nodes': 1, 'gpu_type': None, 'gres': None},
                'module_loads': [], 'extra_pythonpath': [],
                'extra_env_vars': {'OMP_NUM_THREADS': '1'}, 'extra_sbatch_flags': [],
                'dataset_search_dirs': []}
    variants = {
        'common': {'scheduler': 'slurm', 'resources': {'partition': 'common', 'cores': 4,
                   'memory': '8G', 'gpus': 0, 'no_gres': True}},
        'gpu': {'scheduler': 'slurm', 'consolidate': True, 'gres_format': 'gpu:{gpu_type}:{gpus}',
                'resources': {'partition': 'gpu', 'gpus': 1, 'cores': 16, 'memory': '32G',
                              'no_gres': False}, 'module_loads': ['cuda']},
    }
    if execution is not None:
        variants['common']['execution'] = dict(execution)
    return create(root, 'cluster', defaults, variants)


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
        artifacts=(TunerArtifact('study', f'optuna/{execution_id}/study.sqlite', True),),
        trials=40, storage_kind='managed', distributions=('optuna', 'sqlalchemy'),
        continuation_notes='reinitialized from configured seed; suggestion sequence can differ').record()


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


def sweep_workspace(root):
    """A copy of the checkout workspace at `root` with Optuna installed, for Hydra sweep scripts.

    Returns the workspace and the `hydra.searchpath=[...]` override a script that composes
    `paths.config` needs to see package and framework profiles (sites, the Optuna sweepers).
    Run the script with `THETA_WORKSPACE` set to `paths.root`.
    """
    import shutil
    from conftest import checkout_workspace
    root = Path(root)
    shutil.copytree(checkout_workspace(), root, dirs_exist_ok=True)
    paths = WorkspacePaths(root)
    install_optional(paths, 'optuna', root.parent / (root.name + '-archives'))
    roots = [path for path in paths.config_roots() if path != paths.config]
    return paths, 'hydra.searchpath=[' + ','.join('file://' + str(path) for path in roots) + ']'


def sweeper_args(storage, study, n_trials, sampler="TPESampler", **settings):
    """Every Hydra sweeper value, as the Optuna tuner compiles them from a tuning block."""
    values = dict(storage=storage, study_name=study, n_trials=n_trials, n_jobs=1, direction="minimize",
                  pruner="none", pruner_startup_trials=5, pruner_warmup_steps=0, timeout_seconds=0,
                  max_failed_trials=0)
    values.update(settings)
    return ["hydra/sweeper=theta_optuna", f"++hydra.sweeper.sampler._target_=optuna.samplers.{sampler}",
            *(f"hydra.sweeper.{key}={value}" for key, value in values.items())]


#: The Optuna component's own Hydra configs, for tests that launch Hydra directly.
OPTUNA_CONFIGS = Path(__file__).resolve().parents[1] / "optional_components/optuna/configs"
