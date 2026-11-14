"""Optional components are installed through the same native lifecycle as fixtures."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from scripts.package_optional_components import build
from src.app.workspace import WorkspacePaths
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo
from src.app.core.provider_catalog import get_catalog
from src.app.core.contributions import ContributionRegistry
from src.app.sites.scheduler_api import get_scheduler

SOURCE = Path(__file__).resolve().parents[1] / 'optional_components'


def install(paths, name, output):
    archive = build(SOURCE / name, output)
    with ZipFile(archive) as bundle: meta = json.loads(bundle.read('.theta_component.json'))
    component = HubComponent.from_dict(meta)
    component.releases = {component.version: ReleaseInfo(component.version, archive.as_uri(), '')}
    installer = HubInstaller(paths.root, paths.data_dir)
    installer.install(component, install_dependencies=False)
    return installer


def test_clean_distribution_has_no_optional_providers(tmp_path):
    paths = WorkspacePaths(tmp_path)
    assert get_catalog(paths).resolve('tuner', 'optuna') is None
    assert get_catalog(paths).resolve('scheduler', 'slurm') is None
    assert not ContributionRegistry(paths).contributions('settings.site')
    assert not ContributionRegistry(paths).contributions('configuration.tuning')
    with pytest.raises(ValueError, match='not installed'):
        get_scheduler(cfg={'site': {'scheduler': 'slurm'}}, paths=paths)


def test_native_slurm_install_status_cancel_disable_and_preserved_override(tmp_path, monkeypatch):
    from src.app.sites.storage import write_json
    from src.app.sites.execution import receipt_path, refresh_status, cancel
    from src.app.components.configuration import ConfigurationFiles
    paths = WorkspacePaths(tmp_path / 'workspace')
    installer = install(paths, 'slurm', tmp_path / 'archives')
    cfg = {'site': {'scheduler': 'slurm', 'connection': {'host': 'fixture', 'remote_root': '/scratch/fixture'}}}
    provider = get_scheduler(cfg=cfg, paths=paths)
    assert provider.api_version == 1
    resources = provider.validate_allocation({'site': cfg['site'], 'trainer': {'num_nodes': 1}} , paths=paths)
    assert resources['nodes'] == 1
    record = {'schema_version': 1, 'scheduler_component': 'slurm', 'execution_id': 'a' * 32,
              'project_id': 'fixture', 'job_id': '123', 'state': 'running', 'config': cfg}
    write_json(receipt_path(paths.root, record['execution_id']), record)
    calls = []
    def command(args, **kwargs):
        calls.append(args)
        assert args[0] in ('squeue', 'sacct', 'scancel')
        return SimpleNamespace(stdout='123|CANCELLED|0:15|Requested by user|||||\n' if args[0] == 'sacct' else '', returncode=0)
    monkeypatch.setattr('subprocess.run', command)
    assert cancel(paths.root, record)['scheduler_state'] == 'cancelled'
    assert [args[0] for args in calls] == ['scancel', 'squeue', 'sacct']
    override = ConfigurationFiles(paths).create_override('site/ncshare/_defaults')
    override.write_text('# @package site\nscheduler: slurm\nconnection: {host: user-edited}\n')
    before = override.read_bytes()
    registry = ContributionRegistry(paths)
    assert registry.contributions('settings.site')
    assert registry.contributions('monitor.scheduler', config=cfg)
    registry.set_enabled('scheduler', 'slurm', False)
    assert not registry.contributions('settings.site')
    with pytest.raises(ValueError, match='disabled'):
        get_scheduler(cfg=cfg, paths=paths)
    registry.set_enabled('scheduler', 'slurm', True)
    from src.app.components.lifecycle import uninstall
    package = get_catalog(paths).package('slurm')
    assert uninstall(installer, {**package.manifest, 'target_dir': str(package.root)})
    assert override.read_bytes() == before
    assert not registry.contributions('settings.site')


def test_optuna_archive_has_setup_and_saved_import_identity(tmp_path, monkeypatch):
    from src.app.core.provider_loader import load_module
    paths = WorkspacePaths(tmp_path / 'workspace')
    install(paths, 'optuna', tmp_path / 'archives')
    monkeypatch.setenv('THETA_WORKSPACE', str(paths.root))
    registry = ContributionRegistry(paths)
    assert registry.contributions('configuration.tuning')
    package = get_catalog(paths).package('optuna')
    current = load_module(package, 'component.py', paths=paths)
    historical = importlib.import_module('src.app.tuners.optuna.component')
    assert historical.OptunaTuner is current.OptunaTuner
    assert current.OptunaTuner().api_version == 1


def test_independent_scheduler_uses_the_same_dispatch(tmp_path):
    paths = WorkspacePaths(tmp_path)
    root = paths.user('package') / 'independent-scheduler'
    root.mkdir(parents=True)
    (root / 'scheduler.py').write_text('class Scheduler:\n api_version = 1\n def validate_allocation(self, cfg, *, paths=None): return {"workspace": str(paths.root)}\n')
    (root / '.theta_component.json').write_text(json.dumps({'id': 'independent-scheduler', 'name': 'Independent', 'version': '1.0.0',
      'kind': 'package', 'storage_version': 2, 'api_version': 1,
      'capabilities': [{'id': 'lattice-batch', 'kind': 'scheduler', 'entry_point': 'scheduler.py:Scheduler', 'scheduler_api': 1}],
      'payload': {'implementations': ['scheduler.py'], 'resources': [], 'configs': {}}}))
    cfg = {'site': {'scheduler': 'lattice-batch'}}
    from src.app.sites.execution import validate_allocation
    assert validate_allocation(cfg, paths=paths) == {'workspace': str(paths.root)}


def test_installed_optuna_runs_a_real_study_without_upstream_sweeper(tmp_path, monkeypatch):
    pytest.importorskip('optuna')
    from src.app.core.provider_loader import load_module
    paths = WorkspacePaths(tmp_path / 'workspace')
    install(paths, 'optuna', tmp_path / 'archives')
    package = get_catalog(paths).package('optuna')
    sweeper = load_module(package, 'sweeper.py', paths=paths)
    from hydra.core.override_parser.overrides_parser import OverridesParser
    parser = OverridesParser.create()
    distribution = sweeper._distribution_from_override(parser.parse_override('rate=range(0.1,0.5,0.1)'))
    assert distribution.step == 0.1 and distribution.low == 0.1
    import optuna
    storage = load_module(package, 'utils.py', paths=paths).get_optuna_storage('sqlite:///' + str(tmp_path / 'study.sqlite'))
    study = optuna.create_study(storage=storage, study_name='standalone', direction='minimize')
    study.optimize(lambda trial: trial.suggest_float('rate', 0.1, 0.5), n_trials=2)
    assert len(study.trials) == 2
    assert study.best_value is not None


def test_site_aliases_come_from_provider_forwarding_files(tmp_path):
    from src.app.sites import profiles
    from src.app.sites.catalog import create, defaults_path, load, normalize, all_profiles
    paths = WorkspacePaths(tmp_path / 'workspace')
    install(paths, 'slurm', tmp_path / 'archives')
    available = all_profiles(paths)
    assert normalize('ncshare_common', available) == 'ncshare/common'
    assert normalize('arc', available) == 'arc/default'
    assert 'ncshare_common' not in profiles(paths) and 'ncshare/common' in profiles(paths)
    assert defaults_path(paths, 'ncshare_gpu') == defaults_path(paths, 'ncshare/gpu')
    # An authored variant that only inherits its defaults is a real profile, not an alias.
    create(paths.root, 'lab', {'scheduler': 'slurm'}, {'fast': {}})
    available = all_profiles(paths)
    assert normalize('lab/fast', available) == 'lab/fast' and 'lab/fast' in profiles(paths)
    assert load(paths, 'ncshare_common') == load(paths, 'ncshare/common')


def test_new_site_records_only_an_explicit_scheduler(tmp_path):
    import yaml
    from src.app.sites import scheduler
    from src.app.sites.catalog import create, load
    root = tmp_path / 'workspace'
    create(root, 'unset')
    defaults = (root / 'src/usr/sites/unset/configs/site/unset/_defaults.yaml').read_text()
    assert 'installed scheduler component' in defaults and yaml.safe_load(defaults)['scheduler'] is None
    with pytest.raises(ValueError, match='must name a scheduler capability'):
        scheduler({'site': load(root, 'unset/default')})
    create(root, 'chosen', scheduler='lattice-batch')
    assert scheduler({'site': load(root, 'chosen/default')}) == 'lattice-batch'


def test_site_check_uses_the_selected_schedulers_remote_requirements(tmp_path, monkeypatch):
    from src.app.sites import transfer
    from src.app.sites.catalog import create, configure
    paths = WorkspacePaths(tmp_path)
    root = paths.user('package') / 'independent-scheduler'
    root.mkdir(parents=True)
    (root / 'scheduler.py').write_text(
        'class Scheduler:\n api_version = 1\n'
        ' def remote_requirements(self, site):\n'
        '  return [("lattice-submit", None), ("lattice-acct", "accounting unavailable")]\n')
    (root / '.theta_component.json').write_text(json.dumps({'id': 'independent-scheduler', 'name': 'Independent',
      'version': '1.0.0', 'kind': 'package', 'storage_version': 2, 'api_version': 1,
      'capabilities': [{'id': 'lattice-batch', 'kind': 'scheduler', 'entry_point': 'scheduler.py:Scheduler', 'scheduler_api': 1}],
      'payload': {'implementations': ['scheduler.py'], 'resources': [], 'configs': {}}}))
    create(paths.root, 'lattice', scheduler='lattice-batch')
    configure(paths.root, 'lattice', {'host': 'fixture', 'remote_root': '/scratch/fixture'})
    monkeypatch.setattr(transfer, 'require_local_rsync', lambda config: 'rsync')
    scripts = []
    def ssh(config, arguments, **kwargs):
        scripts.append(arguments)
        return 'rsync  version 3.2.7\n' if arguments[0] == 'bash' else 'Python 3.11\n'
    monkeypatch.setattr(transfer, 'ssh', ssh)
    assert transfer.check(paths.root, 'lattice')['site'] == 'lattice'
    script = scripts[0][-1]
    assert script.startswith('command -v lattice-submit && rsync --version && test -w .')
    assert "command -v lattice-acct || printf '%s\\n' 'accounting unavailable'" in script
    assert 'sbatch' not in script
    (root / 'scheduler.py').write_text('class Scheduler:\n api_version = 1\n'
                                       ' def remote_requirements(self, site):\n  return [("x; rm -rf /", None)]\n')
    from src.app.core.provider_loader import invalidate
    invalidate(paths)
    with pytest.raises(ValueError, match='remote requirement'):
        transfer.check(paths.root, 'lattice')
