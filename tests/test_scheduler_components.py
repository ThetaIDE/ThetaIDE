"""Tuners and schedulers are installed components; the host dispatches to whichever is selected."""
import json

import pytest
from src.app.workspace import WorkspacePaths
from src.app.core.provider_catalog import get_catalog
from src.app.core.contributions import ContributionRegistry
from src.app.sites.scheduler_api import get_scheduler
from component_fixtures import install_fixture


def test_clean_distribution_has_no_tuner_or_scheduler_providers(tmp_path):
    paths = WorkspacePaths(tmp_path)
    assert get_catalog(paths).resolve('tuner', 'search') is None
    assert get_catalog(paths).resolve('scheduler', 'batch') is None
    assert not ContributionRegistry(paths).contributions('settings.site')
    assert not ContributionRegistry(paths).contributions('configuration.tuning')
    with pytest.raises(ValueError, match='not installed'):
        get_scheduler(cfg={'site': {'scheduler': 'batch'}}, paths=paths)


def test_installed_scheduler_status_cancel_disable_and_uninstall(tmp_path):
    from src.app.components.installer import HubInstaller
    from src.app.components.lifecycle import uninstall
    from src.app.sites.storage import write_json
    from src.app.sites.execution import receipt_path, cancel
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_fixture(paths, 'batch-scheduler', tmp_path / 'archives')
    cfg = {'site': {'scheduler': 'batch', 'connection': {'host': 'fixture', 'remote_root': '/scratch/fixture'}}}
    provider = get_scheduler(cfg=cfg, paths=paths)
    assert provider.api_version == 2
    assert provider.validate_allocation({'site': cfg['site'], 'trainer': {'num_nodes': 1}}, paths=paths)['nodes'] == 1
    record = {'schema_version': 1, 'scheduler_component': 'batch', 'execution_id': 'a' * 32,
              'project_id': 'fixture', 'job_id': '1', 'state': 'running', 'config': cfg}
    write_json(receipt_path(paths.root, record['execution_id']), record)
    assert cancel(paths.root, record)['scheduler_state'] == 'cancelled'
    registry = ContributionRegistry(paths)
    assert registry.contributions('settings.site')
    assert registry.contributions('monitor.scheduler', config=cfg)
    registry.set_enabled('scheduler', 'batch', False)
    assert not registry.contributions('settings.site')
    with pytest.raises(ValueError, match='disabled'):
        get_scheduler(cfg=cfg, paths=paths)
    registry.set_enabled('scheduler', 'batch', True)
    package = get_catalog(paths).package('batch-scheduler')
    assert uninstall(HubInstaller(paths.root, paths.data_dir), {**package.manifest, 'target_dir': str(package.root)})
    assert not registry.contributions('settings.site')


def test_uninstalling_a_site_package_preserves_the_users_override(tmp_path):
    from src.app.components.configuration import ConfigurationFiles
    from src.app.components.installer import HubInstaller
    from src.app.components.lifecycle import uninstall
    paths = WorkspacePaths(tmp_path / 'workspace')
    install_fixture(paths, 'template-site', tmp_path / 'archives')
    override = ConfigurationFiles(paths).create_override('site/lattice/_defaults')
    override.write_text('# @package site\nscheduler: batch\nconnection: {host: user-edited}\n')
    before = override.read_bytes()
    package = get_catalog(paths).package('template-site')
    assert uninstall(HubInstaller(paths.root, paths.data_dir), {**package.manifest, 'target_dir': str(package.root)})
    assert override.read_bytes() == before


def test_independent_scheduler_uses_the_same_dispatch(tmp_path):
    paths = WorkspacePaths(tmp_path)
    root = paths.user('package') / 'independent-scheduler'
    root.mkdir(parents=True)
    (root / 'scheduler.py').write_text('class Scheduler:\n api_version = 2\n def validate_allocation(self, cfg, *, paths=None): return {"workspace": str(paths.root)}\n')
    (root / '.theta_component.json').write_text(json.dumps({'id': 'independent-scheduler', 'name': 'Independent', 'version': '1.0.0',
      'kind': 'package', 'storage_version': 3, 'api_version': 1,
      'capabilities': [{'id': 'lattice-batch', 'kind': 'scheduler', 'entry_point': 'scheduler.py:Scheduler', 'scheduler_api': 2}],
      'payload': {'implementations': ['scheduler.py'], 'resources': [], 'configs': {}}}))
    cfg = {'site': {'scheduler': 'lattice-batch'}}
    from src.app.sites.execution import validate_allocation
    assert validate_allocation(cfg, paths=paths) == {'workspace': str(paths.root)}


def test_site_profiles_are_nested_and_a_bare_name_selects_default(tmp_path):
    from src.app.sites import profiles
    from src.app.sites.catalog import create, defaults_path, load, normalize, all_profiles
    paths = WorkspacePaths(tmp_path / 'workspace')
    from component_fixtures import cluster_site
    cluster_site(paths)
    available = all_profiles(paths)
    assert normalize('cluster', available) == 'cluster/default'
    assert 'cluster/cpu' in profiles(paths) and 'cluster_cpu' not in available
    assert defaults_path(paths, 'cluster') == defaults_path(paths, 'cluster/gpu')
    with pytest.raises(ValueError, match='Unknown site profile'):
        load(paths, 'cluster_cpu')
    # An authored variant that only inherits its defaults is a real profile.
    create(paths.root, 'lab', {'scheduler': 'batch'}, {'fast': {}})
    available = all_profiles(paths)
    assert normalize('lab/fast', available) == 'lab/fast' and 'lab/fast' in profiles(paths)


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
        'class Scheduler:\n api_version = 2\n'
        ' def remote_requirements(self, site):\n'
        '  return [("lattice-submit", None), ("lattice-acct", "accounting unavailable")]\n')
    (root / '.theta_component.json').write_text(json.dumps({'id': 'independent-scheduler', 'name': 'Independent',
      'version': '1.0.0', 'kind': 'package', 'storage_version': 3, 'api_version': 1,
      'capabilities': [{'id': 'lattice-batch', 'kind': 'scheduler', 'entry_point': 'scheduler.py:Scheduler', 'scheduler_api': 2}],
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
    (root / 'scheduler.py').write_text('class Scheduler:\n api_version = 2\n'
                                       ' def remote_requirements(self, site):\n  return [("x; rm -rf /", None)]\n')
    from src.app.core.provider_loader import invalidate
    invalidate(paths)
    with pytest.raises(ValueError, match='remote requirement'):
        transfer.check(paths.root, 'lattice')
