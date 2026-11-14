"""Extension discovery is headless; activation and teardown are explicit."""
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from src.app.core.contributions import (ContributionRegistry, ContributionScope, load_contribution)
from src.app.core.provider_loader import load_capability
from src.app.core.provider_catalog import get_catalog
from src.app.pipeline.exceptions import ConfigurationError
from src.app.workspace import WorkspacePaths

FIXTURE = Path(__file__).parent / 'fixtures/contribution_provider'


def fixture(tmp_path, *, origin='user', executable=False):
    root = tmp_path / ('src/usr/packages/tools' if origin == 'user' else 'src/plugins/components/installed/tools')
    shutil.copytree(FIXTURE, root)
    if not executable:
        (root / 'provider.py').write_text('raise AssertionError("discovery executed provider code")')
    return WorkspacePaths(tmp_path, application=tmp_path / 'empty-application'), root


def test_readiness_enablement_and_removal_are_metadata_only(tmp_path):
    paths, root = fixture(tmp_path)
    registry = ContributionRegistry(paths)
    assert registry.status('site', 'fixture-site').state == 'setup'
    assert [c.slot for c in registry.contributions()] == ['settings.site']
    ready = {'connection': {'host': 'compute.example'}}
    assert registry.status('site', 'fixture-site', config=ready).state == 'ready'
    assert len(registry.contributions(config=ready)) == 2
    registry.set_enabled('site', 'fixture-site', False)
    assert registry.status('site', 'fixture-site').state == 'disabled'
    assert not registry.contributions(config=ready)
    selected = get_catalog(paths).resolve('site', 'fixture-site')
    with pytest.raises(ConfigurationError, match='disabled'):
        load_capability(*selected, paths=paths)
    registry.set_enabled('site', 'fixture-site', True)
    assert len(registry.contributions(config=ready)) == 2
    (root / '.theta_component.json').unlink()
    assert registry.status('site', 'fixture-site').state == 'absent'
    assert registry.contributions(config=ready) == []


def test_incompatible_extension_is_browsable_but_not_activated(tmp_path):
    paths, root = fixture(tmp_path)
    manifest = root / '.theta_component.json'
    data = json.loads(manifest.read_text()); data['capabilities'][0]['contribution_api'] = 99
    manifest.write_text(json.dumps(data))
    registry = ContributionRegistry(paths)
    assert registry.status('site', 'fixture-site').state == 'incompatible'
    assert not registry.contributions()
    with pytest.raises(ConfigurationError, match='host supports 1'):
        load_capability(*get_catalog(paths).resolve('site', 'fixture-site'), paths=paths)


def test_contribution_scope_cleans_up_disable_remove_and_workspace_switch(tmp_path):
    paths, root = fixture(tmp_path, executable=True)
    registry = ContributionRegistry(paths)
    calls = []
    scope = ContributionScope(lambda c: load_contribution(c, calls, paths=paths))
    scope.reconcile(registry.contributions('settings.site'))
    scope.reconcile(registry.contributions('settings.site'))
    assert calls == ['activated']
    registry.set_enabled('site', 'fixture-site', False)
    scope.reconcile(registry.contributions('settings.site'))
    assert calls == ['activated', 'closed']
    registry.set_enabled('site', 'fixture-site', True)
    scope.reconcile(registry.contributions('settings.site'))
    (root / '.theta_component.json').unlink()
    scope.reconcile(registry.contributions('settings.site'))
    assert calls == ['activated', 'closed', 'activated', 'closed']
    scope.close(); scope.close()
    assert not scope.active


def test_state_is_workspace_scoped_and_requirements_need_no_imports(tmp_path, monkeypatch):
    first, _ = fixture(tmp_path / 'first')
    second, _ = fixture(tmp_path / 'second')
    registry = ContributionRegistry(first)
    registry.set_enabled('site', 'fixture-site', False)
    assert registry.status('site', 'fixture-site').state == 'disabled'
    assert ContributionRegistry(second).status('site', 'fixture-site').state == 'setup'
    assert not (second.data_dir / 'extensions.json').exists()
    from importlib.metadata import PackageNotFoundError
    def missing(name): raise PackageNotFoundError(name)
    monkeypatch.setattr('src.app.core.contributions.version', missing)
    _, capability = get_catalog(second).resolve('site', 'fixture-site')
    capability.metadata['readiness']['distributions'] = ['not-installed']
    status = ContributionRegistry(second).status('site', 'fixture-site', config={'connection': {'host': 'compute'}})
    assert status.state == 'setup' and 'install dependency not-installed' in status.reason


def test_lifecycle_rolls_back_all_new_mounts_even_when_cleanup_fails(tmp_path):
    paths, _ = fixture(tmp_path)
    contribution = ContributionRegistry(paths).contributions()[0]
    calls = []
    def mount(c):
        if c.id == 'third': raise ValueError('activation failed')
        def cleanup():
            calls.append(c.id)
            if c.id == 'second': raise RuntimeError('cleanup failed')
        return cleanup
    scope = ContributionScope(mount)
    with pytest.raises(RuntimeError, match='rollback failed'):
        scope.reconcile([replace(contribution, id=name) for name in ('first', 'second', 'third')])
    assert calls == ['second', 'first'] and not scope.active


def test_factory_payload_and_duplicate_identity_are_validated(tmp_path):
    from src.app.components.archive import validate_native_package
    paths, root = fixture(tmp_path)
    manifest = json.loads((root / '.theta_component.json').read_text())
    cap = manifest['capabilities'][0]
    cap['contributions'].append(dict(cap['contributions'][0]))
    with pytest.raises(ValueError, match='Duplicate contribution ID'):
        validate_native_package(manifest, root)
    cap['contributions'].pop()
    (root / 'extra.py').write_text('raise AssertionError("validation executed code")')
    cap['contributions'][0]['entry_point'] = 'extra.py:activate'
    with pytest.raises(ValueError, match='outside declared payload'):
        validate_native_package(manifest, root)


def test_cli_enablement_and_preview_fingerprints_use_workspace_state(tmp_path, capsys):
    from src.app.components.__main__ import main
    from src.app.services.config_service import _inputs_fingerprint
    paths, _ = fixture(tmp_path)
    before = _inputs_fingerprint(paths)
    assert main(['--root', str(tmp_path), 'extension', 'disable', 'site', 'fixture-site']) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'disabled'
    assert _inputs_fingerprint(paths) != before
    assert main(['--root', str(tmp_path), 'extension', 'enable', 'site', 'fixture-site']) == 0
    assert json.loads(capsys.readouterr().out)['state'] == 'setup'


def test_workspace_switch_replaces_live_slot_resources(tmp_path):
    first, _ = fixture(tmp_path / 'first')
    second, _ = fixture(tmp_path / 'second')
    calls = []
    def mount(contribution):
        calls.append(('start', str(contribution.package.root)))
        return lambda: calls.append(('stop', str(contribution.package.root)))
    scope = ContributionScope(mount)
    scope.reconcile(ContributionRegistry(first).contributions())
    scope.reconcile(ContributionRegistry(second).contributions())
    scope.close()
    assert [kind for kind, _ in calls] == ['start', 'stop', 'start', 'stop']
    assert calls[0][1] != calls[2][1]


def test_active_job_lease_prevents_state_change(tmp_path):
    from src.app.components.transactions import provider_lease
    paths, _ = fixture(tmp_path)
    with provider_lease(paths, ['independent-host-tools'], purpose='fixture worker'):
        with pytest.raises(RuntimeError, match='fixture worker'):
            ContributionRegistry(paths).set_enabled('site', 'fixture-site', False)
    assert ContributionRegistry(paths).status('site', 'fixture-site').enabled


def test_tuner_runtime_requires_readiness_but_setup_fields_remain_available(tmp_path):
    from src.app.pipeline.tuners import get_tuner, available_tuners
    from src.app.pipeline.tuner_execution import prepare_plan
    paths = WorkspacePaths(tmp_path, application=tmp_path / 'empty-application')
    shutil.copytree(Path(__file__).parent / 'fixtures/contribution_tuner', tmp_path / 'src/usr/tuners/lattice')
    assert get_tuner(name='lattice', paths=paths).settings()
    assert 'lattice' in available_tuners(paths=paths)
    cfg = {'tuning': {'component': 'lattice'}}
    with pytest.raises(ConfigurationError, match='configure tuning.options.lattice.token'):
        get_tuner(cfg, paths=paths)
    with pytest.raises(ConfigurationError, match='configure tuning.options.lattice.token'):
        prepare_plan(cfg, 'a'*32, paths=paths)
    cfg['tuning']['options'] = {'lattice': {'token': 'configured'}}
    assert get_tuner(cfg, paths=paths)
    ContributionRegistry(paths).set_enabled('tuner', 'lattice', False)
    assert 'lattice' not in available_tuners(paths=paths)
    with pytest.raises(ConfigurationError, match='disabled'):
        get_tuner(cfg, paths=paths)
