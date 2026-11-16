"""Explicit remote bundles: what an experiment selects, what stays home, and verification."""
import json
import shutil
from pathlib import Path

import pytest

from src.app.sites import bundle, transfer
from src.app.sites.storage import read_json, write_json
from src.app.workspace import WorkspacePaths

TUNED = """# @package _global_
defaults:
  - /experiment/thetaide/cartpole_ppo_reference
  - _self_

experiment_id: tuned
tuning:
  component: optuna
  metric: reward
  direction: maximize
"""


@pytest.fixture(scope='module')
def workspace(tmp_path_factory):
    """The fixture workspace with Optuna installed, an unrelated authored package and a tuned recipe."""
    from conftest import build_fixture_workspace
    from component_fixtures import install_optional
    base = tmp_path_factory.mktemp('bundles')
    root = build_fixture_workspace(base / 'workspace')
    install_optional(root, 'optuna', base / 'archives')
    unrelated = root / 'src/usr/models/unrelated-models'
    shutil.copytree(root / 'src/usr/models/test-models', unrelated)
    manifest = read_json(unrelated / '.theta_component.json')
    manifest['id'] = 'unrelated-models'
    for capability in manifest['capabilities']:
        capability['id'] = 'unrelated_' + capability['id']
        capability.pop('aliases', None)
    manifest['payload']['configs'] = {}
    shutil.rmtree(unrelated / 'configs')
    write_json(unrelated / '.theta_component.json', manifest)
    (WorkspacePaths(root).config / 'experiment/thetaide/tuned.yaml').write_text(TUNED)
    results = WorkspacePaths(root).results / 'logs/thetaide/cartpole_ppo_baseline/run.csv'
    results.parent.mkdir(parents=True)
    results.write_text('step\n')
    dataset = WorkspacePaths(root).datasets / 'big.npz'
    dataset.parent.mkdir(parents=True, exist_ok=True)
    dataset.write_bytes(b'data')
    return root


@pytest.fixture
def selected(workspace, monkeypatch):
    monkeypatch.setenv('THETA_WORKSPACE', str(workspace))
    return workspace


def providers(manifest):
    return {item['id'] for item in manifest['providers']}


def test_bundle_holds_what_the_experiment_selects(selected, tmp_path):
    manifest = bundle.prepare(selected, 'thetaide/cartpole_ppo_reference', [], tmp_path / 'bundle')
    assert manifest['schema'] == 'theta.bundle/v1'
    assert manifest['experiment'] == {'selector': 'thetaide/cartpole_ppo_reference', 'overrides': [], 'site': 'local/default'}
    # The algorithm, environment and model the recipe names, and nothing it does not.
    assert {'sb3-dev', 'test-models', 'gym_classic_control_dev'} <= providers(manifest)
    assert not {'optuna', 'unrelated-models'} & providers(manifest)
    files = set(manifest['files'])
    assert 'run_pipeline.py' in files and 'src/app/workspace.py' in files
    assert not any(name.startswith(('frontend/', 'src/plugins/app/')) for name in files)
    assert not any(name.startswith(('src/usr/results', 'src/usr/datasets', 'src/usr/models/unrelated-models',
                                    'src/plugins/components/installed/optuna')) for name in files)
    assert 'src/usr/config/experiment/thetaide/cartpole_ppo_reference.yaml' in files
    assert 'src/usr/config/experiment/thetaide/_base.yaml' in files
    assert 'src/usr/config/experiment/thetaide/tuned.yaml' not in files  # Other experiments stay home.
    assert any(item['selector'] == 'experiment/thetaide/cartpole_ppo_reference.yaml' and item['origin'] == 'workspace'
               for item in manifest['configs'])
    # Requirements are those of the selected providers; this experiment has no tuner (5e).
    assert not any(line.startswith(('optuna', 'sqlalchemy')) for line in manifest['requirements']['providers'])
    assert (tmp_path / 'bundle' / bundle.REQUIREMENTS).read_text().splitlines() == manifest['requirements']['providers']
    assert manifest['application']['name'] == 'theta-ide'
    assert bundle.verify(tmp_path / 'bundle')['bundle_sha256'] == manifest['bundle_sha256']


def test_selected_tuner_and_its_requirements_travel_with_the_bundle(selected, tmp_path):
    manifest = bundle.prepare(selected, 'thetaide/tuned', [], tmp_path / 'bundle')
    assert 'optuna' in providers(manifest)
    tuner = next(item for item in manifest['providers'] if item['id'] == 'optuna')
    assert tuner['path'] == 'src/plugins/components/installed/optuna' and tuner['digest']
    assert any(line.startswith('optuna') for line in manifest['requirements']['providers'])


def test_unchanged_sources_reproduce_the_bundle_digest(selected, tmp_path):
    first = bundle.prepare(selected, 'thetaide/cartpole_ppo_reference', [], tmp_path / 'first')
    second = bundle.prepare(selected, 'thetaide/cartpole_ppo_reference', [], tmp_path / 'second')
    assert first['bundle_sha256'] == second['bundle_sha256']


def test_verification_rejects_changed_unknown_and_excluded_content(selected, tmp_path):
    directory = tmp_path / 'bundle'
    manifest = bundle.prepare(selected, 'thetaide/cartpole_ppo_reference', [], directory)
    smuggled = {'src/usr/results/logs/x.csv': '0' * 64, **manifest['files']}
    write_json(directory / bundle.MANIFEST, {**manifest, 'files': smuggled, 'bundle_sha256': bundle._set_digest(smuggled)})
    with pytest.raises(ValueError, match='may not contain src/usr/results'):
        bundle.verify(directory)
    for schema in ('theta.bundle/v0', None):
        write_json(directory / bundle.MANIFEST, {**manifest, 'schema': schema})
        with pytest.raises(ValueError, match='Unsupported bundle schema'):
            bundle.verify(directory)
    write_json(directory / bundle.MANIFEST, manifest)
    (directory / 'run_pipeline.py').write_text('# changed\n')
    with pytest.raises(ValueError, match='failed verification: run_pipeline.py'):
        bundle.verify(directory)


def test_activation_and_archive_reproduce_the_bundle(selected, tmp_path):
    stage = tmp_path / 'upload'
    manifest = bundle.prepare(selected, 'thetaide/cartpole_ppo_reference', [], stage)
    remote = tmp_path / 'remote'
    remote.mkdir()
    assert transfer.activate(remote, stage) == {'bundle_sha256': manifest['bundle_sha256'], 'experiment': 'thetaide/cartpole_ppo_reference'}
    current = bundle.current(remote)
    archive = remote / transfer.ARCHIVES / (manifest['bundle_sha256'] + '.zip')
    extracted = bundle.extract(archive, tmp_path / 'extracted')
    assert extracted['bundle_sha256'] == current['bundle_sha256'] == manifest['bundle_sha256']
    # The activated workspace composes the experiment from its bundle alone.
    bundle.check(remote, current)


def test_bundle_without_a_provider_it_needs_fails_its_check(selected, tmp_path):
    directory = tmp_path / 'bundle'
    manifest = bundle.prepare(selected, 'thetaide/cartpole_ppo_reference', [], directory)
    shutil.rmtree(directory / 'src/usr/algorithms/sb3-dev')
    with pytest.raises(ValueError, match='lacks'):
        bundle.check(directory, manifest)


def test_push_refuses_an_experiment_of_another_site(selected, tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, 'connection', lambda *_: {'host': 'fake', 'remote_root': '/remote'})
    monkeypatch.setattr(transfer, 'require_local_rsync', lambda config: 'rsync')
    monkeypatch.setattr(transfer, 'ssh', lambda *_a, **_k: pytest.fail('network attempted'))
    with pytest.raises(ValueError, match='runs on site local/default, not cluster/common'):
        transfer.push(selected, 'cluster/common', 'thetaide/cartpole_ppo_reference')
