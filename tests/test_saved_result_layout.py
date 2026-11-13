"""Saved executions stay below experiments and coexist with legacy run roots."""
import json
import zipfile
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.result_paths import iter_run_dirs, result_experiment_id, saved_run_id
from src.app.services import catalog


@pytest.mark.parametrize('identity', ['run', 'nested/run'])
def test_saved_ids_group_with_legacy_ids(identity):
    for suffix in ('_20261112-120000', '_20261112-120000-2', '_20261112-120000_abcdef12', '__' + 'a' * 32):
        nested = saved_run_id(identity, suffix)
        assert nested == identity + '/' + suffix.lstrip('_')
        assert result_experiment_id(nested) == identity
        assert result_experiment_id(identity + suffix) == identity
        assert saved_run_id(nested, suffix) == nested


@pytest.fixture
def mixed_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(catalog, 'LOGS_DIR', tmp_path / 'results/logs')
    for root in catalog.RESULT_ROOTS:
        for identity in ('run', 'run/20261112-120000'):
            folder = tmp_path / 'results' / root / 'g' / identity
            folder.mkdir(parents=True)
            (folder / 'config.yaml').write_text('seed: 1\n')
    for identity in ('run', 'run/20261112-120000'):
        agent = catalog.LOGS_DIR / 'g' / identity / 'agent/version_0'
        agent.mkdir(parents=True)
        (agent / 'metrics.csv').write_text('step,eval/reward\n1,42\n')
    return tmp_path


def test_legacy_and_nested_runs_are_discovered_without_mixing_artifacts(mixed_results):
    assert {p.relative_to(catalog.LOGS_DIR).as_posix() for p in iter_run_dirs(catalog.LOGS_DIR)} == {
        'g/run', 'g/run/20261112-120000'}
    manifest = catalog.run_manifest('g', 'run')
    assert manifest['agents'] == {'agent': ['version_0']}
    assert all('/20261112-120000/' not in f['path'] for f in manifest['artifacts'])
    destination = mixed_results / 'old.zip'
    catalog.archive_run('g', 'run', str(destination))
    with zipfile.ZipFile(destination) as archive:
        assert all('/20261112-120000/' not in name for name in archive.namelist())
    catalog.delete_run('g', 'run')
    for root in catalog.RESULT_ROOTS:
        assert (mixed_results / 'results' / root / 'g/run/20261112-120000/config.yaml').exists()
        assert not (mixed_results / 'results' / root / 'g/run/config.yaml').exists()


def test_overwriting_legacy_run_preserves_saved_runs(mixed_results):
    from src.app.pipeline.local_runner import _setup_output_dirs
    _setup_output_dirs(OmegaConf.create({'group': 'g', 'experiment_id': 'run', 'no_plot': False, 'plots': ['losses']}))
    for root in ('logs', 'checkpoints', 'plots'):
        assert (mixed_results / 'results' / root / 'g/run/20261112-120000/config.yaml').exists()
        assert not (mixed_results / 'results' / root / 'g/run/config.yaml').exists()


def test_timestamp_queue_ids_always_nest_and_handle_collisions(tmp_path, monkeypatch):
    pytest.importorskip('PyQt6')
    from frontend.app import Window
    from src.app import services
    monkeypatch.setattr(services, 'LOGS_DIR', tmp_path)
    window = SimpleNamespace(current_experiment='g/nested/run', issued_ids=set())
    first = Window.unique_experiment_id(window, 'nested/run')
    assert first.startswith('nested/run/')
    (tmp_path / 'g' / first).mkdir(parents=True)
    second = Window.unique_experiment_id(window, 'nested/run')
    assert second != first
    assert Window.unique_experiment_id(window, first, is_recover=True) == first


def test_results_tree_groups_new_and_old_saves():
    pytest.importorskip('PyQt6')
    from PyQt6.QtWidgets import QApplication
    from frontend.results.explorer import RunExplorer
    application = QApplication.instance() or QApplication([])
    explorer = RunExplorer()
    names = ['nested/run_20261111-120000', 'nested/run/20261112-120000',
             'nested/run/' + 'a' * 32, 'nested/run/20261112-120000_abcdef12']
    explorer.set_runs([{'group': 'g', 'experiment_id': name, 'agents': [{'name': 'agent'}]} for name in names])
    group = explorer.tree.topLevelItem(0)
    assert group.childCount() == 1
    assert group.child(0).text(0) == 'run'
    assert group.child(0).childCount() == len(names)
    assert explorer.select_run('g', names[1])
    assert {source.experiment_id for source in explorer.checked_sources()} == {names[1]}
    explorer.deleteLater()


@pytest.mark.parametrize("identity", ["nested/run", "nested/run/" + "a" * 32])
def test_variability_results_are_nested(tmp_path, monkeypatch, identity):
    from src.app.pipeline.variability import run_execution
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr('src.app.pipeline.local_runner._run_methods_sequential', lambda *args: None)
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': identity, 'plotting': {'enabled': False}})
    context = {'methods': {'agent': {'name': 'agent'}}, 'sanitized_extra_args': []}
    plan = {'training_seeds': [1], 'parallel_runs': 1, 'evaluation_seed': None}
    run_execution(cfg, context, plan)
    manifest_path, = tmp_path.glob('results/executions/g/' + identity + '/*/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['result_experiment_id'].startswith('nested/run/')
    assert result_experiment_id(manifest['result_experiment_id']) == 'nested/run'
    if identity != 'nested/run':
        assert manifest['result_experiment_id'] == identity
    assert (tmp_path / manifest['runs'][0]['logs']).is_dir()


def test_renaming_nested_run_keeps_experiment_group(mixed_results):
    from src.app.services.result_rename import rename_results
    pytest.importorskip('PyQt6')
    from PyQt6.QtWidgets import QApplication
    from frontend.results.explorer import RunExplorer
    application = QApplication.instance() or QApplication([])
    snapshot = catalog.LOGS_DIR / 'g/run/20261112-120000/config.yaml'
    snapshot.write_text('experiment_id: run/20261112-120000\n')
    rename_results('run', 'g', 'run/20261112-120000', 'best')
    summary = catalog.run_config_summary(catalog.LOGS_DIR / 'g/run/best')
    explorer = RunExplorer()
    explorer.set_runs([{'group': 'g', 'experiment_id': 'run/best', 'config': summary,
                        'agents': [{'name': 'agent'}]}])
    assert explorer.tree.topLevelItem(0).child(0).text(0) == 'run'
    explorer.deleteLater()


def test_renaming_experiment_moves_nested_runs_and_execution_references(mixed_results):
    from src.app.services.result_rename import rename_results
    manifest = mixed_results / 'results/executions/g/run/execution/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'group': 'g', 'experiment_id': 'run',
                                   'result_experiment_id': 'run/20261112-120000',
                                   'runs': [{'experiment_id': 'run/20261112-120000', 'status': 'completed'}]}))
    renamed = rename_results('experiment', 'g', 'run', 'renamed')
    assert renamed['runs']['g/run/20261112-120000'] == 'g/renamed/20261112-120000'
    document = json.loads((mixed_results / 'results/executions/g/renamed/execution/manifest.json').read_text())
    assert document['result_experiment_id'] == document['runs'][0]['experiment_id'] == 'renamed/20261112-120000'
    for root in catalog.RESULT_ROOTS:
        assert (mixed_results / 'results' / root / 'g/renamed/20261112-120000/config.yaml').exists()
