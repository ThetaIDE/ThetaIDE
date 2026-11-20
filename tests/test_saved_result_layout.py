"""Saved executions stay below experiments and coexist with runs written to the experiment folder."""
import json
import zipfile
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.result_paths import iter_run_dirs, result_experiment_id, saved_run_id, saved_run_label
from src.app.services import catalog


@pytest.mark.parametrize('identity', ['run', 'nested/run'])
def test_saved_ids_nest_inside_the_experiment(identity):
    for suffix in ('_20261112-120000', '_20261112-120000-2', '_20261112-120000_abcdef12', '__' + 'a' * 32):
        nested = saved_run_id(identity, suffix)
        assert nested == identity + '/' + suffix.lstrip('_')
        assert result_experiment_id(nested) == identity
        assert saved_run_id(nested, suffix) == nested
        if suffix.startswith('__'):
            # A cluster execution published beside its experiment belongs to that experiment.
            assert result_experiment_id(identity + suffix) == identity
            assert saved_run_label(identity + suffix) == 'a' * 32
        else:
            # A timestamped sibling folder is an experiment of its own, not a save of this one.
            assert result_experiment_id(identity + suffix) == identity + suffix


@pytest.fixture
def mixed_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    for root in catalog.RESULT_ROOTS:
        for identity in ('run', 'run/20261112-120000'):
            folder = tmp_path / 'src/usr/results' / root / 'g' / identity
            folder.mkdir(parents=True)
            (folder / 'config.yaml').write_text('seed: 1\n')
    for identity in ('run', 'run/20261112-120000'):
        agent = (tmp_path / 'src/usr/results/logs') / 'g' / identity / 'agent/version_0'
        agent.mkdir(parents=True)
        (agent / 'metrics.csv').write_text('step,eval/reward\n1,42\n')
    return tmp_path


def test_direct_and_nested_runs_are_discovered_without_mixing_artifacts(mixed_results):
    assert {p.relative_to((mixed_results / 'src/usr/results/logs')).as_posix() for p in iter_run_dirs((mixed_results / 'src/usr/results/logs'))} == {
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
        assert (mixed_results / 'src/usr/results' / root / 'g/run/20261112-120000/config.yaml').exists()
        assert not (mixed_results / 'src/usr/results' / root / 'g/run/config.yaml').exists()


def test_overwriting_direct_run_preserves_saved_runs(mixed_results):
    from src.app.pipeline.local_runner import _setup_output_dirs
    _setup_output_dirs(OmegaConf.create({'group': 'g', 'experiment_id': 'run', 'no_plot': False, 'plots': ['losses']}))
    for root in ('logs', 'checkpoints', 'plots'):
        assert (mixed_results / 'src/usr/results' / root / 'g/run/20261112-120000/config.yaml').exists()
        assert not (mixed_results / 'src/usr/results' / root / 'g/run/config.yaml').exists()


def test_timestamp_queue_ids_always_nest_and_handle_collisions(tmp_path, monkeypatch):
    pytest.importorskip('PyQt6')
    from frontend.app import Window
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(tmp_path)
    window = SimpleNamespace(current_experiment='g/nested/run', issued_ids=set(), workspace_paths=paths)
    first = Window.unique_experiment_id(window, 'nested/run')
    assert first.startswith('nested/run/')
    (paths.results / 'logs/g' / first).mkdir(parents=True)
    second = Window.unique_experiment_id(window, 'nested/run')
    assert second != first
    assert Window.unique_experiment_id(window, first, is_recover=True) == first


def test_results_tree_groups_nested_saves_and_keeps_siblings_apart():
    pytest.importorskip('PyQt6')
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    from frontend.results.explorer import RunExplorer
    application = QApplication.instance() or QApplication([])
    explorer = RunExplorer()
    names = ['nested/run/20261112-120000', 'nested/run/' + 'a' * 32, 'nested/run/20261112-120000_abcdef12']
    sibling = 'nested/run_20261111-120000'
    explorer.set_runs([{'group': 'g', 'experiment_id': name, 'agents': [{'name': 'agent'}]} for name in names + [sibling]])
    group = explorer.tree.topLevelItem(0)
    assert group.childCount() == 1
    folder = group.child(0)  # the experiments' subdirectory below the group is its own row
    assert (folder.text(0), folder.data(0, Qt.ItemDataRole.UserRole)) == ('nested', 'folder')
    experiments = {folder.child(i).text(0): folder.child(i) for i in range(folder.childCount())}
    assert set(experiments) == {'run', 'run_20261111-120000'}
    assert experiments['run'].childCount() == len(names)
    assert explorer.select_run('g', names[1])
    assert folder.isExpanded() and experiments['run'].isExpanded()
    assert {source.experiment_id for source in explorer.checked_sources()} == {names[1]}
    explorer.search.setText('run_2026')
    assert not folder.isHidden() and experiments['run'].isHidden()
    explorer.search.setText('nested')
    assert not experiments['run'].isHidden() and not experiments['run_20261111-120000'].isHidden()
    explorer.tree.setCurrentItem(folder)
    assert len(explorer.deletion_targets()) == len(names) + 1
    explorer.deleteLater()


def test_results_tree_shows_the_trial_and_seed_even_for_one_seed():
    pytest.importorskip('PyQt6')
    from PyQt6.QtWidgets import QApplication
    from frontend.results.data import Manifest
    from frontend.results.explorer import RunExplorer

    application = QApplication.instance() or QApplication([])
    explorer = RunExplorer()
    identity = 'path/g_run/20261112-120000'
    version = 'trial_4/seed_73'
    explorer.set_runs([{'group': 'g', 'experiment_id': identity, 'agents': [{'name': 'ppo'}], 'finished': True}])
    explorer.add_manifest(Manifest('g', identity, {'ppo': [version]}, []))
    group = explorer.tree.topLevelItem(0)
    folder = group.child(0)
    experiment = folder.child(0)
    stamp = experiment.child(0)
    method = stamp.child(0)
    trial = method.child(0)
    seed = trial.child(0)
    assert [item.text(0) for item in (group, folder, experiment, stamp, method, trial, seed)] == [
        'g', 'path', 'g_run', '20261112-120000', 'ppo', 'trial_4', 'seed_73']
    icons = [item.icon(0) for item in (folder, experiment, method)]
    assert all(not icon.isNull() for icon in icons)
    assert len({icon.cacheKey() for icon in icons}) == 3
    assert explorer.select_version(f'g/{identity}', 'ppo', version)
    assert [source.version for source in explorer.checked_sources()] == [version]
    explorer.deleteLater()


@pytest.mark.parametrize("identity", ["nested/run", "nested/run/" + "a" * 32])
def test_variability_results_are_nested(tmp_path, monkeypatch, identity):
    from src.app.pipeline.variability import run_execution
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path))
    monkeypatch.setattr('src.app.pipeline.local_runner._run_methods_sequential', lambda *args: None)
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': identity, 'plotting': {'enabled': False}})
    context = {'methods': {'agent': {'name': 'agent'}}, 'sanitized_extra_args': []}
    plan = {'training_seeds': [1], 'parallel_runs': 1, 'evaluation_seed': None}
    run_execution(cfg, context, plan)
    manifest_path, = tmp_path.glob('src/usr/results/executions/g/' + identity + '/*/manifest.json')
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
    snapshot = (mixed_results / 'src/usr/results/logs') / 'g/run/20261112-120000/config.yaml'
    snapshot.write_text('experiment_id: run/20261112-120000\n')
    rename_results('run', 'g', 'run/20261112-120000', 'best')
    summary = catalog.run_config_summary((mixed_results / 'src/usr/results/logs') / 'g/run/best')
    explorer = RunExplorer()
    explorer.set_runs([{'group': 'g', 'experiment_id': 'run/best', 'config': summary,
                        'agents': [{'name': 'agent'}]}])
    assert explorer.tree.topLevelItem(0).child(0).text(0) == 'run'
    explorer.deleteLater()


def test_renaming_experiment_moves_nested_runs_and_execution_references(mixed_results):
    from src.app.services.result_rename import rename_results
    manifest = mixed_results / 'src/usr/results/executions/g/run/execution/manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'group': 'g', 'experiment_id': 'run',
                                   'result_experiment_id': 'run/20261112-120000',
                                   'runs': [{'experiment_id': 'run/20261112-120000', 'status': 'completed'}]}))
    renamed = rename_results('experiment', 'g', 'run', 'renamed')
    assert renamed['runs']['g/run/20261112-120000'] == 'g/renamed/20261112-120000'
    document = json.loads((mixed_results / 'src/usr/results/executions/g/renamed/execution/manifest.json').read_text())
    assert document['result_experiment_id'] == document['runs'][0]['experiment_id'] == 'renamed/20261112-120000'
    for root in catalog.RESULT_ROOTS:
        assert (mixed_results / 'src/usr/results' / root / 'g/renamed/20261112-120000/config.yaml').exists()


def test_failed_launch_nested_in_an_experiment_is_listed(tmp_path):
    logs = tmp_path / 'logs'
    (logs / 'g' / 'exp').mkdir(parents=True)
    (logs / 'g' / 'exp' / 'a.stderr.log').write_text('')
    (logs / 'g' / 'exp' / '20261104-090046').mkdir()
    (logs / 'g' / 'exp' / '20261104-090046' / 'b.stderr.log').write_text('boom')
    assert [path.relative_to(logs).as_posix() for path in iter_run_dirs(logs)] == [
        'g/exp', 'g/exp/20261104-090046']


def test_an_experiment_that_also_holds_experiments_is_one_row():
    pytest.importorskip('PyQt6')
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    from frontend.results.explorer import RunExplorer
    application = QApplication.instance() or QApplication([])
    explorer = RunExplorer()
    explorer.set_runs([{'group': 'g', 'experiment_id': name, 'agents': [{'name': 'agent'}], 'finished': True}
                       for name in ('Cartpole', 'Cartpole/smoke', 'Final Sweep__' + 'b' * 32)])
    group = explorer.tree.topLevelItem(0)
    rows = {group.child(i).text(0): group.child(i) for i in range(group.childCount())}
    assert set(rows) == {'Cartpole', 'Final Sweep'}
    cartpole = rows['Cartpole']
    assert cartpole.data(0, Qt.ItemDataRole.UserRole) == 'experiment'
    assert [cartpole.child(i).text(0) for i in range(cartpole.childCount())] == ['Run', 'smoke']
    assert rows['Final Sweep'].child(0).text(0) == 'b' * 32
    explorer.deleteLater()
