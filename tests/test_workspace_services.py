"""Exercise reusable result/launch APIs independently of process context."""
import json
from concurrent.futures import ThreadPoolExecutor

from omegaconf import OmegaConf

from src.app.workspace import WorkspacePaths
from src.app.services import catalog, executions
from src.app.services.result_rename import rename_results
from src.app.pipeline.experiment_snapshot import save_run_config, save_experiment_snapshot


def make_run(paths, value):
    version = paths.results / 'logs/g/run/m/version_0'
    version.mkdir(parents=True)
    (version / 'metrics.csv').write_text(f'transitions,eval/reward\n1,{value}\n')
    (version / 'runtime.json').write_text(json.dumps({'end_time_iso': '2026-11-13T00:00:00'}))
    (version.parent.parent / 'config.yaml').write_text('group: g\nexperiment_id: run\n')


def test_result_services_are_isolated_in_concurrent_workspaces(tmp_path, monkeypatch):
    unrelated = tmp_path / 'launch'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    workspaces = [WorkspacePaths(tmp_path / name, tmp_path / (name + '-state')) for name in ('one', 'two')]
    for value, paths in enumerate(workspaces, 11): make_run(paths, value)
    monkeypatch.setenv('THETA_WORKSPACE', str(tmp_path / 'wrong'))
    def read(paths):
        for _ in range(3):
            runs = catalog.list_runs(workspace=paths)
            assert len(runs) == 1
            assert runs[0]['agents'][0]['latest_reward'] == (11 if paths == workspaces[0] else 12)
            assert catalog.results_dir(workspace=paths) == paths.results
            assert catalog.read_run_file('g', 'run', 'logs/g/run/m/version_0/metrics.csv', workspace=paths).startswith(b'transitions')
            manifest = catalog.run_manifest('g', 'run', workspace=paths)
            assert manifest['experiment_id'] == 'run'
        saved = json.loads((paths.results / '.cache/catalog-summaries.json').read_text())
        assert all(key.startswith(str(paths.results)) for group in (saved['agents'], saved['configs'], saved['metadata']) for key in group)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(read, workspaces))
    rename_results('run', 'g', 'run', 'renamed', workspace=workspaces[0])
    assert (workspaces[0].results / 'logs/g/renamed').is_dir()
    assert (workspaces[1].results / 'logs/g/run').is_dir()
    catalog.delete_run('g', 'renamed', workspace=workspaces[0])
    assert (workspaces[1].results / 'logs/g/run').is_dir()
    assert not list(unrelated.iterdir())


def test_execution_and_snapshot_services_do_not_infer_launch_directory(tmp_path, monkeypatch):
    unrelated = tmp_path / 'launch'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    paths = WorkspacePaths(tmp_path / 'workspace')
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'run', 'seed': 7})
    save_experiment_snapshot(cfg, {'workspace': paths, 'experiment_yaml': b'seed: 7\n'})
    save_run_config(cfg, paths=paths)
    assert (paths.results / 'logs/g/run/config.yaml').is_file()
    target = paths.results / 'executions/g/run/id/manifest.json'
    target.parent.mkdir(parents=True)
    target.write_text('{"schema_version": 1, "runs": []}')
    assert executions.list_executions(workspace=paths) == [str(target)]
    assert executions.read_execution(target, workspace=paths)['manifest']['runs'] == []
    assert not list(unrelated.iterdir())


def test_local_coordinator_passes_selected_workspace_to_each_boundary(tmp_path, monkeypatch):
    from src.app.pipeline import local_runner, datasets
    paths = WorkspacePaths(tmp_path / 'workspace')
    paths.root.mkdir()
    unrelated = tmp_path / 'launch'
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'run', 'no_plot': True,
                            'methods': {'m': {'model': 'example'}}, 'paradigm': 'online_rl'})
    calls = []
    monkeypatch.setattr(local_runner, 'resolve_engine', lambda *args, paths: ('src/app/train.py', None))
    monkeypatch.setattr(local_runner, 'paradigm_uses_static_dataset', lambda *args, paths: False)
    monkeypatch.setattr(local_runner, 'build_method_overrides', lambda **kw: calls.append(kw['workspace']) or [])
    monkeypatch.setattr(local_runner, 'run_experiment', lambda *args, **kw: calls.append(kw['paths']))
    local_runner.run_local_training(cfg, {'workspace': paths, 'methods': {'m': {'model': 'example'}},
                                          'sanitized_extra_args': [], 'storage_url': None})
    assert calls == [paths, paths]
    assert (paths.results / 'logs/g/run/config.yaml').is_file()
    assert not list(unrelated.iterdir())
