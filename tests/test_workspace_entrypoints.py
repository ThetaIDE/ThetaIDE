"""Entry points select a workspace once; reusable services never fall back to the launch directory."""
import json
import os
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from src.app.workspace import WorkspacePaths

REPO = Path(__file__).resolve().parents[1]


def decoy(tmp_path, monkeypatch):
    """A launch directory that looks like a workspace, plus the explicitly selected one."""
    launch = tmp_path / 'launch'
    (launch / 'src/usr').mkdir(parents=True)
    selected = WorkspacePaths(tmp_path / 'selected', tmp_path / 'selected-state')
    selected.root.mkdir()
    monkeypatch.chdir(launch)
    monkeypatch.setenv('THETA_WORKSPACE', str(selected.root))
    monkeypatch.setenv('THETA_DATA_DIR', str(selected.data_dir))
    return launch, selected


def test_omitted_runner_context_uses_the_process_selection(tmp_path, monkeypatch):
    from src.app.pipeline import local_runner
    launch, selected = decoy(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(local_runner, 'resolve_engine', lambda *args, paths: ('src/app/train.py', None))
    monkeypatch.setattr(local_runner, 'paradigm_uses_static_dataset', lambda *args, paths: False)
    monkeypatch.setattr(local_runner, 'build_method_overrides', lambda **kw: calls.append(kw['workspace'].root) or [])
    monkeypatch.setattr(local_runner, 'run_experiment', lambda *args, **kw: calls.append(kw['paths'].root))
    cfg = OmegaConf.create({'group': 'g', 'experiment_id': 'run', 'no_plot': True,
                            'methods': {'m': {'model': 'example'}}, 'paradigm': 'online_rl'})
    local_runner.run_local_training(cfg, {'methods': {'m': {'model': 'example'}},
                                          'sanitized_extra_args': [], 'storage_url': None})
    assert calls == [selected.root, selected.root]
    assert (selected.results / 'logs/g/run/config.yaml').is_file()
    assert not (launch / 'results').exists()


def test_omitted_execution_context_uses_the_process_selection(tmp_path, monkeypatch):
    from src.app.services import executions
    launch, selected = decoy(tmp_path, monkeypatch)
    for root in (launch, selected.root):
        target = root / 'results/executions/g/run/id/manifest.json'
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({'schema_version': 1, 'runs': [], 'owner': str(root)}))
    listed = executions.list_executions()
    assert listed == [str(selected.root / 'results/executions/g/run/id/manifest.json')]


def test_configure_workspace_selects_without_rebinding_module_constants(tmp_path, monkeypatch):
    from src.app import services
    from src.app.services import catalog, config_service
    from src.app.pipeline import compose
    before = (services.LOGS_DIR, catalog.LOGS_DIR, catalog.CONFIG_DIR, compose.CONFIG_DIR, config_service.CONFIG_DIR)
    for name in ('THETA_WORKSPACE', 'THETA_DATA_DIR'):
        monkeypatch.setenv(name, str(tmp_path / 'previous'))  # restored after the test
    selected = WorkspacePaths(tmp_path / 'selected', tmp_path / 'state')
    services.configure_workspace(selected)
    assert (services.LOGS_DIR, catalog.LOGS_DIR, catalog.CONFIG_DIR, compose.CONFIG_DIR, config_service.CONFIG_DIR) == before
    assert WorkspacePaths.for_process() == selected
    assert catalog.results_dir() == selected.results


def test_desktop_job_managers_default_to_the_selected_workspace(tmp_path, monkeypatch):
    from frontend.runner_manager import JobManager as RunnerManager
    launch, selected = decoy(tmp_path, monkeypatch)
    manager = RunnerManager()
    try:
        assert manager.project_root == selected.root
        assert manager.logs_dir == selected.results / 'logs'
        assert manager.jobs_dir == selected.results / 'jobs'
    finally:
        manager.deleteLater()
    assert not (launch / 'results').exists()


def test_plugin_context_reports_the_hosts_workspace_and_state(tmp_path):
    from types import SimpleNamespace
    from frontend.plugins.context import PluginContext
    selected = WorkspacePaths(tmp_path / 'workspace', tmp_path / 'external-state')
    window = SimpleNamespace(workspace_paths=selected, store=SimpleNamespace(root=tmp_path / 'elsewhere/.thetaide/runs'))
    context = PluginContext.__new__(PluginContext)
    context._window = window
    assert context.workspace_paths() is selected
    assert context.get_workspace_dir() == selected.root
    assert context.get_data_dir() == selected.data_dir


def test_cli_help_needs_no_optional_scheduler(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith('THETA_')}
    env['THETA_WORKSPACE'] = str(tmp_path)
    result = subprocess.run([sys.executable, str(REPO / 'run_pipeline.py'), '--help'], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert 'site=<profile>' in result.stdout
    for name in ('ncshare', 'Slurm', 'Optuna'):
        assert name not in result.stdout
