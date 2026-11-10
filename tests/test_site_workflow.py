from pathlib import Path
from src.app.sites import transfer
from src.app.sites.slurm import bootstrap_command, submission_command, shortcuts, ssh_arguments


def test_user_configuration_and_setup_are_offline(tmp_path, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network attempted')))
    transfer.configure(tmp_path, 'cluster', 'alias', '/scratch/project')
    profile = tmp_path / 'src/usr/sites/cluster/configs/site/cluster/_defaults.yaml'
    from src.app.sites.catalog import configure
    configure(tmp_path, 'cluster', {'rsync_executable': '/custom/rsync'})
    assert transfer.connection(tmp_path, 'cluster')['rsync_executable'] == '/custom/rsync'
    script = bootstrap_command(tmp_path, 'cluster')
    assert 'site push cluster' in script and 'pip install' in script
    assert 'python3 -m venv' in script
    paths = shortcuts(tmp_path, 'cluster')
    assert len(paths) == 3
    assert any('src/usr/sites/cluster/ssh_config' in value for value in ssh_arguments(tmp_path, transfer.connection(tmp_path, 'cluster')))
    command = submission_command(transfer.connection(tmp_path, 'cluster'), 'g/test', 'cluster')
    assert command.index('--verify') < command.index('dry_run=true') < command.rindex('run_pipeline.py')


def test_launch_button_hands_off_to_user_terminal(monkeypatch):
    from frontend.app import Window
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr('frontend.site_launch.launch', calls.append)
    host = SimpleNamespace(composed_config={'site': {'scheduler': 'slurm'}})
    Window.launch_training(host)
    assert calls == [host]
