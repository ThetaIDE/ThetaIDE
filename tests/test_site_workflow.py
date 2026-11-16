from pathlib import Path
from src.app.sites import transfer
from src.app.sites.transport import ssh_arguments
from component_fixtures import optional_module
_slurm_commands = optional_module('slurm', 'commands.py')
bootstrap_command, submission_command, shortcuts = (
    _slurm_commands.bootstrap_command, _slurm_commands.submission_command, _slurm_commands.shortcuts)


def test_user_configuration_and_setup_are_offline(tmp_path, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network attempted')))
    transfer.configure(tmp_path, 'cluster', 'alias', '/scratch/project')
    profile = tmp_path / 'src/usr/sites/cluster/configs/site/cluster/_defaults.yaml'
    from src.app.sites.catalog import configure
    configure(tmp_path, 'cluster', {'rsync_executable': '/custom/rsync'})
    assert transfer.connection(tmp_path, 'cluster')['rsync_executable'] == '/custom/rsync'
    script = bootstrap_command(tmp_path, 'cluster', 'g/test')
    assert 'site push cluster g/test' in script
    # Setup installs the requirements the pushed bundle lists, not those of every installed package.
    assert 'pip install -r .thetaide/requirements.txt' in script
    assert 'python3 -m venv' in script
    paths = shortcuts(tmp_path, 'cluster')
    assert len(paths) == 3
    push = next(Path(path) for path in paths if '/theta-push-' in path).read_text()
    assert push.rstrip().endswith('cluster "$@"')  # The shortcut takes the experiment to push.
    assert any('src/usr/sites/cluster/ssh_config' in value for value in ssh_arguments(tmp_path, transfer.connection(tmp_path, 'cluster')))
    command = submission_command(transfer.connection(tmp_path, 'cluster'), 'g/test', 'cluster')
    assert command == 'venv/bin/python3 run_pipeline.py g/test'


def test_launch_button_hands_off_to_user_terminal(monkeypatch):
    from frontend.app import Window
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr('frontend.site_launch.launch', calls.append)
    host = SimpleNamespace(composed_config={'site': {'scheduler': 'slurm'}})
    Window.launch_training(host)
    assert calls == [host]
