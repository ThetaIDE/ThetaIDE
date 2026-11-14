"""SSH connection budgets are shared by remote commands and rsync."""
import shlex
from types import SimpleNamespace

import pytest

from src.app.sites import transfer
from src.app.sites.transport import rsync_shell, ssh_arguments, ssh_session
from pathlib import Path
import os


@pytest.mark.parametrize('configured, expected', [(None, 60), (120, 120)])
def test_remote_commands_and_rsync_share_connection_timeout(tmp_path, monkeypatch, configured, expected):
    config = {'host': 'cluster', 'remote_root': '/project', '_root': str(tmp_path)}
    if configured is not None:
        config['ssh_connect_timeout_seconds'] = configured
    commands = []

    def run(arguments, **kwargs):
        commands.append(arguments)
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(transfer.subprocess, 'run', run)
    transfer.ssh(config, ['true'])
    transport = shlex.split(rsync_shell(tmp_path, config))
    assert commands[0][:len(transport)] == transport
    assert f'-oConnectTimeout={expected}' in transport
    assert '-oBatchMode=yes' in transport


@pytest.mark.parametrize('timeout', [0, -1, True, '120', 1.5, None])
def test_invalid_connection_timeout_is_rejected(tmp_path, timeout):
    with pytest.raises(ValueError, match='ssh_connect_timeout_seconds must be a positive integer'):
        ssh_arguments(tmp_path, {'ssh_connect_timeout_seconds': timeout})


def test_interactive_shell_keeps_ssh_connection_preferences(tmp_path):
    assert ssh_arguments(tmp_path, {}, interactive=True) == ['ssh', '-t']


@pytest.mark.skipif(os.name == 'nt', reason='Unix control sockets')
def test_sync_connection_is_shared_scoped_and_closed_on_failure(tmp_path, monkeypatch):
    config = {'host': 'cluster', '_root': str(tmp_path), 'remote_root': '/project'}
    calls = []
    monkeypatch.setattr('src.app.sites.transport.subprocess.run', lambda args, **kw: calls.append(args))
    with pytest.raises(RuntimeError, match='failed'):
        with ssh_session(tmp_path, config):
            args = ssh_arguments(tmp_path, config)
            assert shlex.split(rsync_shell(tmp_path, config)) == args
            assert '-oControlMaster=auto' in args
            socket = Path(next(arg.split('=', 1)[1] for arg in args if arg.startswith('-oControlPath=')))
            socket.touch()  # Model a master created lazily by the first SSH command.
            assert not any('ControlPath' in arg for arg in ssh_arguments(tmp_path, {'host': 'other'}))
            raise RuntimeError('failed')
    assert calls == [[*args, '-O', 'exit', '--', 'cluster']]
    assert not socket.parent.exists()
    assert not any('ControlPath' in arg for arg in ssh_arguments(tmp_path, config))
