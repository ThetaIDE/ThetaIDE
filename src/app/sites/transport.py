"""Shared SSH/rsync transport, independent of scheduler implementations."""
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
import os
import shlex
import subprocess
import sys
import tempfile
from .storage import safe_path

_session = ContextVar('site_ssh_session', default=None)


@contextmanager
def ssh_session(root, config):
    """Share one sync-owned connection, including remote helpers and rsync."""
    if os.name == 'nt':
        # Native Windows OpenSSH does not support Unix control sockets.
        yield
        return
    with tempfile.TemporaryDirectory(prefix='theta-ssh-', dir='/tmp') as directory:
        socket = str(Path(directory) / 'control')
        identity = (str(Path(root).resolve()), config.get('host'), config.get('ssh_config'))
        token = _session.set((identity, socket))
        try:
            yield
        finally:
            try:
                if Path(socket).exists():
                    subprocess.run([*ssh_arguments(root, config), '-O', 'exit', '--', config['host']],
                                   capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass  # ControlPersist also bounds the lifetime after a failed close.
            finally:
                _session.reset(token)


def ssh_arguments(root, config, interactive=False):
    args = ['ssh']
    if config.get('ssh_config'):
        args += ['-F', str(safe_path(root, config['ssh_config']))]
    if interactive: args += ['-t']
    else:
        timeout = config.get('ssh_connect_timeout_seconds', 60)
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ValueError('connection.ssh_connect_timeout_seconds must be a positive integer')
        args += ['-oBatchMode=yes', f'-oConnectTimeout={timeout}']
        session = _session.get()
        identity = (str(Path(root).resolve()), config.get('host'), config.get('ssh_config'))
        if session and session[0] == identity:
            args += ['-oControlMaster=auto', '-oControlPersist=60', '-oControlPath=' + session[1]]
    return args


def rsync_shell(root, config):
    return shlex.join(ssh_arguments(root, config))


