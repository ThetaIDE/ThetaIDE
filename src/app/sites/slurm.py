"""Reusable SLURM site workflow; user transport configuration lives in src/usr/sites."""
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


def submission_command(config, experiment, site):
    """The saved experiment owns its site/options; the shell is already at project root."""
    python = Path(config['python'])
    remote_root = Path(config['remote_root'])
    if python.is_absolute() and python.is_relative_to(remote_root):
        python = python.relative_to(remote_root)
    return shlex.join([str(python), 'run_pipeline.py', experiment])


def launch_in_terminal(root, site, experiment):
    """Called inside the local terminal PTY. Upload, then replace this process with SSH."""
    import os
    from . import transfer
    root = Path(root).resolve()
    config = transfer.connection(root, site)
    print(f'Theta: preparing {experiment} for {site}', flush=True)
    transfer.push(root, site, live_output=True)
    print('\nProject upload complete. Opening the cluster SSH terminal.', flush=True)
    print('Once connected, paste the copied command and press Enter to validate and submit.', flush=True)
    print('Run command (also copied by the desktop):\n' + submission_command(config, experiment, site), flush=True)
    command = [*ssh_arguments(root, config, True), '--', config['host'],
               'cd ' + shlex.quote(config['remote_root']) + ' && exec bash -l']
    os.execvp(command[0], command)
    return 0


def bootstrap_command(root, site):
    from .transfer import connection
    from src.app.workspace import WorkspacePaths
    import json
    root = Path(root).resolve()
    from .catalog import load
    if load(root, site).get('scheduler') == 'local':
        return shlex.join([sys.executable, '-m', 'pip', 'install', '-e', '.'])
    config = connection(root, site)
    # Collect dependencies from the same native providers that project sync uploads.
    requirements = set()
    for manifest in WorkspacePaths(root).manifests():
        requirements.update(json.loads(manifest.read_text()).get('dependencies', {}).get('pip', []))
    profile = load(root, site)
    runtime_python = profile.get('python_executable')
    if runtime_python in (None, 'auto'):
        runtime_python = str(Path(config['remote_root']) / profile.get('venv_dir', 'venv') / 'bin/python3')
    elif not Path(runtime_python).is_absolute():
        runtime_python = str(Path(config['remote_root']) / runtime_python)
    environment_dir = str(Path(runtime_python).parent.parent)
    if Path(runtime_python).parent.name != 'bin':
        raise ValueError('Generate setup for a venv bin/python interpreter, or provision the custom interpreter manually')
    setup = ('mkdir -p ' + shlex.quote(config['remote_root']) + ' && cd ' + shlex.quote(config['remote_root']) +
             ' && python3 -m venv ' + shlex.quote(environment_dir))
    install = ('cd ' + shlex.quote(config['remote_root']) + ' && ' +
               shlex.join([config['python'], '-m', 'pip', 'install', '-e', '.[optuna]', *sorted(requirements)]))
    # Only emit text: the user chooses when to execute SSH, upload and package installation.
    return '\n'.join(['set -e', shlex.join([*ssh_arguments(root, config, True), '--', config['host'], setup]),
        shlex.join([sys.executable, str(root / 'run_pipeline.py'), 'site', 'push', site, '--root', str(root)]),
        shlex.join([*ssh_arguments(root, config, True), '--', config['host'], install])])


def shortcuts(root, site):
    from .transfer import connection
    config = connection(root, site)
    directory = safe_path(root, config.get('_shortcuts', {}).get('directory', 'src/usr/sites/' + site.split('/')[0] + '/bin'))
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    import hashlib
    tag = hashlib.sha256(site.encode()).hexdigest()[:12]
    commands = {
        'ssh': shlex.join([sys.executable, '-m', 'src.app.sites.slurm', site, '--root', str(root)]),
        'push': shlex.join([sys.executable, str(Path(root) / 'run_pipeline.py'), 'site', 'push', site, '--root', str(root)]),
        'sync': shlex.join([sys.executable, str(Path(root) / 'run_pipeline.py'), 'site', 'sync', site, '--root', str(root)]),
    }
    for action, command in commands.items():
        path = directory / f'theta-{action}-{tag}'
        path.write_text('#!/bin/sh\ncd ' + shlex.quote(str(root)) + ' || exit 1\nexec ' + command + '\n')
        path.chmod(0o700)
        paths.append(str(path))
    return paths


def verify_environment(root):
    import json
    from importlib.metadata import version, PackageNotFoundError
    from packaging.requirements import Requirement
    from src.app.workspace import WorkspacePaths
    from .storage import digest, read_json
    root = Path(root).resolve()
    source = read_json(root / '.thetaide/source.json')
    if not source.get('files'): raise ValueError('Upload the project before submitting')
    for relative, expected in source['files'].items():
        if digest(safe_path(root, relative)) != expected:
            raise ValueError(f'Uploaded source differs: {relative}; upload again')
    for manifest in WorkspacePaths(root).manifests():
        for value in json.loads(manifest.read_text()).get('dependencies', {}).get('pip', []):
            requirement = Requirement(value)
            if requirement.marker and not requirement.marker.evaluate(): continue
            try: installed = version(requirement.name)
            except PackageNotFoundError as exc:
                raise ValueError(f'Missing component dependency {value}; run the Settings setup script') from exc
            if requirement.specifier and installed not in requirement.specifier:
                raise ValueError(f'Component dependency {value} conflicts with installed {installed}')


if __name__ == '__main__':
    import argparse, os
    from .transfer import connection
    parser = argparse.ArgumentParser()
    parser.add_argument('site', nargs='?')
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        verify_environment(args.root)
    else:
        config = connection(args.root, args.site)
        command = [*ssh_arguments(args.root, config, True), '--', config['host'],
                   'cd ' + shlex.quote(config['remote_root']) + ' && exec bash -l']
        os.execvp(command[0], command)
