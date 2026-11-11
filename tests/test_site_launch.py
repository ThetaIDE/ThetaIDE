"""Cluster Run uses a new shell in the terminal pane for upload and SSH."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from frontend import site_launch
from src.app.sites import transfer
from src.app.sites.slurm import launch_in_terminal
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication(['test-site-launch'])


def test_run_creates_and_selects_new_shell_in_existing_terminal_pane(tmp_path, monkeypatch):
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import QTabWidget
    from frontend.terminal import TerminalPanel

    # Control readiness ourselves so the test never opens a real SSH connection.
    monkeypatch.setattr(QWebEngineView, 'load', lambda *_: None)
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    monkeypatch.setattr(transfer, 'push', lambda *_a, **_k: pytest.fail('GUI thread started upload'))
    window = QWidget()
    window.tabs = QTabWidget(window)
    window.tabs.addTab(QWidget(), 'Experiment')
    # A custom terminal start folder must not change the cluster upload root.
    window.terminal_panel = TerminalPanel(cwd=str(tmp_path / 'custom'), parent=window)
    window.tabs.addTab(window.terminal_panel, 'Terminal')
    first = window.terminal_panel.terminal
    window.compose_valid = True
    window.composed_config = {'site': {'name': 'cluster', 'scheduler': 'slurm'}}
    window.current_experiment = 'cartpole/single'
    try:
        site_launch.launch(window)
        panel = window.terminal_panel
        terminal = panel.terminal
        assert window.tabs.currentWidget() is panel
        assert panel.shell_count() == 2 and panel.terminals[0] is first
        assert terminal is not first
        assert terminal.cwd == str(tmp_path)
        assert first.cwd == str(tmp_path / 'custom')
        assert terminal.pty.command == [sys.executable, '-u', '-m', 'src.app.sites', 'launch',
                                        '--root', str(tmp_path), 'cluster', '--experiment', 'cartpole/single']
        assert QApplication.clipboard().text() == 'venv/bin/python3 run_pipeline.py cartpole/single'
        # Each explicit Run uses the existing New shell workflow.
        site_launch.launch(window)
        assert panel.shell_count() == 3 and panel.terminal is not terminal
        assert not hasattr(window, '_site_launch_dialog')
    finally:
        window.terminal_panel.close()
        window.close()


@pytest.mark.parametrize('started', [False, True])
def test_close_shell_terminates_pty_and_blocks_late_startup(tmp_path, monkeypatch, started):
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from frontend.terminal import TerminalPanel

    monkeypatch.setattr(QWebEngineView, 'load', lambda *_: None)
    panel = TerminalPanel(cwd=str(tmp_path))
    terminal = panel.new_shell(command=[sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        if started:
            terminal._on_terminal_ready()
            assert terminal.pty.is_alive()
            process = terminal.pty.process
        panel.close_shell()
        assert terminal not in panel.terminals
        assert not terminal.pty.is_alive()
        assert terminal.pty.master_fd is None
        if started:
            assert process.poll() is not None
        # A queued renderer callback cannot reopen a shell that was removed.
        terminal._on_terminal_ready()
        terminal.restart_session()
        assert not terminal.pty.is_alive()
    finally:
        panel.close()


def test_terminal_upload_finishes_before_interactive_ssh(tmp_path, monkeypatch, capsys):
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    events = []
    def upload(root, site, **kwargs):
        events.append(('upload', root, site))
        assert kwargs == {'live_output': True}
        print('rsync progress visible')
    def replace(program, command): events.append(('ssh', program, command))
    monkeypatch.setattr(transfer, 'push', upload)
    monkeypatch.setattr(transfer, 'sync', lambda *_a, **_k: pytest.fail('Run attempted to download results'))
    monkeypatch.setattr(os, 'execvp', replace)
    assert launch_in_terminal(tmp_path, 'cluster', 'cartpole/single') == 0
    assert [item[0] for item in events] == ['upload', 'ssh']
    command = events[1][2]
    assert command[0] == 'ssh' and '-t' in command and 'fake' in command
    assert not any('sbatch' in str(item) for item in command)
    output = capsys.readouterr().out
    assert output.index('rsync progress visible') < output.index('Opening the cluster SSH terminal')
    assert 'paste the copied command' in output
    assert 'venv/bin/python3 run_pipeline.py cartpole/single' in output
    assert 'dry_run=true' not in output and '--verify' not in output and 'site=' not in output


def test_failed_upload_never_opens_ssh(tmp_path, monkeypatch):
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    def fail(*_a, **_k): raise RuntimeError('Permission denied')
    monkeypatch.setattr(transfer, 'push', fail)
    monkeypatch.setattr(os, 'execvp', lambda *_: pytest.fail('SSH opened after failed upload'))
    with pytest.raises(RuntimeError, match='Permission denied'):
        launch_in_terminal(tmp_path, 'cluster', 'cartpole/single')


def test_live_transfer_streams_output_before_completion(capfd):
    transfer.run_transfer([sys.executable, '-u', '-c',
        'import sys; print("rsync: sending files",flush=True); print("rsync: upload failed",file=sys.stderr,flush=True)'],
        live_output=True)
    out, error = capfd.readouterr()
    assert 'rsync: sending files' in out and 'rsync: upload failed' in error


def test_live_failure_retains_diagnostics_in_terminal(capfd):
    with pytest.raises(subprocess.CalledProcessError):
        transfer.run_transfer([sys.executable, '-u', '-c',
            'import sys; print("rsync: permission denied",file=sys.stderr,flush=True); sys.exit(1)'], live_output=True)
    assert 'rsync: permission denied' in capfd.readouterr().err


def test_minimal_command_uses_project_relative_interpreter_and_quotes_experiment():
    import shlex
    from src.app.sites.slurm import submission_command
    config = {'remote_root': '/home/user/theta', 'python': '/home/user/theta/venv/bin/python3'}
    command = submission_command(config, 'cartpole/my run', 'ncshare/common')
    assert shlex.split(command) == ['venv/bin/python3', 'run_pipeline.py', 'cartpole/my run']
    assert '&&' not in command and 'site=' not in command
    config['python'] = '/home/user/env/bin/python3'
    assert shlex.split(submission_command(config, 'g/e', 'cluster'))[0] == '/home/user/env/bin/python3'


def test_upload_output_and_manual_ssh_input_share_one_real_pty(tmp_path):
    import time
    from frontend.terminal.pty_session import PtySession
    mock_ssh = tmp_path / 'mock_ssh.py'
    mock_ssh.write_text('print("SSH READY",flush=True)\ncommand=input()\nprint("USER COMMAND: "+command,flush=True)\n')
    driver = tmp_path / 'launch.py'
    project = Path(__file__).resolve().parents[1]
    driver.write_text(
        'import sys,time\n'
        f'sys.path.insert(0,{str(project)!r})\n'
        'from src.app.sites import slurm,transfer\n'
        'transfer.connection=lambda *_: {"host":"mock","remote_root":"/home/user/theta","python":"venv/bin/python3"}\n'
        'def push(*args,**kwargs):\n'
        ' assert kwargs["live_output"]\n'
        ' print("RSYNC START",flush=True)\n'
        ' time.sleep(.1)\n'
        ' print("RSYNC 100%",flush=True)\n'
        'transfer.push=push\n'
        f'slurm.ssh_arguments=lambda *_: [sys.executable,{str(mock_ssh)!r}]\n'
        f'slurm.launch_in_terminal({str(tmp_path)!r},"cluster","g/exp")\n')
    session = PtySession(command=[sys.executable, str(driver)], cwd=str(tmp_path))
    chunks, exits = [], []
    session.data_ready.connect(chunks.append)
    session.process_exited.connect(exits.append)
    try:
        assert session.start()
        deadline = time.monotonic() + 10
        while 'SSH READY' not in ''.join(chunks) and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.01)
        output = ''.join(chunks)
        assert output.index('RSYNC START') < output.index('RSYNC 100%') < output.index('SSH READY')
        assert 'USER COMMAND:' not in output
        session.write('venv/bin/python3 run_pipeline.py g/exp\n')
        deadline = time.monotonic() + 5
        while not exits and time.monotonic() < deadline:
            app.processEvents(); time.sleep(.01)
        assert 'USER COMMAND: venv/bin/python3 run_pipeline.py g/exp' in ''.join(chunks)
        assert exits == [0]
    finally:
        session.close()
