"""The Run window appears before work, and upload/SSH share its terminal."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QWidget, QPushButton

from frontend import site_launch
from src.app.sites import transfer
from src.app.sites.slurm import launch_in_terminal
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication(['test-site-launch'])


@pytest.mark.parametrize('dismiss', ['x', 'escape', 'parent'])
def test_run_opens_terminal_immediately_and_reuses_visible_window(tmp_path, monkeypatch, dismiss):
    import frontend.terminal
    panels = []
    class Terminal(QWidget):
        session_started = pyqtSignal()
        session_exited = pyqtSignal(int)
        def __init__(self, cwd, command, parent=None):
            super().__init__(parent)
            self.cwd, self.command = cwd, command
            assert not window.start_button.isEnabled()
            self.closed = False
            panels.append(self)
        def focus_terminal(self):
            self.setFocus()
        def close(self):
            self.closed = True
            return super().close()
    monkeypatch.setattr(frontend.terminal, 'TerminalWidget', Terminal)
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    monkeypatch.setattr(transfer, 'push', lambda *_a, **_k: pytest.fail('GUI thread started hidden upload'))
    window = QWidget()
    window.resize(1200, 800)
    window.show()
    window.compose_valid = True
    window.composed_config = {'site': {'name': 'cluster', 'scheduler': 'slurm'}}
    window.current_experiment = 'cartpole/single'
    window.start_button = QPushButton('Play', window)
    window.start_button.clicked.connect(lambda: site_launch.launch(window))
    site_launch.launch(window)
    dialog = window._site_launch_dialog
    assert dialog.isVisible() and len(panels) == 1
    assert not window.start_button.isEnabled() and window._site_terminal_loading
    assert window.start_button.graphicsEffect().opacity() == .35
    assert not dialog.isWindow() and dialog.parentWidget() is window
    assert dialog.body_layout.count() == 2
    assert dialog.body_layout.itemAt(1).widget() is panels[0]
    assert dialog.geometry() == window.rect()
    window.resize(900, 600)
    app.processEvents()
    assert dialog.geometry() == window.rect()
    assert dialog.rect().contains(dialog.card.geometry())
    # Clicking the backdrop keeps the terminal open.
    QTest.mouseClick(dialog, Qt.MouseButton.LeftButton)
    assert dialog.isVisible()
    window.start_button.click()
    assert len(panels) == 1
    command = panels[0].command
    assert command[:5] == [sys.executable, '-u', '-m', 'src.app.sites', 'launch']
    assert command[-2:] == ['--experiment', 'cartpole/single']
    assert str(tmp_path) in command and 'cluster' in command
    assert QApplication.clipboard().text() == 'venv/bin/python3 run_pipeline.py cartpole/single'
    panels[0].session_started.emit()
    assert window.start_button.isEnabled() and not window._site_terminal_loading
    assert window.start_button.graphicsEffect() is None
    site_launch.launch(window)
    assert window._site_launch_dialog is dialog and len(panels) == 1
    if dismiss == 'x':
        dialog.close_button.click()
    elif dismiss == 'escape':
        child = QLineEdit(panels[0])
        child.show()
        child.setFocus()
        override = QKeyEvent(QEvent.Type.ShortcutOverride, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(child, override)
        assert override.isAccepted()
        QTest.keyClick(child, Qt.Key.Key_Escape)
    else:
        window.close()
    assert not dialog.isVisible()
    assert panels[0].closed and not window._site_launch_pending
    window.close()


@pytest.mark.parametrize('started', [False, True])
@pytest.mark.parametrize('dismiss', ['escape', 'x', 'parent'])
def test_overlay_close_terminates_pty_and_blocks_late_startup(tmp_path, started, dismiss):
    from frontend.terminal.terminal_widget import TerminalWidget

    window = QWidget()
    window.resize(900, 600)
    window.show()
    dialog = site_launch.TerminalOverlay(window)
    # A real PTY process stands in for the upload driver, which execs SSH in
    # place. No cluster connection is needed to verify its lifetime.
    terminal = TerminalWidget(cwd=str(tmp_path), parent=dialog,
                              command=[sys.executable, '-c', 'import time; time.sleep(60)'])
    dialog.body_layout.addWidget(terminal)
    dialog.finished.connect(lambda _: terminal.close())
    dialog.show()
    try:
        if started:
            terminal._on_terminal_ready()
            assert terminal.pty.is_alive()
            process = terminal.pty.process
        if dismiss == 'escape':
            QTest.keyClick(terminal.web_view.focusProxy() or terminal.web_view, Qt.Key.Key_Escape)
        elif dismiss == 'x':
            dialog.close_button.click()
        else:
            window.close()
        assert not dialog.isVisible()
        assert not terminal.pty.is_alive()
        assert terminal.pty.master_fd is None
        if started:
            assert process.poll() is not None
        # Queued renderer callbacks and restart requests must not reopen SSH.
        terminal._on_terminal_ready()
        terminal.restart_session()
        app.processEvents()
        assert not terminal.pty.is_alive()
    finally:
        terminal.close()
        window.close()


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
