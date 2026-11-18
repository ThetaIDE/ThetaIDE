"""Cluster Run uses a new shell in the terminal pane for upload and SSH."""
import subprocess
import sys

import pytest
from PyQt6.QtWidgets import QApplication, QWidget

from frontend import site_launch
from src.app.sites import transfer
from component_fixtures import install_fixture
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication(['test-site-launch'])


def test_run_creates_and_selects_new_shell_in_existing_terminal_pane(tmp_path, monkeypatch):
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWidgets import QTabWidget
    from frontend.terminal import TerminalPanel

    # Control readiness ourselves so the test never opens a real SSH connection.
    monkeypatch.setattr(QWebEngineView, 'load', lambda *_: None)
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    install_fixture(tmp_path, 'batch-scheduler', tmp_path.parent / (tmp_path.name + '-archives'))
    transfer.configure(tmp_path, 'cluster', 'fake', '/home/user/theta')
    monkeypatch.setattr(transfer, 'push', lambda *_a, **_k: pytest.fail('GUI thread started upload'))
    window = QWidget()
    window.workspace_paths = WorkspacePaths(tmp_path)
    window.tabs = QTabWidget(window)
    window.tabs.addTab(QWidget(), 'Experiment')
    # A custom terminal start folder must not change the cluster upload root.
    window.terminal_panel = TerminalPanel(cwd=str(tmp_path / 'custom'), parent=window)
    window.tabs.addTab(window.terminal_panel, 'Terminal')
    first = window.terminal_panel.terminal
    window.compose_valid = True
    # A configured site composes its connection; the scheduler's launch contribution requires it.
    window.composed_config = {'site': {'name': 'cluster', 'scheduler': 'batch',
                                       'connection': {'host': 'fake', 'remote_root': '/home/user/theta'}}}
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
        assert QApplication.clipboard().text() == 'batch-submit cartpole/single --site cluster'
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


