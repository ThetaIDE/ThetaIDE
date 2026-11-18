"""Real process stderr, shutdown/final publication, and remote Results log previews."""
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from src.app.pipeline.datasets import _execute_training
from src.app.services import catalog
from src.app.services.process_output import ProcessOutput
from src.app.sites import execution
from src.app.sites.storage import write_json

REPO = Path(__file__).resolve().parents[1]


def test_nested_training_errors_keep_stderr_identity_and_combined_log(tmp_path, capsys):
    code = "import sys; print('training progress', flush=True); raise ImportError('specific missing dependency')"
    combined = tmp_path / 'worker.log'
    with pytest.raises(subprocess.CalledProcessError):
        _execute_training([sys.executable, '-u', '-c', code], dict(os.environ), combined)
    captured = capsys.readouterr()
    assert 'training progress' in captured.out
    assert 'specific missing dependency' not in captured.out
    assert 'ImportError: specific missing dependency' in captured.err
    assert 'training progress' in combined.read_text()
    assert 'ImportError: specific missing dependency' in combined.read_text()
    assert 'ImportError: specific missing dependency' in combined.with_suffix('.err').read_text()
    assert 'training progress' not in combined.with_suffix('.err').read_text()


def test_both_output_pipes_drain_without_deadlock_or_lost_tail():
    code = "import sys; sys.stderr.write('e'*200000+'\\nerror tail'); sys.stderr.flush(); sys.stdout.write('o'*200000+'\\nstdout tail'); sys.stdout.flush()"
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    combined, errors = io.StringIO(), io.StringIO()
    capture = ProcessOutput(process, combined, errors)
    try:
        assert process.wait(timeout=10) == 0
        capture.finish()
        assert errors.getvalue() == 'e' * 200000 + '\nerror tail'
        assert 'stdout tail' in combined.getvalue() and 'error tail' in combined.getvalue()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_allocation_cli_keeps_supervisor_traceback(monkeypatch, capsys, tmp_path):
    from src.app.sites.__main__ import main
    def fail(*args):
        raise RuntimeError('supervisor setup failed')
    monkeypatch.setattr(execution, 'run', fail)
    assert main(['_run', '--root', str(tmp_path), 'a' * 32]) == 2
    assert 'Traceback' in capsys.readouterr().err


@pytest.mark.parametrize('files,expected', [
    ({'pipeline.err': 'real traceback', 'pipeline.log': 'combined', 'scheduler.err.preview.log': 'cancelled'}, 'pipeline.err'),
    ({'pipeline.err': '', 'pipeline.log': 'combined', 'scheduler.err.preview.log': 'cancelled'}, 'scheduler.err.preview.log'),
    ({'pipeline.err': '', 'pipeline.log': 'old combined traceback', 'scheduler.err.preview.log': ''}, 'pipeline.log'),
])
def test_remote_failed_run_opens_nonempty_log_without_local_job_id(tmp_path, monkeypatch, files, expected):
    from PyQt6.QtWidgets import QApplication
    from frontend.local_backend import LocalBackend
    from frontend.results.panel import ResultsPanel
    app = QApplication.instance() or QApplication([])
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(tmp_path)
    directory = tmp_path / 'src/usr/results/sites/execution'
    directory.mkdir(parents=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    write_json(tmp_path / '.thetaide/site_sync.json', {'cluster': {'executions': [dict(
        execution_id='execution', group='demo', result_experiment_id='run', state='failed',
        publication_manifest={'sealed': True})]}})
    # Real catalog and backend dispatch, synchronous only to keep the UI check deterministic.
    backend = LocalBackend(workspace=paths)
    backend._submit = lambda function, *args, callback, **kwargs: callback(function(*args), None)
    panel = ResultsPanel(backend)
    try:
        panel.open_run('demo', 'run', failure=True)
        browser = panel.browser('Files')
        assert browser.current[1]['path'] == f'sites/execution/{expected}'
        assert files[expected] in browser.viewers['text'].browser.toPlainText()
        assert catalog.run_manifest('demo', 'run', completed_only=True, workspace=paths)['artifacts'][0]['kind'] == 'log'
    finally:
        panel.close()


def test_output_storage_failure_still_drains_and_stops_the_child():
    class Unwritable:
        def write(self, text):
            raise OSError('storage unavailable')
    code = "import sys; sys.stderr.write('e'*200000); sys.stderr.flush(); sys.stdout.write('o'*200000); sys.stdout.flush()"
    process = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    errors, console = io.StringIO(), io.StringIO()
    capture = ProcessOutput(process, Unwritable(), errors, stderr=console)
    try:
        capture.write_error('shutdown diagnostic')  # This must not prevent process cleanup.
        assert process.wait(timeout=10) == 0
        with pytest.raises(OSError, match='retain process output'):
            capture.finish()
        assert 'shutdown diagnostic' in errors.getvalue() and 'e' * 200000 in errors.getvalue()
        assert console.getvalue() == errors.getvalue()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
