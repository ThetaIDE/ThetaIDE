"""Real process stderr, shutdown/final publication, and remote Results log previews."""
import builtins
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from src.app.pipeline.datasets import _execute_training
from src.app.services import catalog
from src.app.services.process_output import ProcessOutput
from src.app.sites import execution, transfer
from src.app.sites.snapshot import freeze
from src.app.sites.storage import read_json, workspace_id, write_json

REPO = Path(__file__).resolve().parents[1]


def project(root, pipeline):
    (root / 'src').mkdir(parents=True)
    (root / 'src/worker.py').write_text('# fixture\n')
    (root / 'in/config').mkdir(parents=True)
    (root / 'in/config/config.yaml').write_text('defaults: [{site: local}]\n')
    (root / 'run_pipeline.py').write_text(pipeline)
    workspace_id(root)
    return root


def submit(root, monkeypatch, *, managed=None):
    cfg = {'group': 'demo', 'experiment_id': 'run',
           'site': {'name': 'cluster', 'python_executable': sys.executable}, 'resources': {}}
    if managed:
        cfg['site']['execution'] = managed
    # Submission is faked; workers and their output pipes are real local processes.
    with monkeypatch.context() as patch:
        patch.setattr(execution.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='321\n', stderr=''))
        return execution.submit(root, 'demo/run', [], OmegaConf.create(cfg))


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


def test_cluster_pipeline_err_contains_original_nested_exception(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', '321')
    pipeline = f'''import os, pathlib, sys
sys.path.insert(0, {str(REPO)!r})
from src.app.pipeline.datasets import _execute_training
print('pipeline started', flush=True)
_execute_training([sys.executable, '-c', "raise RuntimeError('original child failure')"], dict(os.environ), pathlib.Path('results/executions/demo/child.log'))
'''
    root = project(tmp_path / 'control', pipeline)
    record = submit(root, monkeypatch)
    assert execution.run(root, record['execution_id']) == 1
    directory = execution.receipt_path(root, record['execution_id']).parent
    assert 'original child failure' in (directory / 'pipeline.err').read_text()
    assert 'CalledProcessError' in (directory / 'pipeline.err').read_text()
    assert 'pipeline started' not in (directory / 'pipeline.err').read_text()
    assert 'pipeline started' in (directory / 'pipeline.log').read_text()
    assert 'original child failure' in (directory / 'pipeline.log').read_text()


def test_batch_term_reaches_supervisor_and_seals_error_logs(tmp_path, monkeypatch):
    ready = tmp_path / 'ready'
    pipeline = f'''import pathlib,signal,sys,time
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
pathlib.Path({str(ready)!r}).touch()
print('training started', flush=True)
try:
    while True: time.sleep(.02)
finally:
    print('training cleanup ran', file=sys.stderr, flush=True)
'''
    root = project(tmp_path / 'control', pipeline)
    upload = tmp_path / 'upload'
    freeze(root, upload)
    transfer.activate(root, upload)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    work = tmp_path / 'work'
    record = submit(root, monkeypatch, managed={'schema_version': 1, 'runtime': 'scratch', 'work_root': str(work),
                    'scratch_root': str(scratch), 'stage_out_seconds': 4, 'scratch_reserve_bytes': 1})
    directory = execution.receipt_path(root, record['execution_id']).parent
    script = (directory / 'job.sbatch').read_text()
    assert script.splitlines()[-1].startswith('exec ')
    # Use the actual generated shell with the working tree as the framework import root.
    # The test recipe remains frozen in the isolated fixture's source archive.
    harness = f"import sys; sys.path.insert(0, {str(REPO)!r}); from src.app.sites.__main__ import main; sys.exit(main())"
    script = script.replace('-m src.app.sites _run', shlex.join(['-c', harness, '_run']))
    batch = tmp_path / 'batch.sh'
    batch.write_text(script)
    environment = dict(os.environ, SLURM_JOB_ID='321', PYTHONDONTWRITEBYTECODE='1')
    with (tmp_path / 'supervisor.err').open('w') as errors:
        process = subprocess.Popen(['/bin/bash', str(batch)], env=environment, stdout=subprocess.DEVNULL,
                                   stderr=errors, start_new_session=True)
        try:
            deadline = time.monotonic() + 15
            while not ready.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail('worker did not start: ' + (tmp_path / 'supervisor.err').read_text())
                time.sleep(.02)
            os.kill(process.pid, signal.SIGTERM)
            assert process.wait(timeout=15) == 0
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    final = read_json(directory / 'submission.json')
    assert final['state'] == 'cancelled' and final['ended_at']
    assert final['publication_state'] == 'sealed'
    manifest = read_json(work / 'results/slurm' / record['execution_id'] / 'publication.json')
    assert manifest['sealed']
    item = next(item for item in manifest['files'] if item['path'].endswith('/pipeline.err'))
    text = b''.join((work / chunk['source']).read_bytes()[chunk.get('offset', 0):chunk.get('offset', 0)+chunk.get('length', chunk['size'])]
                    for chunk in item['chunks']).decode()
    assert 'training cleanup ran' in text and 'SIGTERM' in text
    assert 'training cleanup ran' in (tmp_path / 'supervisor.err').read_text()
    assert not list(scratch.iterdir())  # Cleanup happens only after durable sealing.


def test_sb3_import_failure_preserves_the_dependency_cause(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location('sb3_error_runner', REPO / 'src/usr/algorithms/sb3-dev/sb3_runner.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    original = builtins.__import__
    def fail(name, *args, **kwargs):
        if name == 'stable_baselines3':
            raise ModuleNotFoundError("No module named 'transitive_dependency'")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', fail)
    with pytest.raises(ModuleNotFoundError, match='transitive_dependency'):
        runner.main.__wrapped__(OmegaConf.create({}))
    captured = capsys.readouterr()
    assert 'could not be imported' in captured.err
    assert 'is not installed' not in captured.out + captured.err


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
    logs = tmp_path / 'results/logs'
    monkeypatch.setattr(catalog, 'LOGS_DIR', logs)
    directory = tmp_path / 'results/slurm/execution'
    directory.mkdir(parents=True)
    for name, text in files.items():
        (directory / name).write_text(text)
    write_json(tmp_path / '.thetaide/site_sync.json', {'cluster': {'executions': [dict(
        execution_id='execution', group='demo', result_experiment_id='run', state='failed',
        publication_manifest={'sealed': True})]}})
    # Real catalog and backend dispatch, synchronous only to keep the UI check deterministic.
    backend = LocalBackend()
    backend._submit = lambda function, *args, callback, **kwargs: callback(function(*args), None)
    panel = ResultsPanel(backend)
    try:
        panel.open_run('demo', 'run', failure=True)
        browser = panel.browser('Files')
        assert browser.current[1]['path'] == f'slurm/execution/{expected}'
        assert files[expected] in browser.viewers['text'].browser.toPlainText()
        assert catalog.run_manifest('demo', 'run', completed_only=True)['artifacts'][0]['kind'] == 'log'
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
