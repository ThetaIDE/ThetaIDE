"""Local transport preflight and actionable errors, without cluster connections."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from src.app.sites import transfer


def test_upload_rejects_apple_openrsync_before_network_or_staging(tmp_path, monkeypatch):
    monkeypatch.setattr(transfer, 'connection', lambda *_: {'rsync_executable': 'rsync'})
    calls = []
    def version(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='openrsync: protocol version 29\nrsync version 2.6.9 compatible\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', version)
    monkeypatch.setattr(transfer, 'ssh', lambda *_a, **_k: pytest.fail('network attempted before compatibility check'))
    monkeypatch.setattr(transfer, 'freeze', lambda *_: pytest.fail('staging started before compatibility check'))
    with pytest.raises(ValueError, match='brew install rsync'):
        transfer.push(tmp_path, 'ncshare/common')
    assert calls == [['rsync', '--version']]
    assert not (tmp_path / '.thetaide').exists()


def test_preflight_honors_explicit_modern_rsync_path(monkeypatch):
    calls = []
    def version(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout='rsync  version 3.4.1  protocol version 32\n', stderr='')
    monkeypatch.setattr(subprocess, 'run', version)
    assert transfer.require_local_rsync({'rsync_executable': '/opt/homebrew/bin/rsync'}) == '/opt/homebrew/bin/rsync'
    assert calls == [['/opt/homebrew/bin/rsync', '--version']]


def test_missing_rsync_has_actionable_error(monkeypatch):
    def missing(*args, **kwargs): raise FileNotFoundError('missing executable')
    monkeypatch.setattr(subprocess, 'run', missing)
    with pytest.raises(ValueError, match='Cannot run configured rsync'):
        transfer.require_local_rsync({'rsync_executable': '/missing/rsync'})


@pytest.mark.parametrize('cancellable', [False, True])
def test_transfer_failure_preserves_actual_stderr(cancellable):
    import threading
    with pytest.raises(RuntimeError, match='unrecognized option.*--protect-args'):
        transfer.run_transfer([sys.executable, '-c',
            'import sys; print("unrecognized option: --protect-args",file=sys.stderr); sys.exit(1)'],
            cancel_event=threading.Event() if cancellable else None)
