"""Per-transfer console output, isolated from other workers and process stdout."""
from contextlib import contextmanager
from contextvars import ContextVar
import codecs
import os
import shlex
import signal
import subprocess
import tempfile
import time

_sink = ContextVar('site_transfer_output', default=None)


@contextmanager
def output_to(callback):
    token = _sink.set(callback)
    try:
        yield
    finally:
        _sink.reset(token)


def emit(text):
    callback = _sink.get()
    if callback:
        for line in str(text or '').splitlines():
            callback(line)


def command(arguments):
    emit('$ ' + shlex.join([str(arg) for arg in arguments]))


def progress_flags():
    return ['--progress', '--human-readable'] if _sink.get() else []


def streaming():
    return _sink.get() is not None


def run_streamed(arguments, cancel_event=None, *, on_poll=None):
    """Stream rsync without blocking cancellation or buffering its entire output."""
    command(arguments)
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    pending, tail, offset = '', '', 0
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(arguments, stdout=output, stderr=output, start_new_session=True)
        deadline = time.monotonic() + 3600

        def drain(final=False):
            nonlocal pending, tail, offset
            while True:
                chunk = os.pread(output.fileno(), 65536, offset)
                if not chunk:
                    break
                offset += len(chunk)
                text = decoder.decode(chunk)
                tail = (tail + text)[-65536:]
                pending += text.replace('\r', '\n')
                while '\n' in pending:
                    line, pending = pending.split('\n', 1)
                    if line: emit(line)
                if len(pending) >= 8192:
                    emit(pending)
                    pending = ''
            if final:
                pending += decoder.decode(b'', final=True)
                if pending: emit(pending)

        try:
            while process.poll() is None:
                drain()
                if (cancel_event and cancel_event.is_set()) or time.monotonic() > deadline:
                    raise RuntimeError('Transfer cancelled or timed out; verified objects retained for retry')
                if on_poll: on_poll()
                try: process.wait(timeout=.2)
                except subprocess.TimeoutExpired: pass
            drain(final=True)
            emit(f'Process exited with code {process.returncode}')
            if process.returncode:
                raise RuntimeError(f'rsync failed (exit {process.returncode}): {tail.strip() or "no error output"}')
            return subprocess.CompletedProcess(arguments, process.returncode)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                drain(final=True)
