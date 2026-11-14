"""TensorBoard server run by the desktop app (QProcess) for the TensorBoard tab."""

import sys
import time
import urllib.request

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.plugins.app.bundled.tensorboard.server import TensorBoardServer  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])


def pump(predicate, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met")


def test_not_running_until_started(tmp_path):
    server = TensorBoardServer(logdir=tmp_path)
    assert server.available()
    assert not server.is_running() and server.url is None


@pytest.mark.slow
def test_start_signals_ready_then_stop_ends_it(tmp_path):
    server = TensorBoardServer(logdir=tmp_path)
    ready, stopped = [], []
    server.ready.connect(ready.append)
    server.stopped.connect(stopped.append)
    server.start()
    pump(lambda: ready)
    url = ready[0]
    assert url.startswith("http://127.0.0.1:")
    with urllib.request.urlopen(url + "data/environment", timeout=5) as response:
        assert response.status == 200

    server.start()  # starting again reuses the running server
    assert server.url == url

    server.stop()
    pump(lambda: stopped)
    assert stopped == [""]  # a deliberate stop is not reported as an error
    assert not server.is_running()
    with pytest.raises(OSError):
        urllib.request.urlopen(url + "data/environment", timeout=2)
