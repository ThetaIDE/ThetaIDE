"""Manual sync UI must stay offline until explicitly invoked."""
import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PyQt6.QtWidgets import QApplication

from frontend.site_sync import SiteSyncControl
from src.app.sites import transfer
from src.app.workspace import WorkspacePaths

app = QApplication.instance() or QApplication([])


def test_sync_is_explicit_and_reports_completion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(WorkspacePaths, 'for_process', classmethod(lambda cls: cls(tmp_path)))
    transfer.configure(tmp_path, 'cluster', 'alias', '/cluster/project')
    monkeypatch.setattr(transfer, 'sync', lambda root, site, mode: calls.append((site, mode)) or {'last_sync': '2026-11-09T12:00:00Z'})
    widget = SiteSyncControl('metrics')
    widget.populate()
    assert calls == []
    completed = []
    widget.synced.connect(completed.append)
    widget.start('cluster', 'sync')
    assert not widget.button.isEnabled()
    widget.backend.wait()
    app.processEvents()
    assert calls == [('cluster', 'metrics')]
    assert completed == ['cluster']
    assert '12:00:00' in widget.status.text()
    assert widget.button.isEnabled()


def test_remote_stop_never_cancels_a_local_job():
    from frontend.app import Window
    host = SimpleNamespace(selected={'remote': {'site': 'cluster'}}, active={'simulated': False},
                           jobs=SimpleNamespace(cancel=lambda *args: (_ for _ in ()).throw(AssertionError('cancel called'))))
    Window.stop_run(host)


def test_remote_queue_command_is_inactive():
    from frontend.app import Window
    host = SimpleNamespace(composed_config={'site': {'scheduler': 'slurm'}})
    Window.add_to_queue(host, start=True)
