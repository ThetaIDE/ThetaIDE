import threading
import time

from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from frontend.hub.client import HubClient
from frontend.hub.dialog import HubView
from frontend.hub.models import HubComponent

app = QApplication.instance() or QApplication([])
fetch_async = HubClient.fetch_index_async


def wait_for(predicate):
    deadline = time.monotonic() + 5
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    assert predicate()


def test_async_update_batches_callbacks_and_replaces_stale_card(tmp_path, monkeypatch):
    client = HubClient(tmp_path, tmp_path / '.thetaide')
    component = HubComponent(id='example', name='Example', kind='model', version='2.0.0',
                             is_installed=True, installed_version='1.0.0')
    client.components = [component]
    monkeypatch.setattr(client, 'fetch_index_async', lambda **kw: None)
    installed = ['1.0.0']
    monkeypatch.setattr(client.installer, 'check_installed', lambda c: (True, installed[0]))
    view = HubView(client, show_footer=False)
    view._populate_cards(client.components)
    threads, finished, progress = [], [], []
    client.componentChanged.connect(lambda *args: threads.append(QThread.currentThread()))
    client.indexLoaded.connect(lambda *args: threads.append(QThread.currentThread()))
    client.installFinished.connect(lambda *args: finished.append(args))
    client.installProgress.connect(lambda cid, percent: progress.append(percent))

    def install(**kwargs):
        assert QThread.currentThread() != app.thread()
        kwargs['progress_callback'](100, 100)
        client.installer.on_change_callback('example', 'install')
        assert not threads  # No UI notifications in the middle of installation.
        installed[0] = '2.0.0'
        return True

    monkeypatch.setattr(client.installer, 'install', install)
    view.cards['example']._on_action_clicked()
    wait_for(lambda: bool(finished))
    assert finished[0][1] is True
    assert all(thread == app.thread() for thread in threads)
    assert max(progress) < 100
    card = view.cards['example']
    assert card.component.installed_version == '2.0.0'
    assert card.btn_action.text() == 'Uninstall'
    assert card.btn_action.isEnabled() and card.progress_bar.isHidden()
    view.deleteLater()


def test_failed_update_is_visible_without_footer(tmp_path, monkeypatch):
    client = HubClient(tmp_path, tmp_path / '.thetaide')
    c = HubComponent(id='example', name='Example', kind='model', version='2.0.0',
                     is_installed=True, installed_version='1.0.0')
    client.components = [c]
    monkeypatch.setattr(client, 'fetch_index_async', lambda **kw: None)
    monkeypatch.setattr(client.installer, 'check_installed', lambda c: (True, '1.0.0'))
    def fail(**kwargs):
        kwargs['progress_callback'](100, 100)
        raise ValueError('Local source changes must be preserved')
    monkeypatch.setattr(client.installer, 'install', fail)
    view = HubView(client, show_footer=False)
    view._populate_cards(client.components)
    finished = []
    client.installFinished.connect(lambda *args: finished.append(args))
    view.cards['example']._on_action_clicked()
    wait_for(lambda: bool(finished))
    card = view.cards['example']
    assert not finished[0][1]
    assert card.operation_error.text() == 'Local source changes must be preserved'
    assert not card.operation_error.isHidden()
    assert card.progress_bar.isHidden() and card.btn_action.isEnabled()
    view.deleteLater()


def test_fetch_publishes_on_main_thread(tmp_path, monkeypatch):
    client = HubClient(tmp_path, tmp_path / '.thetaide')
    seen, gate = [], threading.Event()
    def fetch(**kwargs):
        gate.wait(2)
        return []
    monkeypatch.setattr(client, 'fetch_index_sync', fetch)
    client.indexLoaded.connect(lambda comps: seen.append(QThread.currentThread()))
    fetch_async(client)
    gate.set()
    wait_for(lambda: bool(seen))
    assert seen == [app.thread()]
