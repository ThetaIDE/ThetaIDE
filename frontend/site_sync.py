"""Manual site transfers, with all network work on a dedicated background worker."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QHBoxLayout, QLineEdit, QMenu, QMessageBox, QPushButton, QVBoxLayout, QWidget

from src.app.sites import profiles
from src.app.sites import transfer
from src.app.workspace import WorkspacePaths
from .local_backend import LocalBackend
from .widgets import ComboBox, label


class SiteSyncControl(QWidget):
    synced = pyqtSignal(str)
    transfer_progress = pyqtSignal(object, object)

    def __init__(self, mode, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.paths = WorkspacePaths.for_process()
        self.backend = LocalBackend(self)  # separate pool: a transfer must not block chart/config reads
        self.button = QPushButton('Sync Metrics' if mode == 'metrics' else 'Sync Results')
        self.button.setObjectName('siteSyncButton')
        self.button.setToolTip('Pull committed scalar metrics and status. Choose diagnostics explicitly for monitor arrays; Results retrieves all owned outputs.')
        self.menu = QMenu(self.button)
        self.button.setMenu(self.menu)
        self.menu.aboutToShow.connect(self.populate)
        self.status = label('', 'muted')
        self.transfer_progress.connect(lambda done, total: self.status.setText(f'{done / 1048576:.1f}/{total / 1048576:.1f} MiB'))
        self.cancel_button = QPushButton('Cancel transfer')
        self.cancel_button.hide()
        self.cancel_button.clicked.connect(lambda: self.cancel_event.set())
        from src.app.sites.storage import read_json
        previous = read_json(self.paths.root / '.thetaide/site_sync.json')
        latest = max((value.get('last_sync', '') for value in previous.values()), default='')
        if latest:
            self.status.setText('Last synced ' + latest[11:19])
            self.status.setToolTip(latest)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.button)
        layout.addWidget(self.status)
        layout.addWidget(self.cancel_button)

    def populate(self):
        self.menu.clear()
        for site, config in transfer.connections(self.paths.root).items():
            if not config.get('host') or not config.get('remote_root'): continue
            self.menu.addAction('Pull ' + site, lambda checked=False, site=site: self.start(site, 'sync'))
            if self.mode == 'metrics':
                self.menu.addAction('Pull diagnostics from ' + site, lambda checked=False, site=site: self.start(site, 'diagnostics'))
            self.menu.addAction('Sync Project to ' + site, lambda checked=False, site=site: self.start(site, 'push'))
            self.menu.addAction('Check ' + site, lambda checked=False, site=site: self.start(site, 'check'))
        self.menu.addSeparator()
        notice = self.menu.addAction('Edit site profiles in Components')
        notice.setEnabled(False)

    def start(self, site, action):
        import threading
        self.cancel_event = threading.Event()
        self.cancel_button.setVisible(action in ('sync', 'diagnostics'))
        self.button.setEnabled(False)
        self.status.setText(f'{site}: {"syncing" if action == "sync" else action}…')
        if action == 'diagnostics': function = lambda: transfer.sync(self.paths.root, site, self.mode, diagnostics=True, cancel_event=self.cancel_event, progress=self.transfer_progress.emit)
        elif action == 'sync': function = lambda: transfer.sync(self.paths.root, site, self.mode, cancel_event=self.cancel_event, progress=self.transfer_progress.emit)
        elif action == 'push': function = lambda: transfer.push(self.paths.root, site)
        else: function = lambda: transfer.check(self.paths.root, site)
        def done(result, error):
            self.button.setEnabled(True)
            self.cancel_button.hide()
            if error:
                self.status.setText(f'{site}: sync failed' if action in ('sync', 'diagnostics') else f'{site}: {action} failed')
                self.status.setToolTip(error)
                QMessageBox.warning(self, 'Site Transfer', error)
                return
            self.status.setText(f'{site}: synced {result["last_sync"][11:19]}' if action in ('sync', 'diagnostics') else f'{site}: {action} complete')
            self.status.setToolTip(str(result.get('last_sync') or result))
            if action in ('sync', 'diagnostics'): self.synced.emit(site)
            if action == 'push': QMessageBox.information(self, 'Submit on the cluster', result['command'])
        self.backend._submit(function, callback=done)
