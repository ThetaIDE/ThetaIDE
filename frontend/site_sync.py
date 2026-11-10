"""Pull-only site controls, with network work on a dedicated background worker."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QMenu, QMessageBox, QPushButton, QWidget

from src.app.sites import transfer
from src.app.workspace import WorkspacePaths
from .local_backend import LocalBackend
from .widgets import label


class SiteSyncControl(QWidget):
    synced = pyqtSignal(str)
    transfer_progress = pyqtSignal(object, object)
    console_output = pyqtSignal(str)

    def __init__(self, mode, parent=None, *, run_provider=None):
        super().__init__(parent)
        self.mode = mode
        self.run_provider = run_provider
        self.console_output.connect(self._console_log)
        self.paths = WorkspacePaths.for_process()
        self.backend = LocalBackend(self)  # separate pool: a transfer must not block chart/config reads
        self.button = QPushButton('Sync Metrics' if mode == 'metrics' else 'Sync Results')
        self.button.setObjectName('siteSyncButton')
        self.button.setToolTip('Download metrics and status from the cluster. Choose diagnostics for monitor arrays.' if mode == 'metrics' else 'Download published results and monitor data from the cluster.')
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

    def _console_log(self, message):
        log = getattr(self.window(), 'log', None)
        if callable(log):
            log(message)

    def populate(self):
        self.menu.clear()
        if self.run_provider is not None:
            run = self.run_provider() or {}
            remote = run.get('remote') or {}
            site, execution_id = remote.get('site'), remote.get('execution_id')
            action = self.menu.addAction('Sync experiment')
            action.setEnabled(bool(site and execution_id))
            action.setToolTip('Download all published outputs for this experiment run, including diagnostics and checkpoints.'
                              if site and execution_id else 'Select a synced remote experiment run first.')
            if site and execution_id:
                action.triggered.connect(lambda checked=False, site=site, execution_id=execution_id:
                                         self.start(site, 'experiment', execution_id=execution_id))
            self.menu.addSeparator()
        for site, config in transfer.connections(self.paths.root).items():
            if not config.get('host') or not config.get('remote_root'): continue
            self.menu.addAction(('Pull metrics from ' if self.mode == 'metrics' else 'Pull results from ') + site, lambda checked=False, site=site: self.start(site, 'sync'))
            if self.mode == 'metrics':
                self.menu.addAction('Pull diagnostics from ' + site, lambda checked=False, site=site: self.start(site, 'diagnostics'))
        self.menu.addSeparator()
        notice = self.menu.addAction('Edit site profiles in Components')
        notice.setEnabled(False)

    def start(self, site, action, *, execution_id=None):
        if action not in ('sync', 'diagnostics', 'experiment'):
            raise ValueError('Sync controls only download results. Use Run to upload experiment code.')
        if action == 'experiment' and not execution_id:
            raise ValueError('Select a remote experiment run before syncing the experiment.')
        transfer_mode = 'results' if action == 'experiment' else self.mode
        options = {'execution_id': execution_id} if execution_id else {}
        import threading
        self.cancel_event = threading.Event()
        self.cancel_button.show()
        self.button.setEnabled(False)
        self.status.setText(f'{site}: downloading…')
        if action == 'diagnostics': function = lambda: transfer.sync(self.paths.root, site, transfer_mode, **options, diagnostics=True, cancel_event=self.cancel_event, progress=self.transfer_progress.emit)
        else: function = lambda: transfer.sync(self.paths.root, site, transfer_mode, **options, cancel_event=self.cancel_event, progress=self.transfer_progress.emit)
        def done(result, error):
            self.button.setEnabled(True)
            self.cancel_button.hide()
            if error:
                self.status.setText(f'{site}: download failed')
                self.status.setToolTip(error)
                self.console_output.emit(f'[site {site}] {action} failed: {error}')
                QMessageBox.warning(self, 'Site Transfer', error)
                return
            self.status.setText(f'{site}: synced {result["last_sync"][11:19]}')
            self.status.setToolTip(str(result.get('last_sync') or result))
            prefix = f'[site {site}] {action} complete'
            updated, unchanged = result.get('transferred', 0), result.get('skipped', 0)
            if updated:
                summary = f'{updated} files saved, {unchanged} unchanged'
            elif unchanged:
                summary = f'No new files; everything is up to date ({unchanged} unchanged)'
            else:
                summary = 'No published files available to download'
            if 'bytes' in result:
                summary += f'; {result["bytes"]:,} bytes downloaded'
            self.console_output.emit(f'{prefix}: {summary}.')
            self.console_output.emit(f'[site {site}] Local destination: {self.paths.root / "results"}')
            for error in result.get('diagnostic_report_errors', []):
                self.console_output.emit(f'[site {site}] Monitor report warning: {error}')
            self.synced.emit(site)
        def logged_transfer():
            from src.app.sites.transfer_output import output_to
            emit = lambda message: self.console_output.emit(f'[site {site}] {message}')
            with output_to(emit):
                emit(f'Starting download from {site} (mode={transfer_mode}, execution={execution_id or "all"}, workspace={self.paths.root})')
                return function()
        self.backend._submit(logged_transfer, callback=done)


class ExperimentSiteSyncControl(SiteSyncControl):
    """Pull the selected experiment's site, including monitor diagnostics."""

    def __init__(self, parent=None, *, button=None):
        super().__init__('results', parent)
        from .tree_tools import square_tool_button
        self._toolbar_button = button is not None
        previous = self.button
        self.layout().removeWidget(previous)
        previous.hide()
        previous.deleteLater()
        self.button = button if button is not None else square_tool_button('sync', 'Sync site data')
        if not self._toolbar_button:
            self.layout().insertWidget(0, self.button)
        self.button.setToolTip('Pull published results, metrics and monitor data from this experiment’s site.')
        self.button.hide()
        self.button.clicked.connect(lambda: self.start(self.site, 'sync'))
        self.site = None
        self.status.clear()
        self.hide()

    def set_config(self, config):
        from src.app.sites import site_name
        site = site_name(config or {})
        site = site if site != 'local' else None
        if site != self.site:
            self.status.clear()
        self.site = site
        title = f'Sync monitor & results from {site}' if site else 'Sync monitor & results'
        self.button.setAccessibleName(title)
        self.button.setToolTip(title + ': pull published results, metrics and monitor data.')
        self.button.setVisible(site is not None)
        self.setVisible(site is not None and (not self._toolbar_button or bool(self.status.text())))

    def start(self, site, action):
        self.show()
        super().start(site, action)
