"""Pull-only site controls, with network work on a dedicated background worker."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QHBoxLayout, QMenu, QMessageBox, QPushButton, QWidget

from src.app.sites import transfer
from src.app.workspace import WorkspacePaths
from .local_backend import LocalBackend
from .widgets import label


class SiteSyncControl(QWidget):
    watch_in_terminal = False
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
        if self.run_provider is not None:
            self.button.setMenu(None)
            self.button.setText('Sync')
            self.button.setToolTip('Download all published results for the displayed experiment run.')
            self.button.clicked.connect(self.sync_run)
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
        if self.run_provider is not None:
            layout.addWidget(self.status)
            layout.addWidget(self.button)
        else:
            layout.addWidget(self.button)
            layout.addWidget(self.status)
        layout.addWidget(self.cancel_button)

    def _console_log(self, message):
        log = getattr(self.window(), 'log', None)
        if callable(log):
            log(message)
        terminal = getattr(self, '_sync_terminal', None)
        if terminal is not None:
            terminal.append_output(message + '\r\n')

    def _open_terminal(self, site, mode):
        if not self.watch_in_terminal:
            return
        window = self.window()
        panel, tabs = getattr(window, 'terminal_panel', None), getattr(window, 'tabs', None)
        if panel is None or tabs is None:
            return
        self._return_view = tabs.currentWidget()
        terminal = panel.new_shell(output_only=True, title=f'Sync {mode} · {site}')
        self._sync_terminal = terminal
        def closed():
            if self._sync_terminal is terminal:
                if self._transferring:
                    self.cancel_event.set()
                self._sync_terminal = None
        terminal.closed.connect(closed)
        tabs.setCurrentWidget(panel)
        terminal.focus_terminal()

    def _finish_terminal(self, success):
        terminal = getattr(self, '_sync_terminal', None)
        if terminal is None:
            return
        terminal.finish_output(0 if success else 1)
        window = self.window()
        panel, tabs = window.terminal_panel, window.tabs
        # Do not interrupt navigation or another transfer the user selected.
        if (tabs.currentWidget() is panel and panel.terminal is terminal
                and self._return_view is not None and tabs.indexOf(self._return_view) >= 0):
            tabs.setCurrentWidget(self._return_view)
        self._sync_terminal = None
        selected = panel.terminal
        if terminal in panel.terminals:
            panel.close_shell(panel.terminals.index(terminal))
        if selected is not terminal and selected in panel.terminals:
            panel.set_current_shell(panel.terminals.index(selected))

    def refresh_target(self):
        if self.run_provider is not None:
            remote = (self.run_provider() or {}).get('remote') or {}
            self.button.setEnabled(bool(remote.get('site') and remote.get('execution_id'))
                                   and not getattr(self, '_transferring', False))

    def sync_run(self):
        remote = (self.run_provider() or {}).get('remote') or {}
        if remote.get('site') and remote.get('execution_id'):
            self.start(remote['site'], 'experiment', execution_id=remote['execution_id'])

    def populate(self):
        self.menu.clear()
        if self.run_provider is not None:
            return
        for site, config in transfer.connections(self.paths.root).items():
            if not config.get('host') or not config.get('remote_root'): continue
            self.menu.addAction(('Pull metrics from ' if self.mode == 'metrics' else 'Pull results from ') + site, lambda checked=False, site=site: self.start(site, 'sync'))
            if self.mode == 'metrics':
                self.menu.addAction('Pull diagnostics from ' + site, lambda checked=False, site=site: self.start(site, 'diagnostics'))
        self.menu.addSeparator()
        notice = self.menu.addAction('Edit site profiles in Components')
        notice.setEnabled(False)

    def start(self, site, action, *, execution_id=None, experiment=None):
        if getattr(self, '_transferring', False):
            return
        if action not in ('sync', 'diagnostics', 'experiment'):
            raise ValueError('Sync controls only download results. Use Run to upload experiment code.')
        if action == 'experiment' and not (execution_id or experiment):
            raise ValueError('Select a remote experiment run before syncing the experiment.')
        transfer_mode = 'results' if action == 'experiment' else self.mode
        options = {'execution_id': execution_id} if execution_id else {}
        if experiment: options['experiment'] = experiment
        self._transferring = True
        import threading
        self.cancel_event = threading.Event()
        self._open_terminal(site, transfer_mode)
        self.cancel_button.show()
        self.button.setEnabled(False)
        self.status.setText(f'{site}: downloading…')
        if action == 'diagnostics': function = lambda: transfer.sync(self.paths.root, site, transfer_mode, **options, diagnostics=True, cancel_event=self.cancel_event, progress=self.transfer_progress.emit)
        else: function = lambda: transfer.sync(self.paths.root, site, transfer_mode, **options, cancel_event=self.cancel_event, progress=self.transfer_progress.emit)
        def done(result, error):
            self._transferring = False
            self.button.setEnabled(True)
            self.refresh_target()
            self.cancel_button.hide()
            if error:
                self.status.setText(f'{site}: download failed')
                self.status.setToolTip(error)
                self.console_output.emit(f'[site {site}] {action} failed: {error}')
                self._finish_terminal(False)
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
            self._finish_terminal(True)
        def logged_transfer():
            from src.app.sites.transfer_output import output_to
            emit = lambda message: self.console_output.emit(f'[site {site}] {message}')
            with output_to(emit):
                emit(f'Starting download from {site} (mode={transfer_mode}, execution={execution_id or "all"}, workspace={self.paths.root})')
                return function()
        self.backend._submit(logged_transfer, callback=done)


class ExperimentSiteSyncControl(SiteSyncControl):
    """Pull the selected experiment's site, including monitor diagnostics."""

    watch_in_terminal = True

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
        self.button.clicked.connect(self.sync_experiment)
        self.site = None
        self.experiment = None
        self.status.clear()
        self.hide()

    def set_config(self, config, experiment=None):
        from src.app.sites import site_name
        site = site_name(config or {})
        local = site in ('local', 'local/default') or ((config or {}).get('site') or {}).get('scheduler') == 'local'
        site = None if local else site
        if site != self.site or experiment != self.experiment:
            self.status.clear()
        self.site = site
        self.experiment = experiment
        title = f'Sync {experiment or "experiment"} from {site}' if site else 'Sync'
        self.button.setAccessibleName(title)
        self.button.setToolTip(title + ': pull published results, metrics and monitor data.')
        self.button.setVisible(site is not None)
        self.setVisible(site is not None and not self._toolbar_button)

    def sync_experiment(self):
        if self.site and self.experiment:
            self.start(self.site, 'experiment', experiment=self.experiment)

    def start(self, site, action, **options):
        if not self._toolbar_button:
            self.show()
        super().start(site, action, **options)
