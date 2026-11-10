"""Visible project upload and SSH handoff, with user-mediated job submission."""
from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtWidgets import (QApplication, QFrame, QHBoxLayout, QVBoxLayout, QMessageBox,
                             QPushButton, QToolButton, QWidget, QGraphicsOpacityEffect)
from src.app.workspace import WorkspacePaths
from src.app.sites import transfer
from src.app.sites.slurm import bootstrap_command, shortcuts, submission_command


class TerminalOverlay(QWidget):
    """A terminal surface anchored above the main window's content."""

    finished = pyqtSignal(int)

    def __init__(self, window):
        super().__init__(window)
        self.setObjectName('siteTerminalOverlay')
        self.setStyleSheet('#siteTerminalOverlay { background: rgba(0, 0, 0, 110); }')
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.card = QFrame(self)
        self.card.setObjectName('card')
        self.card.setAutoFillBackground(True)
        self.body_layout = QVBoxLayout(self.card)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        self.body_layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(0, 2, 4, 2)
        header.addStretch()
        self.close_button = QToolButton(self.card)
        self.close_button.setText('×')
        self.close_button.setAccessibleName('Close cluster terminal')
        self.close_button.setToolTip('Close terminal (Escape)')
        self.close_button.clicked.connect(self.close)
        header.addWidget(self.close_button)
        self.body_layout.addLayout(header)

    def _place(self):
        self.setGeometry(self.parentWidget().rect())
        margin = min(24, self.width() // 20, self.height() // 20)
        width = min(1000, max(1, self.width() - 2 * margin))
        height = min(650, max(1, self.height() - 2 * margin))
        self.card.setGeometry((self.width() - width) // 2, (self.height() - height) // 2,
                              width, height)

    def showEvent(self, event):
        self._place()
        QApplication.instance().installEventFilter(self)
        super().showEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.parentWidget():
            if event.type() == QEvent.Type.Resize:
                self._place()
            elif event.type() == QEvent.Type.Close:
                self.close()
        # Filter the focused WebEngine child too: Escape must dismiss the overlay
        # rather than being consumed by the terminal or an application shortcut.
        if (self.isVisible() and isinstance(obj, QWidget)
                and (obj is self or self.isAncestorOf(obj))
                and event.type() in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress)
                and event.key() == Qt.Key.Key_Escape):
            event.accept()
            if event.type() == QEvent.Type.KeyPress:
                self.close()
            return True
        return super().eventFilter(obj, event)

    def closeEvent(self, event):
        QApplication.instance().removeEventFilter(self)
        super().closeEvent(event)
        if event.isAccepted():
            self.finished.emit(0)


def _terminal_loading(window, loading):
    window._site_terminal_loading = loading
    button = getattr(window, 'start_button', None)
    if button is not None:
        if loading:
            window._site_play_enabled = button.isEnabled()
            window._site_play_tooltip = button.toolTip()
            effect = QGraphicsOpacityEffect(button)
            effect.setOpacity(0.35)
            button.setGraphicsEffect(effect)
            button.setEnabled(False)
            button.setToolTip('Opening cluster terminal…')
            button.repaint()  # paint the disabled icon before terminal construction
        else:
            button.setGraphicsEffect(None)
        if not loading and not callable(getattr(window, 'update_launch_state', None)):
            button.setEnabled(getattr(window, '_site_play_enabled', True))
            button.setToolTip(getattr(window, '_site_play_tooltip', ''))
    update = getattr(window, 'update_launch_state', None)
    if callable(update): update()


def launch(window):
    from src.app.sites import site_name
    if getattr(window, '_site_terminal_loading', False): return
    if not getattr(window, 'compose_valid', False):
        QMessageBox.warning(window, 'Site launch', 'Resolve experiment validation errors before launching.')
        return
    previous = getattr(window, '_site_launch_dialog', None)
    if getattr(window, '_site_launch_pending', False) and previous and previous.isVisible():
        previous.raise_()
        previous.terminal.focus_terminal()
        return

    _terminal_loading(window, True)
    dialog = None
    try:
        paths = WorkspacePaths.for_process()
        site = site_name(window.composed_config)
        experiment = window.current_experiment
        viewer = getattr(window, 'config_viewer', None)
        if viewer and viewer.is_dirty and not viewer.save_to_disk(show_error=True):
            _terminal_loading(window, False)
            return
        config = transfer.connection(paths.root, site)
        from .terminal import TerminalWidget
        import sys
        # The terminal owns upload -> SSH. Its startup signal restores Play once
        # the terminal renderer and PTY are ready; close/error also release it.
        dialog = TerminalOverlay(window)
        dialog.setWindowTitle(f'{site} — project upload and SSH')
        layout = dialog.body_layout
        command = [sys.executable, '-u', '-m', 'src.app.sites', 'launch', '--root', str(paths.root),
                   site, '--experiment', experiment]
        terminal = TerminalWidget(cwd=str(paths.root), command=command, parent=dialog)
        layout.addWidget(terminal)
        dialog.terminal = terminal
        window._site_launch_dialog = dialog
        window._site_launch_pending = True
        QApplication.clipboard().setText(submission_command(config, experiment, site))
        def ready(*_):
            if getattr(window, '_site_launch_dialog', None) is dialog and getattr(window, '_site_terminal_loading', False):
                _terminal_loading(window, False)
        terminal.session_started.connect(ready)
        terminal.session_exited.connect(ready)
        def closed(_):
            terminal.close()
            if getattr(window, '_site_launch_dialog', None) is dialog:
                window._site_launch_pending = False
                ready()
        dialog.finished.connect(closed)
        dialog.show()
        dialog.raise_()
        terminal.focus_terminal()
    except Exception as exc:
        if dialog is not None: dialog.close()
        window._site_launch_pending = False
        _terminal_loading(window, False)
        QMessageBox.warning(window, 'Cluster terminal', str(exc))


class SiteSetup(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        from .widgets import ComboBox, label
        from src.app.sites import profiles
        from src.app.sites.storage import safe_path
        self.paths = WorkspacePaths.for_process()
        layout = QVBoxLayout(self)
        layout.addWidget(label('Cluster sites', 'cardTitle'))
        self.site = ComboBox()
        self.site.setEditable(False)
        self.site.addItems(list(profiles(self.paths)))
        layout.addWidget(self.site)
        for title, action in [('Copy first-run upload and package setup', self.copy_setup),
                              ('Create project SSH / push / sync shortcuts', self.make_shortcuts),
                              ('Open editable site configuration', self.open_config)]:
            button = QPushButton(title); button.clicked.connect(action); layout.addWidget(button)
        self.status = label('Address, SSH and transfer settings belong to the selected site component in Components.', 'muted')
        self.status.setWordWrap(True); layout.addWidget(self.status)

    def showEvent(self, event):
        from src.app.sites import profiles
        selected = self.site.currentText()
        self.site.blockSignals(True)
        self.site.clear()
        self.site.addItems(list(profiles(self.paths)))
        index = self.site.findText(selected)
        if index >= 0: self.site.setCurrentIndex(index)
        self.site.blockSignals(False)
        super().showEvent(event)

    def copy_setup(self):
        try: command = bootstrap_command(self.paths.root, self.site.currentText())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Site setup', str(exc)); return
        QApplication.clipboard().setText(command)
        self.status.setText('Setup script copied. Run it in a local terminal; review cluster module and package requirements first.')

    def make_shortcuts(self):
        try: paths = shortcuts(self.paths.root, self.site.currentText())
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, 'Site shortcuts', str(exc)); return
        self.status.setText('\n'.join(paths))

    def open_config(self):
        from PyQt6.QtCore import QUrl
        from PyQt6.QtGui import QDesktopServices
        from src.app.sites.storage import safe_path
        from src.app.sites.catalog import defaults_path
        path = defaults_path(self.paths.root, self.site.currentText())
        if path.is_file(): QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
