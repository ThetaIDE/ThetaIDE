"""User-mediated remote submission: upload, copy, then open an interactive SSH terminal."""
from PyQt6.QtWidgets import QApplication, QDialog, QVBoxLayout, QMessageBox, QPushButton, QWidget
from src.app.workspace import WorkspacePaths
from src.app.sites import transfer
from src.app.sites.slurm import bootstrap_command, shortcuts, submission_command, ssh_arguments
from .local_backend import LocalBackend


def launch(window):
    from src.app.sites import site_name
    if not getattr(window, 'compose_valid', False):
        QMessageBox.warning(window, 'Site launch', 'Resolve experiment validation errors before launching.')
        return
    paths = WorkspacePaths.for_process()
    site = site_name(window.composed_config)
    experiment = window.current_experiment
    viewer = getattr(window, 'config_viewer', None)
    if viewer and viewer.is_dirty and not viewer.save_to_disk(show_error=True): return
    try: config = transfer.connection(paths.root, site)
    except ValueError as exc:
        QMessageBox.warning(window, 'Site connection', str(exc)); return
    if getattr(window, '_site_launch_pending', False): return
    window._site_launch_pending = True
    backend = LocalBackend(window)
    window._site_launch_backend = backend
    def ready(result, error):
        window._site_launch_pending = False
        if error:
            QMessageBox.warning(window, 'Project upload', error); return
        QApplication.clipboard().setText(submission_command(config, experiment, site))
        from .terminal import TerminalPanel
        import shlex
        dialog = QDialog(window)
        dialog.setWindowTitle(f'{site} — command copied; paste to validate and submit')
        dialog.resize(1000, 650)
        layout = QVBoxLayout(dialog)
        command = [*ssh_arguments(paths.root, config, True), '--', config['host'],
                   'cd ' + shlex.quote(config['remote_root']) + ' && exec bash -l']
        terminal = TerminalPanel(cwd=str(paths.root), command=command, parent=dialog)
        layout.addWidget(terminal)
        dialog.finished.connect(lambda _: [term.close() for term in terminal.terminals])
        window._site_launch_dialog = dialog
        dialog.show()
    backend._submit(lambda: transfer.push(paths.root, site), callback=ready)


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
