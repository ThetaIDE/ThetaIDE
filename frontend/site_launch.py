"""Project upload and SSH handoff in the existing terminal pane."""
import sys

from PyQt6.QtWidgets import QApplication, QVBoxLayout, QMessageBox, QPushButton, QWidget
from src.app.workspace import WorkspacePaths
from src.app.sites import transfer
from src.app.sites.slurm import bootstrap_command, shortcuts, submission_command


def launch(window):
    from src.app.sites import site_name
    if not getattr(window, 'compose_valid', False):
        QMessageBox.warning(window, 'Site launch', 'Resolve experiment validation errors before launching.')
        return
    try:
        paths = WorkspacePaths.for_process()
        site = site_name(window.composed_config)
        experiment = window.current_experiment
        viewer = getattr(window, 'config_viewer', None)
        if viewer and viewer.is_dirty and not viewer.save_to_disk(show_error=True):
            return
        config = transfer.connection(paths.root, site)
        command = [sys.executable, '-u', '-m', 'src.app.sites', 'launch', '--root', str(paths.root),
                   site, '--experiment', experiment]
        QApplication.clipboard().setText(submission_command(config, experiment, site))
        terminal = window.terminal_panel.new_shell(command=command, cwd=str(paths.root))
        window.tabs.setCurrentWidget(window.terminal_panel)
        terminal.focus_terminal()
    except Exception as exc:
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
