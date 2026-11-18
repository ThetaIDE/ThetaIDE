"""Test settings contribution: a read-only list of the workspace's sites."""
from PyQt6.QtWidgets import QComboBox, QLabel, QVBoxLayout, QWidget


class BatchSetup(QWidget):
    def __init__(self, parent=None, *, paths=None):
        super().__init__(parent)
        from src.app.api.sites import profiles
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Batch sites'))
        self.site = QComboBox()
        self.site.setEditable(False)
        self.site.addItems(list(profiles(paths)))
        layout.addWidget(self.site)


def create_settings(context):
    return BatchSetup(context.get('parent'), paths=context['workspace'])
