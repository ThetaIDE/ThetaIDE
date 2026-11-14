"""Host settings slot for optional providers, with ordinary contribution cleanup."""
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget
from src.app.core.contributions import ContributionRegistry, ContributionScope, load_contribution
from src.app.workspace import WorkspacePaths


class ExtensionSettings(QWidget):
    def __init__(self, slot, parent=None, *, workspace=None, config=None):
        super().__init__(parent)
        self.paths = workspace or WorkspacePaths.for_process()
        self.slot, self.config = slot, config or {}
        self.setLayout(QVBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)
        self._suspended = False
        self.scope = ContributionScope(self._mount)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.reconcile)
        self.reconcile()

    def _mount(self, contribution):
        widget = load_contribution(contribution, {'workspace': self.paths, 'parent': self}, paths=self.paths)
        if not isinstance(widget, QWidget):
            raise TypeError('Settings contributions must return a QWidget')
        self.layout().addWidget(widget)
        def close():
            self.layout().removeWidget(widget)
            widget.close()
            widget.setParent(None)
            widget.deleteLater()
        return close

    def reconcile(self):
        if self._suspended: return
        self.scope.reconcile(ContributionRegistry(self.paths).contributions(self.slot, config=self.config))
        self.setMaximumHeight(16777215 if self.scope.active else 0)

    def showEvent(self, event):
        self._suspended = False
        self.reconcile()
        self.timer.start()
        super().showEvent(event)

    def hideEvent(self, event):
        self._suspended = True
        self.timer.stop()
        self.scope.close()
        super().hideEvent(event)

    def closeEvent(self, event):
        self._suspended = True
        self.timer.stop()
        self.scope.close()
        super().closeEvent(event)
