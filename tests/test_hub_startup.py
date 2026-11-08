"""Opening the embedded Hub must never show temporary top-level windows."""

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QApplication, QWidget

from frontend.hub.client import HubClient
from frontend.hub.dialog import HubView
from frontend.hub.models import HubComponent


class WindowShowRecorder(QObject):
    def __init__(self):
        super().__init__()
        self.windows = []

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.Show and isinstance(watched, QWidget) and watched.isWindow():
            # Record immediately: layout insertion can reparent and hide a window
            # before QApplication.topLevelWidgets() would see it after loading.
            self.windows.append((type(watched).__name__, watched.windowTitle()))
        return False


def test_first_hub_load_does_not_flash_windows(tmp_path):
    app = QApplication.instance() or QApplication([])
    client = HubClient(tmp_path, tmp_path / '.thetaide')
    view = HubView(client, show_footer=False)
    view.show()
    app.processEvents()
    recorder = WindowShowRecorder()
    app.installEventFilter(recorder)
    try:
        client.components = [
            HubComponent(id=kind, name=kind.title(), kind=kind, version='1.0.0')
            for kind in ('model', 'method', 'env', 'data', 'eval', 'plot', 'suite', 'experiment', 'plugin')
        ]
        # First delivery builds cards; subsequent deliveries reuse them.
        for _ in range(2):
            client.indexLoaded.emit(client.components)
            app.processEvents()
        assert len(view.cards) == len(client.components)
        assert view.cards['model'].btn_deep_copy.isVisible()
        assert view.cards['experiment'].btn_deep_copy.isHidden()
        assert recorder.windows == []
    finally:
        app.removeEventFilter(recorder)
        view.close()
        view.deleteLater()
