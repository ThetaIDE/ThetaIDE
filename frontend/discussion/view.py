"""The rendered figure (vega.html in a WebEngine view) and the background figure builder.

``FigureView`` offers ``render(spec)``, ``setParam(name, value)`` and ``error(message)`` over a
QWebChannel bridge. Its QWebEngineView is created the first time the view is shown, so the
Discussion tab costs nothing at startup. Calls made before the page is ready are replayed.

``FigureBuilder`` runs a plot component's ``figure_spec`` in a worker thread: regeneration
requests are debounced, a newer request supersedes an older one, and component exceptions
come back as error messages.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QStackedLayout, QWidget

from ..theme import theme_color

HTML_PATH = Path(__file__).with_name("vega.html")
REGENERATE_DEBOUNCE_MS = 400


def embed_config() -> dict:
    """Vega config from the current theme's color tokens."""
    text, muted, border = theme_color("text"), theme_color("muted"), theme_color("border")
    axis = {"domainColor": muted, "tickColor": muted, "gridColor": border, "labelColor": muted,
            "titleColor": text}
    return {
        "background": "transparent",
        "view": {"stroke": border},
        "axis": axis,
        "legend": {"labelColor": muted, "titleColor": text},
        "title": {"color": text, "subtitleColor": muted},
        "header": {"labelColor": muted, "titleColor": text},
        "text": {"color": text},
    }


class FigureBridge(QObject):
    """The object vega.html reaches as ``channel.objects.bridge``."""

    renderRequested = pyqtSignal(str)
    paramRequested = pyqtSignal(str, str)
    errorRequested = pyqtSignal(str)
    busyRequested = pyqtSignal(bool)
    themeRequested = pyqtSignal(str)
    page_ready = pyqtSignal()
    page_rendered = pyqtSignal()
    page_failed = pyqtSignal(str)

    @pyqtSlot()
    def ready(self):
        self.page_ready.emit()

    @pyqtSlot()
    def rendered(self):
        self.page_rendered.emit()

    @pyqtSlot(str)
    def reportError(self, message):  # noqa: N802 - called from JavaScript
        self.page_failed.emit(message)


class FigureView(QWidget):
    """The figure pane. Replays the last render, its live parameters and the busy state once
    the page is ready."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self._stack = QStackedLayout(self)
        self.web = None
        self.bridge = FigureBridge(self)
        self._ready = False
        self._content = None  # ("render", spec_json) or ("error", message)
        self._params: dict[str, str] = {}
        self._busy = False

    # The interface the panel uses.

    def render(self, spec: dict):
        self._content = ("render", json.dumps(spec, allow_nan=False))
        self._params = {}
        if self._ready:
            self.bridge.renderRequested.emit(self._content[1])

    def setParam(self, name: str, value):  # noqa: N802 - mirrors view.signal in the page
        encoded = json.dumps(value, allow_nan=False)
        self._params[name] = encoded
        if self._ready and self._content and self._content[0] == "render":
            self.bridge.paramRequested.emit(name, encoded)

    def error(self, message: str):
        self._content = ("error", message)
        self._params = {}
        if self._ready:
            self.bridge.errorRequested.emit(message)

    def set_busy(self, busy: bool):
        self._busy = busy
        if self._ready:
            self.bridge.busyRequested.emit(busy)

    def clear(self):
        self.error("")

    def apply_theme(self):
        if self.web is not None:
            from PyQt6.QtGui import QColor
            self.web.page().setBackgroundColor(QColor(Qt.GlobalColor.transparent))
        if self._ready:
            self.bridge.themeRequested.emit(self._theme_json())
            self._replay()

    # The page.

    def showEvent(self, event):
        super().showEvent(event)
        if self.web is None:
            self._create_web_view()

    def _create_web_view(self):
        from PyQt6.QtGui import QColor
        from PyQt6.QtWebChannel import QWebChannel
        from PyQt6.QtWebEngineCore import QWebEngineSettings
        from PyQt6.QtWebEngineWidgets import QWebEngineView

        self.web = QWebEngineView(self)
        self.web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.web.page().setBackgroundColor(QColor(Qt.GlobalColor.transparent))
        settings = self.web.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        self.channel = QWebChannel(self.web.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web.page().setWebChannel(self.channel)
        self.bridge.page_ready.connect(self._page_ready)
        self.web.load(QUrl.fromLocalFile(str(HTML_PATH)))
        self._stack.addWidget(self.web)

    def _theme_json(self) -> str:
        return json.dumps({"config": embed_config(), "text": theme_color("text"), "muted": theme_color("muted"),
                           "surface": theme_color("surface")})

    def _page_ready(self):
        self._ready = True
        self.bridge.themeRequested.emit(self._theme_json())
        self._replay()

    def _replay(self):
        if self._content is not None:
            kind, payload = self._content
            if kind == "render":
                self.bridge.renderRequested.emit(payload)
                for name, value in self._params.items():
                    self.bridge.paramRequested.emit(name, value)
            else:
                self.bridge.errorRequested.emit(payload)
        self.bridge.busyRequested.emit(self._busy)


class FigureBuilder(QObject):
    """Builds specs off the GUI thread; only the newest request's result is delivered."""

    started = pyqtSignal()
    built = pyqtSignal(dict)
    failed = pyqtSignal(str)
    finished = pyqtSignal()
    _result = pyqtSignal(int, object, str)

    def __init__(self, paths, parent=None, *, debounce_ms: int = REGENERATE_DEBOUNCE_MS, build=None):
        super().__init__(parent)
        self.paths = paths
        self._build = build or self._build_spec
        self._generation = 0
        self._job = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._start)
        self._result.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        self.busy = False

    def request(self, doc, *, immediate: bool = False):
        """Rebuild ``doc``'s figure: now, or after the debounce interval (restarted by each call)."""
        self._generation += 1
        self._job = (self._generation, doc.plotter, doc.experiment, doc.values(), doc.live_params())
        if immediate:
            self._timer.stop()
            self._start()
        else:
            self._timer.start()

    def cancel(self):
        """Drop the pending request and ignore results still being built."""
        self._timer.stop()
        self._generation += 1
        self._job = None
        if self.busy:
            self.busy = False
            self.finished.emit()

    def _start(self):
        job, self._job = self._job, None
        if job is None:
            return
        if not self.busy:
            self.busy = True
            self.started.emit()
        threading.Thread(target=self._run, args=(job,), name="figure-builder", daemon=True).start()

    def _run(self, job):
        generation, plotter, experiment, values, live = job
        try:
            spec = self._build(plotter, experiment, values, live)
            self._result.emit(generation, spec, "")
        except Exception as exc:  # the component's own errors go to the figure pane
            self._result.emit(generation, None, str(exc) or type(exc).__name__)

    def _deliver(self, generation, spec, error):
        if generation != self._generation:
            return  # superseded by a newer request
        self.busy = False
        if error:
            self.failed.emit(error)
        else:
            self.built.emit(spec)
        self.finished.emit()

    def _build_spec(self, plotter, experiment, values, live):
        from src.app.core.plotting.figures import build_figure_spec
        from .model import resolve_plotter
        package, capability = resolve_plotter(plotter, paths=self.paths)
        return build_figure_spec(package, capability, experiment, values, live, paths=self.paths)
