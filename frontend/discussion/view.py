"""The rendered figure (vega.html in a WebEngine view) and the background figure builder.

``FigureView`` offers ``render(spec)``, ``setParam(name, value)``, ``error(message)``,
``reset_view()`` and ``request_image()`` (answered by ``image_ready``) over a QWebChannel bridge.
Tooltips use the bundled vega-tooltip handler, styled from the theme. Its QWebEngineView is created the first time the view is shown, so the
Discussion tab costs nothing at startup. Calls made before the page is ready are replayed.

``FigureBuilder`` runs a plot component's ``figure_spec`` in a worker thread: regeneration
requests are debounced, a newer request supersedes an older one, and component exceptions
come back as error messages. Built specs are cached in memory, keyed by the plotter and its
version, the target, the regeneration parameters' values and the target's ``target_stamp``,
so reopening an unchanged figure draws at once. A forced request bypasses the cache.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, QTimer, QUrl, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QStackedLayout, QWidget

from ..theme import theme_color

HTML_PATH = Path(__file__).with_name("vega.html")
REGENERATE_DEBOUNCE_MS = 400
SPEC_CACHE_SIZE = 32


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
    resetRequested = pyqtSignal()
    imageRequested = pyqtSignal(float)
    page_ready = pyqtSignal()
    page_rendered = pyqtSignal()
    page_failed = pyqtSignal(str)
    page_image = pyqtSignal(str)

    @pyqtSlot()
    def ready(self):
        self.page_ready.emit()

    @pyqtSlot()
    def rendered(self):
        self.page_rendered.emit()

    @pyqtSlot(str)
    def reportError(self, message):  # noqa: N802 - called from JavaScript
        self.page_failed.emit(message)

    @pyqtSlot(str)
    def imageReady(self, data_url):  # noqa: N802 - called from JavaScript
        self.page_image.emit(data_url)


def image_from_data_url(data_url: str):
    """The QImage of a ``data:image/png;base64,…`` URL; a null image for anything else."""
    import base64
    import binascii

    from PyQt6.QtGui import QImage
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        return QImage()
    try:
        raw = base64.b64decode(data_url[len(prefix):], validate=True)
    except (binascii.Error, ValueError):
        return QImage()
    return QImage.fromData(raw, "PNG")


class FigureView(QWidget):
    """The figure pane. Replays the last render, its live parameters and the busy state once
    the page is ready."""

    image_ready = pyqtSignal(object)  # QImage; null when there is no figure to copy

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
        self.bridge.page_image.connect(lambda url: self.image_ready.emit(image_from_data_url(url)))

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

    def reset_view(self):
        """Redraw the current figure: zoom, pan and legend selections return to the spec's."""
        if self._ready and self._content and self._content[0] == "render":
            self.bridge.resetRequested.emit()

    def request_image(self, scale: float = 2.0):
        """Ask the page for a PNG of the drawn figure; ``image_ready`` answers."""
        from PyQt6.QtGui import QImage
        if self._ready and self._content and self._content[0] == "render":
            self.bridge.imageRequested.emit(scale)
        else:
            self.image_ready.emit(QImage())

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
                           "surface": theme_color("surface"), "border": theme_color("border"),
                           "panel": theme_color("panel")})

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
        self._cache: OrderedDict = OrderedDict()
        self._keys: dict[int, tuple] = {}  # generation -> cache key of the job building it

    def cache_key(self, doc) -> tuple:
        """What a built spec depends on. Live parameters are applied after drawing, so they are not part of it."""
        from src.app.core.plotting.metric_sources import figure_source_targets
        from .model import FIGURE_SOURCES, target_stamp
        live = set(doc.live_params())
        values = {name: value for name, value in doc.values().items() if name not in live}
        package = doc.capability()[0]
        properties = doc.schema().get("properties") or {}
        sources = next((values.get(key) for key, node in properties.items()
                        if node.get(FIGURE_SOURCES)), None)
        targets = figure_source_targets(doc.experiment, sources)
        return (doc.plotter, package.id, package.version, doc.experiment,
                json.dumps(values, sort_keys=True, default=str),
                tuple(target_stamp(self.paths, target) for target in targets))

    def request(self, doc, *, immediate: bool = False, force: bool = False):
        """Rebuild ``doc``'s figure: now, or after the debounce interval (restarted by each call).

        An unchanged figure is served from the cache unless ``force`` is set.
        """
        self._generation += 1
        key = self.cache_key(doc)
        if not force and key in self._cache:
            self._timer.stop()
            self._job = None
            self._cache.move_to_end(key)
            if self.busy:
                self.busy = False
                self.finished.emit()
            self.built.emit(self._cache[key])
            return
        self._keys = {self._generation: key}
        self._job = (self._generation, doc.plotter, doc.experiment, doc.values(), doc.live_params())
        if immediate:
            self._timer.stop()
            self._start()
        else:
            self._timer.start()

    def clear_cache(self):
        self._cache.clear()

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
            result = (generation, self._build(plotter, experiment, values, live), "")
        except Exception as exc:  # the component's own errors go to the figure pane
            result = (generation, None, str(exc) or type(exc).__name__)
        try:
            self._result.emit(*result)
        except RuntimeError:  # the panel closed during the build; nobody is waiting for it
            pass

    def _deliver(self, generation, spec, error):
        if generation != self._generation:
            return  # superseded by a newer request
        self.busy = False
        if error:
            self.failed.emit(error)
        else:
            key = self._keys.pop(generation, None)
            if key is not None:
                self._cache[key] = spec
                while len(self._cache) > SPEC_CACHE_SIZE:
                    self._cache.popitem(last=False)
            self.built.emit(spec)
        self.finished.emit()

    def _build_spec(self, plotter, experiment, values, live):
        from src.app.core.plotting.figures import build_figure_spec
        from .model import resolve_plotter
        package, capability = resolve_plotter(plotter, paths=self.paths)
        return build_figure_spec(package, capability, experiment, values, live, paths=self.paths)
