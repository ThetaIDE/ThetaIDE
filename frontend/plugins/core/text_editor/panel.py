"""Native Qt document tabs around a locally bundled CodeMirror 6 surface."""

from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import QEventLoop, QFileSystemWatcher, QObject, Qt, QTimer, QUrl, pyqtSlot
from PyQt6.QtGui import QColor, QIcon
from PyQt6.QtWebChannel import QWebChannel
from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings, QWebEngineUrlRequestInterceptor
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from frontend.sidetabs import svg_icon
from frontend.theme import current_theme

from .documents import Document, FileConflictError

WEB_ROOT = Path(__file__).parent / "web"


class LocalRequests(QWebEngineUrlRequestInterceptor):
    """The editor can load its own assets and Qt's bridge, without network access."""

    def interceptRequest(self, info):
        url = info.requestUrl()
        allowed = url.scheme() == "qrc" or (
            url.isLocalFile() and Path(url.toLocalFile()).resolve().is_relative_to(WEB_ROOT.resolve())
        )
        info.block(not allowed)


class EditorPage(QWebEnginePage):
    def acceptNavigationRequest(self, url, _kind, _main_frame):
        return url == QUrl.fromLocalFile(str((WEB_ROOT / "index.html").resolve()))


class EditorBridge(QObject):
    def __init__(self, panel):
        super().__init__(panel)
        self.panel = panel

    @pyqtSlot()
    def ready(self):
        self.panel._ready = True
        self.panel.apply_theme()
        self.panel._show_current()

    @pyqtSlot(str, str, int)
    def documentChanged(self, document_id, text, revision):
        self.panel._accept_change(document_id, text, revision)

    @pyqtSlot(str)
    def actionRequested(self, action):
        # Leave the WebChannel callback before opening any native modal dialog.
        QTimer.singleShot(0, lambda: self.panel.perform_action(action))

    @pyqtSlot(int, int)
    def cursorChanged(self, line, column):
        self.panel.cursor_label.setText(f"Ln {line}, Col {column}")


class TextEditorPanel(QWidget):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.documents: list[Document] = []
        self._revisions: dict[str, int] = {}
        self._ready = False
        self._failed = False
        self._busy = False
        self._untitled = 0
        self._locations: dict[str, tuple] = {}
        self.setObjectName("textEditorPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(8)

        toolbar = QHBoxLayout()
        for title, action in (
            ("New", "new"),
            ("Open…", "open"),
            ("Save", "save"),
            ("Save As…", "save_as"),
            ("Undo", "undo"),
            ("Redo", "redo"),
            ("Find / Replace", "find"),
        ):
            button = QPushButton(title)
            button.clicked.connect(lambda _checked=False, a=action: self.perform_action(a))
            toolbar.addWidget(button)
        toolbar.addStretch()
        self.wrap_button = QToolButton()
        self.wrap_button.setText("Wrap")
        self.wrap_button.setCheckable(True)
        self.wrap_button.setChecked(bool(context.window.settings_manager.get("editor", "wrap_lines", default=False)))
        self.wrap_button.toggled.connect(self._set_wrap)
        toolbar.addWidget(self.wrap_button)
        self.recover_button = QPushButton("Recover editor")
        self.recover_button.clicked.connect(self.recover_renderer)
        self.recover_button.hide()
        toolbar.addWidget(self.recover_button)
        layout.addLayout(toolbar)

        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setTabsClosable(False)
        self.tab_bar.setUsesScrollButtons(True)
        self.tab_bar.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.tab_bar.currentChanged.connect(lambda _index: self._show_current())
        self.tab_bar.tabCloseRequested.connect(self.close_document)
        layout.addWidget(self.tab_bar)

        self.banner = QFrame()
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(10, 5, 10, 5)
        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        banner_layout.addWidget(self.banner_label, 1)
        reload_button = QPushButton("Reload from disk")
        reload_button.clicked.connect(self.reload_current)
        banner_layout.addWidget(reload_button)
        keep_button = QPushButton("Keep editing")
        keep_button.clicked.connect(self.banner.hide)
        banner_layout.addWidget(keep_button)
        self.banner.hide()
        layout.addWidget(self.banner)

        self.stack = QStackedWidget()
        empty = QWidget()
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch()
        title = QLabel("Text Editor")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("cardTitle")
        empty_layout.addWidget(title)
        subtitle = QLabel("Open a file or start a new document.\nPython · YAML · JSON · TOML · Markdown · Plain text")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(subtitle)
        empty_layout.addStretch()
        self.stack.addWidget(empty)
        self.web_view = QWebEngineView()
        self.web_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        page = EditorPage(self.web_view)
        self.web_view.setPage(page)
        self.interceptor = LocalRequests(page)
        page.setUrlRequestInterceptor(self.interceptor)
        settings = page.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, False)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        self.bridge = EditorBridge(self)
        self.channel = QWebChannel(page)
        self.channel.registerObject("bridge", self.bridge)
        page.setWebChannel(self.channel)
        page.renderProcessTerminated.connect(self._renderer_failed)
        page.loadFinished.connect(lambda ok: None if ok else self._renderer_failed())
        self.stack.addWidget(self.web_view)
        layout.addWidget(self.stack, 1)

        status = QHBoxLayout()
        self.path_label = QLabel("No document open")
        self.path_label.setTextFormat(Qt.TextFormat.PlainText)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.path_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        status.addWidget(self.path_label, 1)
        self.cursor_label = QLabel()
        status.addWidget(self.cursor_label)
        self.encoding_label = QLabel()
        status.addWidget(self.encoding_label)
        layout.addLayout(status)
        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self.check_external_changes)
        self.watcher.directoryChanged.connect(self.check_external_changes)
        context.window.theme_manager.changed.connect(self.apply_theme)
        context.window.settings_manager.changed.connect(self.apply_theme)
        self.web_view.load(QUrl.fromLocalFile(str((WEB_ROOT / "index.html").resolve())))
        self.apply_theme()

    @property
    def current_document(self) -> Document | None:
        index = self.tab_bar.currentIndex()
        return self.documents[index] if 0 <= index < len(self.documents) else None

    def _js(self, method, *args):
        if self._ready:
            self.web_view.page().runJavaScript(f"window.thetaEditor.{method}({', '.join(json.dumps(a) for a in args)})")

    def _payload(self, doc):
        return {"id": doc.id, "name": doc.title, "text": doc.text, "revision": self._revisions[doc.id]}

    def _accept_change(self, document_id, text, revision):
        # Ignore bridge messages sent before a synchronous snapshot or an explicit reload.
        for index, doc in enumerate(self.documents):
            if doc.id == document_id and revision >= self._revisions[doc.id]:
                self._revisions[doc.id] = revision
                doc.text = text
                self._update_tab(index)
                break

    def sync_documents(self) -> bool:
        """Drain renderer edits before saves, close prompts, and plugin deactivation."""
        if not self._ready:
            if self._failed:
                self._error(
                    "Editor unavailable", "The editor renderer stopped. Keep Theta open to recover your buffers."
                )
                return False
            return True  # initial buffers have not reached the renderer yet
        loop = QEventLoop()
        result = []
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)

        def received(value):
            result.append(value)
            loop.quit()

        self.web_view.page().runJavaScript("window.thetaEditor.snapshot()", received)
        timer.start(3000)
        loop.exec()
        timer.stop()
        if not result or not isinstance(result[0], list):
            self._error("Editor unavailable", "Could not read the latest edits. The documents have been kept open.")
            return False
        for doc in result[0]:
            self._accept_change(doc["id"], doc["text"], doc["revision"])
        return True

    def _add_document(self, doc):
        self.documents.append(doc)
        self._revisions[doc.id] = 0
        index = self.tab_bar.addTab(doc.title)
        close_button = QToolButton(self.tab_bar)
        close_button.setAutoRaise(True)
        close_button.setIcon(svg_icon("close", {(QIcon.Mode.Normal, QIcon.State.Off): "muted"}, size=14))
        close_button.setToolTip(f"Close {doc.title}")
        close_button.setAccessibleName("Close document")
        close_button.setFixedSize(20, 20)
        close_button.setStyleSheet(
            "QToolButton { padding: 0; margin: 0; border: none; background: transparent; }"
            "QToolButton:hover { background: rgba(127, 127, 127, 0.2); }"
        )
        close_button.clicked.connect(lambda: self.close_document(self.documents.index(doc)))
        self.tab_bar.setTabButton(index, QTabBar.ButtonPosition.RightSide, close_button)
        self.tab_bar.setCurrentIndex(index)
        self._update_tab(index)
        self._show_current()
        self._watch_files()
        return doc

    def new_document(self):
        self._untitled += 1
        return self._add_document(Document(name=f"Untitled {self._untitled}"))

    def open_file(self, path, line=None, column=None) -> bool:
        path = Path(path).expanduser().resolve()
        for index, doc in enumerate(self.documents):
            if doc.path == path:
                self._locations[doc.id] = (line, column)
                self.tab_bar.setCurrentIndex(index)
                self._show_current()
                return True
        try:
            doc = Document.open(path)
        except (OSError, ValueError) as exc:
            self._error("Cannot open file", str(exc))
            return False
        self._locations[doc.id] = (line, column)
        self._add_document(doc)
        return True

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open text file", self._start_folder(), "All files (*)")
        return bool(path) and self.open_file(path)

    def _start_folder(self):
        doc = self.current_document
        return str(doc.path.parent if doc and doc.path else Path.cwd())

    def _show_current(self):
        doc = self.current_document
        self.stack.setCurrentIndex(1 if doc else 0)
        self.banner.hide()
        if doc:
            self.path_label.setText(str(doc.path) if doc.path else doc.title)
            self.path_label.setToolTip(self.path_label.text())
            self.encoding_label.setText(
                f"{doc.encoding.upper()} · {'CRLF' if doc.newline == chr(13) + chr(10) else 'CR' if doc.newline == chr(13) else 'LF'}"
            )
            if self._ready:
                line, column = self._locations.pop(doc.id, (None, None))
                self._js("show", self._payload(doc), line, column)
            self.check_external_changes()
        else:
            self.path_label.setText("No document open")
            self.cursor_label.clear()
            self.encoding_label.clear()

    def _update_tab(self, index):
        doc = self.documents[index]
        self.tab_bar.setTabText(index, f"{'● ' if doc.dirty else ''}{doc.title}")
        self.tab_bar.setTabToolTip(index, str(doc.path) if doc.path else doc.title)

    def save_current(self, save_as=False):
        doc = self.current_document
        if doc is None or not self.sync_documents():
            return False
        path = None
        if save_as or doc.path is None:
            name, _ = QFileDialog.getSaveFileName(
                self,
                "Save text file",
                str(doc.path or Path(self._start_folder()) / doc.title),
                "All files (*)",
                options=QFileDialog.Option.DontConfirmOverwrite,
            )
            if not name:
                return False
            path = Path(name).resolve()
        return self.save_document(doc, path)

    def save_document(self, doc, path=None):
        target = Path(path).resolve() if path else doc.path
        if any(other is not doc and other.path == target for other in self.documents):
            self._error("File already open", "That file is open in another tab. Save to a different path.")
            return False
        try:
            try:
                doc.save(target)
            except FileConflictError:
                reply = QMessageBox.warning(
                    self,
                    "File changed on disk",
                    f"{target}\n\nThe file changed outside this editor or already exists. Overwrite it with this buffer?",
                    QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Cancel,
                    QMessageBox.StandardButton.Cancel,
                )
                if reply != QMessageBox.StandardButton.Save:
                    return False
                doc.save(target, force=True)
        except (OSError, ValueError) as exc:
            self._error("Cannot save file", str(exc))
            return False
        self._update_tab(self.documents.index(doc))
        self._watch_files()
        self._show_current()  # updates language after Save As, preserving history
        return True

    def _confirm_close(self, doc):
        if not doc.dirty:
            return True
        reply = QMessageBox.warning(
            self,
            "Unsaved changes",
            f"Save changes to {doc.title}?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if reply == QMessageBox.StandardButton.Cancel:
            return False
        if reply == QMessageBox.StandardButton.Save:
            self.tab_bar.setCurrentIndex(self.documents.index(doc))
            return self.save_current()
        return True

    def close_document(self, index=None):
        if self._busy:
            return False
        index = self.tab_bar.currentIndex() if index is None else index
        if not 0 <= index < len(self.documents):
            return True
        self._busy = True
        try:
            if not self.sync_documents() or not self._confirm_close(self.documents[index]):
                return False
            doc = self.documents.pop(index)
            self._js("remove", doc.id)
            self._revisions.pop(doc.id, None)
            self._locations.pop(doc.id, None)
            self.tab_bar.removeTab(index)
            self._show_current()
            self._watch_files()
            return True
        finally:
            self._busy = False

    def can_close(self):
        if self._busy:
            return False
        self._busy = True
        try:
            return self.sync_documents() and all(self._confirm_close(doc) for doc in list(self.documents))
        finally:
            self._busy = False

    def reload_current(self):
        doc = self.current_document
        if doc is None or doc.path is None or not self.sync_documents():
            return False
        if (
            doc.dirty
            and QMessageBox.question(
                self,
                "Reload file",
                "Discard this buffer's edits and reload from disk?",
                QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Discard
        ):
            return False
        try:
            fresh = Document.open(doc.path)
        except (OSError, ValueError) as exc:
            self._error("Cannot reload file", str(exc))
            return False
        doc.text, doc.saved_text = fresh.text, fresh.saved_text
        doc.encoding, doc.newline = fresh.encoding, fresh.newline
        doc.disk_fingerprint = fresh.disk_fingerprint
        self._revisions[doc.id] += 1
        self._js("replace", self._payload(doc))
        self._update_tab(self.documents.index(doc))
        self._watch_files()
        self._show_current()
        return True

    def _watch_files(self):
        wanted = {str(doc.path) for doc in self.documents if doc.path and doc.path.exists()}
        wanted |= {str(doc.path.parent) for doc in self.documents if doc.path and doc.path.parent.exists()}
        existing = set(self.watcher.files() + self.watcher.directories())
        if existing - wanted:
            self.watcher.removePaths(sorted(existing - wanted))
        if wanted - existing:
            self.watcher.addPaths(sorted(wanted - existing))

    def check_external_changes(self, *_args):
        self._watch_files()
        doc = self.current_document
        try:
            changed = doc is not None and doc.changed_on_disk()
        except OSError:
            changed = True
        self.banner.setVisible(bool(changed))
        if changed:
            self.banner_label.setText(
                "This file changed on disk or was deleted. Reload it, or keep editing and review before saving."
            )

    def perform_action(self, action):
        if self._busy:
            return
        actions = {
            "new": self.new_document,
            "open": self.choose_file,
            "save": self.save_current,
            "save_as": lambda: self.save_current(True),
            "close": self.close_document,
        }
        if action in actions:
            return actions[action]()
        self._js("action", action)

    def _set_wrap(self, value):
        self.context.window.settings_manager.set("editor", "wrap_lines", value)

    def apply_theme(self):
        palette = current_theme()["colors"]
        self.tab_bar.setStyleSheet(
            "QTabBar::tab { padding: 7px 10px; margin-right: 3px; border: none; "
            f"border-bottom: 2px solid transparent; background: {palette['panel']}; color: {palette['muted']}; }}"
            f"QTabBar::tab:selected {{ color: {palette['text']}; border-bottom-color: {palette['accent']}; "
            f"background: {palette['base']}; }}"
            f"QTabBar::tab:hover {{ background: {palette['surface']}; }}"
        )
        for index in range(self.tab_bar.count()):
            button = self.tab_bar.tabButton(index, QTabBar.ButtonPosition.RightSide)
            if button is not None:
                button.setIcon(svg_icon("close", {(QIcon.Mode.Normal, QIcon.State.Off): "muted"}, size=14))
        self.web_view.page().setBackgroundColor(QColor(palette["base"]))
        self.web_view.setZoomFactor(self.context.window.settings_manager.zoom)
        wrap = bool(self.context.window.settings_manager.get("editor", "wrap_lines", default=False))
        self.wrap_button.blockSignals(True)
        self.wrap_button.setChecked(wrap)
        self.wrap_button.blockSignals(False)
        self._js("configure", palette, wrap)

    def _renderer_failed(self, *_args):
        self._ready = False
        self._failed = True
        self.path_label.setText("Editor stopped. Use Recover editor to restore the buffer copies retained in memory.")
        self.recover_button.show()

    def recover_renderer(self):
        if (
            QMessageBox.question(
                self,
                "Recover editor",
                "Restart the editor using its buffer copies? The last keystrokes before the renderer stopped may be missing.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        self._failed = False
        self.recover_button.hide()
        self.web_view.reload()

    def _error(self, title, message):
        QMessageBox.warning(self, title, message)

    def cleanup(self):
        self.context.window.theme_manager.changed.disconnect(self.apply_theme)
        self.context.window.settings_manager.changed.disconnect(self.apply_theme)
        self.watcher.blockSignals(True)
        self._ready = False
        self.web_view.stop()
