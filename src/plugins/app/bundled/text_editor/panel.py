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
    QSplitter,
    QStackedWidget,
    QTabBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.app.api.desktop import current_theme, refresh_tool_icons, style_tool_button, svg_icon

from .documents import Document, FileConflictError
from .explorer import FileExplorer

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
        self.panel.context.mark_startup("editor_renderer_ready")
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
    def cursorChanged(self, _line, _column):
        # Retain the bridge entry point for the bundled editor surface.
        pass


class TextEditorPanel(QWidget):
    def __init__(self, context, parent=None, *, lazy=False):
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

        self.file_explorer = FileExplorer(settings_manager=context.tree_toolbar_store())
        workspace = context.workspace_paths()
        if workspace.research.is_dir():
            self.file_explorer.open_root(workspace.research)
        self.component_tree = self.file_explorer.source_tree
        self.file_explorer.fileActivated.connect(self.open_file)
        self.file_explorer.openFolderRequested.connect(self.choose_folder)
        self.file_explorer.actionRequested.connect(self.perform_action)
        self.file_explorer.toggleRequested.connect(self.toggle_file_tree)
        self._explorer_width = 260
        self.wrap_button = self.file_explorer.action_buttons["wrap"]
        self.wrap_button.setChecked(bool(context.editor_preference("wrap_lines", False)))
        self.wrap_button.toggled.connect(self._set_wrap)
        self.recover_button = self.file_explorer.action_buttons["recover"]
        self.recover_button.clicked.connect(self.recover_renderer)

        self.editor_pane = QWidget()
        editor_layout = QVBoxLayout(self.editor_pane)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(8)
        self.tab_header = QWidget()
        self.tab_header_layout = QHBoxLayout(self.tab_header)
        self.tab_header_layout.setContentsMargins(0, 0, 0, 0)
        self.tab_header_layout.setSpacing(4)

        self.tab_bar = QTabBar()
        self.tab_bar.setExpanding(False)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setTabsClosable(False)
        self.tab_bar.setUsesScrollButtons(True)
        self.tab_bar.setElideMode(Qt.TextElideMode.ElideMiddle)
        self.tab_bar.currentChanged.connect(lambda _index: self._show_current())
        self.tab_bar.tabCloseRequested.connect(self.close_document)
        self.tab_header_layout.addWidget(self.tab_bar, 1)
        editor_layout.addWidget(self.tab_header)

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
        editor_layout.addWidget(self.banner)
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
        self.web_view = None
        editor_layout.addWidget(self.stack, 1)
        self.editor_splitter = QSplitter()
        self.editor_splitter.addWidget(self.file_explorer)
        self.editor_splitter.addWidget(self.editor_pane)
        self.editor_splitter.setStretchFactor(0, 0)
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setCollapsible(1, False)
        self.editor_splitter.setCollapsible(0, False)
        layout.addWidget(self.editor_splitter, 1)
        self.editor_splitter.setSizes([260, 640])

        self.watcher = QFileSystemWatcher(self)
        self.watcher.fileChanged.connect(self.check_external_changes)
        self.watcher.directoryChanged.connect(self.check_external_changes)
        context.add_appearance_listener(self.apply_theme)
        if not lazy:
            self._ensure_renderer()
        self.apply_theme()

    def showEvent(self, event):
        super().showEvent(event)
        self._ensure_renderer()

    def _ensure_renderer(self):
        if self.web_view is not None:
            return
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
        return {"id": doc.id, "name": doc.title, "text": doc.text, "revision": self._revisions[doc.id],
                "readOnly": bool(doc.path and self._hub_read_only(doc.path))}

    def _hub_read_only(self, path):
        return self.context.is_managed_file(path)

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

    def choose_folder(self):
        root = QFileDialog.getExistingDirectory(self, "Open folder", self._start_folder())
        return bool(root) and self.open_folder(root)

    def _start_folder(self):
        doc = self.current_document
        return str(doc.path.parent if doc and doc.path else self.file_explorer.root or Path.cwd())

    def _show_current(self):
        doc = self.current_document
        if doc:
            self._ensure_renderer()
        self.stack.setCurrentIndex(1 if doc else 0)
        self.file_explorer.reveal(doc.path if doc else None)
        self.banner.hide()
        if doc:
            if self._ready:
                line, column = self._locations.pop(doc.id, (None, None))
                self._js("show", self._payload(doc), line, column)
            self.check_external_changes()

    def _update_tab(self, index):
        doc = self.documents[index]
        locked = bool(doc.path and self._hub_read_only(doc.path))
        self.tab_bar.setTabText(index, f"{'● ' if doc.dirty else ''}{doc.title}{' (read-only)' if locked else ''}")
        self.tab_bar.setTabToolTip(index, (str(doc.path) if doc.path else doc.title) + ("\nHub component: make a copy to edit." if locked else ""))

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
        if target and self._hub_read_only(target):
            self._error("Read-only component", "Hub components are read-only. Make a component copy to edit it.")
            return False
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
        if doc.path and doc.path.suffix.lower() in (".yaml", ".yml", ".json"):
            self.context.notify_config_files_changed()
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
            self.context.notify_config_files_changed()
            self.banner_label.setText(
                "This file changed on disk or was deleted. Reload it, or keep editing and review before saving."
            )

    def perform_action(self, action):
        if self._busy:
            return
        actions = {
            "new": self.new_document,
            "open": self.choose_file,
            "open_folder": self.choose_folder,
            "save": self.save_current,
            "save_as": lambda: self.save_current(True),
            "close": self.close_document,
        }
        if action in actions:
            return actions[action]()
        self._js("action", action)

    def _set_wrap(self, value):
        self.context.set_editor_preference("wrap_lines", value)

    def apply_theme(self):
        self.file_explorer.apply_theme()
        refresh_tool_icons(self.tab_header)
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
        if self.web_view is not None:
            self.web_view.page().setBackgroundColor(QColor(palette["base"]))
            self.web_view.setZoomFactor(self.context.ui_zoom())
        wrap = bool(self.context.editor_preference("wrap_lines", False))
        self.wrap_button.blockSignals(True)
        self.wrap_button.setChecked(wrap)
        self.wrap_button.blockSignals(False)
        self._js("configure", palette, wrap)

    def _renderer_failed(self, *_args):
        self._ready = False
        self._failed = True
        self.recover_button.show()
        self.set_file_tree_visible(True)

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
        self.context.remove_appearance_listener(self.apply_theme)
        self.watcher.blockSignals(True)
        self._ready = False
        if self.web_view is not None:
            self.web_view.stop()

    def toggle_file_tree(self):
        self.set_file_tree_visible(self.file_explorer.isHidden())

    def set_file_tree_visible(self, visible):
        button = self.file_explorer.toggle_button
        if visible:
            if not self.file_explorer.isHidden():
                return
            self.tab_header_layout.removeWidget(button)
            self.file_explorer.restore_toggle_button()
            style_tool_button(button, "sidebar_collapse", "Collapse file tree")
            self.file_explorer.show()
            available = self.editor_splitter.width() - self.editor_splitter.handleWidth()
            self.editor_splitter.setSizes([self._explorer_width,
                                           max(1, available - self._explorer_width)])
        else:
            if self.file_explorer.isHidden():
                return
            self._explorer_width = self.editor_splitter.sizes()[0] or self._explorer_width
            self.file_explorer.detach_toggle_button()
            self.tab_header_layout.insertWidget(0, button)
            style_tool_button(button, "sidebar_expand", "Show file tree")
            self.file_explorer.hide()
        button.show()

    def open_folder(self, root, initial_file=None, owned_paths=()):
        try:
            self.file_explorer.open_root(root, owned_paths)
        except (OSError, ValueError) as exc:
            self._error("Cannot open folder", str(exc))
            return False
        self.set_file_tree_visible(True)
        if initial_file:
            return self.open_file(initial_file)
        self.file_explorer.reveal(self.current_document.path if self.current_document else None)
        return True
