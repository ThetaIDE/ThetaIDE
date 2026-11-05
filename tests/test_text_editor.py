"""File safety, the actual CodeMirror bridge, and host editor routing."""

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")
pytest.importorskip("PyQt6.QtWebEngineWidgets")
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QMessageBox

from frontend.editor_service import EditorService
from frontend.plugins.core.text_editor.documents import Document, FileConflictError
from frontend.plugins.core.text_editor.panel import TextEditorPanel
from frontend.settings import SettingsManager
from frontend.theme import ThemeManager

_app = None


def pump_until(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        QApplication.processEvents()
        if predicate():
            return
        QTest.qWait(10)
    assert predicate(), "Timed out waiting for editor"


@pytest.fixture
def qapp():
    global _app
    if QApplication.instance() is None:
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    _app = QApplication.instance() or QApplication(["theta-editor-tests"])
    return _app


@pytest.fixture
def panel(qapp, tmp_path):
    settings = SettingsManager(tmp_path)
    theme = ThemeManager(tmp_path / ".appearance.json")
    window = SimpleNamespace(settings_manager=settings, theme_manager=theme)
    instance = TextEditorPanel(SimpleNamespace(window=window))
    instance.resize(1100, 680)
    instance.show()
    errors = []
    instance._error = lambda title, message: errors.append((title, message))
    instance.test_errors = errors
    pump_until(lambda: instance._ready)
    yield instance
    instance.cleanup()
    instance.close()
    instance.deleteLater()
    QApplication.processEvents()


@pytest.mark.parametrize("raw", [b"x = 1\n", b"x = 1\r\n", b"x = 1\r", b"\xef\xbb\xbfx = 1\r\n"])
def test_round_trip_preserves_encoding_and_line_endings(tmp_path, raw):
    path = tmp_path / "test.py"
    path.write_bytes(raw)
    doc = Document.open(path)
    doc.save()
    assert path.read_bytes() == raw
    doc.text = doc.text.replace("1", "2")
    doc.save()
    assert path.read_bytes() == raw.replace(b"1", b"2")
    assert not doc.dirty


def test_conflicting_save_keeps_external_file_and_buffer(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("original")
    doc = Document.open(path)
    doc.text = "my changes"
    path.write_text("external changes")
    with pytest.raises(FileConflictError):
        doc.save()
    assert path.read_text() == "external changes"
    assert doc.dirty
    doc.save(force=True)
    assert path.read_text() == "my changes"


def test_deleted_file_requires_explicit_recreation(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("original")
    doc = Document.open(path)
    path.unlink()
    assert doc.changed_on_disk()
    with pytest.raises(FileConflictError):
        doc.save()
    doc.save(force=True)
    assert path.read_text() == "original"


def test_failed_atomic_write_keeps_original_and_dirty_buffer(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("original")
    doc = Document.open(path)
    doc.text = "changes"
    with patch("frontend.plugins.core.text_editor.documents.os.replace", side_effect=OSError("disk full")):
        with pytest.raises(OSError):
            doc.save()
    assert path.read_text() == "original"
    assert doc.dirty
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("raw", [b"\x00binary", b"\xff\xfeUTF16", b"a\r\nb\n"])
def test_unsupported_input_is_not_silently_converted(tmp_path, raw):
    path = tmp_path / "test.bin"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        Document.open(path)


def test_oversized_file_rejected(tmp_path):
    path = tmp_path / "large.txt"
    path.write_bytes(b"x" * (10 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="10 MiB"):
        Document.open(path)


def test_symlink_save_preserves_link_and_permissions(tmp_path):
    path = tmp_path / "script.py"
    path.write_text("before")
    path.chmod(0o755)
    link = tmp_path / "linked.py"
    link.symlink_to(path)
    doc = Document.open(link)
    doc.text = "after"
    doc.save()
    assert link.is_symlink()
    assert path.read_text() == "after"
    assert path.stat().st_mode & 0o777 == 0o755


def test_default_provider_routes_locations_and_falls_back(qapp, tmp_path):
    settings = SettingsManager(tmp_path)
    service = EditorService(settings)
    opened = []
    service.register("test", "Test editor", lambda path, **location: opened.append((path, location)) or True)
    settings.default_text_editor = "test"
    assert service.open_file(tmp_path / "a.txt", 12, 3)
    assert opened == [(tmp_path / "a.txt", {"line": 12, "column": 3})]
    service.unregister("test")
    with patch("frontend.editor_service.QDesktopServices.openUrl", return_value=True) as external:
        assert service.open_file(tmp_path / "a.txt")
        external.assert_called_once()


def evaluate(panel, script):
    values = []
    panel.web_view.page().runJavaScript(script, values.append)
    pump_until(lambda: bool(values))
    return values[0]


def test_real_codemirror_edit_save_undo_and_tab_state(panel, tmp_path):
    path = tmp_path / "example.py"
    path.write_text("print('hello')\n")
    assert panel.open_file(path, line=1, column=3)
    pump_until(lambda: evaluate(panel, "!!document.querySelector('.cm-editor')"))
    # Browser input follows the same path as a user typing into the contenteditable.
    evaluate(panel, "document.querySelector('.cm-content').focus(); document.execCommand('insertText', false, 'ABC')")
    assert panel.sync_documents()
    first = panel.current_document
    assert "ABC" in first.text
    assert first.dirty
    assert panel.save_current()
    assert "ABC" in path.read_text()
    assert not first.dirty
    second = panel.new_document()
    panel.tab_bar.setCurrentIndex(0)
    panel._js("action", "undo")
    assert panel.sync_documents()
    assert first.text == "print('hello')\n"
    assert first.dirty  # undoing past the save point is an unsaved change
    assert second.text == ""
    assert not panel.test_errors


def test_open_deduplicates_and_save_as_refuses_another_open_tab(panel, tmp_path):
    first, second = tmp_path / "a.txt", tmp_path / "b.txt"
    first.write_text("a")
    second.write_text("b")
    assert panel.open_file(first)
    assert panel.open_file(first.parent / "." / first.name)
    assert len(panel.documents) == 1
    assert panel.open_file(second)
    assert not panel.save_document(panel.current_document, first)
    assert first.read_text() == "a"


def test_document_tabs_restore_scroll_position(panel, tmp_path):
    path = tmp_path / "long.txt"
    path.write_text("\n".join(f"Line {index}" for index in range(300)))
    panel.open_file(path)
    pump_until(lambda: evaluate(panel, "document.querySelector('.cm-scroller').scrollHeight") > 3000)
    evaluate(panel, "document.querySelector('.cm-scroller').scrollTop = 1200")
    pump_until(lambda: evaluate(panel, "document.querySelector('.cm-scroller').scrollTop") > 1000)
    QTest.qWait(100)
    panel.new_document()
    QTest.qWait(50)
    panel.tab_bar.setCurrentIndex(0)
    pump_until(lambda: evaluate(panel, "document.querySelector('.cm-scroller').scrollTop") > 1000)


def test_close_cancel_and_discard_and_renderer_snapshot(panel):
    doc = panel.new_document()
    evaluate(
        panel, "document.querySelector('.cm-content').focus(); document.execCommand('insertText', false, 'unsaved')"
    )
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
        assert not panel.can_close()
        assert not panel.close_document()
    assert doc.text == "unsaved"
    assert len(panel.documents) == 1
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
        from PyQt6.QtWidgets import QTabBar

        panel.tab_bar.tabButton(0, QTabBar.ButtonPosition.RightSide).click()
    assert not panel.documents


def test_external_replacement_reload_and_conflict_cancel(panel, tmp_path):
    path = tmp_path / "test.md"
    path.write_text("original")
    panel.open_file(path)
    doc = panel.current_document
    path.write_text("external")
    panel.check_external_changes()
    assert not panel.banner.isHidden()
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
        assert not panel.save_current()
    assert path.read_text() == "external"
    assert panel.reload_current()
    assert doc.text == "external"
    assert not doc.dirty
    assert not panel.banner.isVisible()
    assert panel.sync_documents()
    assert doc.text == "external"


def test_save_as_cancel_preserves_unsaved_buffer(panel):
    doc = panel.new_document()
    evaluate(
        panel, "document.querySelector('.cm-content').focus(); document.execCommand('insertText', false, 'new text')"
    )
    with patch("frontend.plugins.core.text_editor.panel.QFileDialog.getSaveFileName", return_value=("", "")):
        assert not panel.save_current()
    assert doc.dirty
    assert doc.path is None


def test_local_editor_blocks_remote_and_unrelated_local_navigation(panel, tmp_path):
    from PyQt6.QtCore import QUrl

    assert not panel.web_view.page().acceptNavigationRequest(QUrl("https://example.com"), None, True)
    assert not panel.web_view.page().acceptNavigationRequest(QUrl.fromLocalFile(str(tmp_path / "a.html")), None, True)


@pytest.fixture
def host(qapp, tmp_path):
    from PyQt6.QtWidgets import QWidget

    from frontend.app import Window

    class QuietTerminal(QWidget):
        def __init__(self, *args, parent=None, **kwargs):
            super().__init__(parent)
            self.terminals = []

        def configure(self, **kwargs):
            pass

        def set_font_size(self, size):
            pass

        def apply_theme(self, theme):
            pass

    with (
        patch("frontend.app.TerminalPanel", QuietTerminal),
        patch.object(Window, "init_default_experiment", lambda self: None),
    ):
        window = Window(tmp_path)
    yield window
    window.close()


def test_host_enable_default_settings_and_disable_cancel(host, tmp_path):
    manager = host.plugin_manager
    assert manager.manifests["text_editor"].is_core
    assert not manager.is_plugin_active("text_editor")
    host.on_plugin_slider_toggled("text_editor", True)
    assert manager.is_plugin_active("text_editor")
    assert host.settings_manager.default_text_editor == "text_editor"
    assert host.settings_view.default_editor_combo.currentData() == "text_editor"
    instance = manager.instances["text_editor"]
    pump_until(lambda: instance.panel._ready)
    host.tabs.setCurrentWidget(instance.panel)
    host.menu_actions["new_experiment"].trigger()
    evaluate(
        instance.panel,
        "document.querySelector('.cm-content').focus(); document.execCommand('insertText', false, 'keep me')",
    )
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Cancel):
        assert not manager.disable_plugin("text_editor")
        host.on_plugin_slider_toggled("text_editor", False)
        assert host.plugin_sliders["text_editor"].isChecked()
        from PyQt6.QtGui import QCloseEvent

        event = QCloseEvent()
        host.closeEvent(event)
        assert not event.isAccepted()
    assert manager.is_plugin_active("text_editor")
    assert host.settings_manager.default_text_editor == "text_editor"
    with patch.object(QMessageBox, "warning", return_value=QMessageBox.StandardButton.Discard):
        assert manager.disable_plugin("text_editor")
    assert "text_editor" not in host.tabs.tabs
    assert host.settings_manager.default_text_editor == "system"
    assert host.settings_view.default_editor_combo.currentData() == "system"
    assert host.command_registry.get_command("text_editor.open") is None


def test_host_settings_file_routing_menu_save_and_restart_preference(host):
    assert host.plugin_manager.enable_plugin("text_editor")
    panel = host.plugin_manager.instances["text_editor"].panel
    pump_until(lambda: panel._ready)
    host.open_settings_file()
    assert host.tabs.currentWidget() is panel
    assert panel.current_document.path == host.settings_manager.workspace_settings_path
    evaluate(
        panel,
        "document.querySelector('.cm-content').focus(); document.execCommand('insertText', false, '# edited in Theta\\n')",
    )
    host.menu_actions["save_config"].trigger()
    assert not panel.current_document.dirty
    assert host.settings_manager.workspace_settings_path.read_text().startswith("# edited in Theta\n")
    host.settings_manager.default_text_editor = "system"
    assert host.plugin_manager.shutdown()
    # Initialize from persisted enabled states, preserving an explicit system-editor preference.
    host.plugin_manager.initialize_plugins()
    assert host.plugin_manager.is_plugin_active("text_editor")
    assert host.settings_manager.default_text_editor == "system"
