"""Regression coverage for settings cascades without starting runner/WebEngine."""
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication, QWidget

from frontend.config_viewer import ConfigViewer
from frontend.hotkeys import HotkeyManager
from frontend.settings import SettingsManager, _parse_toml_file, _write_toml_file
from frontend.settings_view import SettingsView


@pytest.fixture
def settings(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    manager = SettingsManager(tmp_path / "settings")
    yield manager
    manager.deleteLater()
    app.processEvents()


def test_batch_persists_once_and_notifies_complete_state(settings, monkeypatch):
    import frontend.settings as module
    writer = Mock(wraps=_write_toml_file)
    monkeypatch.setattr(module, "_write_toml_file", writer)
    observed = []
    settings.changed.connect(lambda: observed.append(settings.section("hotkeys")))
    updates = {("hotkeys", "bindings"): {"config": "Action + T"},
               ("hotkeys", "panes"): ["none"] * 10}
    settings.set_many(updates)
    settings.set_many(updates)
    settings.set("hotkeys", "bindings", {"config": "Action + T"})
    assert writer.call_count == len(observed) == 1
    assert observed[0]["panes"] == ["none"] * 10
    assert _parse_toml_file(settings.workspace_settings_path)["hotkeys"] == observed[0]
    # Watcher notifications for our own atomic write must be harmless.
    settings._on_file_changed(str(settings.workspace_settings_path))
    assert len(observed) == 1


def test_failed_commit_does_not_publish(settings, monkeypatch):
    import frontend.settings as module
    previous = settings.action_key
    signal = Mock()
    settings.changed.connect(signal)
    monkeypatch.setattr(module, "_write_toml_file", Mock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        settings.set("hotkeys", "action_key", "alt")
    assert settings.action_key == previous
    signal.assert_not_called()


def test_unrelated_settings_preserve_multiple_forms(settings, monkeypatch):
    viewers = [ConfigViewer(settings_manager=settings) for _ in range(3)]
    renders = []
    for viewer in viewers:
        viewer.raw_data = {"description": "unsaved edit"}
        viewer.is_dirty = True
        render = Mock()
        monkeypatch.setattr(viewer, "_render_boxes", render)
        renders.append(render)
    # Also retain a per-tab view override across unrelated global settings.
    viewers[0]._on_view_mode_toggled(True)
    renders[0].reset_mock()
    settings.set("notifications", "enabled", False)
    settings.set("hotkeys", "action_key", "alt")
    assert viewers[0].view_mode == "resolved"
    for viewer, render in zip(viewers, renders):
        render.assert_not_called()
        assert viewer.is_dirty and viewer.raw_data["description"] == "unsaved edit"
    settings.show_method_agent_defaults = True
    for render in renders:
        assert render.call_count == 1
    # External atomic replacement also updates relevant preferences once.
    data = _parse_toml_file(settings.workspace_settings_path)
    data["appearance"]["show_resolved_defaults"] = True
    _write_toml_file(settings.workspace_settings_path, data)
    settings._on_file_changed(str(settings.workspace_settings_path))
    for viewer, render in zip(viewers, renders):
        assert render.call_count == 2
        assert viewer.view_mode == "resolved" and viewer.chk_view_mode.isChecked()
        viewer.deleteLater()


def test_rebinding_clearing_and_legacy_navigation(settings):
    window = QWidget()
    manager = HotkeyManager(window, settings)
    handler = SimpleNamespace(window=SimpleNamespace(settings_manager=settings))
    observed = Mock()
    settings.changed.connect(observed)
    try:
        assert manager.get_pane_id_by_index(2) == "config"
        SettingsView._on_pane_hotkey_changed(handler, "config", "Action + T")
        assert observed.call_count == 1
        assert settings.hotkey_panes[2] == "none"
        assert manager.get_pane_id_by_index(2) is None
        switch = Mock()
        manager.switch_to_pane = switch
        manager._leader_active = True
        manager.eventFilter(window, QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_T,
                                             Qt.KeyboardModifier.NoModifier, "t"))
        switch.assert_called_once_with("config")
        # A stale legacy slot cannot resurrect an explicit binding or clearing.
        settings.set("hotkeys", "panes", ["settings", "components", "config"])
        assert manager.get_pane_id_by_index(2) is None
        SettingsView._on_pane_hotkey_changed(handler, "config", "")
        assert "config" not in settings.hotkey_bindings
        assert manager.get_pane_id_by_index(2) is None
        SettingsView._on_pane_hotkey_changed(handler, "config", "Action + 4")
        assert manager.get_pane_id_by_index(4) == "results"  # first conflicting target
        assert settings.hotkey_bindings["results"] == "Action + 4"
    finally:
        manager.cleanup()
        window.deleteLater()
