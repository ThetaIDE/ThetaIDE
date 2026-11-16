"""Launcher: a rofi-style overlay to jump to any panel, plugin, settings page or command by typing."""

import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QEvent, Qt  # noqa: E402
from PyQt6.QtGui import QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.launcher import LaunchItem, fuzzy_match, match_item, rank  # noqa: E402


def item(key, title, category="Panel", **kw):
    return LaunchItem(key=key, title=title, category=category, action=lambda: None, **kw)


# ── matching ─────────────────────────────────────────────────────────────────

def test_fuzzy_match_letters_in_order_and_prefers_tight_word_start_matches():
    assert fuzzy_match("tnsb", "TensorBoard")[1] == [0, 2, 3, 6]
    assert fuzzy_match("xyz", "TensorBoard") is None
    assert fuzzy_match("", "Anything") == (0, [])
    # a substring beats a scattered match; a word start beats the middle of a word
    assert fuzzy_match("mon", "Training monitor")[0] > fuzzy_match("mon", "Components")[0]
    assert fuzzy_match("board", "TensorBoard")[0] > fuzzy_match("tsbd", "TensorBoard")[0]
    assert fuzzy_match("que", "Job queue")[0] > fuzzy_match("que", "Unique")[0]


def test_every_word_must_match_somewhere_and_titles_are_highlighted():
    tb = item("pane:tensorboard", "TensorBoard", keywords=("tb",))
    score, positions = match_item("tb", tb)
    assert score > 0 and positions == [0, 6]  # the alias matched, but the title's letters still light up
    queue = item("command:toggle_queue", "Start or pause queue", "Command")
    assert match_item("start queue", queue)[1] == [0, 1, 2, 3, 4, 15, 16, 17, 18, 19]
    assert match_item("start zebra", queue) is None
    assert match_item("command", queue) is not None  # the category counts too


def test_descriptions_match_only_as_plain_text():
    hotkeys = item("settings:hotkeys", "Hotkeys", "Settings", detail="Action-key leader navigation")
    queue = item("pane:queue", "Job queue", detail="hidden in the sidebar, opening shows it")
    assert match_item("leader", hotkeys) is not None
    assert match_item("hot", queue) is None  # h…o…t scattered through the sentence is not a match


def test_rank_puts_recent_first_with_no_query_and_best_match_first_with_one():
    items = [item("pane:a", "Alpha"), item("command:z", "Zap", "Command"), item("settings:m", "Mode", "Settings"),
             item("pane:b", "Beta")]
    assert [i.key for i, _ in rank(items, "")] == ["pane:a", "pane:b", "settings:m", "command:z"]
    assert [i.key for i, _ in rank(items, "", recent=["command:z", "pane:b"])][:2] == ["command:z", "pane:b"]
    assert [i.key for i, _ in rank(items, "be")] == ["pane:b"]


# ── in the window ────────────────────────────────────────────────────────────

@pytest.fixture
def window():
    from frontend.app import Window

    temp = tempfile.TemporaryDirectory()
    win = Window(Path(temp.name) / ".thetaide" / "runs")
    win.resize(1200, 800)
    win.show()
    app.processEvents()
    yield win
    win._quitting = True
    win.close()
    temp.cleanup()


def press(widget, key, modifiers=Qt.KeyboardModifier.NoModifier):
    QApplication.sendEvent(widget, QKeyEvent(QEvent.Type.KeyPress, key, modifiers))
    app.processEvents()


def open_launcher(win):
    win.menu_actions["open_launcher"].trigger()
    app.processEvents()
    return win.launcher


def test_shortcut_is_rebindable_and_toggles_the_overlay(window):
    action = window.menu_actions["open_launcher"]
    assert action.shortcut().toString() == "Ctrl+Space"
    launcher = open_launcher(window)
    assert launcher.isVisible() and launcher.input.hasFocus() and launcher.list.count() > 10
    action.trigger()
    assert not launcher.isVisible()
    window.settings_manager.set("shortcuts", "open_launcher", "Ctrl+Shift+P")
    app.processEvents()
    assert action.shortcut().toString() == "Ctrl+Shift+P"


def test_keyboard_navigation_and_escape(window):
    launcher = open_launcher(window)
    first = launcher.list.currentRow()
    press(launcher.input, Qt.Key.Key_Down)
    press(launcher.input, Qt.Key.Key_N, Qt.KeyboardModifier.ControlModifier)
    assert launcher.list.currentRow() == first + 2
    press(launcher.input, Qt.Key.Key_Up)
    assert launcher.list.currentRow() == first + 1
    press(launcher.input, Qt.Key.Key_Up)
    press(launcher.input, Qt.Key.Key_Up)
    assert launcher.list.currentRow() == launcher.list.count() - 1  # wraps around
    press(launcher.input, Qt.Key.Key_Escape)
    assert not launcher.isVisible()


def test_opening_a_settings_page_and_running_a_command(window):
    launcher = open_launcher(window)
    launcher.input.setText("hotkeys")
    launcher.activate_current()
    app.processEvents()
    assert window.tabs.currentWidget() is window.tabs.settings_widget
    assert window.settings_view.active_tab_id == "hotkeys"
    zoom = window.zoom_factor
    open_launcher(window)
    launcher.input.setText("zoom in")
    launcher.activate_current()
    app.processEvents()
    assert window.zoom_factor > zoom
    window.reset_zoom()


def test_turning_on_a_plugin_from_the_launcher(window):
    assert "documentation" not in window.plugin_manager.instances
    launcher = open_launcher(window)
    launcher.input.setText("turn on documentation")
    assert launcher.current_item().key == "plugin:documentation"
    launcher.activate_current()
    app.processEvents()
    assert "documentation" in window.plugin_manager.instances
    assert window.tabs.currentWidget() is window.tabs.tabs["documentation"]["widget"]
    window.plugin_manager.disable_plugin("documentation")


def test_no_matches_and_a_click_on_the_backdrop_closes(window):
    from PyQt6.QtCore import QPointF
    from PyQt6.QtGui import QMouseEvent

    launcher = open_launcher(window)
    launcher.input.setText("zzqqxx")
    assert launcher.list.count() == 0 and launcher.hint.text() == "No matches"
    launcher.activate_current()  # nothing selected: nothing happens
    assert launcher.isVisible()
    click = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5), Qt.MouseButton.LeftButton,
                        Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    launcher.mousePressEvent(click)
    assert not launcher.isVisible()
