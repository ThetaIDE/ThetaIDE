"""Sidebar bins: drag tabs onto each other to group them; bins expand, collapse, rename and are saved."""

import sys

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QMimeData, QPoint, QPointF, Qt  # noqa: E402
from PyQt6.QtGui import QDropEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication, QLabel  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])

from frontend.sidetabs import BIN_MIME, MERGE_ZONE, TAB_MIME, SideTabs  # noqa: E402

IDS = ["components", "config", "monitor", "results", "queue", "console"]


@pytest.fixture
def tabs():
    widget = SideTabs()
    for tab_id in IDS:
        widget.addTab(QLabel(tab_id), tab_id.title(), icon=tab_id, short=tab_id.title(), tab_id=tab_id)
    widget.set_settings_widget(QLabel("settings"))
    widget.resize(300, 900)
    widget.show()
    app.processEvents()
    widget.saved = []
    widget.tabOrderChanged.connect(lambda order: widget.saved.append(list(order)))
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def shown(tabs):
    return [(kind, item_id) for kind, item_id, _w in tabs.visible_items()]


def drop(tabs, kind, item_id, y):
    """Simulate dropping a dragged tab or bin at bar height y."""
    mime = QMimeData()
    mime.setData(TAB_MIME if kind == "tab" else BIN_MIME, item_id.encode())
    for event_type in ("move", "drop"):
        event = QDropEvent(QPointF(30, y), Qt.DropAction.MoveAction, mime, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier)
        if event_type == "move":
            from PyQt6.QtGui import QDragMoveEvent
            event = QDragMoveEvent(QPointF(30, y).toPoint(), Qt.DropAction.MoveAction, mime,
                                   Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
            tabs.bar.dragMoveEvent(event)
        else:
            tabs.bar.dropEvent(event)
    app.processEvents()


def middle_of(widget):
    rect = widget.geometry()
    return rect.top() + rect.height() * 0.5


def edge_of(widget, top=True):
    rect = widget.geometry()
    return rect.top() + rect.height() * (MERGE_ZONE / 2 if top else 1 - MERGE_ZONE / 2)


def test_dropping_a_tab_on_another_makes_a_bin_in_its_place(tabs):
    drop(tabs, "tab", "queue", middle_of(tabs.tabs["monitor"]["button"]))
    (bin_id, entry), = tabs.bins.items()
    assert entry["members"] == ["monitor", "queue"] and entry["name"] == "Bin 1"
    assert tabs.entries == ["components", "config", bin_id, "results", "console"]
    assert tabs.tab_order == ["components", "config", "monitor", "queue", "results", "console"]
    assert shown(tabs)[2:5] == [("bin", bin_id), ("tab", "monitor"), ("tab", "queue")]
    assert tabs.saved  # saved so the layout persists


def test_dragging_from_above_keeps_the_bin_where_the_target_was(tabs):
    tabs.group_tabs("results", "components")
    bin_id = next(iter(tabs.bins))
    assert tabs.entries == ["config", "monitor", bin_id, "queue", "console"]
    assert tabs.bins[bin_id]["members"] == ["results", "components"]


def test_dropping_on_a_bin_header_or_member_adds_to_the_bin(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    drop(tabs, "tab", "console", middle_of(tabs.bins[bin_id]["header"]))
    assert tabs.bins[bin_id]["members"] == ["monitor", "queue", "console"]
    drop(tabs, "tab", "components", middle_of(tabs.tabs["monitor"]["button"]))
    assert tabs.bins[bin_id]["members"] == ["monitor", "components", "queue", "console"]


def test_collapsing_hides_members_and_highlights_the_bin_when_it_hides_the_open_panel(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    tabs.setCurrentWidget(tabs.tabs["queue"]["widget"])
    header = tabs.bins[bin_id]["header"]
    header.click()
    app.processEvents()
    assert not tabs.bins[bin_id]["expanded"]
    assert not tabs.tabs["queue"]["button"].isVisible() and header.isVisible()
    assert header.isChecked()  # the open panel is inside
    header.click()
    app.processEvents()
    assert tabs.tabs["queue"]["button"].isVisible() and not header.isChecked()


def test_rename_inline_and_names_are_cleaned(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    header = tabs.bins[bin_id]["header"]
    header.start_rename()
    header.editor.setText("  Training   runs  ")
    header.editor.returnPressed.emit()
    assert tabs.bins[bin_id]["name"] == "Training runs"
    header.start_rename()
    header.editor.setText("discarded")
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtCore import QEvent
    header.editor.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier))
    assert tabs.bins[bin_id]["name"] == "Training runs" and header.editor is None
    tabs.rename_bin(bin_id, "   ")
    assert tabs.bins[bin_id]["name"] == "Training runs"  # empty names are ignored


def _click_twice(header, gap_ms):
    """Two left clicks on the header as real input arrives: press, release, double-click, release.
    Qt reports the second press as a double-click whenever it's inside the platform interval."""
    from PyQt6.QtCore import QEvent
    from PyQt6.QtGui import QMouseEvent

    class TimedMouseEvent(QMouseEvent):  # PyQt6 has no QMouseEvent.setTimestamp
        def __init__(self, ts, *args):
            super().__init__(*args)
            self._ts = ts

        def timestamp(self):
            return self._ts

    pos = QPointF(header.rect().center())
    glob = QPointF(header.mapToGlobal(header.rect().center()))
    left, none = Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton
    for kind, ts, buttons in ((QEvent.Type.MouseButtonPress, 1000, left), (QEvent.Type.MouseButtonRelease, 1040, none),
                              (QEvent.Type.MouseButtonDblClick, 1000 + gap_ms, left),
                              (QEvent.Type.MouseButtonRelease, 1040 + gap_ms, none)):
        event = TimedMouseEvent(ts, kind, pos, glob, left, buttons, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(header, event)
        app.processEvents()


def test_two_unhurried_clicks_toggle_twice_and_do_not_rename(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    header = tabs.bins[bin_id]["header"]
    _click_twice(header, 380)  # inside the 400 ms platform interval, but not a quick double-click
    assert header.editor is None
    assert tabs.bins[bin_id]["expanded"]  # collapsed, then expanded again


def test_a_quick_double_click_renames_and_leaves_the_bin_as_it_was(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    header = tabs.bins[bin_id]["header"]
    _click_twice(header, 150)
    assert header.editor is not None
    assert tabs.bins[bin_id]["expanded"]
    header.finish_rename(False)


def test_clicking_elsewhere_ends_the_rename_and_keeps_the_name(tabs):
    from PyQt6.QtTest import QTest

    bin_id = tabs.group_tabs("monitor", "queue")
    header = tabs.bins[bin_id]["header"]
    header.start_rename()
    header.editor.setText("Training runs")
    # the sidebar's empty area below the tabs takes no focus, so only the click itself can end editing
    QTest.mouseClick(tabs.bar, Qt.MouseButton.LeftButton, pos=tabs.bar.rect().bottomLeft() + QPoint(4, -4))
    app.processEvents()
    assert header.editor is None and tabs.bins[bin_id]["name"] == "Training runs"


def test_dragging_a_member_out_by_the_edges_leaves_the_bin_and_the_last_one_out_removes_it(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    drop(tabs, "tab", "queue", edge_of(tabs.tabs["console"]["button"], top=False))  # below the last tab
    assert tabs.bins[bin_id]["members"] == ["monitor"] and tabs.entries[-1] == "queue"
    tabs.remove_from_bin("monitor")
    assert tabs.bins == {} and "monitor" in tabs.entries
    assert tabs.tab_order == ["components", "config", "monitor", "results", "console", "queue"]


def test_dropping_between_members_of_an_open_bin_joins_it(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    drop(tabs, "tab", "console", edge_of(tabs.tabs["queue"]["button"], top=True))  # between monitor and queue
    assert tabs.bins[bin_id]["members"] == ["monitor", "console", "queue"]


def test_plain_reordering_still_works_without_bins(tabs):
    tabs.move_tab_to_visible_slot("console", 0)
    assert tabs.tab_order[0] == "console"
    before = len(tabs.saved)
    tabs.move_tab_to_visible_slot("console", 1)  # right after itself: nothing changes
    assert len(tabs.saved) == before


def test_bins_move_as_a_whole_and_never_nest(tabs):
    first = tabs.group_tabs("monitor", "queue")
    second = tabs.group_tabs("results", "console")
    tabs.move_bin_to_visible_slot(second, 0)
    assert tabs.entries[0] == second and tabs.tab_order[:2] == ["results", "console"]
    # Into the middle of the other open bin: lands just after it instead
    items = shown(tabs)
    inside = items.index(("tab", "queue"))
    tabs.move_bin_to_visible_slot(second, inside)
    assert tabs.entries.index(second) == tabs.entries.index(first) + 1
    assert all(len(b["members"]) == 2 for b in tabs.bins.values())


def test_ungroup_puts_the_tabs_back_where_the_bin_was(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    tabs.ungroup(bin_id)
    assert tabs.bins == {} and tabs.entries == ["components", "config", "monitor", "queue", "results", "console"]


def test_hidden_tabs_and_a_bin_of_hidden_tabs(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    tabs.set_tab_visible("monitor", False)
    assert ("tab", "monitor") not in shown(tabs) and ("bin", bin_id) in shown(tabs)
    tabs.set_tab_visible("queue", False)
    assert ("bin", bin_id) not in shown(tabs)  # nothing left to show in it


def test_removing_a_tab_updates_its_bin(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    tabs.removeTab("queue")
    assert tabs.bins[bin_id]["members"] == ["monitor"]
    tabs.removeTab("monitor")
    assert tabs.bins == {} and "monitor" not in tabs.tab_order


def test_state_round_trips_and_survives_missing_tabs(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    tabs.rename_bin(bin_id, "Training")
    tabs.set_bin_expanded(bin_id, False)
    state, order = tabs.bins_state(), list(tabs.tab_order)
    assert state == [{"name": "Training", "members": ["monitor", "queue"], "expanded": False}]

    fresh = SideTabs()
    for tab_id in IDS:
        if tab_id != "queue":  # e.g. a plugin tab that is now switched off
            fresh.addTab(QLabel(tab_id), tab_id, tab_id=tab_id)
    fresh.set_bins(state + [{"name": "Gone", "members": ["nonexistent"]}, "junk"])
    fresh.apply_tab_order(order)
    assert fresh.bins_state() == [{"name": "Training", "members": ["monitor"], "expanded": False}]
    assert fresh.tab_order == ["components", "config", "monitor", "results", "console"]
    fresh.deleteLater()


def test_apply_tab_order_keeps_bins_together(tabs):
    bin_id = tabs.group_tabs("monitor", "queue")
    tabs.apply_tab_order(["queue", "console", "components", "monitor", "config", "results"])
    assert tabs.entries[0] == bin_id and tabs.bins[bin_id]["members"] == ["queue", "monitor"]
    assert tabs.tab_order == ["queue", "monitor", "console", "components", "config", "results"]


