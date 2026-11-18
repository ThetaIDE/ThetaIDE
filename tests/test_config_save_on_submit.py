"""Typed edits are saved when a box is submitted or left, never while the user is typing."""
import sys

import pytest
import yaml

pytest.importorskip("PyQt6")
from PyQt6.QtCore import Qt  # noqa: E402
from PyQt6.QtTest import QTest  # noqa: E402
from PyQt6.QtWidgets import QApplication, QPushButton  # noqa: E402

from frontend.config_viewer import ConfigViewer  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def editor(tmp_path, monkeypatch):
    path = tmp_path / "test_exp.yaml"
    path.write_text("paradigm: online_rl\nseed: 123\ntotal_timesteps: 25000\ndescription: before\n", encoding="utf-8")
    viewer = ConfigViewer()
    viewer.auto_save = True
    viewer.load_file(path, "experiment/test_exp.yaml")
    elsewhere = QPushButton("elsewhere", viewer)
    viewer.resize(1000, 900)
    viewer.show()
    app.processEvents()
    saves = []
    real = viewer.save_to_disk
    monkeypatch.setattr(viewer, "save_to_disk", lambda *a, **k: saves.append(1) or real(*a, **k))
    yield viewer, path, saves, elsewhere
    viewer.close()


def on_disk(path, key):
    return yaml.safe_load(path.read_text())[key]


def type_into(field, text):
    field.setFocus()
    app.processEvents()
    field.selectAll()
    QTest.keyClicks(field, text)


def test_typing_into_a_number_box_saves_only_on_enter(editor):
    viewer, path, saves, _ = editor
    type_into(viewer.spin_seed, "4567")
    assert viewer.raw_data["seed"] == 4567  # the document and pipeline check follow the typing
    assert saves == [] and on_disk(path, "seed") == 123
    QTest.keyClick(viewer.spin_seed, Qt.Key.Key_Return)
    assert saves == [1] and on_disk(path, "seed") == 4567


def test_leaving_a_number_box_saves_once(editor):
    viewer, path, saves, elsewhere = editor
    type_into(viewer.spin_seed, "89")
    assert saves == []
    elsewhere.setFocus()
    app.processEvents()
    assert saves == [1] and on_disk(path, "seed") == 89
    QTest.keyClick(viewer.spin_seed, Qt.Key.Key_Return)  # nothing new to write
    assert saves == [1]


def test_typing_into_a_text_box_saves_when_it_is_left(editor):
    viewer, path, saves, elsewhere = editor
    field = viewer.field_widgets["description"]
    type_into(field, "after")
    assert saves == [] and on_disk(path, "description") == "before"
    elsewhere.setFocus()
    app.processEvents()
    assert on_disk(path, "description") == "after"


def test_edits_without_a_focused_box_save_immediately(editor):
    viewer, path, saves, elsewhere = editor
    elsewhere.setFocus()
    app.processEvents()
    viewer.spin_seed.setValue(7)  # a toggle, choice or programmatic edit is already a submit
    assert saves == [1] and on_disk(path, "seed") == 7


def test_closing_writes_an_edit_still_held_by_a_box(editor):
    viewer, path, saves, _ = editor
    type_into(viewer.spin_seed, "321")
    assert on_disk(path, "seed") == 123
    viewer.flush_pending_save()  # what tab and window close call
    assert on_disk(path, "seed") == 321
