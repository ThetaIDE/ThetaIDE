import pytest
from PyQt6.QtWidgets import QApplication, QLineEdit

from frontend.config_viewer import ConfigViewer


def test_schema_number_editor_preserves_precision_and_null():
    app = QApplication.instance() or QApplication([])
    viewer = ConfigViewer()
    values = []
    editor = viewer._schema_widget(None, {"type": "number"}, values.append)
    assert isinstance(editor, QLineEdit)
    assert editor.text() == ""
    editor.setText("1e-9")
    editor.editingFinished.emit()
    assert values == [1e-9]
    editor.setText("invalid")
    editor.editingFinished.emit()
    assert values == [1e-9]
    editor.clear()
    editor.editingFinished.emit()
    assert values == [1e-9, None]
    assert app is not None


def test_schema_number_editor_honors_explicit_bounds():
    app = QApplication.instance() or QApplication([])
    viewer = ConfigViewer()
    values = []
    editor = viewer._schema_widget(0.5, {"type": "number", "minimum": 0, "maximum": 1}, values.append)
    editor.setText("2")
    editor.editingFinished.emit()
    assert values == []
    editor.setText("0.25")
    editor.editingFinished.emit()
    assert values == [0.25]
    assert app is not None
