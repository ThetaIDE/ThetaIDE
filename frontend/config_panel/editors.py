"""Schema/type editor factory and draft validation, shared by section editors."""

import math
import re

import yaml
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QFileDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QWidget

from ..widgets import ScientificDoubleSpinBox, ToggleSlider, WrappingFlowWidget, integer_field
from .controls import (NUMBER_FIELD_WIDTH, TEXT_FIELD_WIDTH, ComboBox, _make_param_adder_button,
                       _make_subtle_tool_button)
from .document import validation_error


class DirectoryField(QLineEdit):
    """A shared folder picker that opens from the field itself."""

    def __init__(self, value, write, *, initial_directory="", parent=None):
        super().__init__(str(value or ""), parent)
        self.setReadOnly(True)
        self.setMaximumWidth(TEXT_FIELD_WIDTH)
        self.setPlaceholderText("Default location")
        self.setAccessibleName("Save location")
        self._write = write
        self._initial_directory = str(initial_directory)

    def browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Save trajectories to", self.text() or self._initial_directory
        )
        if directory:
            self.setText(directory)
            self.setToolTip(directory)
            self._write(directory)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self.browse()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.browse()
        else:
            super().keyPressEvent(event)


def set_error(widget, error):
    if widget.property("parameterInvalid") == bool(error) and widget.accessibleDescription() == (error or ""):
        return
    widget.setProperty("parameterInvalid", bool(error))
    widget.setAccessibleDescription(error or "")
    widget.setToolTip(error or widget.property("parameterHelp") or "")
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def draft_field(value, schema, write, parse, *, help_text="", width=TEXT_FIELD_WIDTH):
    widget = QLineEdit("" if value is None else str(value))
    widget.setMaximumWidth(width)
    widget.setProperty("parameterHelp", help_text)
    widget.setToolTip(help_text)

    # Keep invalid drafts out of storage, including when autosave fires.
    def commit():
        try:
            parsed = parse(widget.text())
            error = validation_error(parsed, schema)
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            error = str(exc)
        set_error(widget, error)
        if not error:
            write(parsed)

    widget.editingFinished.connect(commit)
    widget.textEdited.connect(lambda _: set_error(widget, None))
    return widget


def schema_kind(schema):
    """The declared value type, treating ``[type, "null"]`` as a nullable ``type``."""
    kind = (schema or {}).get("type")
    if isinstance(kind, list):
        kinds = [k for k in kind if k != "null"]
        return kinds[0] if len(kinds) == 1 else None
    return kind


def numeric_editor(value, schema, write):
    kind = schema_kind(schema)
    # A spin box cannot be cleared, so nullable integers use the text field.
    nullable = isinstance(schema.get("type"), list) and "null" in schema["type"]
    if kind == "integer" and not nullable and isinstance(value, int) and not isinstance(value, bool):
        widget = integer_field(value, minimum=schema.get("minimum"), maximum=schema.get("maximum"))
        widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
        widget.valueChanged.connect(write)
        return widget

    def parse(text):
        if not text.strip():
            return None
        if kind == "integer":
            if not re.fullmatch(r"[+-]?[0-9]+", text.strip()):
                raise ValueError("Enter a whole number")
            return int(text)
        number = float(text)
        if not math.isfinite(number):
            raise ValueError("Enter a finite number")
        return number

    return draft_field(
        value,
        schema,
        write,
        parse,
        width=NUMBER_FIELD_WIDTH,
        help_text="Enter a number; scientific notation is supported.",
    )


class ChoiceListEditor(WrappingFlowWidget):
    """A list of strings as content-sized entries, each with an ×, and a trailing **+** that
    offers the remaining ``choices`` and a custom entry. Entries stay on one line while they fit
    and wrap with the shared ``FlowLayout`` beyond that. Every add or removal writes the whole list."""

    changed = pyqtSignal(list)

    def __init__(self, values, choices, write, *, noun="value", parent=None):
        super().__init__(parent, spacing=6, vertical_alignment=Qt.AlignmentFlag.AlignVCenter, fit_contents=True)
        self.setProperty("parameterChoiceList", True)
        self.values = [str(v) for v in values or []]
        self.choices = [str(c) for c in choices or []]
        self.noun = noun
        self._write = write
        self.flow = self.layout()
        self._render()

    def _render(self):
        while self.flow.count():
            widget = self.flow.takeAt(0).widget()
            if widget is not None:
                widget.hide()
                widget.deleteLater()
        for value in self.values:
            entry = QFrame()
            entry.setProperty("choiceEntry", True)
            entry.setProperty("choiceValue", value)
            row = QHBoxLayout(entry)
            row.setContentsMargins(8, 1, 1, 1)
            row.setSpacing(2)
            row.addWidget(QLabel(value))
            remove = _make_subtle_tool_button("close", f"Remove {value}", size=18, icon_size=10)
            remove.setProperty("choiceRemove", value)
            remove.clicked.connect(lambda _=False, v=value: self.remove(v))
            row.addWidget(remove)
            self.flow.addWidget(entry)
        remaining = [choice for choice in self.choices if choice not in self.values]
        add = _make_param_adder_button(f"Add a {self.noun}", remaining, self.add, parent_widget=self,
                                       custom_prompt=(f"Add {self.noun}", f"{self.noun.capitalize()}:"))
        add.setProperty("choiceAdd", True)
        self.flow.addWidget(add)
        self.updateGeometry()

    def add(self, value):
        if value and value not in self.values:
            self._commit([*self.values, value])

    def remove(self, value):
        self._commit([v for v in self.values if v != value])

    def _commit(self, values):
        self.values = values
        self._write(list(values))
        self.changed.emit(list(values))
        QTimer.singleShot(0, self._render)  # an entry's own × may be the sender


def _value_editor(value, schema, write, *, choices=None):
    schema = schema if isinstance(schema, dict) else {}
    if schema_kind(schema) == "array" and (choices is not None or (schema.get("items") or {}).get("enum")):
        options = choices if choices is not None else schema["items"]["enum"]
        return ChoiceListEditor(value if isinstance(value, list) else [], options, write)
    choices = schema.get("enum") if choices is None else choices
    if isinstance(choices, list) and choices:
        widget = ComboBox()
        widget.setMaximumWidth(TEXT_FIELD_WIDTH)
        widget.addItem("", None)
        for choice in choices:
            widget.addItem(str(choice), choice)
        index = next(
            (
                i
                for i in range(widget.count())
                if type(widget.itemData(i)) is type(value) and widget.itemData(i) == value
            ),
            -1,
        )
        if index < 0 and value is not None:
            widget.insertItem(1, str(value), value)
            index = 1
        widget.setCurrentIndex(max(0, index))
        widget.currentIndexChanged.connect(lambda i: write(widget.itemData(i)))
        return widget
    kind = schema_kind(schema)
    if kind == "string":
        widget = QLineEdit("" if value is None else str(value))
        widget.setMaximumWidth(TEXT_FIELD_WIDTH)
        widget.textChanged.connect(write)
        return widget
    if kind == "boolean" or isinstance(value, bool):
        widget = ToggleSlider(checked=bool(value))
        widget.toggled.connect(write)
        return widget
    if kind in ("integer", "number"):
        return numeric_editor(value, schema, write)
    if kind == "array" or isinstance(value, list):
        return draft_field(
            value,
            {"type": "array", **schema},
            write,
            yaml.safe_load,
            help_text="YAML list, for example [64, 64]",
            width=TEXT_FIELD_WIDTH + 80,
        )
    if isinstance(value, (int, float)):
        widget = ScientificDoubleSpinBox(value) if isinstance(value, float) else integer_field(value)
        widget.valueChanged.connect(write)
        return widget
    widget = QLineEdit("" if value is None else str(value))
    widget.textChanged.connect(write)
    return widget


def value_editor(value, schema, write, *, choices=None):
    """Preserve loaded values, expose validation, and only emit typed valid edits."""
    field = None

    def commit(edited):
        error = validation_error(edited, schema)
        set_error(field, error)
        if not error:
            write(edited)

    field = _value_editor(value, schema, commit, choices=choices)
    error = validation_error(value, schema)
    if error:
        set_error(field, error)
    return field


def number_field(value):
    """Preserve exact integers and round-trip floats without UI-only bounds."""
    return ScientificDoubleSpinBox(value) if isinstance(value, float) else integer_field(value)
