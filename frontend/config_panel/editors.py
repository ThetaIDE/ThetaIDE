"""Schema/type editor factory and draft validation, shared by section editors."""

import math
import re

import yaml
from PyQt6.QtWidgets import QLineEdit

from ..widgets import ScientificDoubleSpinBox, ToggleSlider, integer_field
from .controls import NUMBER_FIELD_WIDTH, TEXT_FIELD_WIDTH, ComboBox
from .document import validation_error


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


def numeric_editor(value, schema, write):
    if schema.get("type") == "integer" and isinstance(value, int) and not isinstance(value, bool):
        widget = integer_field(value, minimum=schema.get("minimum"), maximum=schema.get("maximum"))
        widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
        widget.valueChanged.connect(write)
        return widget

    def parse(text):
        if not text.strip():
            return None
        if schema.get("type") == "integer":
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


def _value_editor(value, schema, write, *, choices=None):
    schema = schema if isinstance(schema, dict) else {}
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
    kind = schema.get("type")
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
