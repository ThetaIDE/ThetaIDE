"""The figure's Parameters box: one ordinary ConfigBox built from the plot component's
parameter_schema with the shared parameter controls (docs/reference/CONFIG_PARAMETER_FORMS.md).

Unset parameters show the schema default as an inherited value; editing one writes an explicit
value. **+** beside the title adds parameters without a default; × on row hover removes an
explicit value, so the default applies again. Required parameters have no ×. Every valid edit
is saved at once: a live parameter emits ``live_changed`` and any other parameter emits
``regenerate_needed``. Invalid drafts never reach the figure file.
"""

from __future__ import annotations

import copy

from PyQt6 import sip
from PyQt6.QtCore import QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLayout, QWidget

from ..config_panel.controls import (ConfigBox, ParameterForm, _make_param_adder_button,
                                     _make_subtle_tool_button, _parameter_row)
from ..config_panel.editors import schema_kind, set_error, value_editor
from ..config_panel.rendering import ParameterNode, render_nodes
from .model import FigureError

INHERITED_TIP = "Default. Editing creates an explicit value."


def initial_value(schema: dict):
    """The value a parameter without a default starts from when it is added."""
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return copy.deepcopy(schema["enum"][0])
    kind = schema_kind(schema)
    if kind in ("integer", "number"):
        value = schema.get("minimum", 0)
        return int(value) if kind == "integer" else value
    return {"string": "", "boolean": False, "array": [], "object": {}}.get(kind)


def _clear(layout: QLayout):
    while layout.count():
        item = layout.takeAt(0)
        widget, child = item.widget(), item.layout()
        if widget is not None:
            widget.hide()
            widget.setParent(None)  # out of the box at once; deleted later
            widget.deleteLater()
        elif child is not None:
            _clear(child)
            child.deleteLater()


class ParametersBox(ConfigBox):
    """Edits one FigureDoc. ``metric_choices`` are offered for ``x-theta-metric`` parameters."""

    live_changed = pyqtSignal(str, object)
    regenerate_needed = pyqtSignal()

    def __init__(self, doc, *, metric_choices=(), parent=None):
        super().__init__(parent=parent, section="figure")
        self.doc = doc
        self.metric_choices = list(metric_choices)
        self.editors: dict[str, QWidget] = {}
        self._render_pending = False
        QApplication.instance().focusChanged.connect(self._focus_changed)
        self.render()

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self):
        self._render_pending = False
        _clear(self.layout)
        self.editors = {}
        properties = self.doc.schema().get("properties") or {}
        defaults = self.doc.defaults()
        required = set(self.doc.required())
        shown = [key for key in properties if key in self.doc.params or key in defaults or key in required]
        addable = [key for key in properties if key not in shown]
        add = _make_param_adder_button("Add parameter", addable, self.add_param, parent_widget=self,
                                       allow_custom=False)
        add.setProperty("parameterPath", "")
        self.set_header("Parameters", add=add)

        values = self.doc.values()
        nodes = []
        for key in shown:
            schema = properties[key]
            value = values.get(key, initial_value(schema))
            editor = self._editor(key, value, schema)
            remove = None
            if key in self.doc.params and key not in required:
                remove = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                remove.setProperty("parameterPath", key)
                remove.clicked.connect(lambda _=False, k=key: self.remove_param(k))
            nodes.append(ParameterNode((key,), schema.get("title") or key, row=_parameter_row(editor, remove=remove)))
        form = ParameterForm()
        render_nodes(form, nodes)
        self.add_layout(form)

    def _editor(self, key, value, schema):
        holder = []
        choices = self.metric_choices if schema.get("x-theta-metric") and self.metric_choices else None
        editor = value_editor(value, schema, lambda edited: self.set_param(key, edited, holder[0]), choices=choices)
        holder.append(editor)
        editor.setProperty("parameterPath", key)
        editor.setProperty("figureParameter", key)
        help_text = schema.get("description") or ""
        if key not in self.doc.params:
            editor.setProperty("inheritedDefault", True)
            help_text = f"{help_text}\n{INHERITED_TIP}".strip()
        if help_text:
            editor.setProperty("parameterHelp", help_text)
            if not editor.accessibleDescription():
                editor.setToolTip(help_text)
        self.editors[key] = editor
        return editor

    def _focus_changed(self, old, new):
        """A value that became explicit gets its × once its editor loses focus, not while typing."""
        if not self._render_pending or sip.isdeleted(self):
            return
        if old is not None and not sip.isdeleted(old) and self.isAncestorOf(old):
            restore = new.property("figureParameter") if new is not None and self.isAncestorOf(new) else None
            self.render()
            if restore in self.editors:
                self.editors[restore].setFocus()

    def _structure_changed(self, editor=None):
        focused = QApplication.focusWidget()
        if editor is not None and focused is not None and (focused is editor or editor.isAncestorOf(focused)):
            self._render_pending = True
        else:
            QTimer.singleShot(0, self.render)

    # ── Edits ────────────────────────────────────────────────────────────────

    def _changed(self, key):
        if key in self.doc.live_params():
            values = self.doc.values()
            if key in values:
                self.live_changed.emit(key, copy.deepcopy(values[key]))
                return
        self.regenerate_needed.emit()

    def set_param(self, key, value, editor=None) -> bool:
        was_explicit = key in self.doc.params
        try:
            changed = self.doc.set_param(key, value)
        except FigureError as exc:
            if editor is not None:
                set_error(editor, str(exc))
            return False
        if not changed:
            return False
        self.doc.save()
        self._changed(key)
        if not was_explicit:
            self._structure_changed(editor)
        return True

    def remove_param(self, key):
        try:
            changed = self.doc.unset_param(key)
        except FigureError:
            return
        if changed:
            self.doc.save()
            self._changed(key)
            QTimer.singleShot(0, self.render)

    def add_param(self, key):
        schema = (self.doc.schema().get("properties") or {}).get(key, {})
        self.set_param(key, initial_value(schema))  # a new explicit value re-renders the box
