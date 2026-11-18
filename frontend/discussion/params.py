"""The figure's parameter box: one ordinary ConfigBox built from the plot component's
parameter_schema with the shared parameter controls (docs/reference/CONFIG_PARAMETER_FORMS.md).

The box is titled with the figure's name; clicking the title renames the figure. Unset
parameters show the schema default as an inherited value; editing one writes an explicit
value. **+** beside the title adds parameters without a default; × on row hover removes an
explicit value, so the default applies again. Required parameters have no ×. A parameter whose
default is null or "" shows **Auto** when empty, and clearing it returns it to the default.

Schema annotations:
- ``x-theta-metric``: the target's logged metrics are offered (a string picks one, an array several).
- ``x-theta-method``: the target's logged methods are offered, the same way.
- ``x-theta-group``: properties sharing a group render in one titled, collapsible block, after
  the ungrouped ones, in order of first appearance.

Every valid edit is saved at once: a live parameter emits ``live_changed`` and any other
parameter emits ``regenerate_needed``. Invalid drafts never reach the figure file.
"""

from __future__ import annotations

import copy

from PyQt6 import sip
from PyQt6.QtCore import QEvent, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLayout, QLineEdit, QWidget

from ..config_panel.controls import (ConfigBox, IndentedHierarchyBlock, ParameterForm,
                                     _collapsible_parameter_label, _make_param_adder_button,
                                     _make_subtle_tool_button, _parameter_row)
from ..config_panel.editors import schema_kind, set_error, value_editor
from ..config_panel.rendering import ParameterNode, render_nodes
from .model import FigureError

INHERITED_TIP = "Default. Editing creates an explicit value."
AUTO = "Auto"
RENAME_TIP = "Click to rename the figure"


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
    """Edits one FigureDoc.

    ``metric_choices`` and ``method_choices`` are offered for ``x-theta-metric`` and
    ``x-theta-method`` parameters. ``collapsed`` holds the names of collapsed groups; it is
    updated in place and ``collapsed_changed`` reports each change.
    """

    live_changed = pyqtSignal(str, object)
    regenerate_needed = pyqtSignal()
    rename_requested = pyqtSignal(str)
    collapsed_changed = pyqtSignal(object)

    def __init__(self, doc, *, metric_choices=(), method_choices=(), collapsed=None, parent=None):
        super().__init__(parent=parent, section="figure")
        self.doc = doc
        self.metric_choices = list(metric_choices)
        self.method_choices = list(method_choices)
        self.collapsed = set(collapsed or ())
        self.editors: dict[str, QWidget] = {}
        self.groups: dict[str, IndentedHierarchyBlock] = {}
        self.name_editor = None
        self._render_pending = False
        QApplication.instance().focusChanged.connect(self._focus_changed)
        self.render()

    # ── Rendering ────────────────────────────────────────────────────────────

    def render(self):
        self._render_pending = False
        self.name_editor = None
        _clear(self.layout)
        self.editors, self.groups = {}, {}
        properties = self.doc.schema().get("properties") or {}
        defaults = self.doc.defaults()
        required = set(self.doc.required())
        shown = [key for key in properties if key in self.doc.params or key in defaults or key in required]
        addable = [key for key in properties if key not in shown]
        add = _make_param_adder_button("Add parameter", addable, self.add_param, parent_widget=self,
                                       allow_custom=False)
        add.setProperty("parameterPath", "")
        self.header = self.set_header(self.doc.name, add=add)
        self.title_label.setToolTip(RENAME_TIP)
        self.title_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self.title_label.installEventFilter(self)

        values = self.doc.values()
        ungrouped, grouped = [], {}
        for key in shown:
            schema = properties[key]
            value = values.get(key, initial_value(schema))
            editor = self._editor(key, value, schema)
            remove = None
            if key in self.doc.params and key not in required:
                remove = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                remove.setProperty("parameterPath", key)
                remove.clicked.connect(lambda _=False, k=key: self.remove_param(k))
            node = ParameterNode((key,), schema.get("title") or key, row=_parameter_row(editor, remove=remove))
            group = schema.get("x-theta-group")
            (grouped.setdefault(str(group), []) if group else ungrouped).append(node)
        form = ParameterForm()
        render_nodes(form, ungrouped)
        for name, nodes in grouped.items():
            block = IndentedHierarchyBlock(level=1)
            block.setProperty("parameterGroup", name)
            render_nodes(block.form_layout, nodes, level=2)
            caption = _collapsible_parameter_label(name, block, expanded=name not in self.collapsed,
                                                   on_toggled=lambda on, n=name: self._group_toggled(n, on))
            caption.setProperty("parameterGroupCaption", name)
            spacer = QWidget()  # the caption row has no editor
            spacer.setStyleSheet("background: transparent;")
            form.add_parameter(("group", name), caption, spacer, children=block)
            self.groups[name] = block
        self.add_layout(form)

    def _choices(self, schema):
        if schema.get("x-theta-metric"):
            return self.metric_choices
        if schema.get("x-theta-method"):
            return self.method_choices
        return None

    def _editor(self, key, value, schema):
        holder = []
        choices = self._choices(schema)
        if choices is not None and schema_kind(schema) != "array" and not choices:
            choices = None  # nothing logged yet: type a value
        editor = value_editor(value, schema, lambda edited: self.set_param(key, edited, holder[0]), choices=choices)
        holder.append(editor)
        editor.setProperty("parameterPath", key)
        editor.setProperty("figureParameter", key)
        if isinstance(editor, QLineEdit) and schema.get("default", 0) in ("", None):
            editor.setPlaceholderText(AUTO)
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

    def _group_toggled(self, name, expanded):
        (self.collapsed.discard if expanded else self.collapsed.add)(name)
        self.collapsed_changed.emit(set(self.collapsed))

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

    # ── Renaming ─────────────────────────────────────────────────────────────

    def eventFilter(self, watched, event):
        if (watched is self.title_label and event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton):
            self.start_rename()
            return True
        if watched is self.name_editor and event.type() == QEvent.Type.KeyPress \
                and event.key() == Qt.Key.Key_Escape:
            self.cancel_rename()
            return True
        return super().eventFilter(watched, event)

    def start_rename(self):
        """Edit the figure's name in place of the title. Enter or leaving the field commits it;
        Escape cancels."""
        if self.name_editor is not None:
            self.name_editor.setFocus()
            return
        editor = QLineEdit(self.doc.name)
        editor.setObjectName("titleEditor")
        editor.setAccessibleName("Figure name")
        editor.setStyleSheet("font-size: 20px; font-weight: 600; padding: 2px 8px; border-radius: 4px;")
        editor.setMinimumWidth(200)
        editor.setMaximumWidth(400)
        editor.installEventFilter(self)
        editor.editingFinished.connect(self._commit_rename)  # Enter or leaving the field
        self.header.insertWidget(self.header.indexOf(self.title_label), editor)
        self.title_label.hide()
        self.name_editor = editor
        editor.selectAll()
        editor.setFocus()

    def _commit_rename(self):
        editor = self.name_editor
        if editor is None:
            return
        name = editor.text().strip()
        if not name or name == self.doc.name:
            self.cancel_rename()
            return
        self.rename_requested.emit(name)

    def rename_failed(self, message):
        """Keep the editor open with the reason, as invalid parameter drafts do."""
        if self.name_editor is not None:
            set_error(self.name_editor, message)

    def cancel_rename(self):
        editor, self.name_editor = self.name_editor, None
        if editor is None:
            return
        editor.blockSignals(True)
        editor.hide()
        editor.deleteLater()
        self.title_label.show()

    # ── Edits ────────────────────────────────────────────────────────────────

    def _changed(self, key):
        if key in self.doc.live_params():
            values = self.doc.values()
            if key in values:
                self.live_changed.emit(key, copy.deepcopy(values[key]))
                return
        self.regenerate_needed.emit()

    def set_param(self, key, value, editor=None) -> bool:
        schema = (self.doc.schema().get("properties") or {}).get(key, {})
        if value in (None, "") and schema.get("default", 0) == value and key not in self.doc.required():
            return self.remove_param(key, rerender=False)  # cleared: back to Auto
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

    def remove_param(self, key, *, rerender=True) -> bool:
        try:
            changed = self.doc.unset_param(key)
        except FigureError:
            return False
        if changed:
            self.doc.save()
            self._changed(key)
            if rerender:
                QTimer.singleShot(0, self.render)
            else:
                self._structure_changed(self.editors.get(key))
        return changed

    def add_param(self, key):
        schema = (self.doc.schema().get("properties") or {}).get(key, {})
        self.set_param(key, initial_value(schema))  # a new explicit value re-renders the box
