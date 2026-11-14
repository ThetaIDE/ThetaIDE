"""Explicit parameter regions, responsive placement and live subtree movement."""

from dataclasses import dataclass

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtWidgets import QFormLayout, QGridLayout, QWidget

from .controls import NUMBER_FIELD_WIDTH, ParameterForm


def row_slots(item):
    """A child form occupies all of its rows, rather than one atomic slot."""
    form = getattr(item, "form_layout", None)
    if form is None:
        return 1
    count = 0
    for index in range(form.rowCount()):
        child = form.itemAt(index, QFormLayout.ItemRole.SpanningRole)
        count += row_slots(child.widget()) if child and child.widget() else 1
    return max(1, count)


@dataclass(frozen=True)
class RowSlot:
    column: int
    row: int
    span: int = 1


class ParameterRegion(QGridLayout):
    """Two columns when measured row minima fit, one column otherwise.

    Display reordering assigns mutable slots in logical order. Slots stay stable
    during resize; only presentation collapses. Every group remains one drag unit.
    """

    def __init__(self, groups, *, identity, align_rows=False):
        super().__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setHorizontalSpacing(14)
        self.setVerticalSpacing(8)
        self.setProperty("equalParameterColumns", True)
        self.setProperty("parameterRegion", identity)
        self._parameter_rows = RegionRows(self, groups, align_rows=align_rows)
        self._narrow = False
        self._placing = False
        self._parameter_rows.apply()

    def collapse_width(self):
        widths = [
            max(NUMBER_FIELD_WIDTH + 120 + 10, part.minimumSize().width())
            for group in self._parameter_rows.groups
            for part in group.parts
        ]
        return 2 * max(widths, default=300) + self.horizontalSpacing()

    def minimumSize(self):
        size = super().minimumSize()
        # Let the containing scroll area actually give us a narrow viewport.
        return QSize(min(size.width(), max(280, self.collapse_width() // 2)), size.height())

    def setGeometry(self, rect):
        narrow = rect.width() < self.collapse_width()
        if narrow != self._narrow and not self._placing:
            self._narrow = narrow
            self._placing = True
            try:
                self._parameter_rows.apply()
            finally:
                self._placing = False
        super().setGeometry(rect)


class RegionRows:
    def __init__(self, layout, groups, *, align_rows=False):
        self.layout, self.groups = layout, groups
        self.align_rows = align_rows
        self.slots = {}

    def apply(self):
        while self.layout.count():
            self.layout.takeAt(0)
        for column in range(self.layout.columnCount()):
            self.layout.setColumnStretch(column, 0)
            self.layout.setColumnMinimumWidth(column, 0)
        rows = [0, 0]
        narrow_row = 0
        self.slots = {}
        for index, group in enumerate(self.groups):
            # Explicit order, independent of sizeHint/expansion/resize.
            column = 0 if rows[0] <= rows[1] else 1
            row = rows[column]
            for part in group.parts:
                span = row_slots(part)
                self.slots[id(part)] = RowSlot(column, row, span)
                actual_row, actual_column = (narrow_row, 0) if self.layout._narrow else (row, column)
                actual_span = 1 if self.layout._narrow else span
                if isinstance(part, QWidget):
                    self.layout.addWidget(part, actual_row, actual_column, actual_span, 1, Qt.AlignmentFlag.AlignTop)
                else:
                    self.layout.addLayout(part, actual_row, actual_column, actual_span, 1)
                row += span
                narrow_row += 1
            rows[column] = row
        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 0 if self.layout._narrow else 1)
        self.layout.invalidate()


def two_column_form(form, *, identity, align_rows=False):
    """Compatibility boundary for declared form rows during section migration."""
    from ..parameter_reorder import form_groups

    groups = form_groups(form)
    while form.rowCount():
        form.takeRow(0)
    for group in groups:
        parts = []
        for caption, field in group.parts:
            if caption is None:
                nested = getattr(field, "form_layout", None)
                if isinstance(nested, ParameterForm):
                    nested.set_compact(True)
                parts.append(field)
            else:
                row = ParameterForm(compact=True)
                row._parameter_column_row = True
                row.addRow(caption, field)
                parts.append(row)
        group.parts = parts
    return ParameterRegion(groups, identity=identity, align_rows=align_rows)


def _two_column_parameter_layout(form, *, align_rows=False):
    """Compatibility entry point; new compositions declare an explicit identity."""
    return two_column_form(form, identity=form.property("parameterContext") or "context", align_rows=align_rows)
