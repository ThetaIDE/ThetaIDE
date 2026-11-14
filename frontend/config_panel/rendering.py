"""One row/group renderer; domain adapters describe nodes and capabilities."""

import json
from dataclasses import dataclass

from .controls import IndentedHierarchyBlock, ParameterForm, _parameter_header


@dataclass
class ParameterNode:
    identity: tuple[str, ...]
    title: str
    row: object = None
    children: list | None = None
    add: object = None
    remove: object = None
    extra_children: object = None


def render_nodes(form, nodes, *, level=1):
    if nodes and not form.property("parameterContext"):
        form.setProperty("parameterContext", json.dumps(nodes[0].identity[:-1]))
    for node in nodes:
        row = node.row
        if row is None:
            row = _parameter_header(node.title, add=node.add, remove=node.remove, heading=False)
            caption = None
            handle = row.caption
        else:
            caption, handle = node.title, None
        children = node.extra_children
        if node.children is not None:
            children = IndentedHierarchyBlock(level=level)
            children.setProperty("parameterPath", ".".join(node.identity))
            render_nodes(children.form_layout, node.children, level=level + 1)
            if node.extra_children is not None:
                children.addRow(node.extra_children)
        group = form.add_parameter(node.identity, caption, row, children=children)
        if handle is not None:
            group.caption = handle


class ParameterRows(list):
    """Compatibility iterable with explicit membership for specialized hosts."""

    def __init__(self, nodes, *, level=1):
        form = ParameterForm()
        render_nodes(form, nodes, level=level)
        self.groups = form._declared_groups or []
        self.context = form.property("parameterContext")
        super().__init__((caption or "", field) for group in self.groups for caption, field in group.parts)
        while form.rowCount():
            form.takeRow(0)

    def mount(self, form):
        if self.context:
            form.setProperty("parameterContext", self.context)
        for group in self.groups:
            caption, row = group.parts[0]
            child = group.parts[1][1] if len(group.parts) > 1 else None
            mounted = form.add_parameter((), caption, row, children=child)
            mounted.key, mounted.caption = group.key, group.caption
