"""Model schema adapter; ordinary rows and nesting use the shared tree renderer."""

from ..catalogs import config_tree
from ..controls import _make_param_adder_button, _make_subtle_tool_button, _parameter_row
from ..rendering import ParameterNode, ParameterRows


def _render_method_model_parameters(
    self,
    method_name,
    model_name,
    values,
    defaults,
    path=(),
    level=1,
    show_defaults=False,
    schema_model_name=None,
    schema_path=(),
):
    tree = self.tree

    def nodes(values, defaults, storage, owner, schema_path):
        schema = tree.model_parameter_schema(owner, schema_path) if tree else {}
        properties = schema.get("properties", {})
        result = []
        for key in dict.fromkeys([*values, *defaults, *properties]):
            present = key in values
            required = key in schema.get("required", [])
            if not present and not show_defaults and not required:
                continue
            path = storage + (key,)
            schema_key = schema_path + (key,)
            child_schema = tree.model_parameter_schema(owner, schema_key) if tree else {}
            value = values.get(key, defaults.get(key, child_schema.get("default")))
            identity = ("methods", method_name, "model", *path)
            remove = None
            if present and not required:
                remove = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                remove.setProperty("parameterPath", ".".join(identity))
                remove.clicked.connect(
                    lambda _, p=path, sm=owner, sp=schema_key: self._remove_method_model_parameter(
                        method_name, model_name, p, sm, sp
                    )
                )
            if child_schema.get("x-theta-domain") == "model":
                from ...config_model import component_selection

                selected, child_values = component_selection(value)
                selected = tree.models[selected].name if tree and selected in tree.models else selected
                selected_schema = {**child_schema, "type": "string"}
                picker = self._schema_widget(
                    selected,
                    selected_schema,
                    lambda name, p=path: self._on_method_child_model_changed(method_name, model_name, p, name or ""),
                    action_path=identity,
                )
                child_defaults = tree.model_parameters(selected) if tree and selected else {}
                children = nodes(child_values, child_defaults, path, selected, ()) if selected else []
                candidates = [
                    k
                    for k in (tree.model_parameter_keys(selected) if tree and selected else [])
                    if k not in child_values and k not in ("name", "architecture", "type")
                ]
                add = (
                    _make_param_adder_button(
                        f"Add {selected} parameter",
                        candidates,
                        lambda child, p=path, sm=selected: self._add_method_model_parameter(
                            method_name, model_name, p + (child,), sm, (child,)
                        ),
                        parent_widget=self,
                        allow_custom=False,
                    )
                    if candidates
                    else None
                )
                row = _parameter_row(picker, add=add, remove=remove)
                result.append(ParameterNode(identity, key, row=row, children=children or None))
            elif isinstance(value, dict) or child_schema.get("type") == "object":
                child_values = value if isinstance(value, dict) else {}
                child_defaults = defaults.get(key, {})
                child_defaults = child_defaults if isinstance(child_defaults, dict) else {}
                candidates = [
                    k
                    for k in dict.fromkeys([*child_defaults, *child_schema.get("properties", {})])
                    if k not in child_values
                ]
                add = (
                    _make_param_adder_button(
                        f"Add {key} parameter",
                        candidates,
                        lambda child, p=path, sm=owner, sp=schema_key: self._add_method_model_parameter(
                            method_name, model_name, p + (child,), sm, sp + (child,)
                        ),
                        parent_widget=self,
                        allow_custom=False,
                    )
                    if candidates
                    else None
                )
                result.append(
                    ParameterNode(
                        identity,
                        key,
                        children=nodes(child_values, child_defaults, path, owner, schema_key),
                        add=add,
                        remove=remove,
                    )
                )
            else:
                widget = self._schema_widget(
                    value,
                    child_schema,
                    lambda value, p=path, sm=owner, sp=schema_key: self._on_method_model_parameter_edited(
                        method_name, model_name, p, value, sm, sp
                    ),
                    action_path=identity,
                )
                widget.setProperty("parameterPath", ".".join(identity))
                row = _parameter_row(widget, remove=remove)
                if present:
                    self._add_sweep_toggle(row, method_name, "model." + ".".join(path), value)
                result.append(ParameterNode(identity, key, row=row, extra_children=getattr(row, "_sweep_block", None)))
        return result

    return ParameterRows(nodes(values, defaults, path, schema_model_name or model_name, schema_path), level=level)
