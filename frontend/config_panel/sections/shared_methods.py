"""Shared-method domain bindings, presented by the ordinary tree renderer."""

import json

from ...config_model import COMPONENT_ID_KEYS
from ..controls import ComboBox, _make_param_adder_button, _make_subtle_tool_button, _parameter_row
from ..rendering import ParameterNode, render_nodes


def build(host, form, values, tree, paradigm, explicit):
    form.setProperty("parameterContext", json.dumps(("methods", "params")))

    def defaults_for(path):
        if not tree or not path or path[0] not in ("agent", "model"):
            return {}
        if len(path) == 1:
            return {name: {} for name in (tree.agent_names if path[0] == "agent" else tree.model_names)}
        defaults = tree.agent_hyperparameters(path[1]) if path[0] == "agent" else tree.model_parameters(path[1])
        for part in path[2:]:
            defaults = defaults.get(part, {}) if isinstance(defaults, dict) else {}
        return defaults if isinstance(defaults, dict) else {}

    def schema_for(path):
        if tree and len(path) >= 2 and path[0] in ("agent", "model"):
            return (tree.agent_parameter_schema if path[0] == "agent" else tree.model_parameter_schema)(
                path[1], path[2:]
            )
        return {}

    def remove_for(path, required):
        if not explicit or required:
            return None
        button = _make_subtle_tool_button("close", f"Remove parameter '{'.'.join(path)}'", size=20, icon_size=12)
        button.clicked.connect(lambda _: host._remove_universal_param(".".join(path)))
        return button

    def nodes(values, parent=()):
        result = []
        required_keys = schema_for(parent).get("required", [])
        for key, value in values.items():
            if not parent and key in ("tune", "search_space"):
                continue
            path = (*parent, key)
            identity = ("methods", "params", *path)
            title = {"agent": "Algorithm", "model": "Model"}.get(key, key) if not parent else key
            remove = remove_for(path, key in required_keys)
            if (
                not parent
                and key in ("agent", "model")
                and isinstance(value, dict)
                and not any(k in value for k in COMPONENT_ID_KEYS)
            ):
                choices = (
                    [a.name for a in tree.agents_for(paradigm)]
                    if tree and key == "agent" and paradigm in tree.paradigms
                    else (list(tree.model_names) if tree and key == "model" else [])
                )
                for selected, params in list(value.items()) or [("", {})]:
                    picker = ComboBox()
                    picker.setObjectName(f"shared_{key}_selector")
                    picker.addItem("Select algorithm..." if key == "agent" else "Select model...", "")
                    for name in choices:
                        picker.addItem(name, name)
                    if selected and selected not in choices:
                        picker.addItem(f"{selected} (unavailable)", selected)
                        picker.model().item(picker.count() - 1).setEnabled(False)
                    picker.setCurrentIndex(max(0, picker.findData(selected)))
                    picker.currentIndexChanged.connect(
                        lambda i, c=picker, k=key, old=selected: host._select_shared_component(k, old, c.itemData(i))
                    )
                    nested_path = (*path, selected) if selected else path
                    defaults = defaults_for(nested_path)
                    params = params if isinstance(params, dict) else {}
                    candidates = [k for k in defaults if k not in params]
                    add = _make_param_adder_button(
                        f"Add {selected} parameter",
                        candidates,
                        lambda child, p=nested_path, d=defaults: host._add_universal_param(
                            ".".join((*p, child)), d.get(child)
                        ),
                        parent_widget=host,
                    )
                    row = _parameter_row(picker, add=add, remove=remove_for(nested_path, False))
                    result.append(
                        ParameterNode(
                            ("methods", "params", *nested_path),
                            title,
                            row=row,
                            children=nodes(params, nested_path) if params else None,
                        )
                    )
            elif isinstance(value, dict):
                defaults = defaults_for(path)
                candidates = [k for k in defaults if k not in value]
                add = _make_param_adder_button(
                    f"Add {title} parameter",
                    candidates,
                    lambda child, p=path, d=defaults: host._add_universal_param(".".join((*p, child)), d.get(child)),
                    parent_widget=host,
                )
                result.append(ParameterNode(identity, title, children=nodes(value, path), add=add, remove=remove))
            else:
                widget = host._schema_widget(
                    value,
                    schema_for(path),
                    lambda edited, p=path: host._on_universal_param_edited(".".join(p), edited),
                    action_path=identity,
                )
                widget.setProperty("parameterPath", ".".join(identity))
                row = _parameter_row(widget, remove=remove)
                if explicit:
                    target = f"agent.{key}" if not parent else ".".join(path)
                    if not parent or parent[0] in ("agent", "model") and key not in COMPONENT_ID_KEYS:
                        host._add_sweep_toggle(row, "params", target, value)
                result.append(
                    ParameterNode(identity, title, row=row, extra_children=getattr(row, "_sweep_block", None))
                )
        return result

    render_nodes(form, nodes(values))
