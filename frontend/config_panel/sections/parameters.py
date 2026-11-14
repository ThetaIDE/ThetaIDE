"""Configuration section composition using shared parameter controls."""

import json

from PyQt6.QtWidgets import (
    QLineEdit,
)

from ...theme import theme_color
from ...widgets import (
    label,
)
from ..catalogs import FIXED_BOX_KEYS
from ..controls import (
    ConfigBox,
    IndentedHierarchyBlock,
    ParameterForm,
    _make_param_adder_button,
    _parameter_row,
)


def _render_remaining_sections(self):
    """Surface structured top-level keys the fixed boxes do not cover.

    Without this an experiment's paradigm- or component-specific configuration
    is absent from the panel even though saving preserves it. Nested
    dictionaries are rendered recursively so their values remain editable
    regardless of how a component structures its parameters. Scalars are
    already rendered in the experiment parameters box.
    """
    leftovers = {
        k: v
        for k, v in self.raw_data.items()
        if k not in FIXED_BOX_KEYS
        and k not in ("data_module", "symbolic_state_loader", "evaluator", "plots", "plot_overrides")
    }
    if isinstance(self.raw_data.get("tuning"), dict):
        # The sweep has its own card; don't render it again as a generic section.
        leftovers.pop("tuning", None)
    leftovers.pop("variability", None)
    leftovers.pop("objective", None)
    leftovers.pop("monitors", None)  # settings live under their stream controls
    if isinstance(leftovers.get("telemetry"), dict):  # telemetry.metrics has its own box (Recorded Metrics)
        rest = {k: v for k, v in leftovers["telemetry"].items() if k not in ("metrics", "monitors")}
        if rest:
            leftovers["telemetry"] = rest
        else:
            leftovers.pop("telemetry")
    if not leftovers:
        return

    sections = {k: v for k, v in leftovers.items() if isinstance(v, (dict, list))}

    for name, value in sections.items():
        box = ConfigBox()
        box.set_header(
            name,
            add=self._dict_parameter_adder((name,), value)
            if isinstance(value, dict)
            else _make_param_adder_button(
                "Add item", [], lambda item, k=name: self._append_list_parameter(k, item), parent_widget=self
            ),
            remove=self._parameter_remove((name,)),
        )
        form = ParameterForm()
        form.setVerticalSpacing(8)
        if isinstance(value, list):
            form.add_parameter((name,), "items", self._list_field(value, lambda v, k=name: self._on_field_edited(k, v)))
        else:
            self._render_experiment_dict_contents(form, value, (name,))
        box.add_layout(form)
        self.boxes_layout.addWidget(box)


def _render_generic_boxes(self):
    """Render clean boxes for modular YAML configs (agent/*.yaml, env/*.yaml, model/*.yaml, etc.)."""
    self.paradigm_badge.hide()
    data = self.raw_data

    stem = self.current_path.stem if self.current_path else ""

    # --- Component Parameters Card ---
    box_props = ConfigBox()
    box_props.set_header("Parameters", add=self._dict_parameter_adder((), data))

    form = ParameterForm()

    desc = str(self.raw_data.get("description", "") or "").strip()
    desc_edit = QLineEdit(desc)
    desc_edit.setPlaceholderText("Add a brief description for this component...")
    desc_edit.setStyleSheet(f"""
        QLineEdit {{
            background: {theme_color("base")};
            border: 1px dashed {theme_color("border")};
            border-radius: 4px;
            color: {theme_color("text")};
            font-size: 12px;
            padding: 4px 8px;
        }}
        QLineEdit:focus {{
            border: 1px solid {theme_color("accent")};
            background-color: {theme_color("surface")};
        }}
    """)
    desc_edit.textChanged.connect(lambda val: self._on_field_edited("description", val.strip()))
    form.add_parameter(
        ("description",), "Description", _parameter_row(desc_edit, remove=self._parameter_remove(("description",)))
    )

    leaf_params = {}
    nested_sections = {}

    for k, v in data.items():
        if k in ("name", "algorithm", "architecture", "experiment_id", "id", "description"):
            continue
        if k in ("type", "env_id") and str(v).lower() == stem.lower():
            continue
        if isinstance(v, str) and v.lower() == stem.lower() and k in ("model", "agent"):
            continue
        if isinstance(v, dict):
            nested_sections[k] = v
        else:
            leaf_params[k] = v

    if leaf_params:
        self._render_generic_dict_contents(form, leaf_params, ())
    elif not desc:
        form.addRow(label("No parameters configured. Use + to add one.", "muted"))

    box_props.add_layout(form)

    self.boxes_layout.addWidget(box_props)

    # --- Nested Sections (Hierarchical Parameters) ---
    for section_name, section_val in nested_sections.items():
        box_sec = ConfigBox()
        box_sec.set_header(
            section_name,
            add=self._dict_parameter_adder((section_name,), section_val),
            remove=self._parameter_remove((section_name,)),
        )

        if isinstance(section_val, dict):
            form_sec = ParameterForm()
            form_sec.setVerticalSpacing(8)
            self._render_generic_dict_contents(form_sec, section_val, (section_name,), level=1)
            box_sec.add_layout(form_sec)
        self.boxes_layout.addWidget(box_sec)


def _render_generic_dict_contents(
    self, parent_layout, d: dict, path: tuple[str, ...], level: int = 1, defaults: dict | None = None
):
    from ..document import MISSING
    from ..rendering import ParameterNode, render_nodes

    form = parent_layout.form_layout if isinstance(parent_layout, IndentedHierarchyBlock) else parent_layout
    form.setProperty("parameterContext", json.dumps(path))

    def nodes(values, prefix, inherited):
        result = []
        for key, value in values.items():
            storage = prefix + (key,)
            remove = self._parameter_remove(storage)
            if isinstance(value, dict):
                child_defaults = inherited.get(key, {})
                child_defaults = child_defaults if isinstance(child_defaults, dict) else {}
                node = ParameterNode(
                    storage,
                    key,
                    children=nodes(value, storage, child_defaults),
                    add=self._dict_parameter_adder(storage, value, child_defaults),
                    remove=remove,
                )
            else:
                widget = self._parameter_editor(key, value, storage)
                binding = self._binding(storage)
                if binding.inherited or binding.explicit is MISSING:
                    widget.setProperty("inheritedDefault", True)
                    widget.setToolTip("Inherited default. Editing creates an explicit override.")
                node = ParameterNode(storage, key, row=_parameter_row(widget, remove=remove))
            result.append(node)
        return result

    render_nodes(form, nodes(d, path, defaults or {}), level=level)
