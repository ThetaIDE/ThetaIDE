"""Configuration section composition using shared parameter controls."""

import copy

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from src.app.services import PROJECT_ROOT

from ...widgets import (
    label,
)
from ..catalogs import config_tree
from ..controls import (
    ComboBox,
    ConfigBox,
    IndentedHierarchyBlock,
    ParameterForm,
    ParameterRow,
    _make_param_adder_button,
    _make_subtle_tool_button,
    _parameter_header,
    _parameter_row,
    _reveal_remove_on_row_hover,
)


def _render_sweep_objective(self, form, data, resolved):
    """The sweep's objective selector owns code actions and parameter children."""
    from src.app.objectives.api import available_objectives, load_objective

    spec = self._effective_objective_spec(data, resolved)
    legacy = not isinstance(spec, dict)
    tuning = data.get("tuning") or resolved.get("tuning") or {}
    metric = str(
        tuning.get("metric")
        or ("eval/reward" if (data.get("paradigm") or resolved.get("paradigm")) == "online_rl" else "val/loss")
    )
    spec = spec if not legacy else {"component": "final_metric", "params": {"metric": metric}}
    tree = self.tree
    workspace = tree.config_root.parent.parent if tree else PROJECT_ROOT
    params = spec.get("params") or {}
    selected = "Reported metric" if legacy else str(spec.get("component") or "final_metric")
    template = "final_metric" if legacy else selected

    def select(name):
        if not name:
            QTimer.singleShot(0, self._render_boxes)
            return
        if name == "Reported metric":
            self.document.remove(("objective",))
            if isinstance(resolved.get("objective"), dict):
                self.document.set(("objective",), None)
        else:
            defaults = {"metric": metric} if name == "final_metric" else {}
            self.document.set(("objective",), {"component": name, "params": defaults})
        self._mark_dirty()
        self._render_boxes()

    schema = {"type": "string", "enum": ["Reported metric", *sorted(set(available_objectives(workspace)) | {template})]}
    try:
        component = load_objective(template, workspace)
        source = PROJECT_ROOT / "src/app/pipeline/objective.py" if legacy else component.source
        schema["x-theta-actions"] = [{"type": "open-in-editor", "path": str(source), "label": "Open objective code"}]
    except (ValueError, OSError) as exc:
        component = None
        form.addRow(label(str(exc), "muted"))
    selector = self._schema_widget(selected, schema, select, action_path=("objective", "component"))
    selector.setObjectName("objectiveComponent")
    choice = selector if isinstance(selector, QComboBox) else selector.findChild(QComboBox)
    if choice is not None:
        choice.setObjectName("sweepObjective")
    add_params = self._dict_parameter_adder(("objective", "params"), params) if not legacy else None
    details = IndentedHierarchyBlock()
    details.setProperty("parameterPath", "objective.params")
    form.add_parameter(("objective",), "Objective", _parameter_row(selector, add=add_params), children=details)
    if legacy:
        field = QLineEdit(metric)
        field.setObjectName("sweepMetric")
        field.textChanged.connect(lambda value: self._on_nested_edited("tuning", "metric", value.strip()))
        details.form_layout.add_parameter(("tuning", "metric"), "Metric", _parameter_row(field))
    if component and not legacy:
        # Render defaults without converting them into explicit overrides.
        def merge(defaults, overrides):
            values = copy.deepcopy(defaults)
            for key, value in overrides.items():
                values[key] = (
                    merge(values[key], value)
                    if isinstance(values.get(key), dict) and isinstance(value, dict)
                    else value
                )
            return values

        values = merge(component.metadata.get("defaults", {}), params)
        if values:
            self._render_generic_dict_contents(details, values, ("objective", "params"))


def _render_loader_binding(self, form, caption, path, context, method=None):
    from src.app.core.data_loaders import input_requirements, loader_metadata, loader_selection

    slot = path[-1]
    selection, source = loader_selection(context, slot, method, paths=self._loader_paths())
    current = self.raw_data
    for key in path:
        current = current.get(key) if isinstance(current, dict) else None
    explicit_name = self._component_name(current)
    choices = self._component_choices(slot)
    needs = [
        r["contract"] for r in input_requirements(context, method, paths=self._loader_paths()) if r["slot"] == slot
    ]
    combo = ComboBox()
    combo.setProperty("parameterPath", ".".join(path))
    inherited, _ = loader_selection(
        context if method else self._get_resolved_defaults(), slot, paths=self._loader_paths()
    )
    inherited_name = inherited["name"] if inherited else "None"
    combo.addItem(f"Inherit ({inherited_name})", "")
    for name, profile in sorted(choices.items()):
        metadata = loader_metadata(
            {"name": profile.get("data_module") or profile.get("name") or name}, paths=self._loader_paths()
        )
        if (
            needs
            and not all(contract in metadata.get("output_contracts", []) for contract in needs)
            and name != explicit_name
        ):
            continue
        combo.addItem(name, name)
    if explicit_name and combo.findData(explicit_name) < 0:
        combo.addItem(explicit_name, explicit_name)
    combo.setCurrentIndex(max(0, combo.findData(explicit_name)))
    combo.currentIndexChanged.connect(lambda index, c=combo, p=path: self._set_loader_binding(p, c.itemData(index)))
    row = ParameterRow()
    row.addWidget(combo)
    params = current.get("params", {}) if isinstance(current, dict) else {}
    defaults = selection["params"] if selection else {}
    if explicit_name:
        row.addWidget(self._dict_parameter_adder(path + ("params",), params, defaults))
    remove = self._parameter_remove(path) if current is not None and (not needs or inherited) else None
    if remove:
        row.addWidget(remove)
        _reveal_remove_on_row_hover(row, remove)
    effective_name = selection["name"] if selection else "Missing"
    row.addWidget(label(f"{effective_name} · {source}", "muted"))
    row.addStretch()
    form.add_parameter(path, caption, row)
    self.field_widgets[".".join(path)] = combo
    if params:
        block = IndentedHierarchyBlock()
        block.setProperty("parameterPath", ".".join(path + ("params",)))
        self._render_generic_dict_contents(block, params, path + ("params",))
        form.add_children(path, block)


def _render_loader_bindings(self, box, context, methods):
    from src.app.core.data_loaders import LOADER_SLOTS, data_loader_errors

    form = ParameterForm()
    self._render_loader_binding(form, "Symbolic state default", ("symbolic_state_loader",), context)
    box.add_layout(form)
    for name, method in methods.items():
        box.add_layout(_parameter_header(name, heading=False))
        block = IndentedHierarchyBlock()
        for slot, caption in zip(LOADER_SLOTS, ("Training loader", "Symbolic state")):
            self._render_loader_binding(block.form_layout, caption, ("methods", name, slot), context, method)
        for message in data_loader_errors(context, method, paths=self._loader_paths()):
            status = label(message, "muted")
            status.setObjectName("dataLoaderStatus")
            status.setWordWrap(True)
            block.form_layout.add_parameter(("methods", name, "loader_error", message), None, status)
        box.add_widget(block)


def _render_component_overrides(self, data: dict, resolved: dict) -> None:
    """Expose component selection and per-component YAML parameters in recipes."""
    from src.app.core.data_loaders import input_requirements
    from src.app.pipeline.config import deep_merge, parse_methods_dict

    context = deep_merge(resolved, self.raw_data)
    methods = parse_methods_dict(context, paths=self._loader_paths())
    legacy_env = context.get("env")
    legacy_inputs = isinstance(legacy_env, dict) and any(
        legacy_env.get(key) is not None for key in ("data_module", "symbolic_observation_builder")
    )
    if "data_module" not in data and (
        legacy_inputs
        or context.get("symbolic_state_loader") is not None
        or any(
            any(slot in method for slot in ("data_module", "symbolic_state_loader"))
            or input_requirements(context, method, paths=self._loader_paths())
            for method in methods.values()
        )
    ):
        data = {**data, "data_module": None}
    for kind, title in (("data_module", "Data Loaders"), ("evaluator", "Evaluators")):
        if kind not in data:
            continue
        explicit = data.get(kind)
        inherited = self._inherited_component(kind, resolved)
        name = self._component_name(explicit)
        inherited_name = self._component_name(inherited)
        choices = self._component_choices(kind)
        box = ConfigBox()
        heading = _parameter_header(
            title,
            add=self._dict_parameter_adder(
                (kind,), explicit if isinstance(explicit, dict) else {}, {"name": inherited_name, "params": {}}
            ),
        )
        if kind != "data_module" or kind in self.raw_data:
            trash = _make_subtle_tool_button("trash", f"Remove {title.lower()} override", size=24, icon_size=14)
            trash.clicked.connect(lambda _, k=kind: self.remove_block(k))
            heading.addWidget(trash)
        box.add_layout(heading)
        row = ParameterRow()
        combo = ComboBox()
        combo.setEditable(True)
        inherited_label = inherited_name or "None"
        combo.addItem(inherited_label, "")
        for choice in sorted(
            (set(choices) | ({name} if name else set())) - ({inherited_name} if inherited_name else set())
        ):
            combo.addItem(choice, choice)
        if name:
            explicit_index = combo.findData(name)
            if explicit_index < 0 and name == inherited_name:
                combo.insertItem(1, name, name)
                explicit_index = 1
            combo.setCurrentIndex(explicit_index if explicit_index >= 0 else 0)
        else:
            combo.setCurrentIndex(0)
        combo.currentTextChanged.connect(
            lambda selected, k=kind, c=combo: self._set_component_choice(
                k, str(c.currentData() or selected) if c.currentIndex() != 0 else ""
            )
        )
        row.addWidget(combo)
        if explicit:
            reset = _make_subtle_tool_button("close", f"Use inherited {title.lower()} component", size=20, icon_size=12)
            reset.clicked.connect(lambda _, k=kind: self._set_component_choice(k, ""))
            row.addWidget(reset)
        elif inherited_name:
            customize = QPushButton("Use with defaults")
            customize.setToolTip("Create an experiment override using this component and its profile defaults")
            customize.clicked.connect(lambda _, k=kind, n=inherited_name: self._set_component_choice(k, n))
            row.addWidget(customize)
        row.addStretch()
        component_form = ParameterForm()
        component_form.add_parameter((kind,), "Training default" if kind == "data_module" else "Component", row)
        box.add_layout(component_form)
        self.field_widgets[kind] = combo

        if name:
            params = explicit.get("params", {}) if isinstance(explicit, dict) else {}
            if not isinstance(params, dict):
                params = {}
            defaults = self._component_default_params(kind, name)
            candidates = {key: val for key, val in defaults.items() if key not in params}
            add = self._dict_parameter_adder((kind, "params"), params, defaults)
            # The component selector owns its subparameters, with its + beside it.
            row.insertWidget(row.indexOf(combo) + 1, add)
            block = IndentedHierarchyBlock()
            form = block.form_layout
            self._render_generic_dict_contents(form, params, (kind, "params"))
            if params:
                component_form.add_children((kind,), block)
            if isinstance(explicit, dict):
                extra = {
                    key: value for key, value in explicit.items() if key not in ("name", "class", "type", "params")
                }
                if extra:
                    extra_form = ParameterForm()
                    self._render_generic_dict_contents(extra_form, extra, (kind,))
                    box.add_layout(extra_form)
        if kind == "data_module":
            self._render_loader_bindings(box, context, methods)
        self.boxes_layout.addWidget(box)

    if data.get("plots", resolved.get("plots")) is None and "plot_overrides" not in data:
        return
    plot_choices = self._component_choices("plots")
    explicit_plots = data.get("plots")
    per_plot_overrides = data.get("plot_overrides") if isinstance(data.get("plot_overrides"), dict) else {}
    if isinstance(explicit_plots, list):
        selected = [str(item) for item in explicit_plots]
    else:
        selected = self._inherited_plot_names(resolved)
        if isinstance(explicit_plots, dict):
            for key, value in explicit_plots.items():
                if value is None:
                    selected = [item for item in selected if item != key]
                elif key in plot_choices or isinstance(value, dict) and key != "style":
                    if key not in selected:
                        selected.append(key)
    box = ConfigBox()
    available = [name for name in plot_choices if name not in selected]
    heading = _parameter_header(
        "Figures and Tables",
        add=_make_param_adder_button(
            "Add figure",
            sorted(available),
            lambda name: self._set_plot_override(name, True, resolved),
            parent_widget=self,
            allow_custom=False,
        ),
    )
    trash = _make_subtle_tool_button("trash", "Remove figures and tables", size=24, icon_size=14)
    trash.clicked.connect(self._remove_figures_box)
    heading.addWidget(trash)
    box.add_layout(heading)
    if not selected:
        box.add_widget(label("No figures selected. Use + to add one.", "muted"))
    box.setProperty("parameterColumns", 2)
    cards = QGridLayout()
    cards.setHorizontalSpacing(14)
    cards.setVerticalSpacing(14)
    cards.setColumnStretch(0, 1)
    cards.setColumnStretch(1, 1)
    cards.setProperty("equalParameterColumns", True)
    for index, name in enumerate(selected):
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(8)
        row = ParameterRow()
        row.addWidget(label(name, "cardTitle"))
        inline_values = explicit_plots.get(name, {}) if isinstance(explicit_plots, dict) else {}
        values = dict(inline_values) if isinstance(inline_values, dict) else {}
        overrides = per_plot_overrides.get(name, {})
        if not isinstance(overrides, dict):
            overrides = {}
        values.update(overrides)
        defaults = self._component_default_params("plots", name)
        candidates = {key: val for key, val in defaults.items() if key not in values}
        row.addWidget(
            _make_param_adder_button(
                f"Add {name} parameter",
                list(candidates),
                lambda key, n=name, options=candidates: self._add_plot_parameter(n, key, options.get(key)),
                parent_widget=self,
            )
        )
        row.addStretch()
        remove = _make_subtle_tool_button("close", f"Remove {name} figure", size=20, icon_size=12)
        remove.clicked.connect(lambda _, n=name: self._set_plot_override(n, False, resolved))
        row.addWidget(remove)
        _reveal_remove_on_row_hover(row, remove)
        card_layout.addLayout(row)
        block = IndentedHierarchyBlock()
        block.form_layout.set_compact(True)
        card_layout.addWidget(block)
        if values:
            # Keep legacy inline options editable while giving explicit overrides precedence.
            for source, params in (
                ("plots", {k: v for k, v in values.items() if k not in overrides}),
                ("plot_overrides", overrides),
            ):
                if params:
                    self._render_generic_dict_contents(block, params, (source, name), defaults=defaults)
        elif candidates:
            use_defaults = QPushButton("Use with defaults")
            use_defaults.clicked.connect(lambda _, n=name: self._add_plot_parameter_defaults(n))
            card_layout.addWidget(use_defaults)
        else:
            card_layout.addWidget(label("No configured parameters", "muted"))
        card_layout.addStretch()
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        cards.addWidget(card, index // 2, index % 2)
    box.add_layout(cards)
    if isinstance(explicit_plots, dict):
        general = {key: value for key, value in explicit_plots.items() if key not in selected and value is not None}
        if general:
            box.add_widget(label("Figure settings", "muted"))
            form = ParameterForm()
            self._render_generic_dict_contents(form, general, ("plots",))
            box.add_layout(form)
    self.boxes_layout.addWidget(box)
