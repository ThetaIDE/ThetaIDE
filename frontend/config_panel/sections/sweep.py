"""Configuration section composition using shared parameter controls."""

import math

from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QMenu,
    QToolButton,
)

from ...widgets import (
    ToggleSlider,
    label,
)
from ..controls import (
    NUMBER_FIELD_WIDTH,
    ComboBox,
    ConfigBox,
    IndentedHierarchyBlock,
    ParameterForm,
    _make_param_adder_button,
    _make_subtle_tool_button,
    _parameter_header,
    _parameter_row,
)
from ..layout import _two_column_parameter_layout


def _render_sweep_box(self, data):
    resolved = self._get_resolved_defaults()
    tuning = self._parameter_context_config().get("tuning")
    has_tuning = isinstance(tuning, dict)
    if not has_tuning and not isinstance(self._effective_objective_spec(data, resolved), dict):
        return
    tuning = tuning if has_tuning else {}
    contributions = self._tuner_contributions()
    title = "Hyperparameter Sweep" if has_tuning or contributions else "Objective"
    box = ConfigBox(title, "Optimize each method independently" if has_tuning else "Score each method")
    box.setProperty("configSection", "sweep")
    box.setProperty("parameterColumns", 2)
    add = _make_param_adder_button("Add optional sweep parameter", [], lambda _: None, allow_custom=False)
    add.setObjectName("sweepAddOption")
    optional_menu = add.menu()
    header = _parameter_header(title, add=add)
    remove = _make_subtle_tool_button("trash", "Remove hyperparameter sweep", size=24, icon_size=14)
    remove.clicked.connect(lambda: self.remove_block("tuning" if has_tuning else "objective"))
    if has_tuning or "objective" in data:
        header.addWidget(remove)
    box.add_layout(header)
    form = ParameterForm()
    self._render_sweep_objective(form, data, resolved)
    if not has_tuning:
        if contributions:
            action = optional_menu.addAction("Enable hyperparameter tuning")
            action.triggered.connect(lambda: self.add_block("tuning"))
        form.add_parameter(
            ("objective", "notice"),
            None,
            label("Objective scoring only." + (" Add sweep settings to tune hyperparameters." if contributions else ""), "muted"),
        )
        box.add_layout(_two_column_parameter_layout(form))
        self.boxes_layout.addWidget(box)
        return
    direction = ComboBox()
    direction.setObjectName("sweepDirection")
    direction.setMinimumWidth(NUMBER_FIELD_WIDTH)
    direction.addItems(["maximize", "minimize"])
    direction.setCurrentText(str(tuning.get("direction") or "minimize"))
    direction.currentTextChanged.connect(lambda v: self._on_nested_edited("tuning", "direction", v))
    form.add_parameter(("tuning", "direction"), "Direction", _parameter_row(direction))
    from src.app.pipeline.tuners import tuner_name, get_tuner
    from src.app.pipeline.exceptions import ConfigurationError

    selection_error = None
    try:
        component = tuner_name({"tuning": tuning})
    except ConfigurationError as exc:
        component, selection_error = None, str(exc)
    selector = ComboBox()
    selector.setObjectName("sweepComponent")
    names = sorted({item.capability.id for item in contributions})
    if component is not None and component not in names:
        names.append(component)
    if component is None:
        selector.addItem("Select a tuner", None)
    for name in names:
        selector.addItem(name, name)
    if component is not None:
        selector.setCurrentIndex(selector.findData(component))
    selector.currentIndexChanged.connect(lambda _: self._on_tuner_selected(selector.currentData()))
    form.add_parameter(("tuning", "component"), "Tuner", _parameter_row(selector))
    try:
        fields = get_tuner(name=component, paths=self._loader_paths()).settings() if component is not None else []
    except Exception as exc:
        fields = []
        unavailable = label(f"Tuner unavailable: {exc}", "muted")
        form.add_parameter(("tuning", "unavailable"), None, unavailable)
    from src.app.core.contributions import ContributionRegistry
    status = ContributionRegistry(self._loader_paths()).status("tuner", component, config=self._parameter_context_config()) if component is not None else None
    if selection_error or (status is not None and status.state != "ready"):
        notice = label(selection_error or status.reason, "muted")
        notice.setObjectName("tunerReadinessNotice")
        notice.setWordWrap(True)
        form.add_parameter(("tuning", "readiness"), None, notice)
    all_options = tuning.get("options", {})
    component_options = all_options.get(component, {}) if isinstance(all_options, dict) else {}
    if not isinstance(component_options, dict):
        component_options = {}
    defaults = {field["key"]: field.get("default") for field in fields}
    by_key = {field["key"]: field for field in fields}
    def option_value(key, default=None):
        fallback = tuning.get(key, default) if by_key.get(key, {}).get("legacy_flat") else default
        return component_options.get(key, fallback)
    groups = {}
    child_menus = {field["parent"]: QMenu(box) for field in fields if field.get("parent")}
    for field in fields:
        conditions = field.get("when", {})
        if any(
            option_value(key, defaults.get(key))
            != expected
            for key, expected in conditions.items()
        ):
            continue
        key, kind = field["key"], field["type"]
        default = field.get("default")
        value = option_value(key, default)
        invalid_number = (kind == "int" and (isinstance(value, bool) or not isinstance(value, int))) or (
            kind == "float" and (isinstance(value, bool) or not isinstance(value, (int, float)))
        )
        if kind in ("int", "float") and not invalid_number:
            lower = field.get("min", 0 if kind == "int" else -1e9)
            upper = field.get("max", 100000 if kind == "int" else 1e9)
            invalid_number = value < lower or value > upper or (kind == "float" and not math.isfinite(value))
        if field.get("managed_hint") and not invalid_number:
            continue
        explicit = key in component_options or (field.get("legacy_flat") and key in tuning)
        required = bool(field.get("required_when")) and all(option_value(k, defaults.get(k)) == v for k, v in field["required_when"].items())
        if field.get("optional") and not explicit and not required:
            parent = field.get("parent")
            menu = child_menus[parent] if parent else optional_menu
            action = menu.addAction(str(field.get("label", key)))
            action.setObjectName("sweepAdd_" + key)
            action.triggered.connect(lambda _, c=component, k=key, v=default: self._add_tuner_option(c, k, v))
            continue
        if kind in ("int", "float"):
            from ..editors import value_editor

            schema = {"type": "integer" if kind == "int" else "number"}
            if "min" in field:
                schema["minimum"] = field["min"]
            if "max" in field:
                schema["maximum"] = field["max"]
            widget = value_editor(value, schema, lambda v, c=component, k=key: self._on_tuner_option_edited(c, k, v))
        elif kind == "choice":
            widget = ComboBox()
            choices = [str(item) for item in field.get("choices", [])]
            if str(value) not in choices:
                choices.append(str(value))
            widget.addItems(choices)
            widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(lambda v, c=component, k=key: self._on_tuner_choice_edited(c, k, v))
        elif kind == "text":
            widget = QLineEdit(str(value or ""))
            widget.textChanged.connect(lambda v, c=component, k=key: self._on_tuner_option_edited(c, k, v))
        elif kind == "bool":
            widget = ToggleSlider(checked=bool(value))
            widget.toggled.connect(lambda v, c=component, k=key: self._on_tuner_option_edited(c, k, v))
        else:
            continue
        widget.setObjectName("sweep_" + key)
        if isinstance(widget, (QComboBox, QAbstractSpinBox)):
            widget.setMinimumWidth(NUMBER_FIELD_WIDTH)
        if field.get("help") and not invalid_number:
            widget.setToolTip(str(field["help"]))
        child_add = None
        if key in child_menus:
            child_add = _make_param_adder_button(
                "Add " + str(field.get("label", key)) + " subparameter", [], lambda _: None, allow_custom=False
            )
            child_add.setObjectName("sweepAddSubOption_" + key)
            child_add.setMenu(child_menus[key])
            child_add.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        remove_option = None
        if field.get("optional") and not required:
            remove_option = _make_subtle_tool_button(
                "close", "Use default for " + str(field.get("label", key)), size=20, icon_size=12
            )
            remove_option.setObjectName("sweepRemoveOption_" + key)
            remove_option.clicked.connect(lambda _, c=component, k=key: self._remove_tuner_option(c, k))
        row = _parameter_row(widget, add=child_add, remove=remove_option)
        parent = field.get("parent")
        target_form = form
        if parent:
            if parent not in groups:
                groups[parent] = IndentedHierarchyBlock()
                groups[parent].setProperty("sweepParent", parent)
                form.add_children(("tuning", "options", component, parent), groups[parent])
            target_form = groups[parent].form_layout
        target_form.add_parameter(("tuning", "options", component, key), str(field.get("label", key)), row)
    add.setEnabled(not optional_menu.isEmpty())
    box.add_layout(_two_column_parameter_layout(form))
    for key, menu in child_menus.items():
        if button := box.findChild(QToolButton, "sweepAddSubOption_" + key):
            button.setEnabled(not menu.isEmpty())
    self.boxes_layout.addWidget(box)
