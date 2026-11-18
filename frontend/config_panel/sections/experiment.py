"""Shared Experiment/Group Defaults composition and runtime declarations."""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QMenu,
    QToolButton,
    QWidget,
)

from ...widgets import (
    FlowLayout,
    ToggleSlider,
    label,
)
from ..catalogs import (
    _recorded_metrics,
    config_tree,
)
from ..controls import (
    ComboBox,
    ConfigBox,
    IndentedHierarchyBlock,
    ParameterForm,
    _make_param_adder_button,
    _make_subtle_tool_button,
    _parameter_row,
    _reveal_remove_on_row_hover,
)
from ..document import MISSING, lookup
from ..editors import DirectoryField, value_editor
from ..layout import two_column_form


def context(host, data, resolved):
    box = ConfigBox(section="experiment")
    host._add_experiment_section_header(box, data, resolved)
    host._add_experiment_description(box, data)
    form = ParameterForm()
    form.setProperty("parameterContext", "experiment.context")
    if hasattr(host, "combo_paradigm"):
        form.add_parameter(("paradigm",), "Paradigm", _parameter_row(host.combo_paradigm))
    if hasattr(host, "combo_env"):
        env = data.get("env")
        params = {k: v for k, v in env.items() if k not in ("name", "id", "type")} if isinstance(env, dict) else {}
        children = None
        if params:
            children = IndentedHierarchyBlock()
            children.setProperty("parameterPath", "env")
            host._render_generic_dict_contents(children, params, ("env",))
        form.add_parameter(
            ("env",),
            "Environment" if host._is_base_config() else "Environment / Task",
            _parameter_row(host.combo_env, add=host._environment_parameter_adder(data, resolved)),
            children=children,
        )
    host._add_experiment_site(form, data, resolved)
    box.add_layout(two_column_form(form, identity="experiment.context", align_rows=True))
    if host._has_experiment_lower_params(data, resolved):
        host._add_experiment_separator(box)
    host._add_experiment_trainer_fields(box, data, resolved)
    return box


def runtime(host, data, resolved, tree, paradigm, show_resolved):
    rules = tree.paradigms.get(paradigm) if tree else None
    timesteps = bool(rules and rules.budget_unit == "timesteps")
    form = ParameterForm()
    form.setProperty("parameterContext", "experiment.runtime")
    declarations = [
        ("total_timesteps", "Total Timesteps", "spin_timesteps", 10000, 100),
        ("seed", "Random Seed", "spin_seed", 42, 0),
        ("eval_episodes", "Eval Episodes", "spin_eval_ep", 100, 0),
        ("epochs", "Epochs", "spin_epochs", 10, 1),
        ("eval_every_n_epochs", "Eval Frequency (Epochs)", "spin_eval_every", 1, 1),
    ]
    for key, title, attribute, default, minimum in declarations:
        value = data.get(key, resolved.get(key, default))
        field = value_editor(
            value, {"type": "integer", "minimum": minimum}, lambda value, k=key: host._on_field_edited(k, value)
        )
        setattr(host, attribute, field)
        host.field_widgets[key] = field
        if key == "eval_episodes" and rules and not rules.field_enabled(key):
            field.setEnabled(False)
            field.setToolTip(rules.disabled_reason(key))
            if hasattr(field, "setSuffix"):
                field.setSuffix(f"   (fixed by {paradigm})")
        visible = key in data or show_resolved
        if key in ("total_timesteps", "eval_episodes"):
            visible &= timesteps
        if key in ("epochs", "eval_every_n_epochs"):
            visible &= not timesteps
        if visible:
            form.add_parameter((key,), title, _parameter_row(field, remove=host._parameter_remove((key,))))
    if timesteps:
        freq_key = "eval_freq_steps" if "eval_freq_steps" in data or "eval_freq_steps" in resolved else "eval_freq"
        if freq_key in data or show_resolved and freq_key in resolved:
            field = value_editor(
                data.get(freq_key, resolved.get(freq_key)),
                {"type": "integer", "minimum": 0},
                lambda value: host._on_field_edited(freq_key, value),
            )
            # 0 turns evaluation off (src/app/telemetry/selection.py: evaluation_off).
            field.setToolTip("Steps between evaluations. 0 turns evaluation off: no evaluation episodes run "
                             "and no eval/ metrics are recorded or streamed.")
            if hasattr(field, "setSpecialValueText"):
                field.setSpecialValueText("Off")
            host.field_widgets[freq_key] = field
            form.add_parameter(
                (freq_key,), "Eval Frequency", _parameter_row(field, remove=host._parameter_remove((freq_key,)))
            )
        row = host._checkpoint_row(data, resolved)
        if row is not None:
            form.add_parameter(("checkpoint_schedule",), "Checkpoints", row, children=row._parameter_block)

    standard = {
        "defaults",
        "paradigm",
        "experiment_id",
        "description",
        "seed",
        "total_timesteps",
        "eval_episodes",
        "eval_freq_steps",
        "eval_freq",
        "epochs",
        "eval_every_n_epochs",
        "checkpoint_every_steps",
        "num_checkpoints",
        "run_in_parallel",
        "timestamp_experiment",
        "methods",
        "env",
        "site",
        "trainer",
        "resources",
        "dataset",
        "dataset_path",
        "no_plot",
        "save_dataset",
        "save_dataset_path",
        "recover",
        "hydra",
        "plots",
        "tuning",
    }
    booleans = []
    for key, value in data.items():
        if key in standard or not isinstance(value, (int, float, str, bool)) and value is not None:
            continue
        field = host._parameter_editor(key, value, (key,))
        row = _parameter_row(field, remove=host._parameter_remove((key,)))
        if key == "batch_size":
            host._add_sweep_toggle(row, "params", "config.batch_size", value)
        if isinstance(value, bool):
            booleans.append((key, row))
        else:
            form.add_parameter((key,), key, row, children=getattr(row, "_sweep_block", None))
    return two_column_form(form, identity="experiment.runtime", align_rows=True), booleans


def build(self):
    """Render standard boxes for an Experiment configuration."""
    data = self.raw_data
    resolved = self._get_resolved_defaults()
    is_base = self._is_base_config()
    show_resolved = is_base or self.view_mode == "resolved"

    # Paradigm is inherited from group base by default.
    # A missing paradigm is an incomplete experiment, not an invitation to
    # silently format it as online RL.  The method form depends on this
    # contract to know whether an algorithm belongs in every method.
    paradigm = data.get("paradigm") or self._inherited_paradigm()
    self.paradigm_badge.hide()

    # Metadata / constraint helper widgets (not rendered into visual layout)
    group_val = data.get("group")
    if not group_val:
        parts = Path(self.current_rel_path).parts
        group_val = parts[1] if len(parts) > 2 else "ungrouped"
    self.txt_group = QLineEdit(str(group_val))
    self.txt_group.textChanged.connect(lambda v: self._on_field_edited("group", v))
    self.field_widgets["group"] = self.txt_group

    # Paradigm + Environment, driven by paradigm profile constraints.
    tree = self.tree
    top_box = None
    if is_base:
        self.combo_paradigm = ComboBox()
        self.combo_paradigm.addItem("— Select Paradigm —", None)
        if tree and tree.paradigms:
            self.combo_paradigm.addItems(sorted(tree.paradigms))
        base_p = data.get("paradigm")
        if base_p and tree and base_p in tree.paradigms:
            self.combo_paradigm.setCurrentText(base_p)
        else:
            self.combo_paradigm.setCurrentIndex(0)

        self.combo_env = ComboBox()
        self.combo_env.addItem("— Select Environment —", None)

        def _refresh_base_env_choices(selected_p):
            curr_data = self.combo_env.currentData()
            self.combo_env.blockSignals(True)
            self.combo_env.clear()
            self.combo_env.addItem("— Select Environment —", None)
            if selected_p and tree and selected_p in tree.paradigms:
                allowed = [e.name for e in tree.environments_for(selected_p) if self._is_environment_choice(e.name)]
            elif tree:
                allowed = sorted(
                    set(e.name for e in tree.visible_environments() if self._is_environment_choice(e.name))
                )
            else:
                allowed = []
            self.combo_env.addItems(allowed)
            candidate = self._get_base_env() or (curr_data if isinstance(curr_data, str) else None)
            if candidate in allowed:
                self.combo_env.setCurrentText(candidate)
            elif self._is_environment_choice(candidate):
                unavailable_index = self._add_unavailable_environment_option(
                    self.combo_env, candidate, selected_p, tree
                )
                self.combo_env.setCurrentIndex(unavailable_index if unavailable_index is not None else 0)
            else:
                self.combo_env.setCurrentIndex(0)
            self.combo_env.blockSignals(False)

        base_env = self._get_base_env()
        if base_p and tree and base_p in tree.paradigms:
            allowed_init = [e.name for e in tree.environments_for(base_p) if self._is_environment_choice(e.name)]
        elif tree:
            allowed_init = sorted(
                set(e.name for e in tree.visible_environments() if self._is_environment_choice(e.name))
            )
        else:
            allowed_init = []
        self.combo_env.addItems(allowed_init)
        if self._is_environment_choice(base_env):
            if base_env in allowed_init:
                self.combo_env.setCurrentText(base_env)
            else:
                unavailable_index = self._add_unavailable_environment_option(self.combo_env, base_env, base_p, tree)
                self.combo_env.setCurrentIndex(unavailable_index if unavailable_index is not None else 0)
        else:
            self.combo_env.setCurrentIndex(0)

        def _on_base_p_changed(p_text):
            val = None if p_text in ("", "— Select Paradigm —") else p_text
            self.document.set(("paradigm",), val)
            _refresh_base_env_choices(val)
            self._mark_dirty()

        def _on_base_e_changed(e_text):
            val = None if e_text in ("", "— Select Environment —") else e_text
            self._set_base_env(val)

        self.combo_paradigm.currentTextChanged.connect(_on_base_p_changed)
        self.combo_env.currentTextChanged.connect(_on_base_e_changed)

        self.field_widgets["paradigm"] = self.combo_paradigm
        self.field_widgets["env"] = self.combo_env

        from .experiment import context

        top_box = context(self, data, resolved)
    else:
        if tree and tree.paradigms:
            self.combo_paradigm = ComboBox()
            self.combo_paradigm.addItem("— Select Paradigm —", None)
            self.combo_paradigm.addItems(sorted(tree.paradigms))
            if paradigm in tree.paradigms:
                self.combo_paradigm.setCurrentText(paradigm)
            else:
                self.combo_paradigm.setCurrentIndex(0)
            rules = tree.paradigms.get(self.combo_paradigm.currentText())
            if rules and rules.description:
                self.combo_paradigm.setToolTip(rules.description)
            self.combo_paradigm.currentTextChanged.connect(self._on_paradigm_changed)
            self.field_widgets["paradigm"] = self.combo_paradigm

            self.combo_env = ComboBox()
            self.combo_env.setMinimumWidth(280)
            self.combo_env.setMaximumWidth(360)
            inherited_env = tree.group(self._group_name() or "").env
            inherited_env = inherited_env or self._inherited_env()
            explicit_env = self._get_explicit_env()
            has_explicit_override = bool(explicit_env and explicit_env != inherited_env)
            inherited_is_compatible = self._environment_is_compatible(tree, inherited_env, paradigm)
            inherit_label = inherited_env or "Default"
            if inherited_env and not inherited_is_compatible:
                if has_explicit_override:
                    # Keep the reset-to-default slot, but don't surface the
                    # stale inherited component as a menu choice when an
                    # explicit environment is already selected.
                    inherit_label = "Default"
                else:
                    exists = self._environment_for_name(tree, inherited_env) is not None
                    inherit_label = f"{inherited_env} ({'incompatible' if exists else 'missing'})"
            self.combo_env.addItem(inherit_label, self.INHERIT)
            if inherited_env and not inherited_is_compatible:
                item = self.combo_env.model().item(0)
                if item is not None:
                    item.setEnabled(False)
            selected_paradigm = self.combo_paradigm.currentText()
            if selected_paradigm in tree.paradigms:
                allowed_envs = [
                    e.name for e in tree.environments_for(selected_paradigm) if self._is_environment_choice(e.name)
                ]
            else:
                allowed_envs = sorted(
                    set(e.name for e in tree.visible_environments() if self._is_environment_choice(e.name))
                )
            self.combo_env.addItems(allowed_envs)
            if explicit_env and explicit_env != inherited_env:
                if self._is_environment_choice(explicit_env) and explicit_env in allowed_envs:
                    self.combo_env.setCurrentText(explicit_env)
                elif self._is_environment_choice(explicit_env):
                    unavailable_index = self._add_unavailable_environment_option(
                        self.combo_env, explicit_env, selected_paradigm, tree
                    )
                    if unavailable_index is not None:
                        self.combo_env.setCurrentIndex(unavailable_index)
            else:
                self.combo_env.setCurrentIndex(0)
            self.combo_env.setToolTip(f"Environments for {self.combo_paradigm.currentText()}")
            self.combo_env.currentTextChanged.connect(lambda v: self._on_experiment_env_changed(v))
            self.field_widgets["env"] = self.combo_env

        # An experiment starts with its research environment. For static
        # paradigms that environment is realized by a dataset; the data
        # module below remains an adapter, not a second kind of dataset.
        from .experiment import context

        box_inputs = context(self, data, resolved)

        rules = tree.paradigms.get(paradigm) if tree else None
        if rules and rules.uses_static_dataset and ("dataset_path" in data or show_resolved):
            self._add_dataset_source_fields(
                box_inputs,
                data,
                resolved,
                required=rules.uses_static_dataset,
                paradigm=paradigm,
            )

        top_box = box_inputs

    # The experiment context and its schedule are deliberately one card:
    # paradigm, source, runtime defaults, and budget describe one run.
    box_budget = top_box or ConfigBox()

    from .experiment import runtime

    form_budget, extra_boolean_items = runtime(self, data, resolved, tree, paradigm, show_resolved)

    # Runtime & execution options (logging, recovery, plotting, and parallel methods)
    OPTIONAL_RUNTIME_FLAGS = [
        (("save_dataset",), "Save trajectories"),
        (("recover",), "Recover Checkpoint"),
        (("no_plot",), "No Plots"),
        (("run_in_parallel",), "Run methods in parallel"),
        (("timestamp_experiment",), "Timestamp experiment"),
    ]
    run_loggers = self._run_logger_toggles()
    OPTIONAL_RUNTIME_FLAGS[0:0] = run_loggers
    ALWAYS_SHOWN = (("run_in_parallel",), ("timestamp_experiment",))
    TOOLTIPS = {
        ("timestamp_experiment",): "Add a timestamp when results already exist to keep previous runs.",
        ("run_in_parallel",): "Train every configured method in a separate process at the same time.",
    }

    runtime_form = FlowLayout(spacing=16)
    runtime_form.setObjectName("runtimeParameters")
    runtime_form.setProperty("parameterReorderFlow", True)
    chk_widgets = {}

    def set_flag(path, value):
        if not self._block_updates:
            self.document.set(path, value)

    def clear_flag(path):
        """Remove an optional flag and any mapping it leaves empty."""
        self.document.remove(path)
        for depth in range(len(path) - 1, 0, -1):
            parent = lookup(self.raw_data, path[:depth])
            if parent == {}:
                self.document.remove(path[:depth])
        self._mark_dirty()
        self._render_boxes()

    # Keep each label, toggle, and optional removal action together as the row wraps.
    active_flags = [
        (path, lbl)
        for path, lbl in OPTIONAL_RUNTIME_FLAGS
        if lookup(data, path) is not MISSING or show_resolved or path in ALWAYS_SHOWN
    ]

    for flag_path, flag_label in active_flags:
        flag_key = ".".join(flag_path)
        explicit = lookup(data, flag_path)
        inherited = lookup(resolved, flag_path)
        value = explicit if explicit is not MISSING else inherited
        flag = QWidget()
        flag.setProperty("parameterBoolean", True)
        flag.setProperty("parameterPath", flag_key)
        row_flag = QHBoxLayout(flag)
        row_flag.setContentsMargins(0, 0, 0, 0)
        row_flag.setSpacing(6)
        row_flag.addWidget(label(flag_label))
        chk = ToggleSlider(checked=bool(value is not MISSING and value))
        if flag_path in TOOLTIPS:
            chk.setToolTip(TOOLTIPS[flag_path])
        if flag_path in ALWAYS_SHOWN or flag_path in (("no_plot",), ("save_dataset",)):
            chk.toggled.connect(lambda value, fp=flag_path: set_flag(fp, value))
        else:
            chk.toggled.connect(
                lambda v, fp=flag_path: set_flag(fp, True) if v else clear_flag(fp)
            )
        row_flag.addWidget(chk)
        chk_widgets[flag_path] = chk
        if flag_path == ("save_dataset",):
            add_location = _make_param_adder_button(
                "Choose where to save trajectories",
                ["Save location"],
                lambda _: self.field_widgets["save_dataset_path"].browse(),
                allow_custom=False,
            )
            row_flag.addWidget(add_location)

        if explicit is not MISSING:
            btn_del_flag = _make_subtle_tool_button("close", f"Remove '{flag_key}' override", size=18, icon_size=11)
            btn_del_flag.clicked.connect(lambda _, fp=flag_path: clear_flag(fp))
            row_flag.addWidget(btn_del_flag)
            _reveal_remove_on_row_hover(row_flag, btn_del_flag)
        elif not is_base and flag_path not in ALWAYS_SHOWN:
            row_flag.addWidget(label(str(chk.isChecked()).lower(), "muted"))

        runtime_form.addWidget(flag)

    for key, row in extra_boolean_items:
        flag = QWidget()
        flag.setProperty("parameterBoolean", True)
        flag.setProperty("parameterPath", key)
        row.setContentsMargins(0, 0, 0, 0)
        row.insertWidget(0, label(key))
        flag.setLayout(row)
        runtime_form.addWidget(flag)

    self.chk_timestamp_experiment = chk_widgets.get(("timestamp_experiment",))
    self.run_logger_toggles = {path[2]: chk_widgets[path] for path, _label in run_loggers if path in chk_widgets}
    self.chk_save_ds = chk_widgets.get(("save_dataset",))
    self.chk_recover = chk_widgets.get(("recover",))
    self.chk_no_plot = chk_widgets.get(("no_plot",))

    box_budget.add_layout(form_budget)
    box_budget.add_layout(runtime_form)
    if self.chk_save_ds is not None:
        trajectory_children = IndentedHierarchyBlock()
        trajectory_children.setProperty("parameterPath", "save_dataset")
        destination = data.get("save_dataset_path", resolved.get("save_dataset_path"))
        workspace = tree.workspace if tree else None
        field = DirectoryField(
            destination,
            lambda value: self._on_field_edited("save_dataset_path", value),
            initial_directory=workspace.root if workspace else Path.home(),
        )
        field.setObjectName("trajectorySaveLocation")
        field.setToolTip(str(destination or "Choose a folder for saved trajectories."))
        self.field_widgets["save_dataset_path"] = field
        trajectory_children.form_layout.add_parameter(
            ("save_dataset_path",), "Save location",
            _parameter_row(field, remove=self._parameter_remove(("save_dataset_path",))),
        )
        trajectory_children.setVisible(self.chk_save_ds.isChecked())
        self.chk_save_ds.toggled.connect(trajectory_children.setVisible)
        box_budget.layout.addWidget(trajectory_children)
    shared_spec = self.raw_data.get("methods", {}).get("params", {})
    if isinstance(shared_spec, dict):
        experiment_sweeps = ParameterForm()
        self._render_method_sweeps(experiment_sweeps, "params", shared_spec, prefix="config.")
        box_budget.add_layout(experiment_sweeps)
    self.boxes_layout.addWidget(box_budget)

    from .methods import build as build_methods

    build_methods(self, data, resolved, tree, paradigm, is_base)

    # --- MODULAR BOX: CHECKPOINT SETTINGS ---
    has_checkpoint = "checkpoint" in data and isinstance(data["checkpoint"], dict)
    show_checkpoint = has_checkpoint or (
        self.view_mode == "resolved" and "checkpoint" in resolved and isinstance(resolved["checkpoint"], dict)
    )

    if show_checkpoint:
        source = data.get("checkpoint", {}) if has_checkpoint else resolved.get("checkpoint", {})
        box = ConfigBox()
        defaults = {"monitor": "val/loss", "mode": "min", "save_top_k": 1, "every_n_epochs": 1}
        remove = _make_subtle_tool_button("trash", "Remove checkpoint overrides", size=24, icon_size=14)
        remove.clicked.connect(lambda: self.remove_block("checkpoint"))
        box.set_header(
            "Checkpoint Settings",
            add=self._dict_parameter_adder(("checkpoint",), source, defaults),
            remove=remove if has_checkpoint else None,
        )
        form = ParameterForm()
        self._render_generic_dict_contents(form, source, ("checkpoint",), defaults=defaults)
        box.add_layout(form)
        self.boxes_layout.addWidget(box)

    self._render_variability_box(data, resolved)
    self._render_sweep_box(data)
    self._render_component_overrides(data, resolved)
    self._render_remaining_sections()

    # --- ADD CONFIGURATION BLOCK DROPDOWN BUTTON ---
    self._render_metrics_box(data, resolved)
    self._render_monitors_box(data, resolved)

    btn_add_block_row = QHBoxLayout()
    self.btn_add_block = _make_subtle_tool_button("plus", "Add overrides", size=26, icon_size=14)
    self.btn_add_block.setObjectName("addBlockButton")
    self.btn_add_block.setText("Add Overrides")
    self.btn_add_block.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
    self.btn_add_block.setFixedSize(132, 28)
    self.btn_add_block.setStyleSheet("""
        QToolButton#addBlockButton {
            background: transparent;
            border: 1px dashed #504945;
            border-radius: 4px;
            padding: 0px;
        }
        QToolButton#addBlockButton:hover {
            background: #32302f;
            border-color: #a89984;
        }
        QToolButton#addBlockButton::menu-indicator {
            image: none;
            width: 0px;
        }
    """)
    self.btn_add_block.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    add_menu = QMenu(self.btn_add_block)
    for kind, title in (("data_module", "Data Loaders"), ("evaluator", "Evaluators"), ("plots", "Figures and Tables")):
        action = add_menu.addAction(title)
        action.triggered.connect(lambda _, k=kind: self.add_component_override(k, resolved))
        present = self._has_component_override(kind, data)
        if kind == "plots" and "plots" not in data:
            present = present or resolved.get("plots") is not None
        action.setEnabled(not present)
    act_ckpt = add_menu.addAction("Checkpoint Settings")
    act_ckpt.triggered.connect(lambda: self.add_block("checkpoint"))
    act_ckpt.setEnabled("checkpoint" not in data)

    act_env = add_menu.addAction("Environment Overrides")
    act_env.triggered.connect(lambda: self.add_block("env"))
    act_env.setEnabled(not isinstance(data.get("env"), dict))

    act_ds = add_menu.addAction("Dataset Configuration")
    act_ds.triggered.connect(lambda: self.add_block("dataset"))
    act_ds.setEnabled("dataset_path" not in data)

    has_sweeper = "tuning" in data
    if self._tuner_contributions():
        act_sw = add_menu.addAction("Hyperparameter Sweep")
        act_sw.triggered.connect(lambda: self.add_block("tuning"))
        act_sw.setEnabled(not has_sweeper)
    act_variability = add_menu.addAction("Training and Evaluation Variability")
    act_variability.triggered.connect(lambda: self.add_block("variability"))
    act_variability.setEnabled("variability" not in data)
    act_metrics = add_menu.addAction("Recorded Metrics")
    act_metrics.setToolTip("Record only some metrics (and skip the work behind the others) to train faster")
    act_metrics.triggered.connect(lambda: self.add_block("metrics"))
    act_metrics.setEnabled(_recorded_metrics(data) is None)
    act_monitors = add_menu.addAction("Monitors")
    act_monitors.triggered.connect(lambda: self.add_block("monitors"))
    telemetry = {}
    for config in (resolved, data):
        if isinstance(config.get("telemetry"), dict):
            telemetry.update(config["telemetry"])
    act_monitors.setEnabled(telemetry.get("monitors") is None)
    self.btn_add_block.setMenu(add_menu)
    btn_add_block_row.addWidget(self.btn_add_block)

    btn_add_block_row.addStretch()
    self.boxes_layout.addLayout(btn_add_block_row)
