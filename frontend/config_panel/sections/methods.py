"""Methods cards retain their specialized layout and domain actions."""

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QVBoxLayout

from ...config_model import component_selection
from ...widgets import ToggleSlider, label
from ..catalogs import (
    TRAINING_HYPERPARAMETERS,
    get_agent_candidate_hyperparameters,
    get_model_candidate_hyperparameters,
    get_training_candidate_hyperparameters,
)
from ..controls import (
    NUMBER_FIELD_WIDTH,
    TEXT_FIELD_WIDTH,
    ComboBox,
    ConfigBox,
    IndentedHierarchyBlock,
    InlineTitleEditor,
    MethodCardsContainer,
    MethodDragHandle,
    ParameterForm,
    ParameterRow,
    TitleLabel,
    _make_param_adder_button,
    _make_subtle_tool_button,
    _parameter_row,
    _parameter_title,
    _reveal_remove_on_row_hover,
)
from ..editors import number_field


def build(self, data, resolved, tree, paradigm, is_base):
    # --- BOX 2 / 3: METHODS & AGENT ARCHITECTURES ---
    methods_dict = data.get("methods", {})
    box_methods = ConfigBox("Methods", "Configured agents, policies, and hyperparameter overrides")
    box_methods.setProperty("specialParameterLayout", True)
    m_layout = QVBoxLayout()
    m_layout.setSpacing(10)

    # Universal / Shared Parameters (methods.params)
    has_params = "params" in methods_dict and isinstance(methods_dict["params"], dict)
    resolved_params = resolved.get("methods", {}).get("params", {}) if isinstance(resolved.get("methods"), dict) else {}
    show_params = (
        has_params
        or getattr(self, "_show_empty_shared_params", False)
        or (self.view_mode == "resolved" and bool(resolved_params))
    )

    # The method-plus creates a blank method; shared parameters have their
    # own explicit action at the opposite edge of this header.
    methods_header = QHBoxLayout()
    methods_header.setSpacing(8)
    lbl_methods_title = _parameter_title("Methods")
    methods_header.addWidget(lbl_methods_title)

    btn_add_method = _make_subtle_tool_button("plus", "Add new empty method", size=24, icon_size=14)
    btn_add_method.clicked.connect(lambda: self._start_add_new_method(empty=True))
    methods_header.addWidget(btn_add_method)

    methods_header.addStretch()
    if not show_params:
        btn_create_params = _make_subtle_tool_button("plus", "Add Shared Parameters (params)", size=24, icon_size=14)
        btn_create_params.clicked.connect(lambda: self.add_block("params"))
        methods_header.addWidget(btn_create_params)
    m_layout.addLayout(methods_header)

    if show_params:
        p_card = QFrame()
        p_card.setObjectName("card")
        p_layout = QVBoxLayout(p_card)
        p_layout.setContentsMargins(14, 12, 14, 12)
        p_layout.setSpacing(8)

        p_header = QHBoxLayout()
        p_header.setSpacing(8)
        lbl_shared = label("Shared Parameters", "cardTitle")
        lbl_shared.setStyleSheet("font-size: 16px; font-weight: 600;")
        p_header.addWidget(lbl_shared)

        curr_params = (
            methods_dict.get("params", {})
            if has_params
            else ({} if getattr(self, "_show_empty_shared_params", False) else resolved_params)
        )

        common_universal = [
            "lr",
            "gamma",
            "epochs",
            "eval_every_n_epochs",
            "weight_decay",
            "batch_size",
            "ent_coef",
            "agent",
            "model",
        ]
        cand_u = [u for u in common_universal if u not in curr_params]
        btn_add_u = _make_param_adder_button(
            "Add shared parameter",
            cand_u,
            lambda pk: self._add_universal_param(pk, {} if pk in ("agent", "model") else None),
            parent_widget=self,
        )
        for action in btn_add_u.menu().actions():
            if action.text() == "agent":
                action.setText("Algorithm")
            elif action.text() == "model":
                action.setText("Model")
        p_header.addWidget(btn_add_u)
        p_header.addStretch()
        if has_params or getattr(self, "_show_empty_shared_params", False):
            btn_rm_params = _make_subtle_tool_button("trash", "Remove Shared Parameters block", size=24, icon_size=14)
            btn_rm_params.clicked.connect(lambda: self.remove_block("params"))
            p_header.addWidget(btn_rm_params)
        p_layout.addLayout(p_header)

        from .shared_methods import build as build_shared

        shared_form = ParameterForm(field_width=round(TEXT_FIELD_WIDTH * 2 / 3))
        build_shared(self, shared_form, curr_params, tree, paradigm, has_params)
        p_layout.addLayout(shared_form)
        shared_sweeps = ParameterForm(field_width=round(TEXT_FIELD_WIDTH * 2 / 3))
        self._render_method_sweeps(shared_sweeps, "params", curr_params, exclude_prefix="config.")
        p_layout.addLayout(shared_sweeps)

        m_layout.addWidget(p_card)

    has_methods = any(k != "params" and isinstance(v, dict) for k, v in methods_dict.items()) if methods_dict else False
    pending_new = getattr(self, "_pending_new_method", False)

    if is_base and not has_methods and not pending_new:
        note_row = QHBoxLayout()
        note_row.addWidget(label("No default method configured for this group.", "muted"))
        note_row.addStretch()
        m_layout.addLayout(note_row)
    elif not is_base and not has_methods and not pending_new:
        pass  # The Methods header already provides the add-method action.
    else:
        # Sub-cards for each method wrapped in MethodCardsContainer for drag reordering
        method_container = MethodCardsContainer(self)

        show_ag_defs = bool(
            self.settings_manager
            and (self.settings_manager.show_method_agent_defaults or self.settings_manager.show_all_default_params)
        ) or (self.view_mode == "resolved")
        show_mo_defs = bool(
            self.settings_manager
            and (self.settings_manager.show_method_model_defaults or self.settings_manager.show_all_default_params)
        ) or (self.view_mode == "resolved")
        show_all_defs = bool(self.settings_manager and self.settings_manager.show_all_default_params) or (
            self.view_mode == "resolved"
        )

        for m_name, m_spec in methods_dict.items():
            if m_name == "params" or not isinstance(m_spec, dict):
                continue
            # Render nested algorithm overrides through the existing method
            # controls; edits still target their original YAML container.
            _, agent_inner = component_selection(m_spec.get("agent"))
            m_spec = {**agent_inner, **m_spec}
            sub_card = QFrame()
            sub_card.setObjectName("card")
            card_vbox = QVBoxLayout(sub_card)
            card_vbox.setContentsMargins(10, 10, 10, 10)
            card_vbox.setSpacing(8)

            sc_layout = ParameterForm(field_width=round(TEXT_FIELD_WIDTH * 2 / 3))
            sc_layout.setContentsMargins(0, 0, 0, 0)
            sc_layout.setVerticalSpacing(8)

            title_row = QHBoxLayout()
            title_row.setSpacing(6)

            btn_grip = MethodDragHandle(m_name, sub_card)
            title_row.addWidget(btn_grip)

            enabled_toggle = ToggleSlider(checked=bool(m_spec.get("enabled", True)), parent=sub_card)
            enabled_toggle.setToolTip("Include this method when running and validating the experiment")
            enabled_toggle.setAccessibleName(f"Run and validate method {m_name}")
            enabled_toggle.toggled.connect(lambda enabled, mn=m_name: self._on_method_enabled_toggled(mn, enabled))
            title_row.addWidget(enabled_toggle)

            m_title = TitleLabel(m_name)
            m_title.setObjectName("cardTitle")
            m_title.setToolTip(f"{m_name}\n(Click to rename)")
            title_row.addWidget(m_title)

            m_edit = InlineTitleEditor(m_name)
            m_edit.setObjectName("methodEditor")
            m_edit.setStyleSheet("font-size: 16px; font-weight: 600; padding: 2px 6px; border-radius: 4px;")
            m_edit.hide()
            title_row.addWidget(m_edit)

            cand_train = get_training_candidate_hyperparameters(set(m_spec.keys()))
            selected_agent, _ = component_selection(m_spec.get("agent"))
            agent_defaults = tree.agent_hyperparameters(selected_agent) if tree and selected_agent else {}
            if agent_defaults:
                cand_train = [
                    key for key in cand_train if key not in ("lr", "batch_size", "gamma") or key in agent_defaults
                ]
            btn_add_tp = None
            if cand_train:
                btn_add_tp = _make_param_adder_button(
                    "Add training parameter",
                    cand_train,
                    lambda p, mn=m_name: self._add_method_param(mn, p),
                    parent_widget=sub_card,
                )
                title_row.addWidget(btn_add_tp)

            committing_m_rename = [False]

            def _start_m_rename(lbl=m_title, edt=m_edit, name=m_name):
                committing_m_rename[0] = False
                lbl.hide()
                edt.setText(name)
                edt.show()
                edt.setFocus()
                edt.selectAll()

            def _cancel_m_rename(lbl=m_title, edt=m_edit):
                committing_m_rename[0] = True
                edt.hide()
                lbl.show()

            def _commit_m_rename(lbl=m_title, edt=m_edit, name=m_name):
                if committing_m_rename[0]:
                    return
                committing_m_rename[0] = True
                new_val = edt.text().strip()
                if not new_val or new_val == name:
                    edt.hide()
                    lbl.show()
                    return
                existing_keys = set(self.raw_data.get("methods", {}).keys()) - {"params", name}
                err = self._validate_method_name(new_val, existing_keys)
                if err:
                    QMessageBox.warning(self, "Invalid Method Name", err)
                    edt.hide()
                    lbl.show()
                    return
                self.rename_method(name, new_val)

            m_title.clicked.connect(_start_m_rename)
            m_title.double_clicked.connect(_start_m_rename)
            m_edit.returnPressed.connect(_commit_m_rename)
            m_edit.escape_pressed.connect(_cancel_m_rename)
            m_edit.editingFinished.connect(_commit_m_rename)

            title_row.addStretch()

            if not is_base:
                btn_dup = _make_subtle_tool_button("copy", f"Duplicate method '{m_name}'", size=24, icon_size=14)
                btn_dup.clicked.connect(lambda _, mn=m_name: self.duplicate_method(mn))
                title_row.addWidget(btn_dup)
            btn_remove = _make_subtle_tool_button(
                "trash", f"Remove method '{m_name}' from this experiment", size=24, icon_size=14
            )
            btn_remove.clicked.connect(lambda _, mn=m_name: self.remove_method(mn))
            title_row.addWidget(btn_remove)
            sc_layout.addRow(title_row)

            # --- AGENT SECTION ---
            agent_raw = m_spec.get("agent")
            agent_val, _ = component_selection(agent_raw)
            agent_val = agent_val or ""
            rules = tree.paradigms.get(paradigm) if tree else None
            # This is a contract of the paradigm, not a proxy for whether
            # a compatible plugin happens to be installed today.
            paradigm_has_agents = bool(rules and rules.allows_algorithm)
            txt_agent = None
            if not paradigm_has_agents and not agent_val:
                pass
            elif tree and tree.paradigms.get(paradigm):
                permitted = [a.name for a in tree.agents_for(paradigm)]
                txt_agent = ComboBox()
                txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                txt_agent.setPlaceholderText("Select algorithm...")
                txt_agent.addItems(permitted)
                if agent_val:
                    if agent_val not in permitted:
                        txt_agent.insertItem(0, agent_val)
                        is_missing = not (
                            tree
                            and (
                                agent_val in tree.agents
                                or any(a.name == agent_val or a.algorithm == agent_val for a in tree.agents.values())
                                or (tree.config_root / "agent" / f"{agent_val}.yaml").is_file()
                            )
                        )
                        if is_missing:
                            txt_agent.setToolTip(
                                f"Missing component: Algorithm '{agent_val}' is not installed in in/config/agent/. Install it via Component Hub."
                            )
                        else:
                            txt_agent.setToolTip(
                                f"'{agent_val}' is not permitted by {paradigm} "
                                f"(allowed: {', '.join(permitted) or 'none declared'})"
                            )
                    else:
                        txt_agent.setToolTip(f"Agents permitted by {paradigm}")
                    txt_agent.setCurrentText(agent_val)
                else:
                    txt_agent.setCurrentIndex(-1)
                    txt_agent.setToolTip(f"Select an agent permitted by {paradigm}")

                def _on_agent_changed(v, mn=m_name, cur_ag=agent_val):
                    if self._block_updates or v == cur_ag:
                        return
                    ms = dict(self.raw_data.get("methods", {}).get(mn, {}))
                    if v:
                        previous = ms.get("agent")
                        _, overrides = component_selection(previous)
                        ms["agent"] = {"name": v, **overrides} if isinstance(previous, dict) else v
                    else:
                        ms.pop("agent", None)
                    self.document.set(("methods", mn), ms)
                    self._render_boxes()

                txt_agent.currentTextChanged.connect(_on_agent_changed)
            else:
                txt_agent = QLineEdit(agent_val)
                txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                txt_agent.setPlaceholderText("Algorithm...")
                txt_agent.textChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "agent", v))
                txt_agent.editingFinished.connect(self._render_boxes)

            agent_keys_present = set()
            if txt_agent is not None:
                txt_agent.setObjectName(f"methodAgent_{m_name}")
                agent_rows = []
                btn_add_ap = None
                if agent_val:
                    cand_agent = get_agent_candidate_hyperparameters(agent_val, tree, set(m_spec.keys()))
                    btn_add_ap = _make_param_adder_button(
                        f"Add {agent_val or 'agent'} parameter",
                        cand_agent,
                        lambda p, mn=m_name: self._add_method_param(mn, p),
                        parent_widget=sub_card,
                    )

                    agent_defaults = dict(tree.agent_hyperparameters(agent_val)) if (tree and agent_val) else {}
                    agent_param_keys = (
                        set(agent_defaults.keys())
                        - set(TRAINING_HYPERPARAMETERS)
                        - {"algorithm", "name", "type", "enabled"}
                    )

                    # 1. Overridden keys in m_spec
                    _m_raw = m_spec.get("model", "")
                    _m_name, _ = component_selection(_m_raw)
                    known_model_params = (
                        set(tree.model_parameter_keys(_m_name)) if (tree and _m_name in tree.models) else set()
                    )
                    for ak, av in list(m_spec.items()):
                        if ak in ("agent", "model", "base") or ak in known_model_params:
                            continue
                        if ak in agent_param_keys or (agent_val and ak.startswith(f"{agent_val}_")):
                            agent_keys_present.add(ak)
                            row_ap = ParameterRow()
                            if isinstance(av, bool):
                                cb_ap = ToggleSlider(checked=av)
                                cb_ap.toggled.connect(
                                    lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v)
                                )
                                row_ap.addWidget(cb_ap)
                            elif isinstance(av, (int, float)):
                                spin_ap = number_field(av)
                                spin_ap.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                spin_ap.valueChanged.connect(
                                    lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v)
                                )
                                row_ap.addWidget(spin_ap)
                            else:
                                param_schema = tree.agent_parameter_schema(agent_val, (ak,)) if tree else {}
                                row_ap.addWidget(
                                    self._schema_widget(
                                        av,
                                        param_schema,
                                        lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v),
                                    )
                                )
                            btn_del_ap = _make_subtle_tool_button("close", f"Remove '{ak}'", size=20, icon_size=12)
                            btn_del_ap.clicked.connect(lambda _, mn=m_name, pk=ak: self._remove_method_param(mn, pk))
                            row_ap.addWidget(btn_del_ap)
                            _reveal_remove_on_row_hover(row_ap, btn_del_ap)
                            self._add_sweep_toggle(row_ap, m_name, f"agent.{ak}", av)
                            agent_rows.append((str(ak), row_ap))

                    # 2. Defaults if show_ag_defs enabled
                    if show_ag_defs:
                        for ak in sorted(agent_param_keys):
                            if ak not in agent_keys_present:
                                def_val = agent_defaults.get(ak)
                                if def_val is None:
                                    continue
                                row_ap = ParameterRow()
                                if isinstance(def_val, bool):
                                    cb_ap = ToggleSlider(checked=def_val)
                                    cb_ap.toggled.connect(
                                        lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v)
                                    )
                                    row_ap.addWidget(cb_ap)
                                elif isinstance(def_val, (int, float)):
                                    spin_ap = number_field(def_val)
                                    spin_ap.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                    spin_ap.valueChanged.connect(
                                        lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v)
                                    )
                                    row_ap.addWidget(spin_ap)
                                else:
                                    param_schema = tree.agent_parameter_schema(agent_val, (ak,)) if tree else {}
                                    row_ap.addWidget(
                                        self._schema_widget(
                                            def_val,
                                            param_schema,
                                            lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v),
                                        )
                                    )
                                lbl_def = QLabel("(default)")
                                lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                                row_ap.addWidget(lbl_def)
                                agent_rows.append((str(ak), row_ap))

                row_ag = ParameterRow()
                row_ag.addWidget(txt_agent)
                if btn_add_ap:
                    row_ag.addWidget(btn_add_ap)
                    _reveal_remove_on_row_hover(row_ag, btn_add_ap)
                row_ag.addStretch()
                sc_layout.add_parameter(("methods", m_name, "agent"), "Algorithm", row_ag)

                if agent_rows:
                    agent_block = IndentedHierarchyBlock(level=1)
                    for r_lbl, r_box in agent_rows:
                        r_box.addStretch()
                        agent_block.form_layout.add_parameter(
                            ("methods", m_name, "agent", r_lbl),
                            r_lbl,
                            r_box,
                            children=getattr(r_box, "_sweep_block", None),
                        )
                    sc_layout.add_children(("methods", m_name, "agent"), agent_block)

            # --- MODEL SECTION ---
            model_raw = m_spec.get("model", "")
            model_name, model_inner = component_selection(model_raw)
            model_name = model_name or ""

            # Older configs may use a model's short-name lookup alias.
            # Display and newly saved selections should use the unique
            # path of the YAML that defines it.
            if tree and model_name in tree.models:
                model_name = tree.models[model_name].name

            if tree and tree.models:
                txt_model = ComboBox()
                txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                txt_model.setPlaceholderText("Select model...")
                known = tree.model_names
                txt_model.addItems(known)
                if model_name:
                    if model_name not in known:
                        txt_model.insertItem(0, model_name)
                        mo_stem = model_name.split("/")[-1]
                        is_missing = not (
                            tree
                            and (
                                model_name in tree.models
                                or mo_stem in tree.models
                                or (tree.config_root / "model" / f"{model_name}.yaml").is_file()
                                or (tree.config_root / "model" / "basics" / f"{mo_stem}.yaml").is_file()
                            )
                        )
                        if is_missing:
                            txt_model.setToolTip(
                                f"Missing component: Model '{model_name}' is not installed in in/config/model/. Install it via Component Hub."
                            )
                        else:
                            txt_model.setToolTip("Architectures defined in in/config/model/")
                    else:
                        txt_model.setToolTip("Architectures defined in in/config/model/")
                    txt_model.setCurrentText(model_name)

                def _on_model_changed(v, mn=m_name, cur_mn=model_name):
                    if self._block_updates or v == cur_mn:
                        return
                    ms = dict(self.raw_data.get("methods", {}).get(mn, {}))
                    if not v:
                        ms.pop("model", None)
                    else:
                        ms["model"] = v
                        old_params = tree.model_parameter_keys(cur_mn) if (tree and cur_mn in tree.models) else []
                        for key in old_params:
                            ms.pop(key, None)
                    self.document.set(("methods", mn), ms)
                    self._render_boxes()

                txt_model.currentTextChanged.connect(_on_model_changed)
            else:
                txt_model = QLineEdit(model_name)
                txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                txt_model.setPlaceholderText("Model...")
                txt_model.textChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "model", v))
                txt_model.editingFinished.connect(self._render_boxes)

            txt_model.setObjectName(f"methodModel_{m_name}")
            model_keys_present = set()
            model_rows = []
            btn_add_mp = None
            if model_name:
                model_defs = dict(tree.model_parameters(model_name)) if tree else {}
                model_keys = set(tree.model_parameter_keys(model_name)) if tree else set(model_defs)
                model_keys.update(model_inner.keys() if isinstance(model_inner, dict) else ())
                model_keys -= {"architecture", "type", "lightning_module", "name"}

                model_values = {}
                for key in model_keys:
                    if isinstance(model_inner, dict) and key in model_inner:
                        model_values[key] = model_inner[key]
                    elif key in m_spec:
                        model_values[key] = m_spec[key]

                model_keys_present = set(model_values)
                model_rows = self._render_method_model_parameters(
                    m_name,
                    model_name,
                    model_values,
                    model_defs,
                    show_defaults=show_mo_defs,
                )
                candidate_keys = get_model_candidate_hyperparameters(model_name, tree, model_keys_present)
                if candidate_keys:
                    btn_add_mp = _make_param_adder_button(
                        f"Add {model_name} parameter",
                        candidate_keys,
                        lambda key, mn=m_name, model=model_name: self._add_method_model_parameter(mn, model, (key,)),
                        parent_widget=sub_card,
                    )

                row_mo = ParameterRow()
                row_mo.addWidget(txt_model)
                if btn_add_mp:
                    row_mo.addWidget(btn_add_mp)
                    _reveal_remove_on_row_hover(row_mo, btn_add_mp)
                slot_widget = self._model_row_slot(m_name, model_name)
                if slot_widget is not None:
                    row_mo.addWidget(slot_widget)
                row_mo.addStretch()
                sc_layout.add_parameter(("methods", m_name, "model"), "Model", row_mo)

                if model_rows:
                    model_block = IndentedHierarchyBlock(level=1)
                    model_rows.mount(model_block.form_layout)
                    sc_layout.add_children(("methods", m_name, "model"), model_block)
            else:
                sc_layout.add_parameter(("methods", m_name, "model"), "Model", _parameter_row(txt_model))

            # --- TRAINING PARAMETERS SECTION (UNINDENTED, NO HEADER) ---
            train_rows = []
            u_params = self.raw_data.get("methods", {}).get("params", {})
            base_def_name, base_def_spec = self._get_group_default_method()
            base_def_spec = base_def_spec or {}

            from ..editors import value_editor

            training = [
                ("lr", "Learning Rate (lr)", 0.0003, {"type": "number", "exclusiveMinimum": 0}),
                ("batch_size", "Batch Size", 64, {"type": "integer", "minimum": 1}),
                ("gamma", "Discount Factor (γ)", 0.99, {"type": "number", "minimum": 0, "maximum": 1}),
            ]
            for key, title, default, schema in training:
                present = key in m_spec
                if not present and not show_all_defs:
                    continue
                value = m_spec.get(key, u_params.get(key, base_def_spec.get(key, default)))
                field = value_editor(
                    value, schema, lambda value, mn=m_name, k=key: self._on_method_param_edited(mn, k, value)
                )
                remove = None
                if present:
                    remove = _make_subtle_tool_button("close", f"Remove {key} parameter", size=20, icon_size=12)
                    remove.clicked.connect(lambda _, mn=m_name, k=key: self._remove_method_param(mn, k))
                row = _parameter_row(field, remove=remove)
                if present:
                    self._add_sweep_toggle(row, m_name, f"agent.{key}", value)
                else:
                    row.insertWidget(row.count() - 1, label("(default)", "muted"))
                train_rows.append((title, row))

            consumed_keys = {
                "agent",
                "model",
                "base",
                "enabled",
                "lr",
                "batch_size",
                "gamma",
                "data_module",
                "symbolic_state_loader",
            }
            consumed_keys.update(agent_keys_present)
            consumed_keys.update(model_keys_present)

            for extra_k, extra_v in list(m_spec.items()):
                if extra_k in consumed_keys:
                    continue
                if isinstance(extra_v, (int, float, str, bool)):
                    row_extra = ParameterRow()
                    if isinstance(extra_v, bool):
                        slider_extra = ToggleSlider(checked=extra_v)
                        slider_extra.toggled.connect(
                            lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                        )
                        row_extra.addWidget(slider_extra)
                    elif isinstance(extra_v, (int, float)):
                        spin_extra = number_field(extra_v)
                        spin_extra.setMaximumWidth(NUMBER_FIELD_WIDTH)
                        spin_extra.valueChanged.connect(
                            lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                        )
                        row_extra.addWidget(spin_extra)
                    else:
                        line_extra = QLineEdit(str(extra_v if extra_v is not None else ""))
                        line_extra.setMaximumWidth(TEXT_FIELD_WIDTH)
                        line_extra.textChanged.connect(
                            lambda v, mn=m_name, ek=extra_k: self._on_method_param_edited(mn, ek, v)
                        )
                        row_extra.addWidget(line_extra)
                    btn_del = _make_subtle_tool_button("close", f"Remove '{extra_k}'", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name, ek=extra_k: self._remove_method_param(mn, ek))
                    row_extra.addWidget(btn_del)
                    _reveal_remove_on_row_hover(row_extra, btn_del)
                    self._add_sweep_toggle(row_extra, m_name, f"agent.{extra_k}", extra_v)
                    train_rows.append((str(extra_k), row_extra))

            if train_rows:
                for t_lbl, t_row in train_rows:
                    t_row.addStretch()
                    sc_layout.add_parameter(
                        ("methods", m_name, "training", t_lbl),
                        t_lbl,
                        t_row,
                        children=getattr(t_row, "_sweep_block", None),
                    )

            self._render_method_sweeps(sc_layout, m_name, m_spec)

            card_vbox.addLayout(sc_layout)
            card_vbox.addStretch()

            method_container.add_method_card(m_name, sub_card)

        if pending_new:
            pending_card = QFrame()
            pending_card.setObjectName("card")
            pc_vbox = QVBoxLayout(pending_card)
            pc_vbox.setContentsMargins(10, 10, 10, 10)
            pc_vbox.setSpacing(8)

            pc_layout = ParameterForm(field_width=round(TEXT_FIELD_WIDTH * 2 / 3))
            pc_layout.setContentsMargins(0, 0, 0, 0)
            pc_layout.setVerticalSpacing(8)

            p_title_row = QHBoxLayout()
            p_title_row.setSpacing(6)

            p_edit = InlineTitleEditor()
            p_edit.setObjectName("newMethodEditor")
            def_name, _ = self._get_group_default_method()
            if def_name:
                p_edit.setPlaceholderText(f"Method name (e.g. {def_name})...")
            else:
                p_edit.setPlaceholderText("Method name...")
            p_edit.setStyleSheet("font-size: 16px; font-weight: 600; padding: 2px 6px; border-radius: 4px;")
            p_title_row.addWidget(p_edit)
            p_title_row.addStretch()

            btn_cancel_p = _make_subtle_tool_button("close", "Cancel creating method", size=24, icon_size=14)
            btn_cancel_p.clicked.connect(self._cancel_new_method)
            p_title_row.addWidget(btn_cancel_p)
            pc_layout.addRow(p_title_row)

            # Agent Algorithm row
            if tree and tree.paradigms.get(paradigm):
                p_permitted = [a.name for a in tree.agents_for(paradigm)]
                p_rules = tree.paradigms[paradigm]
                if p_permitted:
                    p_txt_agent = ComboBox()
                    p_txt_agent.setEditable(True)
                    p_txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                    p_txt_agent.setPlaceholderText("Select or enter algorithm...")
                    p_txt_agent.addItems(p_permitted)
                    init_agent = self._pending_new_method_spec.get("agent")
                    if init_agent:
                        p_txt_agent.setCurrentText(str(init_agent))
                    else:
                        p_txt_agent.setCurrentIndex(-1)
                    p_txt_agent.currentTextChanged.connect(
                        lambda v: (
                            self._pending_new_method_spec.__setitem__("agent", v.strip())
                            if v.strip()
                            else self._pending_new_method_spec.pop("agent", None)
                        )
                    )
                    pc_layout.addRow("Algorithm", p_txt_agent)
                elif p_rules.allows_algorithm:
                    p_txt_agent = QLineEdit()
                    p_txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                    p_txt_agent.setPlaceholderText("Algorithm component ID...")
                    init_agent = self._pending_new_method_spec.get("agent")
                    if init_agent:
                        p_txt_agent.setText(str(init_agent))
                    p_txt_agent.textChanged.connect(
                        lambda v: (
                            self._pending_new_method_spec.__setitem__("agent", v.strip())
                            if v.strip()
                            else self._pending_new_method_spec.pop("agent", None)
                        )
                    )
                    pc_layout.addRow("Algorithm", p_txt_agent)

            # Model row — required for all paradigms
            if tree and tree.models:
                p_txt_model = ComboBox()
                p_txt_model.setEditable(True)
                p_txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                p_txt_model.setPlaceholderText("Select or enter model...")
                p_txt_model.addItems(tree.model_names)
                init_model = self._pending_new_method_spec.get("model")
                if init_model:
                    p_txt_model.setCurrentText(str(init_model))
                else:
                    p_txt_model.setCurrentIndex(-1)
                p_txt_model.currentTextChanged.connect(
                    lambda v: (
                        self._pending_new_method_spec.__setitem__("model", v.strip())
                        if v.strip()
                        else self._pending_new_method_spec.pop("model", None)
                    )
                )
                pc_layout.addRow("Model", p_txt_model)
            else:
                p_txt_model = QLineEdit()
                p_txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                p_txt_model.setPlaceholderText("Model component ID...")
                init_model = self._pending_new_method_spec.get("model")
                if init_model:
                    p_txt_model.setText(str(init_model))
                p_txt_model.textChanged.connect(
                    lambda v: (
                        self._pending_new_method_spec.__setitem__("model", v.strip())
                        if v.strip()
                        else self._pending_new_method_spec.pop("model", None)
                    )
                )
                pc_layout.addRow("Model", p_txt_model)

            committing_p = [False]

            def _commit_p():
                if committing_p[0]:
                    return
                name = p_edit.text().strip()
                if not name:
                    committing_p[0] = True
                    self._cancel_new_method()
                    return

                existing = set(self.raw_data.get("methods", {}).keys()) - {"params"}
                err = self._validate_method_name(name, existing)
                if err:
                    QMessageBox.warning(self, "Invalid Method Name", err)
                    p_edit.setFocus()
                    p_edit.selectAll()
                    return

                committing_p[0] = True
                self.document.set(("methods", name), dict(getattr(self, "_pending_new_method_spec", {})))
                self._pending_new_method = False
                self._pending_new_method_spec = {}
                self._mark_dirty()
                self._render_boxes()

            def _cancel_p():
                if committing_p[0]:
                    return
                committing_p[0] = True
                self._cancel_new_method()

            p_edit.returnPressed.connect(_commit_p)
            p_edit.escape_pressed.connect(_cancel_p)

            def _on_p_blur():
                if committing_p[0]:
                    return
                if not p_edit.text().strip():
                    _cancel_p()

            p_edit.editingFinished.connect(_on_p_blur)

            pc_vbox.addLayout(pc_layout)
            pc_vbox.addStretch()
            method_container.add_method_card("", pending_card)
            QTimer.singleShot(0, p_edit.setFocus)

        m_layout.addWidget(method_container)

    box_methods.add_layout(m_layout)
    self.boxes_layout.addWidget(box_methods)
