"""Configuration section composition using shared parameter controls."""

import copy

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
    QWidget,
)

from src.app.telemetry.selection import SECTIONS as METRIC_SECTIONS

from ...metric_picker import MetricPicker, logged_metrics
from ...widgets import (
    ToggleSlider,
    label,
)
from ..catalogs import _monitor_settings, _recorded_metrics, config_tree
from ..controls import (
    HIERARCHY_INDENT,
    ConfigBox,
    IndentedHierarchyBlock,
    ParameterForm,
    _collapsible_parameter_label,
    _make_param_adder_button,
    _make_subtle_tool_button,
    _parameter_header,
    _parameter_row,
)
from ..layout import _two_column_parameter_layout


def _render_monitors_box(self, data, resolved):
    """Optional stream selection, with group controls and a preserved bypass."""
    from src.app.core.provider_catalog import get_catalog
    from src.app.pipeline.config import deep_merge
    from src.app.telemetry.monitors import MonitorSelection, available_monitors, monitors_from_config
    from src.app.telemetry.selection import selection_from_config
    from src.app.workspace import WorkspacePaths

    effective = deep_merge(resolved, data)
    settings = _monitor_settings(effective)
    if settings is None:
        if effective.get("monitors") is None:
            return
        settings = {"enabled": True, "disabled": []}
    effective.setdefault("group", self._group_name())
    box = ConfigBox(section="monitors")
    box.setProperty("parameterColumns", 2)
    remove = _make_subtle_tool_button("trash", "Remove monitors and stream all metrics", size=24, icon_size=14)
    remove.clicked.connect(lambda: self.remove_block("monitors"))
    add = None
    try:
        tree = self.tree
        paths = tree.workspace if tree else WorkspacePaths.for_process()
        catalog = get_catalog(paths)
        components = sorted(
            {cap.id for package in catalog.packages for cap in package.capabilities if cap.kind == "monitor"}
        )
        add = _make_param_adder_button(
            "Add monitor", components, self._add_standalone_monitor, allow_custom=False, parent_widget=self
        )
        monitors_from_config(effective)  # validate even when the box is bypassed
        selection = MonitorSelection(settings.get("disabled", []))
        recorded = selection_from_config(effective)
        discovery_errors = []
        specs = available_monitors(effective, catalog=catalog, include_shared=False, errors=discovery_errors)
    except (ValueError, TypeError, KeyError, OSError) as exc:
        box.set_header("Monitors", add=add, remove=remove)
        message = label(str(exc), "configError")
        message.setWordWrap(True)
        box.add_widget(message)
        self.boxes_layout.addWidget(box)
        return

    def current_settings():
        return deep_merge(settings, _monitor_settings(self.raw_data) or {})

    def write(key, value):
        if self._block_updates:
            return
        updated = copy.deepcopy(current_settings())
        updated[key] = value
        self.document.set(("telemetry", "monitors"), updated)
        QTimer.singleShot(0, self._render_boxes)

    names = [spec.name for spec in specs]

    def change(targets, checked):
        updated = MonitorSelection(current_settings().get("disabled", [])).with_toggled(targets, checked, known=names)
        write("disabled", list(updated.disabled))

    def recordable(spec):
        return recorded.keeps_section(spec.name) if spec.name.endswith("/") else recorded.keeps(spec.name)

    enabled = ToggleSlider(checked=settings.get("enabled", True))
    enabled.setAccessibleName("Use monitor selections")
    enabled.setObjectName("monitorsEnabled")
    enabled.toggled.connect(lambda checked: write("enabled", checked))
    self.field_widgets["telemetry.monitors.enabled"] = enabled
    removed = MonitorSelection(settings.get("removed", []))
    configured = effective.get("monitors") or {}
    groups = {}
    for spec in specs:
        group = "" if spec.source.startswith("Monitor · ") else spec.group
        groups.setdefault((spec.source, group), []).append(spec)
    for identity in configured:
        if configured[identity] != {"enabled": False}:
            groups.setdefault(("Monitor · " + identity, ""), [])
    for (source, title), members in groups.items():
        removed_names = [spec.name for spec in members if not removed.keeps(spec.name)]
        namespace = f"interp/{source.removeprefix('Monitor · ')}/" if source.startswith("Monitor · ") else None
        if removed_names or namespace and not removed.keeps(namespace):
            action = add.menu().addAction("Restore " + (title or source))

            def restore(items=members, prefix=namespace):
                updated = MonitorSelection(current_settings().get("removed", [])).with_toggled(
                    [prefix] if prefix else [spec.name for spec in items], True, known=names
                )
                write("removed", list(updated.disabled))

            action.triggered.connect(lambda _, callback=restore: callback())
    add.setEnabled(enabled.isChecked() and not add.menu().isEmpty())
    box.set_header("Monitors", leading=enabled, add=add, remove=remove)
    body = box.add_parameter_body(enabled=enabled.isChecked())
    if discovery_errors:
        message = label("\n".join(discovery_errors), "configError")
        message.setWordWrap(True)
        body.layout().addWidget(message)
    form = ParameterForm()
    for (source, title), members in groups.items():
        identity = source.removeprefix("Monitor · ") if source.startswith("Monitor · ") else None
        instance = configured.get(identity) if identity else None
        prefix = f"interp/{identity}/" if identity else None
        members = [spec for spec in members if removed.keeps(spec.name)]
        if not members and (instance is None or not removed.keeps(prefix)):
            continue
        group_toggle = ToggleSlider(
            checked=any(selection.keeps(spec.name) and recordable(spec) for spec in members)
            if members
            else selection.keeps(prefix)
        )
        group_toggle.setProperty("monitorGroup", title or source)
        group_toggle.setProperty("monitorGroupSource", source)
        group_toggle.setAccessibleName("Stream " + (title or source))
        group_toggle.setEnabled(not members or any(recordable(spec) for spec in members))
        group_toggle.setToolTip(source)
        group_toggle.toggled.connect(
            lambda checked, items=members, namespace=prefix: change(
                [spec.name for spec in items] or [namespace], checked
            )
        )
        children = IndentedHierarchyBlock()
        group_key = (source, title)
        expanded = getattr(self, "_monitor_group_expanded", {})
        self._monitor_group_expanded = expanded
        caption = _collapsible_parameter_label(
            title or source,
            children,
            expanded=expanded.get(group_key, True),
            on_toggled=lambda checked, key=group_key: expanded.__setitem__(key, checked),
        )
        delete = _make_subtle_tool_button("close", "Remove monitor set " + (title or source), size=20, icon_size=12)
        if identity:
            delete.clicked.connect(lambda _, name=identity: self._remove_standalone_monitor(name))
        else:
            delete.clicked.connect(
                lambda _, items=members: write(
                    "removed",
                    list(dict.fromkeys([*current_settings().get("removed", []), *(spec.name for spec in items)])),
                )
            )
        group_identity = ("telemetry", "monitors", source, title)
        form.add_parameter(group_identity, caption, _parameter_row(group_toggle, remove=delete), children=children)
        if isinstance(instance, dict):
            settings_header = _parameter_header(
                "Settings", heading=False, add=self._dict_parameter_adder(("monitors", identity), instance)
            )
            settings_block = IndentedHierarchyBlock(level=2)
            self._render_generic_dict_contents(settings_block, instance, ("monitors", identity), level=3)
            group = children.form_layout.add_parameter(
                ("monitors", identity), None, settings_header, children=settings_block
            )
            group.caption = settings_header.caption
            if members:
                children.form_layout.add_parameter(
                    ("monitors", identity, "outputs"), None, label("Streamed outputs", "muted")
                )
        prefixes = {}
        for spec in members:
            parents = [name for name in prefixes if spec.name.startswith(name)]
            target_form, level = prefixes[max(parents, key=len)] if parents else (children.form_layout, 1)
            toggle = ToggleSlider(checked=selection.keeps(spec.name) and recordable(spec))
            toggle.setAccessibleName(spec.name)
            parent_disabled = any(
                pattern != spec.name and (pattern == "*" or (pattern.endswith("/") and spec.name.startswith(pattern)))
                for pattern in selection.disabled
            )
            toggle.setEnabled(not parent_disabled and recordable(spec))
            toggle.setToolTip(
                spec.name
                + ("\n" + spec.description if spec.description else "")
                + ("\nExcluded by Recorded Metrics." if not recordable(spec) else "")
            )
            toggle.toggled.connect(lambda checked, name=spec.name: change([name], checked))
            delete = None
            if spec.name in settings.get("disabled", []):
                delete = _make_subtle_tool_button("close", "Remove exclusion for " + spec.name, size=20, icon_size=12)
                delete.clicked.connect(lambda _, name=spec.name: change([name], True))
            monitor_path = ("telemetry", "monitors", spec.name)
            target_form.add_parameter(monitor_path, spec.label, _parameter_row(toggle, remove=delete))
            if spec.name.endswith("/") and any(
                other.name != spec.name and other.name.startswith(spec.name) for other in members
            ):
                nested = IndentedHierarchyBlock(level=level + 1)
                target_form.add_children(monitor_path, nested)
                prefixes[spec.name] = (nested.form_layout, level + 1)
            self.field_widgets["telemetry.monitors." + spec.name] = toggle
    for name in selection.disabled:
        if name not in names:
            delete = _make_subtle_tool_button("close", "Remove exclusion for " + name, size=20, icon_size=12)
            delete.clicked.connect(lambda _, key=name: change([key], True))
            form.add_parameter(
                ("telemetry", "monitors", "excluded", name),
                "Saved exclusion",
                _parameter_row(label(name + " (unavailable)", "muted"), remove=delete),
            )
    for name in removed.disabled:
        if name not in names and not any(name == f"interp/{identity}/" for identity in configured):
            delete = _make_subtle_tool_button("close", "Restore " + name, size=20, icon_size=12)
            delete.clicked.connect(
                lambda _, key=name: write(
                    "removed",
                    list(
                        MonitorSelection(current_settings().get("removed", []))
                        .with_toggled([key], True, known=names)
                        .disabled
                    ),
                )
            )
            form.add_parameter(
                ("telemetry", "monitors", "removed", name),
                "Removed monitor",
                _parameter_row(label(name + " (unavailable)", "muted"), remove=delete),
            )
    body.layout().addLayout(_two_column_parameter_layout(form))
    self.boxes_layout.addWidget(box)


def _render_metrics_box(self, data, resolved):
    """Keep the specialized metric picker, using shared parameter actions."""
    own = _recorded_metrics(data)
    source = own if own is not None else (_recorded_metrics(resolved) if self.view_mode == "resolved" else None)
    if source is None:
        return
    experiment = self.current_path.stem if self.current_path else ""
    from src.app.workspace import WorkspacePaths

    paths = self.tree.workspace if self.tree else WorkspacePaths.for_process()
    known = logged_metrics(paths.results / "logs", str(self._group_name() or ""), experiment)
    known += [name for name in source if not name.endswith("/") and name not in known]

    def add_metric(name):
        selected = picker.selection()
        if name not in selected:
            if "/" in name and not name.endswith("/"):
                selected = [item for item in selected if item != name.split("/", 1)[0] + "/"]
            selected.append(name)
        self._on_recorded_metrics_changed(selected)
        QTimer.singleShot(0, self._render_boxes)

    def metric_row(name, checkbox, section):
        add = None
        if section:
            add = _make_param_adder_button(
                "Add " + name + " metric",
                [item for item in known if item.startswith(name) and item not in source],
                lambda key, prefix=name: add_metric(key if "/" in key else prefix + key),
                parent_widget=self,
            )
        delete = _make_subtle_tool_button("close", "Remove " + name, size=20, icon_size=12)

        def remove_metric():
            was_checked = checkbox.isChecked()
            checkbox.setChecked(False)
            if section and not was_checked:
                picker._section_toggled(name[:-1])

        delete.clicked.connect(remove_metric)
        wrapper = QWidget()
        wrapper.setObjectName("metricParameterRow")
        wrapper.setStyleSheet("QWidget#metricParameterRow { background: transparent; }")
        wrapper.setLayout(_parameter_row(checkbox, add=add, remove=delete))
        return wrapper

    box = ConfigBox()
    remove = _make_subtle_tool_button("trash", "Record every metric again", size=24, icon_size=14)
    remove.clicked.connect(lambda: self.remove_block("metrics"))
    candidates = sorted(set(known) | {section + "/" for section in METRIC_SECTIONS})
    box.set_header(
        "Recorded Metrics",
        add=_make_param_adder_button(
            "Add recorded metric", [item for item in candidates if item not in source], add_metric, parent_widget=self
        ),
        remove=remove if own is not None else None,
    )
    picker = MetricPicker(source, sorted(known), row_factory=metric_row, indent=HIERARCHY_INDENT + 12)
    picker.changed.connect(self._on_recorded_metrics_changed)
    self.field_widgets["telemetry.metrics"] = picker
    box.add_widget(picker)
    self.boxes_layout.addWidget(box)
