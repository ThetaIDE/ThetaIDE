"""Dedicated Workflows panel for composing experiments as string diagrams in Theta-IDE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import QEvent, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from frontend.sidetabs import svg_icon
from frontend.theme import theme_color
from frontend.widgets import label
from src.app.pipeline.workflow.model import (
    PortDirection,
    PortType,
    WorkflowGraph,
    WorkflowNode,
    WorkflowString,
)
from src.app.pipeline.workflow.presets import BUILTIN_PRESETS, get_preset
from src.app.pipeline.workflow.templates import (
    make_dataset_source_node,
    make_distillation_node,
    make_feature_augmenter_node,
    make_offline_rl_node,
    make_online_rl_node,
    make_plot_evaluator_node,
    make_reward_shaper_node,
    make_supervised_node,
)

from .canvas import WorkflowCanvasView, WorkflowScene
from .diagram_items import NodeBoxItem, StringItem


BUTTON_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "text",
    (QIcon.Mode.Disabled, QIcon.State.Off): "disabled",
}
# Status badge kind -> (icon, theme role); None shows the text alone
STATUS_ICONS = {"valid": ("check", "primary"), "issues": ("warning", "focus")}


class WorkflowsPanel(QWidget):
    """Theta-IDE panel for visual experiment string diagram composition."""
    workflow_changed = pyqtSignal()

    def __init__(self, log_fn=None, parent=None):
        super().__init__(parent)
        self.log_fn = log_fn or (lambda msg: None)
        self.selected_item = None
        self.status_kind = None

        self._init_ui()
        self._apply_icons()
        # Load default preset on startup
        self.load_preset("online_vs_offline")

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(10)

        # ── Actions bar ────────────────────────────────────────────────────────
        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(8)

        # Preset switcher
        actions_bar.addWidget(label("Preset:", "muted"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItem("Select a preset…", "")
        for number, (preset_id, factory) in enumerate(BUILTIN_PRESETS.items(), 1):
            self.preset_combo.addItem(f"{number}. {factory().name}", preset_id)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        actions_bar.addWidget(self.preset_combo)

        # Add node dropdown menu
        self.btn_add_node = QPushButton("Add Node")
        self.btn_add_node.setToolTip("Add a new paradigm, task, or data transform node")
        self.btn_add_node.setMenu(self._build_add_node_menu())  # Qt draws the dropdown arrow
        actions_bar.addWidget(self.btn_add_node)

        # Validate button
        self.btn_validate = QPushButton("Validate Strings")
        self.btn_validate.setToolTip("Check type safety, required inputs, and diagram topology")
        self.btn_validate.clicked.connect(self.validate_current_diagram)
        actions_bar.addWidget(self.btn_validate)

        self.btn_zoom_fit = QPushButton("Zoom Fit")
        self.btn_zoom_fit.setToolTip("Fit all nodes nicely in view")
        self.btn_zoom_fit.clicked.connect(lambda: self.canvas_view.zoom_fit())
        actions_bar.addWidget(self.btn_zoom_fit)

        actions_bar.addStretch()

        self.status_icon = QLabel()
        self.status_icon.hide()
        actions_bar.addWidget(self.status_icon)
        self.status_badge = label("Ready", "muted")
        actions_bar.addWidget(self.status_badge)

        self.btn_load_yaml = QPushButton("Load YAML…")
        self.btn_load_yaml.clicked.connect(self.load_yaml_dialog)
        actions_bar.addWidget(self.btn_load_yaml)

        self.btn_save_yaml = QPushButton("Save YAML…")
        self.btn_save_yaml.clicked.connect(self.save_yaml_dialog)
        actions_bar.addWidget(self.btn_save_yaml)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_workflow)
        actions_bar.addWidget(self.btn_clear)

        root_layout.addLayout(actions_bar)

        # ── Main 3-Way Splitter ────────────────────────────────────────────────
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Palette (Left)
        palette_panel = self._create_palette_panel()
        self.splitter.addWidget(palette_panel)

        # 2. Canvas (Center)
        self.scene = WorkflowScene()
        self.scene.graph_changed.connect(self._on_graph_modified)
        self.scene.selection_changed_custom.connect(self._on_selection_changed)

        self.canvas_view = WorkflowCanvasView(self.scene)
        self.splitter.addWidget(self.canvas_view)

        # 3. Inspector (Right)
        self.inspector_panel = self._create_inspector_panel()
        self.splitter.addWidget(self.inspector_panel)

        # Proportions: Palette 210px, Canvas 800px, Inspector 280px
        self.splitter.setSizes([210, 800, 280])
        root_layout.addWidget(self.splitter, 1)

    # ── Palette Builder ────────────────────────────────────────────────────────

    def _create_palette_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("workflowPalette")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        layout.addWidget(label("NODE LIBRARY", "eyebrow"))

        self.palette_list = QListWidget()
        self.palette_list.setSpacing(4)
        items = [
            ("Online RL", "online_rl"),
            ("Offline RL", "offline_rl"),
            ("Supervised / Prediction", "supervised"),
            ("Model Distillation", "distillation"),
            ("V(s) Feature Augmenter", "feature_augmenter"),
            ("Potential Reward Shaper", "reward_shaper"),
            ("Static Replay Dataset", "dataset_source"),
            ("Multi-Method Plotter", "plot_evaluator"),
        ]
        for title, key in items:
            it = QListWidgetItem(title)
            it.setData(Qt.ItemDataRole.UserRole, key)
            self.palette_list.addItem(it)

        self.palette_list.itemDoubleClicked.connect(lambda item: self._spawn_from_palette(item.data(Qt.ItemDataRole.UserRole)))
        layout.addWidget(self.palette_list, 1)

        self.btn_add_selected = btn_add_selected = self._icon_button("Add to Diagram", "plus")
        btn_add_selected.clicked.connect(self._on_add_palette_clicked)
        layout.addWidget(btn_add_selected)

        hint = label("Tip: Click an output port dot and drag a wire to an input port.", "muted")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px;")
        layout.addWidget(hint)

        return panel

    def _on_add_palette_clicked(self):
        curr = self.palette_list.currentItem()
        if curr:
            self._spawn_from_palette(curr.data(Qt.ItemDataRole.UserRole))

    def _spawn_from_palette(self, key: str):
        center = self.canvas_view.mapToScene(self.canvas_view.viewport().rect().center())
        # Add slight jitter so multiple nodes don't stack exactly
        count = len(self.scene.graph.nodes)
        px = center.x() - 110 + (count % 4) * 30
        py = center.y() - 60 + (count % 4) * 30

        idx = count + 1
        if key == "online_rl":
            node = make_online_rl_node(f"online_node_{idx}", f"Online RL #{idx}", pos_x=px, pos_y=py)
        elif key == "offline_rl":
            node = make_offline_rl_node(f"offline_node_{idx}", f"Offline RL #{idx}", pos_x=px, pos_y=py)
        elif key == "supervised":
            node = make_supervised_node(f"supervised_node_{idx}", f"Supervised #{idx}", pos_x=px, pos_y=py)
        elif key == "distillation":
            node = make_distillation_node(f"distill_node_{idx}", f"Distillation #{idx}", pos_x=px, pos_y=py)
        elif key == "feature_augmenter":
            node = make_feature_augmenter_node(f"augment_node_{idx}", f"Feature Augmenter #{idx}", pos_x=px, pos_y=py)
        elif key == "reward_shaper":
            node = make_reward_shaper_node(f"shaper_node_{idx}", f"Reward Shaper #{idx}", pos_x=px, pos_y=py)
        elif key == "dataset_source":
            node = make_dataset_source_node(f"dataset_node_{idx}", f"Dataset #{idx}", pos_x=px, pos_y=py)
        elif key == "plot_evaluator":
            node = make_plot_evaluator_node(f"eval_node_{idx}", f"Plot Evaluator #{idx}", pos_x=px, pos_y=py)
        else:
            return

        self.scene.add_node_to_scene(node)
        self.log_fn(f"Added node '{node.label}' to diagram.")

    def _build_add_node_menu(self) -> QMenu:
        menu = QMenu(self)
        items = [
            ("Online RL", "online_rl"),
            ("Offline RL", "offline_rl"),
            ("Supervised / Prediction", "supervised"),
            ("Model Distillation", "distillation"),
            ("V(s) Feature Augmenter", "feature_augmenter"),
            ("Potential Reward Shaper", "reward_shaper"),
            ("Static Replay Dataset", "dataset_source"),
            ("Multi-Method Plotter", "plot_evaluator"),
        ]
        for title, key in items:
            menu.addAction(title, lambda k=key: self._spawn_from_palette(k))
        return menu

    # ── Icons and status ───────────────────────────────────────────────────────

    def changeEvent(self, event):
        # A theme switch sets the application palette, which reaches every widget as a PaletteChange
        super().changeEvent(event)
        if event.type() == QEvent.Type.PaletteChange and hasattr(self, "btn_clear"):
            self._apply_icons()

    def _apply_icons(self):
        """Toolbar icons in the current theme's colours."""
        for button, name in ((self.btn_add_node, "plus"), (self.btn_validate, "check"),
                             (self.btn_zoom_fit, "zoom_fit"), (self.btn_load_yaml, "folder"),
                             (self.btn_save_yaml, "save"), (self.btn_clear, "clear"),
                             (self.btn_add_selected, "plus")):
            button.setIcon(svg_icon(name, BUTTON_ICON_COLORS, size=16))
            button.setIconSize(QSize(16, 16))
        self._show_status_icon()

    def _set_status(self, text: str, kind: str | None = None):
        """The toolbar's status text, with a check or warning icon for validation results."""
        self.status_kind = kind
        self.status_badge.setText(text)
        self._show_status_icon()

    def _show_status_icon(self):
        if self.status_kind not in STATUS_ICONS:
            self.status_icon.hide()
            return
        name, role = STATUS_ICONS[self.status_kind]
        icon = svg_icon(name, {(QIcon.Mode.Normal, QIcon.State.Off): role}, size=16)
        self.status_icon.setPixmap(icon.pixmap(16, 16))
        self.status_icon.show()

    def _icon_button(self, text: str, icon_name: str) -> QPushButton:
        button = QPushButton(text)
        button.setIcon(svg_icon(icon_name, BUTTON_ICON_COLORS, size=16))
        button.setIconSize(QSize(16, 16))
        return button

    # ── Inspector Builder ──────────────────────────────────────────────────────

    def _create_inspector_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QFrame()
        container.setObjectName("workflowInspector")
        self.inspector_layout = QVBoxLayout(container)
        self.inspector_layout.setContentsMargins(12, 10, 12, 10)
        self.inspector_layout.setSpacing(10)

        self.inspector_title = label("DIAGRAM OVERVIEW", "eyebrow")
        self.inspector_layout.addWidget(self.inspector_title)

        self.inspector_content = QVBoxLayout()
        self.inspector_content.setSpacing(8)
        self.inspector_layout.addLayout(self.inspector_content)
        self.inspector_layout.addStretch()

        scroll.setWidget(container)
        self._show_diagram_overview()
        return scroll

    def _clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
            else:
                sub = item.layout()
                if sub is not None:
                    self._clear_layout(sub)

    def _clear_inspector_content(self):
        self._clear_layout(self.inspector_content)

    def _on_selection_changed(self, selected_item):
        self.selected_item = selected_item
        self._clear_inspector_content()

        if isinstance(selected_item, NodeBoxItem):
            self._show_node_inspector(selected_item.node)
        elif isinstance(selected_item, StringItem):
            self._show_string_inspector(selected_item.wire)
        else:
            self._show_diagram_overview()

    def _show_diagram_overview(self):
        self.inspector_title.setText("DIAGRAM OVERVIEW")

        # Name
        self.inspector_content.addWidget(label("Workflow Name:", "muted"))
        name_edit = QLineEdit(self.scene.graph.name)
        name_edit.textChanged.connect(lambda txt: setattr(self.scene.graph, "name", txt))
        self.inspector_content.addWidget(name_edit)

        # Description
        self.inspector_content.addWidget(label("Description:", "muted"))
        desc_edit = QPlainTextEdit(self.scene.graph.description)
        desc_edit.setMaximumHeight(80)
        desc_edit.textChanged.connect(lambda: setattr(self.scene.graph, "description", desc_edit.toPlainText()))
        self.inspector_content.addWidget(desc_edit)

        # Metrics
        num_nodes = len(self.scene.graph.nodes)
        num_wires = len(self.scene.graph.strings)
        levels = self.scene.graph.topological_levels()

        stats_box = QGroupBox("Diagram Statistics")
        sb_layout = QVBoxLayout(stats_box)
        sb_layout.addWidget(label(f"Nodes: <b>{num_nodes}</b>"))
        sb_layout.addWidget(label(f"String Wires: <b>{num_wires}</b>"))
        sb_layout.addWidget(label(f"Execution Stages: <b>{len(levels)}</b>"))
        self.inspector_content.addWidget(stats_box)

    def _show_node_inspector(self, node: WorkflowNode):
        self.inspector_title.setText(f"NODE: {node.id}")

        self.inspector_content.addWidget(label("Title Label:", "muted"))
        lbl_edit = QLineEdit(node.label)
        def _on_label(txt):
            node.label = txt
            item = self.scene.node_items.get(node.id)
            if item:
                item.update()
        lbl_edit.textChanged.connect(_on_label)
        self.inspector_content.addWidget(lbl_edit)

        self.inspector_content.addWidget(label(f"Paradigm: <b>{node.paradigm}</b>", "text"))
        self.inspector_content.addWidget(label(f"Category: <i>{node.category}</i>", "muted"))

        self.inspector_content.addWidget(label("Experiment Config Ref:", "muted"))
        ref_edit = QLineEdit(node.experiment_ref)
        ref_edit.setPlaceholderText("e.g. group/experiment")
        ref_edit.textChanged.connect(lambda txt: setattr(node, "experiment_ref", txt))
        self.inspector_content.addWidget(ref_edit)

        self.inspector_content.addWidget(label("Hydra CLI Overrides:", "muted"))
        overrides_edit = QPlainTextEdit("\n".join(node.overrides))
        overrides_edit.setPlaceholderText("key=value (one per line)")
        overrides_edit.setMaximumHeight(80)
        def _on_overrides():
            lines = [l.strip() for l in overrides_edit.toPlainText().splitlines() if l.strip()]
            node.overrides = lines
        overrides_edit.textChanged.connect(_on_overrides)
        self.inspector_content.addWidget(overrides_edit)

        # Ports Summary
        ports_box = QGroupBox("Ports")
        pb_layout = QVBoxLayout(ports_box)
        pb_layout.setSpacing(4)
        if node.inputs:
            pb_layout.addWidget(label("Inputs:", "muted"))
            for p in node.inputs.values():
                pb_layout.addWidget(label(f"• <font color='{p.port_type.color}'><b>{p.name}</b></font> ({p.port_type.value})"))
        if node.outputs:
            pb_layout.addWidget(label("Outputs:", "muted"))
            for p in node.outputs.values():
                pb_layout.addWidget(label(f"• <font color='{p.port_type.color}'><b>{p.name}</b></font> ({p.port_type.value})"))
        self.inspector_content.addWidget(ports_box)

        btn_delete = self._icon_button("Delete Node", "trash")
        btn_delete.clicked.connect(lambda: self.scene.remove_node_from_scene(node.id))
        self.inspector_content.addWidget(btn_delete)

    def _show_string_inspector(self, wire: WorkflowString):
        self.inspector_title.setText("STRING WIRE")

        type_color = wire.port_type.color
        type_name = wire.port_type.display_name
        self.inspector_content.addWidget(label(f"Type: <font color='{type_color}'><b>{type_name}</b></font>"))

        conn_box = QGroupBox("Connection")
        cb_layout = QVBoxLayout(conn_box)
        cb_layout.addWidget(label(f"<b>From:</b> {wire.source_node_id}.{wire.source_port_name}"))
        cb_layout.addWidget(label(f"<b>To:</b> {wire.target_node_id}.{wire.target_port_name}"))
        self.inspector_content.addWidget(conn_box)

        self.inspector_content.addWidget(label("Hydra Param Binding:", "muted"))
        binding_edit = QLineEdit(wire.param_binding)
        binding_edit.setPlaceholderText("e.g. ++mode.dataset_path=${source.dataset_path}")
        binding_edit.textChanged.connect(lambda txt: setattr(wire, "param_binding", txt))
        self.inspector_content.addWidget(binding_edit)

        btn_delete = self._icon_button("Disconnect Wire", "unlink")
        btn_delete.clicked.connect(lambda: self.scene.remove_string_from_scene(wire.id))
        self.inspector_content.addWidget(btn_delete)

    # ── Actions & Preset Handlers ──────────────────────────────────────────────

    def _on_preset_selected(self, index: int):
        preset_id = self.preset_combo.currentData()
        if preset_id:
            self.load_preset(preset_id)

    def load_preset(self, preset_id: str):
        graph = get_preset(preset_id)
        if graph:
            self.scene.set_graph(graph)
            self.canvas_view.zoom_fit()
            self._set_status(f"Loaded: {graph.name}")
            self.log_fn(f"Loaded workflow preset '{graph.name}'.")
            self.preset_combo.blockSignals(True)
            idx = self.preset_combo.findData(preset_id)
            if idx >= 0:
                self.preset_combo.setCurrentIndex(idx)
            self.preset_combo.blockSignals(False)
            if self.selected_item is None:
                self._clear_inspector_content()
                self._show_diagram_overview()

    def validate_current_diagram(self):
        issues = self.scene.graph.validate()
        if not issues:
            QMessageBox.information(
                self,
                "Validation Success",
                f"Diagram '{self.scene.graph.name}' is valid.\n\n"
                f"• {len(self.scene.graph.nodes)} nodes\n"
                f"• {len(self.scene.graph.strings)} typed strings\n"
                f"• All port types and required inputs are satisfied."
            )
            self._set_status("Valid", "valid")
        else:
            msg = "Validation detected issues:\n\n" + "\n".join(f"• {iss}" for iss in issues)
            QMessageBox.warning(self, "Validation Warnings", msg)
            self._set_status(f"{len(issues)} issue{'' if len(issues) == 1 else 's'}", "issues")

    def clear_workflow(self):
        new_g = WorkflowGraph(id="new_workflow", name="Untitled Workflow")
        self.scene.set_graph(new_g)
        self.preset_combo.setCurrentIndex(0)
        self._set_status("Cleared")
        self.log_fn("Workflow diagram cleared.")
        self._clear_inspector_content()
        self._show_diagram_overview()

    def _workflow_dir(self):
        from src.app.workspace import WorkspacePaths
        return WorkspacePaths.for_process().config / "workflow"

    def save_yaml_dialog(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Workflow Diagram",
            str(self._workflow_dir() / f"{self.scene.graph.id}.yaml"),
            "YAML Files (*.yaml *.yml)"
        )
        if path:
            self.scene.graph.save_yaml(path)
            self._set_status(f"Saved: {Path(path).name}")
            self.log_fn(f"Saved workflow YAML: {path}")

    def load_yaml_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Workflow Diagram",
            str(self._workflow_dir()),
            "YAML Files (*.yaml *.yml)"
        )
        if path:
            graph = WorkflowGraph.load_yaml(path)
            self.scene.set_graph(graph)
            self.canvas_view.zoom_fit()
            self._set_status(f"Loaded: {Path(path).name}")
            self.log_fn(f"Loaded workflow YAML: {path}")
            if self.selected_item is None:
                self._clear_inspector_content()
                self._show_diagram_overview()

    def _on_graph_modified(self):
        self.workflow_changed.emit()
        self._set_status("Modified")
        if self.selected_item is None:
            self._clear_inspector_content()
            self._show_diagram_overview()
