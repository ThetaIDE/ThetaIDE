"""In-IDE wizard dialog for scaffolding new component plugins."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.app.services.plugin_scaffolder import PluginScaffolder, ScaffoldOptions, ScaffoldResult
from .theme import theme_color
from .widgets import label


class ScaffoldPluginDialog(QDialog):
    """Wizard dialog for generating a new component plugin scaffold."""

    def __init__(self, workspace_root: Path, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.workspace_root = Path(workspace_root)
        self.scaffolder = PluginScaffolder(workspace_root=self.workspace_root)
        self.scaffold_result: Optional[ScaffoldResult] = None
        self.paradigm_boxes: dict[str, QCheckBox] = {}

        self.setWindowTitle("Scaffold New Component Plugin")
        self.setMinimumWidth(500)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 20, 20, 20)
        root.setSpacing(14)

        # Header
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel("Scaffold New Component Plugin")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        header.addWidget(title)

        sub = QLabel("Generate boilerplate code, runtime protocols, default configs, and manifests.")
        sub.setStyleSheet(f"color: {theme_color('muted')}; font-size: 12px;")
        header.addWidget(sub)
        root.addLayout(header)

        # Configuration Card
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setSpacing(10)

        # Kind Selector
        layout.addWidget(label("COMPONENT ARCHETYPE", "eyebrow"))
        self.combo_kind = QComboBox()
        self.combo_kind.addItem("Execution Site (src/usr/sites/)", "site")
        self.combo_kind.addItem("Model Architecture (src/usr/models/)", "model")
        self.combo_kind.addItem("RL Algorithm / Agent (src/usr/algorithms/)", "algorithm")
        self.combo_kind.addItem("Environment & Grounding (src/usr/environments/)", "env")
        self.combo_kind.addItem("DataModule & Sampler (src/usr/data/)", "datamodule")
        self.combo_kind.addItem("Evaluator & Verification Hook (src/usr/eval/)", "evaluator")
        self.combo_kind.addItem("Experiment Monitor (src/usr/monitors/)", "monitor")
        self.combo_kind.addItem("Plotter & Visualization (plot/)", "plotter")
        self.combo_kind.currentIndexChanged.connect(self._on_kind_changed)
        layout.addWidget(self.combo_kind)

        # Name Input
        layout.addWidget(label("COMPONENT IDENTIFIER", "eyebrow"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("e.g. mamba_policy, soft_actor_critic, drone_sim")
        self.edit_name.textChanged.connect(self._update_preview)
        layout.addWidget(self.edit_name)

        # Paradigms Checkboxes
        layout.addWidget(label("COMPATIBLE PARADIGMS", "eyebrow"))
        paradigm_row = QHBoxLayout()
        paradigm_row.setSpacing(16)
        for p_id, p_label in [
            ("online_rl", "Online RL"),
            ("offline_rl", "Offline RL"),
            ("supervised", "Supervised"),
        ]:
            cb = QCheckBox(p_label)
            self.paradigm_boxes[p_id] = cb
            paradigm_row.addWidget(cb)
        paradigm_row.addStretch()
        layout.addLayout(paradigm_row)

        # Author and Version
        meta_row = QHBoxLayout()
        meta_row.setSpacing(10)

        ver_col = QVBoxLayout()
        ver_col.setSpacing(4)
        ver_col.addWidget(label("VERSION", "eyebrow"))
        self.edit_version = QLineEdit("0.1.0")
        ver_col.addWidget(self.edit_version)
        meta_row.addLayout(ver_col)

        author_col = QVBoxLayout()
        author_col.setSpacing(4)
        author_col.addWidget(label("AUTHOR", "eyebrow"))
        self.edit_author = QLineEdit("Theta Researcher")
        author_col.addWidget(self.edit_author)
        meta_row.addLayout(author_col)
        layout.addLayout(meta_row)

        root.addWidget(card)

        # Preview Card
        self.preview_card = QFrame()
        self.preview_card.setObjectName("card")
        prev_layout = QVBoxLayout(self.preview_card)
        prev_layout.setSpacing(6)
        prev_layout.addWidget(label("GENERATED FILES PREVIEW", "eyebrow"))

        self.lbl_preview = QLabel()
        self.lbl_preview.setStyleSheet(f"font-family: monospace; font-size: 11px; color: {theme_color('muted')};")
        self.lbl_preview.setWordWrap(True)
        prev_layout.addWidget(self.lbl_preview)
        root.addWidget(self.preview_card)

        # Action Buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(self.btn_cancel)

        self.btn_scaffold = QPushButton("Scaffold Component")
        self.btn_scaffold.setDefault(True)
        self.btn_scaffold.setStyleSheet(
            f"background-color: {theme_color('accent')}; color: #ffffff; font-weight: 700; padding: 6px 16px;"
        )
        self.btn_scaffold.clicked.connect(self._on_scaffold_clicked)
        button_row.addWidget(self.btn_scaffold)

        root.addLayout(button_row)
        self._set_default_paradigms("model")
        self._update_preview()

    def _set_default_paradigms(self, kind: str):
        defaults = {
            "model": ["online_rl", "offline_rl", "supervised"],
            "agent": ["online_rl"],
            "env": ["online_rl"],
            "datamodule": ["offline_rl", "supervised"],
            "evaluator": ["online_rl", "offline_rl", "supervised"],
            "plotter": ["online_rl", "offline_rl", "supervised"],
        }.get(kind, ["online_rl"])

        for p_id, cb in self.paradigm_boxes.items():
            cb.setChecked(p_id in defaults)

    def _on_kind_changed(self):
        kind = self.combo_kind.currentData()
        self._set_default_paradigms(kind)
        self._update_preview()

    def _update_preview(self):
        name = self.edit_name.text().strip().lower().replace(" ", "_") or "<name>"
        kind = self.combo_kind.currentData()

        if kind == "model":
            lines = [
                f"• src/usr/models/{name}/{name}_model.py",
                f"• src/usr/models/{name}/__init__.py",
                f"• src/usr/models/{name}/.theta_component.json",
                f"• src/usr/models/{name}/pyproject.toml",
                f"• in/config/model/{name}.yaml",
            ]
        elif kind in ("agent", "algorithm"):
            lines = [
                f"• src/usr/algorithms/{name}/{name}_agent.py",
                f"• src/usr/algorithms/{name}/__init__.py",
                f"• src/usr/algorithms/{name}/.theta_component.json",
                f"• src/usr/algorithms/{name}/pyproject.toml",
                f"• in/config/agent/{name}.yaml",
            ]
        elif kind == "env":
            lines = [
                f"• src/usr/environments/{name}/env.py",
                f"• src/usr/environments/{name}/__init__.py",
                f"• src/usr/environments/{name}/.theta_component.json",
                f"• in/config/env/{name}.yaml",
                f"• src/usr/rules/{name}/default.rules",
            ]
        elif kind == "datamodule":
            lines = [
                f"• src/usr/data/{name}/{name}_data_module.py",
                f"• src/usr/data/{name}/__init__.py",
                f"• src/usr/data/{name}/.theta_component.json",
            ]
        elif kind == "evaluator":
            lines = [
                f"• src/usr/eval/{name}/{name}_evaluator.py",
                f"• src/usr/eval/{name}/__init__.py",
                f"• src/usr/eval/{name}/.theta_component.json",
            ]
        elif kind == "monitor":
            lines = [f"• src/usr/monitors/{name}/monitor.py", f"• src/usr/monitors/{name}/catalog.py",
                     f"• src/usr/monitors/{name}/.theta_component.json", f"• in/config/monitor/{name}.yaml"]
        elif kind in ("plotter", "plot"):
            lines = [
                f"• src/usr/plotters/{name}/{name}_plotter.py",
                f"• src/usr/plotters/{name}/__init__.py",
                f"• src/usr/plotters/{name}/.theta_component.json",
                f"• in/config/plot/{name}.yaml",
            ]
        else:
            lines = []

        self.lbl_preview.setText("\n".join(lines))

    def _on_scaffold_clicked(self):
        name = self.edit_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Invalid Name", "Please enter a valid component name.")
            return

        selected_paradigms = [p for p, cb in self.paradigm_boxes.items() if cb.isChecked()]
        if not selected_paradigms:
            QMessageBox.warning(self, "Invalid Selection", "Please select at least one compatible learning paradigm.")
            return

        opts = ScaffoldOptions(
            name=name,
            kind=self.combo_kind.currentData(),
            author=self.edit_author.text().strip() or "Theta Researcher",
            version=self.edit_version.text().strip() or "0.1.0",
            paradigms=selected_paradigms,
        )

        res = self.scaffolder.scaffold(opts)
        self.scaffold_result = res

        if res.success:
            from .config_viewer import invalidate_config_tree
            invalidate_config_tree()
            self.accept()
        else:
            QMessageBox.warning(self, "Scaffolding Failed", res.error or "Unknown scaffolding error.")
