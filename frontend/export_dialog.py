"""Export dialog for packaging Theta experiments into Hub components or standalone recipes."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions, ExportResult
from src.app.services.component_exporter import (
    ComponentExporter,
    ComponentExportOptions,
    ComponentExportResult,
)
from .theme import theme_color
from .widgets import label


class ExportExperimentDialog(QDialog):
    """Modal dialog allowing users to export an experiment as a Theta Hub package (.zip)
    or as a standalone recipe YAML.
    """

    def __init__(
        self,
        experiment_id: str,
        workspace_root: Path,
        raw_yaml_text: Optional[str] = None,
        group: Optional[str] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.experiment_id = experiment_id.strip()
        self.workspace_root = Path(workspace_root)
        self.raw_yaml_text = raw_yaml_text
        self.group = group or (self.experiment_id.split("/")[0] if "/" in self.experiment_id else self.experiment_id)
        self.is_group_export = bool(group) or ("/" not in self.experiment_id and not self.experiment_id.endswith(".yaml"))
        self.exporter = ExperimentExporter(workspace_root=self.workspace_root)
        self.export_result: Optional[ExportResult] = None

        title_text = f"Export Experiment Plugin: {self.group}" if self.is_group_export else "Export Experiment"
        self.setWindowTitle(title_text)
        self.setMinimumWidth(540)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        # Header
        header = QVBoxLayout()
        header.setSpacing(4)
        title_lbl = f"Export Experiment Plugin: {self.group}" if self.is_group_export else "Export Experiment"
        title = QLabel(title_lbl)
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        header.addWidget(title)

        sub_lbl = (
            f"Package entire '{self.group}' experiment suite (all recipes, group defaults, environment, evaluators, and plotters) for 1-click install in Theta Hub."
            if self.is_group_export
            else f"Export '{self.experiment_id}' for distribution via Theta Hub or standalone execution."
        )
        sub = QLabel(sub_lbl)
        sub.setStyleSheet(f"color: {theme_color('muted')}; font-size: 12px;")
        sub.setWordWrap(True)
        header.addWidget(sub)
        root_layout.addLayout(header)

        # Format Card
        format_card = QFrame()
        format_card.setObjectName("card")
        format_layout = QVBoxLayout(format_card)
        format_layout.setSpacing(10)

        format_layout.addWidget(label("EXPORT FORMAT", "eyebrow"))

        self.btn_group = QButtonGroup(self)
        self.radio_zip = QRadioButton("Theta Hub Component Archive (.zip)")
        self.radio_zip.setChecked(True)
        self.radio_zip.setStyleSheet("font-weight: 600;")
        self.btn_group.addButton(self.radio_zip)
        format_layout.addWidget(self.radio_zip)

        zip_hint = QLabel("Self-contained package with manifest (.theta_component.json), environment rules, and model dependencies. Ready for 1-click install in Theta Hub.")
        zip_hint.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px; margin-left: 22px;")
        zip_hint.setWordWrap(True)
        format_layout.addWidget(zip_hint)

        self.radio_yaml = QRadioButton("Standalone Recipe YAML (.yaml)")
        self.btn_group.addButton(self.radio_yaml)
        format_layout.addWidget(self.radio_yaml)

        yaml_hint = QLabel("Exports only the configuration recipe YAML for manual reproduction.")
        yaml_hint.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px; margin-left: 22px;")
        format_layout.addWidget(yaml_hint)

        root_layout.addWidget(format_card)

        # Options Card (Archive contents)
        self.options_card = QFrame()
        self.options_card.setObjectName("card")
        opts_layout = QVBoxLayout(self.options_card)
        opts_layout.setSpacing(8)

        opts_layout.addWidget(label("PACKAGE ASSETS", "eyebrow"))

        self.chk_group_all = QCheckBox("Bundle full experiment group (all recipes + group defaults)")
        self.chk_group_all.setChecked(self.is_group_export)
        opts_layout.addWidget(self.chk_group_all)

        self.chk_env = QCheckBox("Include environment configuration & custom rules")
        self.chk_env.setChecked(True)
        opts_layout.addWidget(self.chk_env)

        self.chk_models = QCheckBox("Include custom model architecture definitions")
        self.chk_models.setChecked(True)
        opts_layout.addWidget(self.chk_models)

        self.chk_eval = QCheckBox("Include custom evaluators & reward shaping")
        self.chk_eval.setChecked(True)
        opts_layout.addWidget(self.chk_eval)

        self.chk_plots = QCheckBox("Include visualization suites & plotters")
        self.chk_plots.setChecked(True)
        opts_layout.addWidget(self.chk_plots)

        self.chk_data = QCheckBox("Include custom data modules & loaders")
        self.chk_data.setChecked(True)
        opts_layout.addWidget(self.chk_data)

        self.chk_checkpoints = QCheckBox("Include trained checkpoint weights (if available)")
        self.chk_checkpoints.setChecked(False)
        opts_layout.addWidget(self.chk_checkpoints)

        root_layout.addWidget(self.options_card)

        # Metadata Row (Version & Author)
        self.meta_card = QFrame()
        self.meta_card.setObjectName("card")
        meta_layout = QHBoxLayout(self.meta_card)
        meta_layout.setSpacing(12)

        ver_col = QVBoxLayout()
        ver_col.setSpacing(4)
        ver_col.addWidget(label("VERSION", "eyebrow"))
        self.edit_version = QLineEdit("1.0.0")
        ver_col.addWidget(self.edit_version)
        meta_layout.addLayout(ver_col)

        author_col = QVBoxLayout()
        author_col.setSpacing(4)
        author_col.addWidget(label("AUTHOR", "eyebrow"))
        self.edit_author = QLineEdit("Theta Researcher")
        author_col.addWidget(self.edit_author)
        meta_layout.addLayout(author_col)

        root_layout.addWidget(self.meta_card)

        # Output Destination Card
        dest_card = QFrame()
        dest_card.setObjectName("card")
        dest_layout = QVBoxLayout(dest_card)
        dest_layout.setSpacing(6)

        dest_layout.addWidget(label("DESTINATION", "eyebrow"))

        dest_row = QHBoxLayout()
        dest_row.setSpacing(8)
        self.edit_dest = QLineEdit()
        self._update_default_destination()
        dest_row.addWidget(self.edit_dest)

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        dest_row.addWidget(self.btn_browse)
        dest_layout.addLayout(dest_row)

        root_layout.addWidget(dest_card)

        # Format switch connection
        self.radio_zip.toggled.connect(self._on_format_changed)

        # Buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(self.btn_cancel)

        self.btn_export = QPushButton("Export")
        self.btn_export.setDefault(True)
        self.btn_export.setStyleSheet(
            f"background-color: {theme_color('accent')}; color: #ffffff; font-weight: 700; padding: 6px 16px;"
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        button_row.addWidget(self.btn_export)

        root_layout.addLayout(button_row)

    def _update_default_destination(self):
        clean_name = self.experiment_id.replace("/", "_").replace(".yaml", "")
        if self.radio_zip.isChecked():
            exports_dir = self.workspace_root / "results" / "exports"
            exports_dir.mkdir(parents=True, exist_ok=True)
            default_path = exports_dir / f"{clean_name}_v{self.edit_version.text().strip() or '1.0.0'}.zip"
        else:
            default_path = self.workspace_root / f"{clean_name}.yaml"
        self.edit_dest.setText(str(default_path))

    def _on_format_changed(self, zip_checked: bool):
        self.options_card.setVisible(zip_checked)
        self.meta_card.setVisible(zip_checked)
        self._update_default_destination()

    def _on_browse_clicked(self):
        clean_name = self.experiment_id.replace("/", "_").replace(".yaml", "")
        if self.radio_zip.isChecked():
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Theta Hub Component",
                self.edit_dest.text() or f"{clean_name}.zip",
                "Theta Package (*.zip)",
            )
        else:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Recipe YAML",
                self.edit_dest.text() or f"{clean_name}.yaml",
                "YAML (*.yaml)",
            )
        if path:
            self.edit_dest.setText(path)

    def _on_export_clicked(self):
        target_path_str = self.edit_dest.text().strip()
        if not target_path_str:
            QMessageBox.warning(self, "Invalid Path", "Please specify an export destination path.")
            return

        target_path = Path(target_path_str)

        if self.radio_yaml.isChecked():
            # Export YAML only
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                if self.raw_yaml_text:
                    content = self.raw_yaml_text
                else:
                    from src.app.workspace import WorkspacePaths
                    exp_yaml = WorkspacePaths(self.workspace_root).resolve_config(f"experiment/{self.experiment_id}.yaml")
                    content = exp_yaml.read_text(encoding="utf-8") if exp_yaml.is_file() else ""
                target_path.write_text(content, encoding="utf-8")
                self.accept()
            except Exception as exc:
                QMessageBox.warning(self, "Export Failed", f"Could not write YAML: {exc}")
            return

        # Export ZIP Hub Component
        is_bundle = getattr(self, "is_group_export", False) or self.chk_group_all.isChecked()
        opts = ExportOptions(
            include_checkpoints=self.chk_checkpoints.isChecked(),
            include_environment=self.chk_env.isChecked(),
            include_models=self.chk_models.isChecked(),
            include_rules=self.chk_env.isChecked(),
            include_evaluators=self.chk_eval.isChecked(),
            include_plotters=self.chk_plots.isChecked(),
            include_data_modules=self.chk_data.isChecked(),
            include_all_group_experiments=is_bundle,
            output_path=target_path,
            author=self.edit_author.text().strip() or "Theta Researcher",
            version=self.edit_version.text().strip() or "1.0.0",
        )

        if is_bundle:
            res = self.exporter.export_group(self.group, options=opts)
        else:
            res = self.exporter.export(self.experiment_id, options=opts)
        self.export_result = res

        if res.success:
            self.accept()
        else:
            QMessageBox.warning(self, "Export Failed", res.error or "Unknown error exporting experiment.")


class ExportComponentDialog(QDialog):
    """Modal dialog allowing users to export a component plugin as a Theta Hub package (.zip)
    or as a standalone git-ready directory outside/inside Theta-IDE.
    """

    def __init__(
        self,
        config_path: Path,
        rel_path: Optional[str] = None,
        workspace_root: Optional[Path] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.config_path = Path(config_path)
        self.rel_path = rel_path
        self.workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self.exporter = ComponentExporter(workspace_root=self.workspace_root)
        self.export_result: Optional[ComponentExportResult] = None

        from .component_actions import find_component_assets
        self.assets = find_component_assets(self.config_path, self.rel_path, workspace_dir=self.workspace_root)
        self.comp_id = self.assets["comp_id"]
        self.comp_name = self.assets.get("comp_name") or self.comp_id
        self.kind = self.assets.get("kind") or "model"
        self.meta = self.assets.get("meta") or {}

        self.setWindowTitle(f"Export Component: {self.comp_name}")
        self.setMinimumWidth(540)
        self.setModal(True)
        self._init_ui()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(20, 20, 20, 20)
        root_layout.setSpacing(16)

        # Header
        header = QVBoxLayout()
        header.setSpacing(4)
        title = QLabel(f"Export {self.kind.capitalize()}: {self.comp_name}")
        title.setStyleSheet("font-size: 16px; font-weight: 700;")
        header.addWidget(title)

        sub = QLabel(
            f"Package '{self.comp_name}' for distribution on Theta Hub or publish as an external repository."
        )
        sub.setStyleSheet(f"color: {theme_color('muted')}; font-size: 12px;")
        sub.setWordWrap(True)
        header.addWidget(sub)
        root_layout.addLayout(header)

        # Format Card
        format_card = QFrame()
        format_card.setObjectName("card")
        format_layout = QVBoxLayout(format_card)
        format_layout.setSpacing(10)

        format_layout.addWidget(label("EXPORT FORMAT", "eyebrow"))

        self.btn_group = QButtonGroup(self)
        self.radio_zip = QRadioButton("Theta Hub Component Archive (.zip)")
        self.radio_zip.setChecked(True)
        self.radio_zip.setStyleSheet("font-weight: 600;")
        self.btn_group.addButton(self.radio_zip)
        format_layout.addWidget(self.radio_zip)

        zip_hint = QLabel(
            "Self-contained package archive with manifest (.theta_component.json), code, and configurations. Ready for 1-click install in Theta Hub."
        )
        zip_hint.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px; margin-left: 22px;")
        zip_hint.setWordWrap(True)
        format_layout.addWidget(zip_hint)

        self.radio_folder = QRadioButton("Standalone Git Repository / Directory")
        self.radio_folder.setStyleSheet("font-weight: 600;")
        self.btn_group.addButton(self.radio_folder)
        format_layout.addWidget(self.radio_folder)

        folder_hint = QLabel(
            "Export to a standalone directory outside or inside Theta-IDE with git/pip repository scaffolding (.gitignore, pyproject.toml, README.md) ready for GitHub."
        )
        folder_hint.setStyleSheet(f"color: {theme_color('muted')}; font-size: 11px; margin-left: 22px;")
        folder_hint.setWordWrap(True)
        format_layout.addWidget(folder_hint)

        root_layout.addWidget(format_card)

        # Metadata Row (Version, Author, Description)
        meta_card = QFrame()
        meta_card.setObjectName("card")
        meta_layout = QVBoxLayout(meta_card)
        meta_layout.setSpacing(10)

        meta_layout.addWidget(label("METADATA", "eyebrow"))

        meta_row = QHBoxLayout()
        meta_row.setSpacing(12)

        ver_col = QVBoxLayout()
        ver_col.setSpacing(4)
        ver_col.addWidget(label("VERSION", "eyebrow"))
        existing_ver = self.meta.get("version") or "1.0.0"
        self.edit_version = QLineEdit(str(existing_ver))
        ver_col.addWidget(self.edit_version)
        meta_row.addLayout(ver_col)

        author_col = QVBoxLayout()
        author_col.setSpacing(4)
        author_col.addWidget(label("AUTHOR", "eyebrow"))
        author_default = "Theta Researcher"
        if isinstance(self.meta.get("author"), dict):
            author_default = self.meta["author"].get("name") or author_default
        elif isinstance(self.meta.get("author"), str):
            author_default = self.meta["author"]
        self.edit_author = QLineEdit(author_default)
        author_col.addWidget(self.edit_author)
        meta_row.addLayout(author_col)

        meta_layout.addLayout(meta_row)

        desc_col = QVBoxLayout()
        desc_col.setSpacing(4)
        desc_col.addWidget(label("DESCRIPTION", "eyebrow"))
        self.edit_desc = QLineEdit(str(self.meta.get("description") or f"{self.comp_name} {self.kind} component"))
        desc_col.addWidget(self.edit_desc)
        meta_layout.addLayout(desc_col)

        root_layout.addWidget(meta_card)

        # Output Destination Card
        dest_card = QFrame()
        dest_card.setObjectName("card")
        dest_layout = QVBoxLayout(dest_card)
        dest_layout.setSpacing(6)

        dest_layout.addWidget(label("DESTINATION", "eyebrow"))

        dest_row = QHBoxLayout()
        dest_row.setSpacing(8)
        self.edit_dest = QLineEdit()
        self._update_default_destination()
        dest_row.addWidget(self.edit_dest)

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self._on_browse_clicked)
        dest_row.addWidget(self.btn_browse)
        dest_layout.addLayout(dest_row)

        root_layout.addWidget(dest_card)

        # Format switch connection
        self.radio_zip.toggled.connect(self._on_format_changed)
        self.edit_version.textChanged.connect(lambda _: self._update_default_destination())

        # Buttons
        button_row = QHBoxLayout()
        button_row.setSpacing(10)
        button_row.addStretch()

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(self.btn_cancel)

        self.btn_export = QPushButton("Export Component")
        self.btn_export.setDefault(True)
        self.btn_export.setStyleSheet(
            f"background-color: {theme_color('accent')}; color: #ffffff; font-weight: 700; padding: 6px 16px;"
        )
        self.btn_export.clicked.connect(self._on_export_clicked)
        button_row.addWidget(self.btn_export)

        root_layout.addLayout(button_row)

    def _update_default_destination(self):
        clean_name = self.comp_id.replace("/", "_")
        ver = self.edit_version.text().strip() or "1.0.0"
        exports_dir = self.workspace_root / "results" / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        if self.radio_zip.isChecked():
            default_path = exports_dir / f"{clean_name}_v{ver}.zip"
        else:
            default_path = exports_dir / clean_name
        self.edit_dest.setText(str(default_path))

    def _on_format_changed(self, zip_checked: bool):
        self._update_default_destination()

    def _on_browse_clicked(self):
        clean_name = self.comp_id.replace("/", "_")
        if self.radio_zip.isChecked():
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Theta Hub Component Package",
                self.edit_dest.text() or f"{clean_name}.zip",
                "Theta Package (*.zip)",
            )
        else:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Export Directory",
                self.edit_dest.text() or str(self.workspace_root / "results" / "exports"),
            )
        if path:
            self.edit_dest.setText(path)

    def _on_export_clicked(self):
        target_path_str = self.edit_dest.text().strip()
        if not target_path_str:
            QMessageBox.warning(self, "Invalid Path", "Please specify an export destination path.")
            return

        target_path = Path(target_path_str)
        fmt = "zip" if self.radio_zip.isChecked() else "folder"

        opts = ComponentExportOptions(
            export_format=fmt,
            output_path=target_path,
            author=self.edit_author.text().strip() or "Theta Researcher",
            version=self.edit_version.text().strip() or "1.0.0",
            description=self.edit_desc.text().strip() or f"{self.comp_name} component",
            include_configs=True,
            include_rules=True,
            generate_git_scaffold=True,
        )

        res = self.exporter.export(self.config_path, options=opts)
        self.export_result = res

        if res.success:
            self.accept()
        else:
            QMessageBox.warning(self, "Export Failed", res.error or "Unknown error exporting component.")
