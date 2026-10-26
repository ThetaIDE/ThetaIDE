"""Dedicated Components panel for managing modular configurations (agent, env, model, paradigms, site, hydra, etc.)."""
import shutil
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_tree import ConfigTreeWidget
from .config_viewer import ConfigViewer
from .hub import HubClient, HubView
from .widgets import YamlHighlighter, label


class ComponentsPanel(QWidget):
    """Panel in Theta-IDE dedicated to modular configuration components outside experiments:
    agent profiles, environment definitions, model architectures, paradigms, site profiles,
    and global configs.
    """
    component_saved = pyqtSignal(str)  # rel_path

    def __init__(self, hub_client=None, log_fn=None, parent=None):
        super().__init__(parent)
        self.hub_client = hub_client
        self.log_fn = log_fn or (lambda msg: None)
        self.active_path = None
        self.active_rel_path = None
        self.hub_view = None
        self.is_hub_active = False
        self._init_ui()
        self.init_default_component()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(0)

        # ── Horizontal Splitter ───────────────────────────────────────────────
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Components Tree (extends to top on the left)
        self.components_tree = ConfigTreeWidget(mode="components")
        self.components_tree.setMinimumWidth(320)
        self.components_tree.file_selected.connect(self.on_component_selected)
        self.splitter.addWidget(self.components_tree)

        # Square buttons located at the top of the filetree
        self.btn_save = self.components_tree.btn_save
        self.btn_plus = getattr(self.components_tree, "btn_plus", getattr(self.components_tree, "btn_new", None))
        self.btn_new = self.btn_plus
        self.btn_hub = self.components_tree.btn_hub
        self.btn_toggle_raw = self.components_tree.btn_toggle_raw
        if self.btn_save is not None:
            self.btn_save.clicked.connect(self.save_current)
        if self.btn_hub is not None:
            self.btn_hub.clicked.connect(self.toggle_hub)
        if self.btn_toggle_raw is not None:
            self.btn_toggle_raw.clicked.connect(lambda: self.toggle_raw_preview())

        # 2. Right Side Section (contains viewer / preview)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # ── Content Stack (Switch between Parameters and Component Hub) ──────
        self.content_stack = QStackedWidget()

        # Page 0: Parameters configuring panels (Viewer + Raw YAML Preview)
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.viewer = ConfigViewer()
        self.viewer.is_component_mode = True
        self.viewer.auto_save = True
        self.viewer.btn_save = self.btn_save
        self.viewer.config_changed.connect(self._on_viewer_changed)
        self.viewer.save_requested.connect(self._on_viewer_saved)
        self.viewer.dirty_state_changed.connect(self.components_tree.set_file_dirty)
        self.viewer.file_renamed.connect(self._on_component_file_renamed)
        self.viewer.edit_code_requested.connect(self.open_active_component_in_editor)
        self.viewer.export_component_requested.connect(self.prompt_export_active_component)
        self.viewer.delete_component_requested.connect(self.prompt_delete_active_component)
        if self.btn_save is not None:
            self.viewer.dirty_changed.connect(self.btn_save.setEnabled)
        self.content_splitter.addWidget(self.viewer)

        # 3. Raw YAML Preview Panel (hidden by default)
        raw_panel = QWidget()
        raw_layout = QVBoxLayout(raw_panel)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.setSpacing(6)

        raw_header = QHBoxLayout()
        raw_header.setSpacing(8)
        raw_header.addWidget(label("RAW YAML CONFIG", "eyebrow"))
        raw_header.addStretch()

        btn_close_raw = QToolButton()
        btn_close_raw.setText("✕")
        btn_close_raw.setToolTip("Close raw YAML preview")
        btn_close_raw.clicked.connect(lambda: self.toggle_raw_preview(False))
        raw_header.addWidget(btn_close_raw)
        raw_layout.addLayout(raw_header)

        self.raw_status = label("Source file view", "muted")
        raw_layout.addWidget(self.raw_status)

        self.raw_edit = QPlainTextEdit()
        self.raw_edit.setReadOnly(True)
        self.raw_highlighter = YamlHighlighter(self.raw_edit.document())
        raw_layout.addWidget(self.raw_edit, 1)

        self.raw_panel = raw_panel
        self.raw_panel.hide()
        self.content_splitter.addWidget(raw_panel)

        self.content_splitter.setSizes([1000, 0])
        self.content_stack.addWidget(self.content_splitter)

        right_layout.addWidget(self.content_stack, 1)

        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([320, 1000])
        root_layout.addWidget(self.splitter, 1)

    def init_default_component(self):
        """Select a default component config if available."""
        for path in sorted(self.components_tree.root_dir.rglob("*.yaml")):
            if path.name.startswith("_"):
                continue
            rel = path.relative_to(self.components_tree.root_dir).as_posix()
            if self.components_tree.select_file(rel):
                return

    def on_component_selected(self, file_path, rel_path):
        """Handle component file selection from the tree."""
        if hasattr(self, "viewer") and self.viewer and self.viewer.is_dirty:
            self.save_current()
        self.show_params()
        self.active_path = Path(file_path)
        self.active_rel_path = rel_path
        self.viewer.load_file(self.active_path, rel_path)
        self._update_component_action_buttons()
        self._update_raw_yaml_view()

    def _update_component_action_buttons(self):
        """Update tooltips and state for pencil, export, and trash buttons."""
        if not self.active_path or not self.active_path.exists():
            return
        try:
            from .component_actions import find_component_assets
            assets = find_component_assets(self.active_path, self.active_rel_path)
            comp_name = assets["comp_name"]
            is_comp = assets["is_component"]
            target_code = assets["primary_code_file"]

            is_hub = assets.get("is_hub_installed", False)

            if hasattr(self.viewer, "btn_edit_code"):
                if target_code and target_code.is_file():
                    self.viewer.btn_edit_code.setToolTip(f"Open {target_code.name} in editor")
                else:
                    self.viewer.btn_edit_code.setToolTip(f"Open {comp_name} source code in editor")

            if hasattr(self.viewer, "btn_export_component"):
                self.viewer.btn_export_component.setToolTip(f"Export '{comp_name}' for Theta Hub or GitHub")

            if hasattr(self.viewer, "btn_delete_component"):
                if is_hub:
                    self.viewer.btn_delete_component.setToolTip(f"Uninstall {comp_name} and remove all files")
                elif is_comp:
                    self.viewer.btn_delete_component.setToolTip(f"Delete component {comp_name} and all its files")
                else:
                    self.viewer.btn_delete_component.setToolTip(f"Delete configuration {self.active_path.name}")
        except Exception:
            pass

    def get_settings_manager(self):
        """Resolve a SettingsManager instance from window, parent, or default instance."""
        w = self.window()
        if hasattr(w, "settings_manager") and w.settings_manager is not None:
            return w.settings_manager
        parent = self.parent()
        if parent is not None and hasattr(parent, "settings_manager") and parent.settings_manager is not None:
            return parent.settings_manager
        try:
            from .settings import SettingsManager
            return SettingsManager()
        except Exception:
            return None

    def open_active_component_in_editor(self, file_path=None, rel_path=None):
        """Open the active component's primary source code in an editor."""
        target_cfg = Path(file_path) if file_path else self.active_path
        target_rel = rel_path or self.active_rel_path
        if not target_cfg or not target_cfg.exists():
            return

        from .component_actions import find_component_assets, open_in_editor
        assets = find_component_assets(target_cfg, target_rel)
        target = assets.get("primary_code_file") or assets.get("code_dir") or target_cfg

        if target and target.exists():
            success = open_in_editor(target)
            if success:
                self.log_fn(f"Opened {target.name} in external editor.")
            else:
                QMessageBox.warning(self, "Open in Editor", f"Could not launch editor for {target}.")
        else:
            QMessageBox.warning(self, "Open in Editor", f"No source file found for {assets['comp_name']}.")

    def prompt_export_active_component(self, file_path=None, rel_path=None):
        """Export the active component as a Theta Hub package (.zip) or standalone directory."""
        target_cfg = Path(file_path) if file_path else self.active_path
        target_rel = rel_path or self.active_rel_path
        if not target_cfg or not target_cfg.exists():
            return

        if hasattr(self, "viewer") and self.viewer and self.viewer.is_dirty:
            self.save_current()

        from .export_dialog import ExportComponentDialog
        workspace_root = self.components_tree.root_dir.parent
        dlg = ExportComponentDialog(
            config_path=target_cfg,
            rel_path=target_rel,
            workspace_root=workspace_root,
            parent=self,
        )
        if dlg.exec() and dlg.export_result and dlg.export_result.success:
            res = dlg.export_result
            self.log_fn(f"Exported component '{dlg.comp_name}' to {res.export_path} ({len(res.files_included)} files).")
            QMessageBox.information(
                self,
                "Component Exported",
                f"Successfully exported component '{dlg.comp_name}'!\n\nDestination:\n{res.export_path}\n\nBundled {len(res.files_included)} files.",
            )

    def prompt_delete_active_component(self, file_path=None, rel_path=None):
        """Prompt to uninstall the component or delete its configuration, and remove all files."""
        target_cfg = Path(file_path) if file_path else self.active_path
        target_rel = rel_path or self.active_rel_path
        if not target_cfg or not target_cfg.exists():
            return

        # Protection: Prevent deleting global config.yaml
        if target_cfg.name in ("config.yaml", "config.yml") and target_rel in ("config.yaml", "config.yml"):
            QMessageBox.warning(self, "Cannot Delete", "The root config.yaml cannot be deleted.")
            return

        from .component_actions import find_component_assets, DeleteComponentDialog
        assets = find_component_assets(target_cfg, target_rel)
        comp_name = assets["comp_name"]
        files_to_delete = assets["files_to_delete"]
        is_comp = assets["is_component"]
        is_hub = assets.get("is_hub_installed", False)

        # Check settings for confirm_delete
        settings = self.get_settings_manager()
        should_confirm = True
        if settings is not None:
            should_confirm = bool(settings.get("components", "confirm_delete", default=True))

        if should_confirm:
            dlg = DeleteComponentDialog(
                comp_name, files_to_delete, is_component=is_comp, is_hub_installed=is_hub, parent=self
            )
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            if dlg.dont_ask_again and settings is not None:
                settings.set("components", "confirm_delete", False)

        # Execute uninstallation / deletion
        deleted_count = 0
        installer = self.components_tree._get_installer() if hasattr(self.components_tree, "_get_installer") else None

        # If it is a managed Hub component with metadata, uninstall via installer
        if is_hub and assets.get("meta") and installer:
            try:
                installer.uninstall_by_metadata(assets["meta"])
                deleted_count = len(files_to_delete)
                # The installer hides owned YAMLs, restores pre-install versions,
                # and retains edits/copies. Do not delete those paths again here.
                files_to_delete = []
            except Exception as exc:
                self.log_fn(f"Installer uninstall notice: {exc}")
                QMessageBox.warning(self, "Uninstall Incomplete", f"Component cleanup stopped before deleting YAML files:\n{exc}")
                return

        # Delete any remaining files or directories in files_to_delete
        for item in files_to_delete:
            try:
                if item.is_dir():
                    shutil.rmtree(str(item), ignore_errors=True)
                    deleted_count += 1
                    if installer:
                        installer._prune_empty_component_dirs(item)
                elif item.is_file():
                    item.unlink(missing_ok=True)
                    deleted_count += 1
                    if installer:
                        installer._prune_empty_config_dirs(item)
            except Exception as exc:
                self.log_fn(f"Error removing {item}: {exc}")

        # Notify plugin manager if this was an active plugin
        w = self.window()
        plugin_mgr = getattr(w, "plugin_manager", None)
        if plugin_mgr is not None and is_hub:
            comp_id = assets.get("comp_id")
            if comp_id:
                try:
                    plugin_mgr.disable_plugin(comp_id)
                except Exception:
                    pass

        action_desc = "Uninstalled component" if is_hub else ("Deleted component" if is_comp else "Deleted configuration")
        self.log_fn(f"{action_desc} '{comp_name}' and all associated files.")

        from .config_viewer import invalidate_config_tree
        invalidate_config_tree()
        main_window = self.window()
        config_tabs = getattr(main_window, "config_tab_manager", None)
        if config_tabs is not None:
            config_tabs.refresh_component_choices()

        # Reload components tree and auto-select fallback
        self.reload_components()

    def hideEvent(self, event):
        super().hideEvent(event)
        if hasattr(self, "viewer") and self.viewer and self.viewer.is_dirty:
            self.save_current()

    def _on_component_file_renamed(self, old_rel, new_rel):
        self.active_path = self.viewer.current_path
        self.active_rel_path = new_rel
        self.components_tree.current_rel_path = new_rel
        self.components_tree.populate()
        self._update_component_action_buttons()
        self._update_raw_yaml_view()

    def get_hub_client(self):
        """Resolve a HubClient instance from attributes, window, parent, or fallback."""
        if self.hub_client is not None:
            return self.hub_client
        w = self.window()
        if hasattr(w, "hub_client") and w.hub_client is not None:
            self.hub_client = w.hub_client
            return self.hub_client
        parent = self.parent()
        if parent is not None and hasattr(parent, "hub_client") and parent.hub_client is not None:
            self.hub_client = parent.hub_client
            return self.hub_client
        workspace_dir = (
            self.components_tree.root_dir.parent.parent
            if hasattr(self, "components_tree") and self.components_tree.root_dir
            else Path(__file__).resolve().parents[2]
        )
        data_dir = workspace_dir / ".thetaide"
        self.hub_client = HubClient(
            workspace_dir=workspace_dir,
            data_dir=data_dir,
            on_change_callback=lambda comp_id, act: self.reload_components(),
        )
        return self.hub_client

    def _ensure_hub_view(self):
        """Lazily initialize and embed the HubView widget into the content stack."""
        if self.hub_view is not None:
            return self.hub_view

        client = self.get_hub_client()
        self.hub_view = HubView(
            client=client,
            initial_kind=None,
            mode="components",
            close_button_text="Back to Parameters",
            show_footer=False,
            title="ThetaHub",
            include_plugins=False,
            all_label="All",
            parent=self,
        )
        self.hub_view.close_requested.connect(self.show_params)
        self.content_stack.addWidget(self.hub_view)
        return self.hub_view

    def show_hub(self, initial_kind: str | None = None):
        """Switch view to the embedded Component Hub in place of the parameters panels."""
        if hasattr(self, "viewer") and self.viewer and self.viewer.is_dirty:
            self.save_current()
        hub = self._ensure_hub_view()
        if initial_kind:
            hub._select_kind(initial_kind)
        self.content_stack.setCurrentWidget(hub)
        self.btn_toggle_raw.hide()
        self.btn_hub.setChecked(True)
        self.is_hub_active = True

    def show_params(self):
        """Switch view back to the parameters configuring panels."""
        self.content_stack.setCurrentWidget(self.content_splitter)
        self.btn_toggle_raw.show()
        self.btn_hub.setChecked(False)
        self.is_hub_active = False

    def toggle_hub(self):
        """Toggle between parameters view and embedded Component Hub."""
        if self.is_hub_active:
            self.show_params()
        else:
            self.show_hub()

    def _update_raw_yaml_view(self):
        """Refresh raw YAML editor content."""
        if not self.active_path or not self.active_path.exists():
            self.raw_edit.setPlainText("")
            self.raw_status.setText("No component loaded")
            return
        try:
            content = self.active_path.read_text(encoding="utf-8")
            self.raw_edit.setPlainText(content)
            self.raw_status.setText(f"{self.active_rel_path} • {len(content.splitlines())} lines")
        except Exception as exc:
            self.raw_edit.setPlainText(f"# Error reading file:\n# {exc}")
            self.raw_status.setText(f"Error: {exc}")

    def _on_viewer_changed(self):
        """When viewer data changes, update the raw view if visible."""
        if self.raw_panel.isVisible():
            try:
                yaml_str = yaml.safe_dump(self.viewer.raw_data, sort_keys=False)
                self.raw_edit.setPlainText(yaml_str)
                self.raw_status.setText(f"{self.active_rel_path} • modified (unsaved)")
            except Exception:
                pass

    def _on_viewer_saved(self):
        """Fired when viewer saves to disk."""
        self._update_raw_yaml_view()
        if self.active_rel_path:
            self.log_fn(f"Saved component configuration: {self.active_rel_path}")
            self.component_saved.emit(self.active_rel_path)

    def save_current(self):
        """Save changes to the currently loaded component config."""
        success = self.viewer.save_to_disk()
        if success:
            self._update_raw_yaml_view()
            if self.active_rel_path:
                self.log_fn(f"Saved component configuration: {self.active_rel_path}")
                self.component_saved.emit(self.active_rel_path)

    def new_component(self):
        """Delegate to tree's new component creation dialog."""
        self.components_tree.prompt_new_component()

    def duplicate_component(self):
        """Copy only the selected YAML configuration, sharing its implementation."""
        self.components_tree.prompt_duplicate_component_yaml()

    def deep_copy_component(self):
        """Copy the selected YAML configuration and its implementation."""
        self.components_tree.prompt_deep_copy_component()

    def reload_components(self, target_rel_path: str | None = None, ensure_expanded: str | None = None):
        """Reload the tree and active file from disk without collapsing folders."""
        curr = target_rel_path or self.active_rel_path
        if not ensure_expanded and curr:
            p = Path(curr).parent
            if str(p) and str(p) != ".":
                ensure_expanded = str(p).replace("\\", "/")

        self.components_tree.populate(ensure_expanded=ensure_expanded)
        if curr and (self.components_tree.root_dir / curr).exists():
            if self.is_hub_active:
                self.components_tree.select_file(curr, emit_signal=False)
                self.active_path = self.components_tree.root_dir / curr
                self.active_rel_path = curr
                self.viewer.load_file(self.active_path, curr)
                self._update_component_action_buttons()
                self._update_raw_yaml_view()
            else:
                self.components_tree.select_file(curr)
        else:
            # If the current active file no longer exists (e.g. was uninstalled),
            # try to select a sibling file in the same folder first so we don't jump away.
            selected_alt = False
            if curr:
                parent_dir = self.components_tree.root_dir / Path(curr).parent
                if parent_dir.exists() and parent_dir.is_dir():
                    siblings = sorted([p for p in parent_dir.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])
                    if siblings:
                        sibling_rel = str(siblings[0].relative_to(self.components_tree.root_dir)).replace("\\", "/")
                        if self.is_hub_active:
                            self.components_tree.select_file(sibling_rel, emit_signal=False)
                            self.active_path = self.components_tree.root_dir / sibling_rel
                            self.active_rel_path = sibling_rel
                            self.viewer.load_file(self.active_path, sibling_rel)
                            self._update_component_action_buttons()
                            self._update_raw_yaml_view()
                            selected_alt = True
                        else:
                            if self.components_tree.select_file(sibling_rel):
                                selected_alt = True
            if not selected_alt and not self.is_hub_active:
                self.init_default_component()

        if self.is_hub_active and self.hub_view is not None:
            self.content_stack.setCurrentWidget(self.hub_view)

        self.log_fn("Reloaded component files from disk.")

    def toggle_raw_preview(self, checked=None):
        """Show or hide the raw YAML preview panel."""
        if checked is None:
            checked = self.btn_toggle_raw.isChecked()
        else:
            self.btn_toggle_raw.setChecked(checked)
        self.raw_panel.setVisible(checked)
        if hasattr(self, "content_splitter"):
            if checked:
                self.content_splitter.setSizes([600, 420])
                self._update_raw_yaml_view()
            else:
                self.content_splitter.setSizes([1000, 0])

    def _open_hub(self):
        """Show the embedded Component Hub."""
        self.show_hub()

