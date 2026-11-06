"""Actions and dialogs for managing components: opening source code in editor and uninstalling/deleting components."""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

# Re-export headless asset discovery functions from core services
from src.app.services.component_exporter import find_component_assets, find_workspace_root


def open_in_editor(path: Path) -> bool:
    """Open a file or directory in the user's preferred or default code editor."""
    resolved = path.resolve()
    path_str = str(resolved)

    # 1. Check $VISUAL or $EDITOR environment variables
    env_editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if env_editor:
        try:
            import shlex
            cmd = shlex.split(env_editor) + [path_str]
            subprocess.Popen(cmd)
            return True
        except Exception:
            pass

    # 2. Check for common code editors in PATH
    for editor_cmd in ("code", "cursor", "subl"):
        if shutil.which(editor_cmd):
            try:
                subprocess.Popen([editor_cmd, path_str])
                return True
            except Exception:
                pass

    # 3. macOS: use native 'open'
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", path_str])
            return True
        except Exception:
            pass

    # 4. Windows: use 'os.startfile'
    if sys.platform == "win32":
        try:
            os.startfile(path_str)
            return True
        except Exception:
            pass

    # 5. Linux: use xdg-open
    if shutil.which("xdg-open"):
        try:
            subprocess.Popen(["xdg-open", path_str])
            return True
        except Exception:
            pass

    # 6. Fallback to Qt QDesktopServices
    return QDesktopServices.openUrl(QUrl.fromLocalFile(path_str))


class DeleteComponentDialog(QDialog):
    """Confirmation dialog when uninstalling a component or deleting a config from the components panel.

    Prompts to confirm, lists all files/folders that will be permanently removed,
    and includes a 'Don't ask again' checkbox.
    """

    def __init__(
        self,
        comp_name: str,
        files_to_delete: List[Path],
        is_component: bool = True,
        is_hub_installed: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.is_component = is_component
        self.is_hub_installed = is_hub_installed
        self.dont_ask_again = False

        if is_hub_installed:
            self.setWindowTitle("Uninstall Component")
            action_verb = "Uninstall"
        elif is_component:
            self.setWindowTitle("Delete Component")
            action_verb = "Delete"
        else:
            self.setWindowTitle("Delete Configuration")
            action_verb = "Delete"

        self.setFixedWidth(520)

        layout = QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(22, 20, 22, 20)

        title = QLabel(f"{action_verb} {comp_name}?")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ebdbb2;")
        layout.addWidget(title)

        configs_count = len([f for f in files_to_delete if f.is_file() and f.suffix in (".yaml", ".yml")])
        if is_hub_installed:
            if configs_count > 1:
                desc_text = (
                    f"This file is part of the Hub-installed component suite <b>{comp_name}</b>, which provides "
                    f"<b>{configs_count} configurations</b>.<br><br>"
                    f"Uninstalling removes the source code and hides its YAMLs. Edited YAMLs are preserved in "
                    f"ignored IDE storage and restored when the component is reinstalled:"
                )
            else:
                desc_text = (
                    f"Are you sure you want to uninstall Hub component <b>{comp_name}</b>?<br>"
                    "The source code will be removed and its YAML hidden. Edited YAML is preserved in ignored IDE "
                    "storage and restored when the component is reinstalled:"
                )
        elif is_component:
            if configs_count > 1:
                desc_text = (
                    f"This file is part of local component <b>{comp_name}</b>, which provides "
                    f"<b>{configs_count} configurations</b>.<br><br>"
                    f"Are you sure you want to delete this component, including its source code directory and all configurations?"
                )
            else:
                desc_text = (
                    f"Are you sure you want to delete component <b>{comp_name}</b>?<br>"
                    "This action cannot be undone. Its source code directory and configuration file will be permanently deleted:"
                )
        else:
            desc_text = (
                f"Are you sure you want to delete configuration <b>'{comp_name}'</b>?<br>"
                "This action cannot be undone. The configuration file will be permanently removed from disk:"
            )

        desc = QLabel(desc_text)
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a89984; font-size: 12px; line-height: 1.4;")
        layout.addWidget(desc)

        # File box listing all items to delete
        files_frame = QFrame()
        files_frame.setStyleSheet(
            "background-color: #1d2021; border: 1px solid #3c3836; border-radius: 4px; padding: 6px;"
        )
        fb_layout = QVBoxLayout(files_frame)
        fb_layout.setSpacing(4)
        fb_layout.setContentsMargins(8, 6, 8, 6)

        ws = find_workspace_root()
        for f in files_to_delete:
            try:
                rel = f.relative_to(ws)
            except ValueError:
                rel = f
            suffix = "/" if f.is_dir() else ""
            lbl_f = QLabel(f"• {rel}{suffix}")
            lbl_f.setStyleSheet("color: #ebdbb2; font-family: monospace; font-size: 11px;")
            lbl_f.setWordWrap(True)
            fb_layout.addWidget(lbl_f)

        layout.addWidget(files_frame)

        self.chk_dont_ask = QCheckBox("Don't ask again")
        self.chk_dont_ask.setStyleSheet("color: #a89984; font-size: 12px; margin-top: 4px;")
        layout.addWidget(self.chk_dont_ask)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch()

        btn_action = QPushButton(f"{action_verb} Permanently")
        btn_action.setStyleSheet(
            "background-color: #cc241d; color: #ebdbb2; font-weight: bold; padding: 6px 14px; border-radius: 4px;"
        )
        btn_action.clicked.connect(self._on_confirm)
        btn_row.addWidget(btn_action)

        layout.addLayout(btn_row)

    def _on_confirm(self):
        self.dont_ask_again = self.chk_dont_ask.isChecked()
        self.accept()


def component_host_attribute(widget, name):
    """Find host services from embedded Hub views as well as Hub dialogs."""
    while widget is not None:
        value = getattr(widget, name, None)
        if value is not None:
            return value
        widget = widget.parent()
    return None


class ComponentDevelopmentActions:
    """Shared user actions for component-tree, Components and Hub surfaces."""

    def __init__(self, workspace, parent, client=None):
        from src.app.services.local_components import LocalComponents
        self.service = LocalComponents(workspace)
        self.parent = parent
        self.client = client or component_host_attribute(parent, "hub_client")
        self.editor = component_host_attribute(parent, "editor_service")

    def changed(self, component_id, action):
        from .config_viewer import invalidate_config_tree
        invalidate_config_tree()
        if self.client:
            self.client.refresh_local_components()
            self.client.componentChanged.emit(component_id, action)
            self.client.indexLoaded.emit(self.client.components)

    def edit(self, config):
        from PyQt6.QtWidgets import QMessageBox
        try:
            assets = find_component_assets(Path(config), workspace_dir=self.service.workspace)
            if assets.get("code_dir"):
                assets, context = self.service.context(config)
                paths = self.service.owned_paths(assets)
                if self.editor:
                    success = self.editor.open_folder(assets["code_dir"], initial_file=assets["primary_code_file"], owned_paths=[*paths, context])
                else:
                    success = open_in_editor(assets["code_dir"])
            else:
                target = assets.get("primary_code_file") or Path(config)
                success = self.editor.open_file(target) if self.editor else open_in_editor(target)
            if not success:
                raise OSError("Could not open the installed component in the selected editor.")
            return True
        except Exception as exc:
            QMessageBox.warning(self.parent, "Open in Editor", str(exc))
            return False

    def deep_copy(self, config, name=None):
        """Begin an empty inline name, or commit that name to a saved copy."""
        if name is None:
            begin = component_host_attribute(self.parent, "begin_deep_copy_component")
            tree = component_host_attribute(self.parent, "components_tree")
            if begin is None and tree is not None:
                begin = tree.begin_deep_copy_component
            if begin is not None:
                show_params = component_host_attribute(self.parent, "show_params")
                if show_params:
                    show_params()
                begin(Path(config))
            else:
                self._begin_inline_copy(config)
            return None
        if not name.strip() or any(char in name for char in '/\\<>:"|?*'):
            raise ValueError("Enter a component name without slashes or special characters.")
        new_id = re.sub(r"[^a-z0-9_]+", "_", name.strip().lower()).strip('_')
        if not new_id or not new_id[0].isalpha():
            new_id = "model_" + new_id
        # Save parameter edits and honor dirty source buffers before collecting.
        widget = self.parent
        while widget is not None:
            viewer = getattr(widget, "viewer", None)
            if viewer is not None and getattr(viewer, "is_dirty", False):
                viewer.save_to_disk()
                if viewer.is_dirty:
                    return None
            widget = widget.parent()
        if self.editor and not self.editor.prepare_component_copy():
            return None
        copied = self.service.deep_copy(config, new_id, name.strip())
        self.changed(new_id, "copy")
        return copied

    def _begin_inline_copy(self, config):
        """Standalone Hub dialogs have no component tree; name a new row in place."""
        from PyQt6.QtWidgets import QWidget

        from .config_tree import InlineGroupEditor
        previous = getattr(self.parent, "_cancel_pending_copy", None)
        if previous is not None:
            previous()
        row = QWidget(self.parent)
        layout = QVBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("New component"))
        editor = InlineGroupEditor()
        editor.setObjectName("componentCopyName")
        editor.setPlaceholderText("Component name…")
        error = QLabel()
        error.setObjectName("componentCopyError")
        error.setWordWrap(True)
        error.hide()
        layout.addWidget(editor)
        layout.addWidget(error)
        self.parent.layout().addWidget(row)
        self.parent._copy_name_row = row
        finished = False
        def cancel():
            nonlocal finished
            if finished:
                return
            finished = True
            self.parent._copy_name_row = None
            row.deleteLater()
        def commit():
            nonlocal finished
            if finished:
                return
            if not editor.text().strip():
                cancel()
                return
            finished = True
            try:
                copied = self.deep_copy(config, editor.text())
                if copied is None:
                    finished = False
                    editor.setFocus()
                    return
            except Exception as exc:
                finished = False
                error.setText(str(exc))
                error.show()
                editor.setFocus()
                return
            panel = component_host_attribute(self.parent, "components_panel")
            if panel:
                panel.show_params()
                panel.components_tree._select_created_component_file(copied)
                tabs = component_host_attribute(self.parent, "tabs")
                if tabs:
                    tabs.setCurrentWidget(panel)
                if isinstance(self.parent.window(), QDialog):
                    self.parent.window().accept()
            row.deleteLater()
        self.parent._cancel_pending_copy = cancel
        editor.returnPressed.connect(commit)
        editor.editingFinished.connect(commit)
        editor.escape_pressed.connect(cancel)
        editor.lost_focus_empty.connect(cancel)
        editor.setFocus()

    def delete_local(self, component, installer):
        from PyQt6.QtWidgets import QMessageBox
        try:
            paths, references = self.service.deletion_details(component, installer)
            detail = "Move these locations to Trash:\n" + "\n".join(map(str, paths))
            detail += "\nReferenced experiments:\n" + ("\n".join(map(str, references)) or "None")
            if QMessageBox.question(self.parent, "Delete local component", detail) != QMessageBox.StandardButton.Yes:
                return False
            self.service.delete(component, installer)
            self.changed(component.id, "delete")
            return True
        except Exception as exc:
            QMessageBox.warning(self.parent, "Delete failed", str(exc))
            return False
