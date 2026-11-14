"""Full-screen, simplified, boxed configuration viewer for Theta-IDE."""
import copy
import json
import math
import re
import weakref
from pathlib import Path
from typing import Any, Mapping

import yaml
from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QObject, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDrag, QIcon, QPainter, QPen, QPixmap
from PyQt6 import sip
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_model import COMPONENT_ID_KEYS, ConfigTree, component_selection, default_method_model, is_group_defaults_path
from .config_model import add_method as add_method_to
from .config_model import remove_method as remove_method_from
from .sidetabs import svg_icon
from .theme import theme_color
from .widgets import (
    ComboBox as BaseComboBox, DoubleSpinBox, SpinBox, label, SmoothScrollArea, ToggleSlider,
    TagCloudWidget, LayerStackWidget, HierarchicalComboBox, ExpandingTextEdit, FlowLayout, WrappingFlowWidget, integer_field,
    ScientificDoubleSpinBox,
)
from .metric_picker import MetricPicker, logged_metrics
from .parameter_reorder import FlowRows, FormRows, ParameterReorder
from src.app.core.provider_discovery import discovery_cached, provider_discovery_snapshot
from src.app.telemetry.selection import SECTIONS as METRIC_SECTIONS

# Compatibility exports for callers while sections migrate.
from .config_panel.controls import (
    CONTENT_WIDTH, NUMBER_FIELD_WIDTH, TEXT_FIELD_WIDTH, HIERARCHY_INDENT,
    PARAM_LABEL_WIDTH, PARAM_FIELD_WIDTH, SUBTLE_ICON_COLORS, ComboBox,
    ConfigBox, ParameterForm, IndentedHierarchyBlock, _RowHoverReveal,
    _make_subtle_tool_button, _reveal_remove_on_row_hover, _parameter_row,
    _parameter_title, _collapsible_parameter_label, _parameter_header,
)
from .config_panel.document import ConfigDocument
from .config_panel.bindings import ParameterBinding, component_identity


from .config_panel.controls import (
    CATEGORICAL_CHOICES, _fit_inline_control, _SweepTypePicker, _make_param_adder_button,
    TitleLabel, InlineTitleEditor, MethodDragHandle, MethodCardsContainer,
    _make_categorical_combo, _SectionMoveControls,
)


CompactDoubleSpinBox = ScientificDoubleSpinBox  # compatibility import


from src.app.workspace import WorkspacePaths

from .config_panel.editors import number_field
from .config_panel.document import _leading_directives, _parse_yaml_value, parse_list_field
from .config_panel.layout import _two_column_parameter_layout
from .config_panel.catalogs import (
    _get_available_paradigms, _get_component_kind, get_agent_candidate_hyperparameters, get_model_candidate_hyperparameters, get_training_candidate_hyperparameters, get_method_candidate_hyperparameters, _dataset_workspace, discover_datasets, _recorded_metrics, _monitor_settings, invalidate_config_tree, set_config_tree, config_tree, _read_group_base, _component_profiles,
    TRAINING_HYPERPARAMETERS, CHECKPOINT_KEYS, DEFAULT_CHECKPOINTS, FIXED_BOX_KEYS,
)


class ConfigViewer(QWidget):
    """Modern, wide, boxed configuration viewer that displays experiment and

    component configurations in themed cards.
    """
    config_changed = pyqtSignal()
    save_requested = pyqtSignal()
    dirty_state_changed = pyqtSignal(str, bool)
    dirty_changed = pyqtSignal(bool)
    file_renamed = pyqtSignal(str, str)
    group_renamed = pyqtSignal(str, str)
    dir_renamed = pyqtSignal(str, str)
    edit_code_requested = pyqtSignal(object, str)
    export_component_requested = pyqtSignal(object, str)
    delete_component_requested = pyqtSignal(object, str)

    INHERIT = "Default"

    @property
    def tree(self):
        return self._workspace_tree if self._workspace is not None else config_tree()

    def refresh_tree(self):
        if self._workspace is not None:
            self._workspace_tree = ConfigTree(self._workspace.config, workspace=self._workspace)
        else:
            invalidate_config_tree()
            config_tree(force_reload=True)
        return self.tree

    def __init__(self, settings_manager=None, parent=None, plugin_manager=None, *, workspace=None):
        if isinstance(settings_manager, QWidget) and parent is None:
            parent = settings_manager
            settings_manager = None
        super().__init__(parent)
        self._workspace = workspace
        self._workspace_tree = ConfigTree(workspace.config, workspace=workspace) if workspace is not None else None
        self.setStyleSheet("""
            QComboBox { padding-right: 6px; }
            QComboBox::drop-down { width: 0px; border: 0; background: transparent; }
            QComboBox::down-arrow, QComboBox::down-arrow:disabled { image: none; width: 0px; height: 0px; }
        """)
        self.settings_manager = settings_manager
        self.plugin_manager = plugin_manager
        if self.plugin_manager is not None:
            self.plugin_manager.pluginStateChanged.connect(self._on_plugin_state_changed)
        self.is_component_mode = False
        self.current_path = None
        self.current_rel_path = None
        self.raw_data = {}
        self.is_dirty = False
        self._block_updates = False
        self._is_editing_title = False
        self._committing_rename = False
        self._is_editing_group = False
        self._committing_group_rename = False
        self.field_widgets = {}
        self._preamble = ""
        self.auto_save = False
        # Field edits (typing) auto-save after a pause instead of on every keystroke: each save
        # rewrites the YAML, invalidates the config tree cache (a rescan of every config on the
        # next lookup) and triggers a pipeline compose.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(400)
        self._autosave_timer.timeout.connect(self.flush_pending_save)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._render_boxes)
        self._sweep_refresh_timer = QTimer(self)
        self._sweep_refresh_timer.setSingleShot(True)
        self._sweep_refresh_timer.timeout.connect(self._refresh_sweep_box)
        self._parameter_orders = {}
        self._parameter_reorders = []
        self._pending_new_method = False
        self._pending_new_method_spec = {}
        self.spin_timesteps = SpinBox()
        self.spin_seed = SpinBox()
        self.spin_eval_ep = SpinBox()
        self.txt_description = QLineEdit()
        self.chk_tb = None
        self.chk_save_ds = None
        self.chk_recover = None
        self.chk_no_plot = None
        self.view_mode = "resolved" if (settings_manager and settings_manager.show_resolved_defaults) else "overrides"
        self._form_settings = self._form_settings_snapshot()
        if self.settings_manager:
            self.settings_manager.changed.connect(self._on_settings_changed)

        self._init_ui()

    def _on_plugin_state_changed(self, plugin_id: str, _enabled: bool) -> None:
        if plugin_id == "tensorboard" and not self.is_component_mode:
            self._render_boxes()

    def _plugin_enabled(self, plugin_id: str) -> bool:
        if self.plugin_manager is not None:
            active_check = getattr(self.plugin_manager, "is_plugin_active", None)
            if callable(active_check):
                return active_check(plugin_id)
            return self.plugin_manager.is_plugin_enabled(plugin_id)
        return True

    def _form_settings_snapshot(self):
        sm = self.settings_manager
        return tuple(bool(sm and getattr(sm, key)) for key in (
            "show_resolved_defaults", "show_method_agent_defaults",
            "show_method_model_defaults", "show_all_default_params",
        ))

    def _on_settings_changed(self):
        state = self._form_settings_snapshot()
        previous = self._form_settings
        if state == previous:
            return
        self._form_settings = state
        if state[0] != previous[0]:
            self.view_mode = "resolved" if state[0] else "overrides"
            self.chk_view_mode.blockSignals(True)
            self.chk_view_mode.setChecked(state[0])
            self.chk_view_mode.blockSignals(False)
        self._render_boxes()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        # Header Info Bar
        self.header_bar = QWidget()
        hb_layout = QHBoxLayout(self.header_bar)
        hb_layout.setContentsMargins(12, 6, 12, 6)
        hb_layout.setSpacing(10)

        self.title_layout = QHBoxLayout()
        self.title_layout.setContentsMargins(0, 0, 0, 0)
        self.title_layout.setSpacing(0)
        hb_layout.addLayout(self.title_layout)

        self.group_badge = TitleLabel("")
        self.group_badge.setObjectName("heading")
        self.group_badge.setToolTip("Group (click to rename)")
        self.group_badge.clicked.connect(self._start_group_edit)
        self.group_badge.double_clicked.connect(self._start_group_edit)
        self.group_badge.hide()
        self.title_layout.addWidget(self.group_badge)

        self.group_editor = QLineEdit()
        self.group_editor.setObjectName("groupEditor")
        self.group_editor.setStyleSheet(
            "font-size: 20px; font-weight: 600; padding: 2px 8px; border-radius: 4px;"
        )
        self.group_editor.setMinimumWidth(120)
        self.group_editor.setMaximumWidth(300)
        self.group_editor.hide()
        self.group_editor.returnPressed.connect(self._commit_group_edit)
        self.group_editor.installEventFilter(self)
        self.title_layout.addWidget(self.group_editor)

        self.breadcrumb_sep = QLabel(" / ")
        self.breadcrumb_sep.setObjectName("heading")
        self.breadcrumb_sep.hide()
        self.title_layout.addWidget(self.breadcrumb_sep)

        self.file_title = TitleLabel("No config loaded")
        self.file_title.clicked.connect(self._start_title_edit)
        self.file_title.double_clicked.connect(self._start_title_edit)

        self.title_layout.addWidget(self.file_title)

        self.title_editor = QLineEdit()
        self.title_editor.setObjectName("titleEditor")
        self.title_editor.setStyleSheet(
            "font-size: 20px; font-weight: 600; padding: 2px 8px; border-radius: 4px;"
        )
        self.title_editor.setMinimumWidth(200)
        self.title_editor.setMaximumWidth(400)
        self.title_editor.hide()
        self.title_editor.returnPressed.connect(self._commit_title_edit)
        self.title_editor.installEventFilter(self)
        self.title_layout.addWidget(self.title_editor)

        self.paradigm_badge = label("", "badge")
        self.paradigm_badge.hide()

        self.dirty_status = label("", "muted")
        self.dirty_status.hide()

        hb_layout.addStretch()

        self.btn_edit_code = _make_subtle_tool_button(
            "pencil", "Open component source code in editor", size=26, icon_size=15
        )
        self.btn_edit_code.setObjectName("btnEditCode")
        self.btn_edit_code.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_edit_code.hide()
        self.btn_edit_code.clicked.connect(self._on_edit_code_clicked)
        hb_layout.addWidget(self.btn_edit_code)

        self.btn_export_component = _make_subtle_tool_button(
            "export", "Export component package for Theta-Hub or GitHub", size=26, icon_size=15
        )
        self.btn_export_component.setObjectName("btnExportComponent")
        self.btn_export_component.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_export_component.hide()
        self.btn_export_component.clicked.connect(self._on_export_component_clicked)
        hb_layout.addWidget(self.btn_export_component)

        self.btn_delete_component = _make_subtle_tool_button(
            "trash", "Uninstall component and all its files", size=26, icon_size=15
        )
        self.btn_delete_component.setObjectName("btnDeleteComponent")
        self.btn_delete_component.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_delete_component.hide()
        self.btn_delete_component.clicked.connect(self._on_delete_component_clicked)
        hb_layout.addWidget(self.btn_delete_component)

        # Maintained for programmatic and test compatibility; hidden from visual header
        self.chk_view_mode = ToggleSlider()
        self.chk_view_mode.setAccessibleName("Show Resolved Defaults")
        self.chk_view_mode.setObjectName("chkViewMode")
        self.chk_view_mode.setChecked(self.view_mode == "resolved")
        self.chk_view_mode.toggled.connect(self._on_view_mode_toggled)
        self.chk_view_mode.hide()

        self.btn_save = None

        root_layout.addWidget(self.header_bar)

        # Main Scroll Area holding the boxes
        self.scroll = SmoothScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setObjectName("configScrollArea")
        self.scroll.verticalScrollBar().setSingleStep(16)

        self.container = QWidget()
        self.container.setMaximumWidth(CONTENT_WIDTH)
        self.boxes_layout = QVBoxLayout(self.container)
        self.boxes_layout.setContentsMargins(12, 10, 12, 16)
        self.boxes_layout.setSpacing(14)

        self.scroll.setWidget(self.container)
        root_layout.addWidget(self.scroll, 1)

    def eventFilter(self, obj, event):
        if obj is getattr(self, "title_editor", None):
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_title_edit()
                    return True
            elif event.type() == QEvent.Type.FocusOut:
                if getattr(self, "_is_editing_title", False) and not getattr(self, "_committing_rename", False):
                    current_stem = self.current_path.stem if self.current_path else ""
                    raw_text = self.title_editor.text().strip()
                    if raw_text and raw_text != current_stem:
                        self._commit_title_edit()
                    else:
                        self._cancel_title_edit()
                return False
        elif obj is getattr(self, "group_editor", None):
            if event.type() == QEvent.Type.KeyPress:
                if event.key() == Qt.Key.Key_Escape:
                    self._cancel_group_edit()
                    return True
            elif event.type() == QEvent.Type.FocusOut:
                if getattr(self, "_is_editing_group", False) and not getattr(self, "_committing_group_rename", False):
                    grp = self._group_name() or ""
                    raw_text = self.group_editor.text().strip()
                    if raw_text.startswith("[") and raw_text.endswith("]"):
                        raw_text = raw_text[1:-1].strip()
                    if raw_text and raw_text != grp:
                        self._commit_group_edit()
                    else:
                        self._cancel_group_edit()
                return False
        return super().eventFilter(obj, event)

    def _start_group_edit(self, event=None):
        if self._hub_read_only():
            return
        grp = self._group_name()
        if not grp or not self.current_path or not self.current_rel_path:
            return
        self._is_editing_group = True
        self.group_badge.hide()
        self.group_editor.setText(grp)
        self.group_editor.show()
        self.group_editor.setFocus()
        self.group_editor.selectAll()

    def _cancel_group_edit(self):
        self._is_editing_group = False
        self.group_editor.hide()
        self.group_badge.show()

    def _commit_group_edit(self):
        if not getattr(self, "_is_editing_group", False) or getattr(self, "_committing_group_rename", False):
            return
        self._committing_group_rename = True
        try:
            old_grp = self._group_name()
            new_grp = self.group_editor.text().strip()
            if new_grp.startswith("[") and new_grp.endswith("]"):
                new_grp = new_grp[1:-1].strip()

            if not new_grp or new_grp == old_grp:
                self._cancel_group_edit()
                return

            invalid_chars = set('/\\:*?"<>|')
            if any(c in invalid_chars for c in new_grp):
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    "The group name cannot contain invalid characters: / \\ : * ? \" < > |",
                )
                self._cancel_group_edit()
                return

            if self.current_path and self.current_rel_path:
                rel_parts_count = len(Path(self.current_rel_path).parts)
                cur = self.current_path
                for _ in range(rel_parts_count):
                    cur = cur.parent
                config_root = cur
            else:
                tree = self.tree
                if not tree or not hasattr(tree, "config_root"):
                    from .config_tree import find_config_root
                    config_root = find_config_root()
                else:
                    config_root = tree.config_root

            old_group_dir = config_root / "experiment" / old_grp
            if not old_group_dir.exists() or not old_group_dir.is_dir():
                QMessageBox.warning(self, "Rename Failed", f"Group folder '{old_grp}' does not exist on disk.")
                self._cancel_group_edit()
                return

            new_group_dir = config_root / "experiment" / new_grp
            if new_group_dir.exists() and old_group_dir.resolve() != new_group_dir.resolve():
                QMessageBox.warning(
                    self,
                    "Folder Already Exists",
                    f"A group folder named '{new_grp}' already exists.",
                )
                self._cancel_group_edit()
                return

            if old_group_dir.name.lower() == new_group_dir.name.lower() and old_group_dir.name != new_group_dir.name:
                temp_dir = old_group_dir.with_name(f"{old_group_dir.name}__theta_tmp_grp")
                old_group_dir.rename(temp_dir)
                temp_dir.rename(new_group_dir)
            else:
                old_group_dir.rename(new_group_dir)

            rel_parts = list(Path(str(self.current_rel_path)).parts)
            if len(rel_parts) >= 2 and rel_parts[0] == "experiment":
                rel_parts[1] = new_grp
            new_rel = "/".join(rel_parts)
            old_rel = self.current_rel_path

            self.current_rel_path = new_rel
            self.current_path = config_root / new_rel

            self._is_editing_group = False
            self.group_editor.hide()
            self.group_badge.setText(str(new_grp))
            self.group_badge.show()

            self.file_renamed.emit(old_rel, new_rel)
            self.dir_renamed.emit(f"experiment/{old_grp}", f"experiment/{new_grp}")
            self.group_renamed.emit(old_grp, new_grp)
            self.config_changed.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Rename Failed", f"Could not rename group:\n{exc}")
            self._cancel_group_edit()
        finally:
            self._committing_group_rename = False

    def _start_title_edit(self, event=None):
        if self._hub_read_only():
            return
        if not self.current_path or not self.current_rel_path:
            return
        self._is_editing_title = True
        self.file_title.hide()
        current_stem = self.current_path.stem
        self.title_editor.setText(current_stem)
        self.title_editor.show()
        self.title_editor.setFocus()
        self.title_editor.selectAll()

    def _on_edit_code_clicked(self):
        if self.current_path:
            self.edit_code_requested.emit(self.current_path, self.current_rel_path or "")

    def _on_export_component_clicked(self):
        if self.current_path:
            self.export_component_requested.emit(self.current_path, self.current_rel_path or "")

    def _on_delete_component_clicked(self):
        if self.current_path:
            self.delete_component_requested.emit(self.current_path, self.current_rel_path or "")

    def _cancel_title_edit(self):
        self._is_editing_title = False
        self.title_editor.hide()
        self.file_title.show()

    def _commit_title_edit(self):
        if not getattr(self, "_is_editing_title", False) or getattr(self, "_committing_rename", False):
            return
        self._committing_rename = True
        try:
            raw_text = self.title_editor.text().strip()
            new_title = raw_text
            if new_title.endswith(".yaml"):
                new_title = new_title[:-5].strip()
            elif new_title.endswith(".yml"):
                new_title = new_title[:-4].strip()

            old_title = self.current_path.stem if self.current_path else ""
            if not new_title or new_title == old_title:
                self._cancel_title_edit()
                return

            invalid_chars = set('/\\:*?"<>|')
            if any(c in invalid_chars for c in new_title):
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    "The config name cannot contain invalid characters: / \\ : * ? \" < > |",
                )
                self._cancel_title_edit()
                return

            old_path = self.current_path
            old_rel = self.current_rel_path
            new_filename = f"{new_title}{old_path.suffix or '.yaml'}"
            new_path = old_path.with_name(new_filename)

            is_same_file = False
            if old_path.exists() and new_path.exists():
                try:
                    is_same_file = old_path.samefile(new_path)
                except Exception:
                    is_same_file = (old_path.name.lower() == new_path.name.lower())

            if new_path.exists() and not is_same_file:
                QMessageBox.warning(
                    self,
                    "File Already Exists",
                    f"A configuration file named '{new_filename}' already exists in this folder.",
                )
                self._cancel_title_edit()
                return

            is_experiment = (
                (old_rel and old_rel.startswith("experiment/"))
                or "methods" in self.raw_data
                or any("_base" in str(d) for d in self.raw_data.get("defaults", []))
            )
            for key in ("algorithm", "name", "architecture", "id"):
                if key in self.raw_data:
                    self.raw_data[key] = new_title

            if is_experiment:
                self.raw_data.pop("experiment_id", None)
                self.raw_data.pop("group", None)
            elif old_rel and old_rel.startswith("agent/"):
                self.raw_data["algorithm"] = new_title
            elif old_rel and old_rel.startswith("env/"):
                self.raw_data["name"] = new_title
            elif old_rel and old_rel.startswith("model/"):
                self.raw_data["architecture"] = new_title


            # Check if methods dict has a method named after old_title
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                methods = self.raw_data["methods"]
                if old_title in methods:
                    methods[new_title] = methods.pop(old_title)

            # Rename file on disk if it exists
            if old_path.exists():
                if old_path.name.lower() == new_path.name.lower() and old_path.name != new_path.name:
                    # Case-only rename: use intermediate temporary file to guarantee APFS/FAT directory entry update
                    temp_path = old_path.with_name(f"{old_path.stem}__theta_tmp_rename{old_path.suffix}")
                    old_path.rename(temp_path)
                    temp_path.rename(new_path)
                else:
                    old_path.rename(new_path)

            self.current_path = new_path
            new_rel = str(Path(old_rel).with_name(new_filename)).replace("\\", "/")
            self.current_rel_path = new_rel

            # Save updated content to disk
            yaml_str = yaml.safe_dump(self.raw_data, sort_keys=False)
            self.current_path.write_text(self._preamble + yaml_str, encoding="utf-8")
            self.is_dirty = False
            self._update_dirty_ui()

            # End edit mode
            self._is_editing_title = False
            self.title_editor.hide()
            self.file_title.setText(new_title)
            self.file_title.setToolTip(f"{new_rel}\n(Click to rename)")
            self.file_title.show()


            self._render_boxes()

            self.file_renamed.emit(old_rel, new_rel)
            self.dirty_state_changed.emit(old_rel, False)
            self.dirty_state_changed.emit(new_rel, False)
            self.config_changed.emit()
            self.save_requested.emit()
        except Exception as exc:
            QMessageBox.critical(self, "Rename Failed", f"Could not rename file:\n{exc}")
            self._cancel_title_edit()
        finally:
            self._committing_rename = False

    @property
    def txt_exp_id(self):
        class _TitleProxy:
            def __init__(proxy_self, viewer):
                proxy_self._viewer = viewer
            def text(proxy_self):
                return proxy_self._viewer.file_title.text()
            def setText(proxy_self, val):
                proxy_self._viewer.file_title.setText(val)
        return _TitleProxy(self)

    def reload_data(self, force_tree_refresh: bool = True):
        """Reload the config and refresh all visual cards/dropdowns."""
        if force_tree_refresh:
            self.refresh_tree()
        if self.current_path and self.current_path.exists():
            if not self.is_dirty:
                self.load_file(self.current_path, self.current_rel_path or "")
            else:
                self._render_boxes()
        elif self.raw_data:
            self._render_boxes()

    @provider_discovery_snapshot()
    def load_file(self, file_path: Path, rel_path: str):
        """Parse YAML file and render visual boxes."""
        self._deferred_load = False
        if getattr(self, "auto_save", False) and self.is_dirty:
            if not self.save_to_disk():
                return

        self._render_timer.stop()
        self._sweep_refresh_timer.stop()
        self._show_empty_shared_params = False
        self._monitor_group_expanded = {}
        self.current_path = Path(file_path)
        self.current_rel_path = rel_path
        self.is_dirty = False
        self._pending_new_method = False
        self._pending_new_method_spec = {}

        try:
            content = self.current_path.read_text(encoding="utf-8")
            self._preamble = _leading_directives(content)
            data = yaml.safe_load(content) or {}
            self.raw_data = data if isinstance(data, dict) else {"content": data}
            self._clean_redundant_params()
        except Exception as exc:
            self.file_title.setText(f"Error loading {self.current_path.stem}")
            return

        self._render_boxes()
        self._update_dirty_ui()

    def _hub_read_only(self):
        if not self.current_path:
            return False
        tree = self.tree
        workspace = tree.workspace if tree else WorkspacePaths.for_process()
        from frontend.hub.installer import HubInstaller
        return HubInstaller(workspace.root, workspace.data_dir).is_installed_file(self.current_path)

    @provider_discovery_snapshot()
    def _render_boxes(self):
        """Rebuild structural changes safely, retaining the current viewport."""
        if getattr(self, "_deferred_load", False):
            return
        if getattr(self, "_rendering_boxes", False):
            self._render_timer.start(0)
            return
        scroll_position = self.scroll.verticalScrollBar().value()
        # A settings/provider update can arrive while a label owns the mouse.
        # Restore that live layout before deleting any of its widgets.
        for controller in self._parameter_reorders:
            if not sip.isdeleted(controller) and controller.dragged is not None:
                controller.finish(False)
        self._sweep_refresh_timer.stop()
        self._rendering_boxes = True
        self.setUpdatesEnabled(False)
        try:
            self._build_boxes()
            for i in range(self.boxes_layout.count()):
                child = self.boxes_layout.itemAt(i).widget()
                if child is not None:
                    child.show()
            self.container.layout().activate()
            self.container.adjustSize()
            self.scroll.verticalScrollBar().setValue(scroll_position)
        finally:
            self._block_updates = False
            self._rendering_boxes = False
            self.setUpdatesEnabled(True)

    def _build_boxes(self):
        """Clear and build the configuration boxes."""
        self._block_updates = True
        self.field_widgets.clear()
        self._rendered_sweeps = set()

        # Clear existing layout
        while self.boxes_layout.count():
            item = self.boxes_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
            elif item.layout():
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    sw = sub.widget()
                    if sw:
                        sw.setParent(None)
                        sw.deleteLater()

        # Update header
        group = self._group_name()
        is_base = self._is_base_config()
        self.title_layout.setSpacing(10 if is_base else 0)

        if is_base:
            self.group_badge.setText(str(group) if group else "group")
            self.group_badge.setToolTip(f"{self.current_rel_path}\n(Base configuration for {group})")
            self.group_badge.show()
            self.breadcrumb_sep.hide()
            self.file_title.setText("Group Defaults")
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Base configuration for {group})")
        elif group:
            rel_parts = Path(str(self.current_rel_path or "")).parts
            if len(rel_parts) > 3:
                sub_crumbs = " / ".join(rel_parts[2:-1])
                self.group_badge.setText(f"{group} / {sub_crumbs}")
            else:
                self.group_badge.setText(str(group))
            self.group_badge.setToolTip(f"Group: {group}\n(Click to rename group)")
            self.group_badge.show()
            self.breadcrumb_sep.show()
            self.file_title.setText(self.current_path.stem)
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Click to rename)")
        else:
            self.group_badge.hide()
            self.breadcrumb_sep.hide()
            self.file_title.setText(self.current_path.stem if self.current_path else "")
            self.file_title.setToolTip(f"{self.current_rel_path}\n(Click to rename)")

        if getattr(self, "is_component_mode", False) and self.current_path:
            component_type, _ = _get_component_kind(str(self.current_rel_path or ""))
            self.file_title.setText(f"{component_type}: {self.current_path.stem}")

        if getattr(self, "is_component_mode", False) and self.current_path:
            self.btn_edit_code.show()
            self.btn_export_component.show()
            self.btn_delete_component.show()
        else:
            self.btn_edit_code.hide()
            self.btn_export_component.hide()
            self.btn_delete_component.hide()


        is_experiment = (
            (self.current_rel_path and self.current_rel_path.startswith("experiment/") and "/methods/" not in self.current_rel_path) or
            "methods" in self.raw_data or
            any("_base" in str(d) for d in self.raw_data.get("defaults", []))
        )

        if self.settings_manager:
            self.view_mode = "resolved" if self.settings_manager.show_resolved_defaults else "overrides"
            self.chk_view_mode.blockSignals(True)
            self.chk_view_mode.setChecked(self.view_mode == "resolved")
            self.chk_view_mode.blockSignals(False)

        if is_experiment:
            self._render_experiment_boxes()
        else:
            self._render_generic_boxes()

        self._place_sweep_subtitles()
        if is_experiment:
            self._setup_experiment_box_order()
        self._setup_parameter_reordering()
        self.boxes_layout.addStretch()
        read_only = self._hub_read_only()
        self.container.setEnabled(not read_only)
        if read_only:
            notice = QLabel("Hub component · Read-only. Make a component copy to edit.")
            notice.setWordWrap(True)
            self.boxes_layout.insertWidget(0, notice)
            self.file_title.setToolTip("Hub component is read-only. Make a component copy to edit it.")
        self._block_updates = False

    def _setup_parameter_reordering(self, only_box=None):
        """Attach the same sibling-list behavior to ordinary parameter layouts."""
        if only_box is None:
            self._parameter_reorders = []
        path = str(self.current_path.resolve()) if self.current_path else str(self.current_rel_path or "")
        saved = self.settings_manager.get("config_viewer", "parameter_orders", default=[]) if self.settings_manager else []
        orders = self._parameter_orders.setdefault(path, next(
            (dict(entry.get("groups", {})) for entry in saved if entry.get("path") == path), {}))
        for i in range(self.boxes_layout.count()):
            box = self.boxes_layout.itemAt(i).widget()
            if not isinstance(box, ConfigBox):
                continue
            if only_box is not None and box is not only_box:
                continue
            title = next((w.text() for w in box.findChildren(QLabel)
                          if w.objectName() in ("heading", "cardTitle")), "Parameters")
            layouts = []
            for grid in box.findChildren(QGridLayout):
                if hasattr(grid, "_parameter_rows"):
                    layouts.append(grid._parameter_rows)
            layouts.extend(FormRows(form) for form in box.findChildren(QFormLayout)
                           if not getattr(form, "_parameter_column_row", False))
            layouts.extend(FlowRows(layout) for layout in box.findChildren(FlowLayout)
                           if layout.property("parameterReorderFlow"))
            # Name child levels before applying a saved parent order.
            for rows in layouts:
                for group in rows.groups:
                    if group.caption is None:
                        continue
                    caption = group.caption
                    text = caption.text() if isinstance(caption, QLabel) else next(
                        (w.text() for w in caption.findChildren(QLabel)), "")
                    for child in group.widgets():
                        if isinstance(child, IndentedHierarchyBlock):
                            child._parameter_reorder_context = text
            occurrences = {}
            for rows in layouts:
                if not any(group.caption is not None for group in rows.groups):
                    continue
                ancestors = []
                parent = rows.layout.parentWidget()
                while parent is not None and parent is not box:
                    context = getattr(parent, "_parameter_reorder_context", None)
                    if context is None:
                        context = parent.property("parameterPath")
                    if context is None:
                        card_title = next((w.text() for w in parent.findChildren(
                            QLabel, options=Qt.FindChildOption.FindDirectChildrenOnly)
                            if w.objectName() == "cardTitle"), None)
                        context = card_title
                    if context is not None:
                        ancestors.insert(0, context)
                    parent = parent.parentWidget()
                context = rows.layout.property("parameterContext") or rows.layout.objectName()
                base = json.dumps([title, ancestors, context])
                index = occurrences.get(base, 0)
                occurrences[base] = index + 1
                scope = json.dumps([base, index])
                region = rows.layout.property("parameterRegion")
                declared = getattr(rows.layout, "_declared_groups", None)
                stable = json.dumps(["region", region]) if region else (
                    json.dumps(["parameters", context]) if declared is not None and context else None)
                previous = orders.get(scope, [])
                if stable:
                    # Old context/runtime grids shared an empty context and
                    # differed by occurrence. Retain those entries for rollback.
                    if not previous and region in ("experiment.context", "experiment.runtime"):
                        old_base = json.dumps([title, ancestors, ""])
                        old_scope = json.dumps([old_base, 0 if region.endswith("context") else 1])
                        previous = orders.get(old_scope, [])
                    previous = orders.get(stable, previous)
                    scope = stable
                def save(order, s=scope, p=path, groups=orders):
                    groups[s] = list(order)
                    if self.settings_manager:
                        stored = {entry["path"]: entry["groups"] for entry in self.settings_manager.get(
                            "config_viewer", "parameter_orders", default=[])}
                        stored[p] = groups
                        self.settings_manager.persist_key("config_viewer", "parameter_orders",
                            [{"path": file, "groups": value} for file, value in stored.items()])
                controller = ParameterReorder(rows, box, self.scroll, previous, save)
                self._parameter_reorders.append(controller)

    def _on_view_mode_toggled(self, checked: bool):
        self.view_mode = "resolved" if checked else "overrides"
        self._render_boxes()

    def _get_resolved_defaults(self) -> dict:
        from .config_panel.domain import _get_resolved_defaults
        return _get_resolved_defaults(self)

    def _group_name(self):
        from .config_panel.domain import _group_name
        return _group_name(self)

    def env_selection(self):
        from .config_panel.domain import env_selection
        return env_selection(self)

    def _inherited_paradigm(self):
        from .config_panel.domain import _inherited_paradigm
        return _inherited_paradigm(self)

    def _inherited_env(self):
        from .config_panel.domain import _inherited_env
        return _inherited_env(self)

    def _is_base_config(self) -> bool:
        from .config_panel.domain import _is_base_config
        return _is_base_config(self)

    def _get_explicit_env(self) -> str | None:
        from .config_panel.domain import _get_explicit_env
        return _get_explicit_env(self)

    @staticmethod
    def _is_environment_choice(name: Any) -> bool:
        from .config_panel.domain import _is_environment_choice
        return _is_environment_choice(name=name)

    @staticmethod
    def _environment_for_name(tree: ConfigTree | None, name: str | None):
        from .config_panel.domain import _environment_for_name
        return _environment_for_name(tree=tree, name=name)

    def _environment_is_compatible(self, tree: ConfigTree | None, name: str | None, paradigm: str | None) -> bool:
        from .config_panel.domain import _environment_is_compatible
        return _environment_is_compatible(self, tree=tree, name=name, paradigm=paradigm)

    def _add_unavailable_environment_option(
        self, combo: ComboBox, name: str, paradigm: str | None, tree: ConfigTree | None = None
    ) -> int | None:
        """Keep an existing incompatible or missing choice visible but impossible to select."""
        if not self._is_environment_choice(name):
            return None
        tree = tree or self.tree
        environment = self._environment_for_name(tree, name)
        exists = environment is not None
        hidden = exists and environment.name in getattr(tree, "_hidden_names", {}).get("env", set())
        reason = "hidden" if hidden else (f"incompatible with {paradigm}" if exists and paradigm else "missing")
        combo.addItem(f"{name} ({reason})", name)
        index = combo.count() - 1
        item = combo.model().item(index)
        if item is not None:
            item.setEnabled(False)
        return index

    def _set_experiment_env(self, env_name: str | None):
        from .config_panel.domain import _set_experiment_env
        return _set_experiment_env(self, env_name=env_name)

    def _get_base_env(self) -> str | None:
        from .config_panel.domain import _get_base_env
        return _get_base_env(self)

    def _set_base_env(self, env_name: str | None):
        from .config_panel.domain import _set_base_env
        return _set_base_env(self, env_name=env_name)

    def _clean_redundant_params(self):
        from .config_panel.domain import _clean_redundant_params
        return _clean_redundant_params(self)

    def _setup_experiment_box_order(self):
        boxes = [self.boxes_layout.itemAt(i).widget() for i in range(self.boxes_layout.count())]
        boxes = [box for box in boxes if isinstance(box, ConfigBox)]
        orders = self.__dict__.setdefault("_experiment_box_orders", {})
        path = str(self.current_path.resolve()) if self.current_path else str(self.current_rel_path or "")
        saved = {entry["path"]: entry["order"] for entry in
                 (self.settings_manager.get("config_viewer", "box_orders", default=[]) if self.settings_manager else [])}
        prior = orders.get(path, saved.get(path, []))
        prior = list(dict.fromkeys(key for key in prior if isinstance(key, str)))
        keyed = []
        for box in boxes:
            title = next((w for w in box.findChildren(QLabel) if w.objectName() in ("heading", "cardTitle")), None)
            if title is not None:
                keyed.append((title.text(), box, title))
        positions = [self.boxes_layout.indexOf(box) for _, box, _ in keyed]
        keyed.sort(key=lambda entry: prior.index(entry[0]) if entry[0] in prior else len(prior) + boxes.index(entry[1]))
        for _, box, _ in keyed:
            self.boxes_layout.removeWidget(box)
        for index, (_, box, _) in zip(positions, keyed):
            self.boxes_layout.insertWidget(index, box)
        # Keep absent sections in the saved sequence: toggling defaults or
        # removing/re-adding an optional card must not lose its position.
        orders[path] = prior + [key for key, _, _ in keyed if key not in prior]

        def attach(layout, title, controls):
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item.widget() is title:
                    if isinstance(layout, QHBoxLayout):
                        leading_index = next((n for n in range(i) if layout.itemAt(n).widget()
                                              and layout.itemAt(n).widget().property("parameterHeaderLeading")), i)
                        layout.insertWidget(leading_index, controls)
                    elif isinstance(layout, QVBoxLayout):
                        layout.removeWidget(title)
                        row = QHBoxLayout()
                        row.addWidget(controls)
                        row.addWidget(title)
                        row.addStretch()
                        layout.insertLayout(i, row)
                    return True
                if item.layout() and attach(item.layout(), title, controls):
                    return True
            return False

        for key, box, title in keyed:
            controls = _SectionMoveControls(title, box, lambda direction, k=key: self._move_experiment_box(k, direction))
            attach(box.layout, title, controls)

    def _move_experiment_box(self, key, direction):
        path = str(self.current_path.resolve()) if self.current_path else str(self.current_rel_path or "")
        order = self._experiment_box_orders.get(path, [])
        if key not in order:
            return
        visible_keys = set(self._visible_experiment_boxes())
        visible = [key for key in order if key in visible_keys]
        index = visible.index(key) if key in visible else -1
        target = index + direction
        if index < 0 or not 0 <= target < len(visible):
            return
        first, second = order.index(key), order.index(visible[target])
        order[first], order[second] = order[second], order[first]
        if self.settings_manager:
            saved = {entry["path"]: entry["order"] for entry in
                     self.settings_manager.get("config_viewer", "box_orders", default=[])}
            saved[path] = list(order)
            self.settings_manager.persist_key("config_viewer", "box_orders",
                                              [{"path": p, "order": o} for p, o in saved.items()])
        self._render_boxes()

    def _visible_experiment_boxes(self):
        return [title.text() for i in range(self.boxes_layout.count())
                if isinstance(box := self.boxes_layout.itemAt(i).widget(), ConfigBox)
                if (title := next((w for w in box.findChildren(QLabel)
                                   if w.objectName() in ("heading", "cardTitle")), None)) is not None]

    def _render_experiment_boxes(self):
        from .config_panel.sections.experiment import build
        return build(self)

    @staticmethod
    def _has_component_override(kind: str, data: dict) -> bool:
        from .config_panel.domain import _has_component_override
        return _has_component_override(kind=kind, data=data)

    @staticmethod
    def _effective_objective_spec(data, resolved):
        from .config_panel.domain import _effective_objective_spec
        return _effective_objective_spec(data=data, resolved=resolved)

    def _render_sweep_objective(self, form, data, resolved):
        from .config_panel.sections.components import _render_sweep_objective
        return _render_sweep_objective(self, form=form, data=data, resolved=resolved)

    def _render_variability_box(self, data, resolved):
        from .config_panel.sections.variability import _render_variability_box
        return _render_variability_box(self, data=data, resolved=resolved)

    def add_component_override(self, kind: str, resolved: dict | None = None) -> None:
        from .config_panel.domain import add_component_override
        return add_component_override(self, kind=kind, resolved=resolved)

    @provider_discovery_snapshot()
    def _component_choices(self, kind: str) -> dict[str, dict]:
        tree = self.tree
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        paths = self._loader_paths()
        return self._component_choices_for(kind, paradigm, paths, tree.config_root if tree else paths.config)

    @discovery_cached
    def _component_choices_for(self, kind, paradigm, paths, config_root):
        from src.app.core.provider_metadata import capability_metadata, selector_metadata
        from src.app.core.provider_catalog import DuplicateProvider
        if kind == "symbolic_state_loader":
            from src.app.core.data_loaders import state_loader_choices
            profiles = state_loader_choices(paths=paths)
            capability_kind, selector = "datamodule", "data_module"
        else:
            directory, selector, capability_kind = {
                "data_module": ("data", "data_module", "datamodule"),
                "evaluator": ("eval", "evaluator", "evaluator"),
                "plots": ("plot", "plotter", "plotter"),
            }[kind]
            profiles = {}
            for root in dict.fromkeys((config_root, *paths.config_roots())):
                for name, values in _component_profiles(root, directory, selector, workspace=paths).items():
                    profiles.setdefault(name, values)
            for name in selector_metadata(capability_kind, paths):
                profiles.setdefault(name, {})
        allowed = {}
        for name, values in profiles.items():
            try:
                metadata = capability_metadata(capability_kind, str(values.get(selector) or values.get("name") or name), paths)
            except DuplicateProvider:
                continue
            if kind in ("data_module", "symbolic_state_loader"):
                role = values.get("role", metadata.get("role", "training"))
                if role != ("training" if kind == "data_module" else "symbolic_state"):
                    continue
            supported = (values.get("compatible_paradigms") or values.get("paradigms")
                         or metadata.get("compatible_paradigms") or metadata.get("paradigms"))
            if not paradigm or not supported or paradigm in supported:
                allowed[name] = values
        return allowed

    @staticmethod
    def _component_name(value: Any) -> str:
        from .config_panel.domain import _component_name
        return _component_name(value=value)

    def _component_default_params(self, kind: str, name: str) -> dict:
        from .config_panel.domain import _component_default_params
        return _component_default_params(self, kind=kind, name=name)

    def _inherited_component(self, kind: str, resolved: dict) -> Any:
        from .config_panel.domain import _inherited_component
        return _inherited_component(self, kind=kind, resolved=resolved)

    def _set_component_choice(self, kind: str, name: str) -> None:
        from .config_panel.domain import _set_component_choice
        return _set_component_choice(self, kind=kind, name=name)

    def _add_component_parameter(self, kind: str, key: str, default: Any) -> None:
        from .config_panel.domain import _add_component_parameter
        return _add_component_parameter(self, kind=kind, key=key, default=default)

    def _inherited_plot_names(self, resolved: dict) -> list[str]:
        from .config_panel.domain import _inherited_plot_names
        return _inherited_plot_names(self, resolved=resolved)

    def _set_plot_override(self, name: str, enabled: bool, resolved: dict | None = None) -> None:
        from .config_panel.domain import _set_plot_override
        return _set_plot_override(self, name=name, enabled=enabled, resolved=resolved)

    def _add_plot_parameter(self, name: str, key: str, default: Any) -> None:
        from .config_panel.domain import _add_plot_parameter
        return _add_plot_parameter(self, name=name, key=key, default=default)

    def _loader_paths(self):
        from .config_panel.domain import _loader_paths
        return _loader_paths(self)

    def _set_loader_binding(self, path, name):
        from .config_panel.domain import _set_loader_binding
        return _set_loader_binding(self, path=path, name=name)

    def _render_loader_binding(self, form, caption, path, context, method=None):
        from .config_panel.sections.components import _render_loader_binding
        return _render_loader_binding(self, form=form, caption=caption, path=path, context=context, method=method)

    def _render_loader_bindings(self, box, context, methods):
        from .config_panel.sections.components import _render_loader_bindings
        return _render_loader_bindings(self, box=box, context=context, methods=methods)

    def _render_component_overrides(self, data: dict, resolved: dict) -> None:
        from .config_panel.sections.components import _render_component_overrides
        return _render_component_overrides(self, data=data, resolved=resolved)

    def _remove_figures_box(self) -> None:
        # A null selection masks a group-level selection when Hydra composes it.
        from .config_panel.domain import _remove_figures_box
        return _remove_figures_box(self)

    def _add_plot_parameter_defaults(self, name: str) -> None:
        from .config_panel.domain import _add_plot_parameter_defaults
        return _add_plot_parameter_defaults(self, name=name)

    def _add_experiment_section_header(self, box: ConfigBox, data: dict, resolved: dict) -> None:
        """Render the Experiment/Group Defaults heading and its override picker."""
        section_header = QHBoxLayout()
        section_header.setSpacing(8)
        is_base = self._is_base_config()
        section_title = _parameter_title("Group Defaults" if is_base else "Experiment")
        section_header.addWidget(section_title)

        tree = self.tree
        paradigm = data.get("paradigm") or self._inherited_paradigm()
        rules = tree.paradigms.get(paradigm) if tree and paradigm else None
        context_candidates = [
            ("trainer", "Trainer", {}),
        ]
        if rules and rules.uses_static_dataset:
            context_candidates.append(("dataset_path", "Dataset", None))
        available_context = [(key, title, default) for key, title, default in context_candidates if key not in data]
        available_context.extend(self._experiment_budget_candidates(data, resolved, rules))
        if available_context:
            btn_add_context = _make_param_adder_button(
                "Add experiment parameter override",
                [title for _, title, _ in available_context],
                lambda selected: self._add_selected_experiment_param(
                    next(item for item in available_context if item[1] == selected)
                ),
                parent_widget=self,
                allow_custom=False,
            )
            section_header.addWidget(btn_add_context)
        section_header.addStretch()
        box.add_layout(section_header)

    def _add_experiment_description(self, box: ConfigBox, data: dict) -> None:
        """Render experiment description before its paradigm and environment."""
        form = ParameterForm()
        self.txt_description = ExpandingTextEdit(str(data.get("description") or ""))
        self.txt_description.textChanged.connect(
            lambda: self._on_field_edited("description", self.txt_description.toPlainText().strip())
            if self.txt_description.toPlainText().strip()
            else (self.raw_data.pop("description", None), self._mark_dirty())
        )
        btn_del_desc = _make_subtle_tool_button("close", "Clear description", size=20, icon_size=12)
        btn_del_desc.clicked.connect(lambda: self.txt_description.setText(""))
        form.add_parameter(("description",), "Description", _parameter_row(self.txt_description, remove=btn_del_desc, expand=True))
        self.field_widgets["description"] = self.txt_description
        box.add_layout(form)

    @staticmethod
    def _site_selection(data):
        from .config_panel.domain import _site_selection
        return _site_selection(data=data)

    def _set_experiment_site(self, value):
        # Store a Hydra group selection, preserving unrelated defaults and overrides.
        from .config_panel.domain import _set_experiment_site
        return _set_experiment_site(self, value=value)

    def _add_experiment_site(self, form, data, resolved):
        tree = self.tree
        inherited = self._site_selection(resolved) or "local"
        selected = self._site_selection(data)
        self.combo_site = ComboBox()
        self.combo_site.setObjectName("experimentSiteSelector")
        self.combo_site.addItem(f"{inherited} (default)", self.INHERIT)
        choices = list(dict.fromkeys(["local", *(tree.sites if tree else ()), *([selected] if selected else [])]))
        for name in choices: self.combo_site.addItem(name, name)
        self.combo_site.setCurrentIndex(self.combo_site.findData(selected) if selected else 0)
        self.combo_site.currentIndexChanged.connect(
            lambda _: self._set_experiment_site(None if self.combo_site.currentData() == self.INHERIT else self.combo_site.currentData()))
        self.field_widgets["site"] = self.combo_site
        defaults = {"partition": None, "account": None, "qos": None, "cores": 4, "memory": "8G",
                    "time": "01:00:00", "gpus": 0, "nodes": 1}
        resources = data.get("resources") or {}
        add = _make_param_adder_button("Add resource override", [key for key in defaults if key not in resources],
            lambda key: self._add_site_resource(key, defaults[key]), parent_widget=self, allow_custom=False)
        # Site always has a value; only an explicit selector override is removable.
        remove = None
        if selected:
            remove = _make_subtle_tool_button("close", "Use inherited site", size=20, icon_size=12)
            remove.clicked.connect(lambda: self._set_experiment_site(None))
        block = None
        if resources:
            block = IndentedHierarchyBlock(level=1)
            self._render_generic_dict_contents(block, resources, ("resources",), defaults=defaults)
        form.add_parameter(("site",), "Site", _parameter_row(self.combo_site, add=add, remove=remove), children=block)

    def _add_site_resource(self, key, value):
        from .config_panel.domain import _add_site_resource
        return _add_site_resource(self, key=key, value=value)

    def _add_experiment_separator(self, box: ConfigBox) -> None:
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {theme_color('border')}; border: none;")
        box.add_widget(divider)

    def _has_experiment_lower_params(self, data: dict, resolved: dict) -> bool:
        from .config_panel.domain import _has_experiment_lower_params
        return _has_experiment_lower_params(self, data=data, resolved=resolved)

    def _environment_parameter_adder(self, data, resolved):
        values = data.get("env") if isinstance(data.get("env"), dict) else {}
        defaults = resolved.get("env") if isinstance(resolved.get("env"), dict) else {}
        return self._dict_parameter_adder(("env",), values, {**defaults, "num_envs": defaults.get("num_envs", 4),
                                                                      "num_steps": defaults.get("num_steps", 128)})

    def _add_environment_parameters(self, form: QFormLayout, data: dict) -> None:
        """Keep environment children in a spanning row directly below their selector."""
        env = data.get("env")
        if isinstance(env, dict):
            params = {key: value for key, value in env.items() if key not in ("name", "id", "type")}
            if params:
                block = IndentedHierarchyBlock()
                block.setProperty("parameterPath", "env")
                self._render_generic_dict_contents(block, params, ("env",))
                form.addRow(block)

    def _add_experiment_trainer_fields(self, box: ConfigBox, data: dict, resolved: dict) -> None:
        """Render trainer overrides in the experiment's lower parameter area."""
        trainer = data.get("trainer")
        if not isinstance(trainer, dict) and self._is_base_config():
            trainer = resolved.get("trainer", {})
        if isinstance(trainer, dict):
            defaults = {"accelerator": "cpu", "max_epochs": 25, **(resolved.get("trainer") or {})}
            form = ParameterForm()
            self._render_generic_dict_contents(form, {"trainer": trainer}, (), defaults={"trainer": defaults})
            box.add_layout(form)

    def _add_experiment_context_param(self, path: str) -> None:
        from .config_panel.domain import _add_experiment_context_param
        return _add_experiment_context_param(self, path=path)

    def _set_experiment_dataset_override(self) -> None:
        from .config_panel.domain import _set_experiment_dataset_override
        return _set_experiment_dataset_override(self)

    def _experiment_budget_candidates(self, data: dict, resolved: dict, rules) -> list[tuple]:
        from .config_panel.domain import _experiment_budget_candidates
        return _experiment_budget_candidates(self, data=data, resolved=resolved, rules=rules)

    def _add_selected_experiment_param(self, item: tuple) -> None:
        from .config_panel.domain import _add_selected_experiment_param
        return _add_selected_experiment_param(self, item=item)

    def _remove_experiment_context_param(self, path: str) -> None:
        from .config_panel.domain import _remove_experiment_context_param
        return _remove_experiment_context_param(self, path=path)

    def _add_dataset_source_fields(self, box, data, resolved, *, required: bool, paradigm: str) -> None:
        """Add a static environment source and its representation adapter."""
        ds_header = QHBoxLayout()
        ds_header.addWidget(label("Static Environment Source", "cardTitle"))
        ds_header.addStretch()
        if required:
            ds_header.addWidget(label(f"(required by {paradigm})", "badge"))
        elif "dataset_path" in data:
            btn_rm_ds = _make_subtle_tool_button("trash", "Remove dataset source override", size=24, icon_size=14)
            btn_rm_ds.clicked.connect(lambda: self.remove_block("dataset"))
            ds_header.addWidget(btn_rm_ds)
        box.add_layout(ds_header)

        ds_form = ParameterForm()
        ds_form.setVerticalSpacing(10)
        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        self.combo_dataset_path = ComboBox()
        self.combo_dataset_path.setMaximumWidth(320)
        self.combo_dataset_path.setEditable(True)
        for dataset_path in discover_datasets(paths=self.tree.workspace if self.tree else None):
            self.combo_dataset_path.addItem(dataset_path)
        current_path = str(data.get("dataset_path") or resolved.get("dataset_path") or "")
        self.combo_dataset_path.setEditText(current_path)

        status = QLabel("")

        def _update_source_status(path_text):
            from src.app.pipeline.datasets import resolve_dataset_for_method
            try:
                source = path_text.strip()
                if "${" in source:
                    from omegaconf import OmegaConf
                    source = OmegaConf.create({**resolved, **data, "dataset_path": source})["dataset_path"]
                path = None
                if source:
                    path = resolve_dataset_for_method(
                        "dataset preview", {}, {"dataset_path": source}, paths=self.tree.workspace if self.tree else _dataset_workspace()
                    )
            except Exception:  # The full preflight reports invalid references; this is a file-status hint.
                path = None
            if path is not None and path.exists():
                try:
                    size_mb = path.stat().st_size / (1024 * 1024)
                    status.setText(f"✓ Exists ({size_mb:.1f} MB)")
                    status.setStyleSheet("color: #b8bb26; font-size: 11px; font-weight: bold;")
                except OSError:
                    status.setText("✓ Exists")
                    status.setStyleSheet("color: #b8bb26; font-size: 11px;")
            elif path_text.strip():
                status.setText("⚠ Not Found")
                status.setStyleSheet("color: #fabd2f; font-size: 11px;")
            else:
                status.setText("")

        _update_source_status(current_path)
        self.combo_dataset_path.currentTextChanged.connect(
            lambda value: (self._on_field_edited("dataset_path", value.strip()), _update_source_status(value))
        )
        source_row.addWidget(self.combo_dataset_path)
        browse = QToolButton()
        browse.setText("Browse…")
        browse.clicked.connect(self._browse_dataset_path)
        source_row.addWidget(browse)
        source_row.addWidget(status)
        source_row.addStretch()
        ds_form.add_parameter(("dataset_path",), "Dataset / Static Environment", source_row)

        box.add_layout(ds_form)

    def _scalar_widget(self, value, write):
        from .config_panel.editors import value_editor
        return value_editor(value, {}, write)

    def _ruleset_domain_choices(self):
        """Browse the complete runtime catalog, independent of the simulator."""
        tree = self.tree
        if tree is None:
            return []
        return tree.ruleset_names()

    def _ruleset_widget(self, value, write):
        combo = HierarchicalComboBox()
        combo.setMaximumWidth(TEXT_FIELD_WIDTH)
        combo.addItem("", None)
        for choice in self._ruleset_domain_choices():
            combo.addItem(choice, choice)
        index = combo.findData(value)
        if index < 0 and value is not None:
            # Preserve legacy bare names and unavailable resources until edited.
            combo.insertItem(1, str(value), value)
            index = 1
        combo.setCurrentIndex(max(0, index))
        combo.currentIndexChanged.connect(lambda i: write(combo.itemData(i)))
        combo.currentTextChanged.connect(combo.setToolTip)
        combo.setToolTip(combo.currentText())
        return combo

    def _parameter_context_config(self):
        from .config_panel.domain import _parameter_context_config
        return _parameter_context_config(self)

    def _schema_widget(self, value, schema, write, *, action_path=()):
        actions = schema.get("x-theta-actions", []) if isinstance(schema, dict) else []
        current = [value]
        def update(edited):
            current[0] = edited
            write(edited)
        field = self._schema_value_widget(value, schema, update)
        if not actions:
            return field
        wrapper = QWidget()
        wrapper._parameter_editor = field
        row = _parameter_row(field)
        wrapper.setLayout(row)
        for action in actions:
            if not isinstance(action, dict) or action.get("type") != "open-in-editor":
                continue
            button = _make_subtle_tool_button("pencil", action.get("label", "Open in editor"), size=20, icon_size=14)
            button.setProperty("parameterOpen", True)
            button.setProperty("parameterPath", ".".join(action_path))
            row.insertWidget(row.count() - 1, button)
            _reveal_remove_on_row_hover(row, button)
            effective_action = {**action, "domain": schema.get("x-theta-domain")}
            button.clicked.connect(lambda _checked=False, a=effective_action, b=button: self._open_parameter_action(
                a, current[0], action_path or tuple(str(wrapper.property("parameterPath") or "").split(".")), b))
        return wrapper

    def _open_parameter_action(self, action, value, path, button):
        from .parameter_actions import opening_targets
        tree = self.tree
        workspace = tree.workspace if tree else WorkspacePaths.for_process()
        try:
            config = self._parameter_context_config()
            targets = opening_targets(action, value=value, workspace=workspace.root, context={
                "workspace_root": str(workspace.root), "config": config, "parameter_path": list(path),
                "config_path": str(self.current_rel_path or ""),
            })
            service = getattr(self.window(), "editor_service", None)
            if service is None:
                raise ValueError("The host editor service is unavailable")
            def open_target(target):
                if target.is_dir():
                    return service.open_folder(target)
                return service.open_file(target)
            if len(targets) == 1:
                if not open_target(targets[0][1]):
                    raise ValueError("The editor could not open the component's resource")
            else:
                menu = QMenu(button)
                for title, target in targets:
                    menu.addAction(title, lambda p=target: open_target(p))
                menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        except Exception as exc:
            QMessageBox.warning(self, "Cannot open parameter file", str(exc))

    def _schema_value_widget(self, value, schema, write):
        from .config_panel.editors import value_editor
        schema = schema if isinstance(schema, dict) else {}
        choices = None
        if schema.get("x-theta-domain") == "model":
            tree = self.tree
            choices = tree.model_names if tree else []
        elif schema.get("x-theta-domain") == "ruleset":
            return self._ruleset_widget(value, write)
        return value_editor(value, schema, write, choices=choices)

    def _component_parameter_schema(self, path: tuple[str, ...]) -> dict[str, Any]:
        from .config_panel.domain import _component_parameter_schema
        return _component_parameter_schema(self, path=path)

    @staticmethod
    def _schema_has_choices(schema: dict[str, Any]) -> bool:
        return bool(schema.get("enum")) or schema.get("x-theta-domain") in ("model", "ruleset")

    def _list_field(self, items, write):
        from .config_panel.editors import draft_field
        def parse(text):
            parsed = yaml.safe_load("[" + text + "]")
            if not isinstance(parsed, list):
                raise ValueError("Enter a comma-separated list")
            return parsed
        import json
        def display(item):
            if isinstance(item, str):
                try:
                    parsed = yaml.safe_load(item)
                    if type(parsed) is str and parsed == item and "," not in item:
                        return item
                except yaml.YAMLError:
                    pass
            return json.dumps(item)
        text = ", ".join(display(item) for item in items)
        return draft_field(text, {"type": "array"}, write, parse,
                           help_text="Comma-separated list; quote strings to preserve their type.")

    def _render_remaining_sections(self):
        from .config_panel.sections.parameters import _render_remaining_sections
        return _render_remaining_sections(self)

    def _render_experiment_dict_contents(self, parent_layout, values: Mapping[str, Any], path: tuple[str, ...], level: int = 1):
        """Experiments use the same recursive parameter renderer as components."""
        self._render_generic_dict_contents(parent_layout, dict(values), path, level)

    def _render_generic_boxes(self):
        from .config_panel.sections.parameters import _render_generic_boxes
        return _render_generic_boxes(self)

    def _parameter_required(self, path):
        from .config_panel.domain import _parameter_required
        return _parameter_required(self, path=path)

    def _parameter_remove(self, path, title=None):
        from .config_panel.document import lookup, MISSING
        if self._parameter_required(path):
            return None
        if lookup(self.raw_data, path) is MISSING:
            return None
        if path[:2] == ("objective", "params"):
            explicit = self.raw_data
            for part in path:
                if not isinstance(explicit, dict) or part not in explicit:
                    return None
                explicit = explicit[part]
        button = _make_subtle_tool_button("close", f"Remove '{title or path[-1]}'", size=20, icon_size=12)
        button.setProperty("parameterPath", ".".join(path))
        button.clicked.connect(lambda _: self._remove_generic_path(path))
        return button

    def _dict_parameter_adder(self, path, values, defaults=None):
        properties = self._component_parameter_schema(path).get("properties", {})
        candidates = {key: copy.deepcopy(value) for key, value in (defaults or {}).items() if key not in values}
        for key, schema in properties.items():
            if key not in values and "default" in schema:
                candidates.setdefault(key, copy.deepcopy(schema["default"]))
        def add(key):
            value = copy.deepcopy(candidates[key]) if key in candidates else self._prompt_component_parameter_value(
                key, self._component_parameter_schema(path + (key,)))
            self._on_generic_path_edited(path + (key,), value)
            QTimer.singleShot(0, self._render_boxes)
        button = _make_param_adder_button(
            "Add parameter" + (" to " + path[-1] if path else ""),
            sorted(set(candidates) | (set(properties) - set(values))), add, parent_widget=self,
        )
        button.setProperty("parameterPath", ".".join(path))
        return button

    def _parameter_editor(self, key, value, path):
        write = lambda updated: self._on_generic_path_edited(path, updated)
        schema = self._component_parameter_schema(path)
        if self._schema_has_choices(schema) or schema.get("type") == "number":
            widget = self._schema_widget(value, schema, write, action_path=path)
            widget.setProperty("parameterPath", ".".join(path))
            return widget
        if key in ("compatible_paradigms", "paradigms"):
            widget = TagCloudWidget(tags=value if isinstance(value, list) else ([value] if value else []),
                                    available_options=_get_available_paradigms())
            widget.tagsChanged.connect(write)
        elif (key in ("hidden_sizes", "hidden_dims", "latent_sizes") and isinstance(value, list)
              and getattr(self.settings_manager, "visual_layer_editor", True)):
            widget = LayerStackWidget(layers=value)
            widget.layersChanged.connect(write)
        elif key.lower() in CATEGORICAL_CHOICES:
            widget = _make_categorical_combo(key, value, write)
        else:
            widget = self._schema_widget(value, schema, write)
        widget.setProperty("parameterPath", ".".join(path))
        return widget

    def _render_generic_dict_contents(self, parent_layout, d: dict, path: tuple[str, ...], level: int = 1, defaults: dict | None = None):
        from .config_panel.sections.parameters import _render_generic_dict_contents
        return _render_generic_dict_contents(self, parent_layout=parent_layout, d=d, path=path, level=level, defaults=defaults)

    def _prompt_add_generic_leaf_param(self):
        """Prompt user to add a new parameter to the component configuration."""
        key, ok = self._prompt_component_parameter_key(())
        if not ok or not key.strip():
            return
        key = key.strip()
        self.raw_data[key] = self._prompt_component_parameter_value(key, self._component_parameter_schema((key,)))
        self._mark_dirty()
        self._render_boxes()

    def _prompt_add_nested_generic_param(self, path: tuple[str, ...]):
        """Prompt user to add a new parameter inside a nested section."""
        path_str = ".".join(path)
        key, ok = self._prompt_component_parameter_key(path)
        if not ok or not key.strip():
            return
        key = key.strip()
        value = self._prompt_component_parameter_value(key, self._component_parameter_schema(path + (key,)))
        self._on_generic_path_edited(path + (key,), value)
        self._render_boxes()

    def _prompt_component_parameter_key(self, path: tuple[str, ...]):
        schema = self._component_parameter_schema(path)
        properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        target = self.raw_data
        for part in path:
            target = target.get(part, {}) if isinstance(target, dict) else {}
        candidates = [key for key in properties if not isinstance(target, dict) or key not in target]
        if candidates:
            return QInputDialog.getItem(self, "Add Parameter", "Parameter name:", candidates, 0, True)
        return QInputDialog.getText(self, "Add Parameter", "Parameter name:")

    def _prompt_component_parameter_value(self, key: str, schema: dict[str, Any]):
        choices = schema.get("enum") if isinstance(schema, dict) else None
        if isinstance(schema, dict) and schema.get("x-theta-domain") == "model":
            tree = self.tree
            choices = tree.model_names if tree else []
        elif isinstance(schema, dict) and schema.get("x-theta-domain") == "ruleset":
            dialog = QDialog(self)
            dialog.setWindowTitle("Parameter Value")
            layout = QVBoxLayout(dialog)
            layout.addWidget(QLabel(f"Value for '{key}':"))
            combo = self._ruleset_widget(schema.get("default"), lambda value: None)
            layout.addWidget(combo)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            selected = combo.currentData() if dialog.exec() == QDialog.DialogCode.Accepted else schema.get("default")
            dialog.deleteLater()
            return selected
        if isinstance(choices, list) and choices:
            labels = [str(value) for value in choices]
            selected, ok = QInputDialog.getItem(self, "Parameter Value", f"Value for '{key}':", labels, 0, False)
            return choices[labels.index(selected)] if ok else choices[0]
        if isinstance(schema, dict) and "default" in schema:
            return schema["default"]
        val_str, ok = QInputDialog.getText(self, "Parameter Value", f"Initial value for '{key}':")
        return _parse_yaml_value(val_str) if ok else None

    @property
    def document(self):
        # raw_data is intentionally still assignable by loaders and callers.
        # Resolve state on access so bindings never retain a previous file.
        return ConfigDocument(
            self.raw_data, self._get_resolved_defaults(),
            read_only=self._hub_read_only(),
            on_change=lambda result: self._mark_dirty(),
            selector_identity=component_identity,
        )

    def _binding(self, path):
        return ParameterBinding(self.document, tuple(path), self._component_parameter_schema(path),
                                self._parameter_required(path))

    def _on_generic_path_edited(self, path: tuple[str, ...], value: Any):
        if self._block_updates or not path:
            return
        return self._binding(path).set(value)

    def _append_list_parameter(self, key, item):
        self.raw_data.setdefault(key, []).append(_parse_yaml_value(item))
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _remove_generic_field(self, key: str):
        self.raw_data.pop(key, None)
        self._mark_dirty()
        self._render_boxes()

    def _remove_generic_path(self, path: tuple[str, ...]):
        if not path or self._block_updates:
            return
        result = self._binding(path).remove()
        if result.changed:
            self._render_boxes()

    # Typed commits; only explicit raw-value dialogs parse YAML.
    def _on_field_edited(self, key, value):
        if self._block_updates:
            return
        return self.document.set((key,), value)

    def _validate_method_name(self, name: str, existing_names: set[str]) -> str | None:
        from .config_panel.domain import _validate_method_name
        return _validate_method_name(self, name=name, existing_names=existing_names)

    def duplicate_method(self, name: str) -> str | None:
        from .config_panel.domain import duplicate_method
        return duplicate_method(self, name=name)

    def _get_group_default_method(self) -> tuple[str | None, dict | None]:
        from .config_panel.domain import _get_group_default_method
        return _get_group_default_method(self)

    def _start_add_new_method(self, *, empty: bool = False):
        """Initiate adding a new method card with focus on the name editor.

        ``empty`` is used by the Methods header action: it deliberately does
        not clone a group's default method into a new experiment variant.
        """
        tree = self.tree
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        if not tree or paradigm not in tree.paradigms:
            QMessageBox.warning(
                self,
                "Select a Paradigm First",
                "Theta needs the experiment's paradigm before it can determine whether a method requires an algorithm or is model-only.",
            )
            return
        if self._is_base_config():
            methods = self.raw_data.get("methods", {})
            if any(k != "params" and isinstance(v, dict) for k, v in methods.items()):
                return
            self._pending_new_method = True
            self._pending_new_method_spec = {}
            self._render_boxes()
            return

        def_name, def_spec = self._get_group_default_method()
        self._pending_new_method = True
        self._pending_new_method_spec = {} if empty else (copy.deepcopy(def_spec) if def_spec else {})
        self._render_boxes()

    def _cancel_new_method(self):
        """Cancel creating a new method."""
        self._pending_new_method = False
        self._pending_new_method_spec = {}
        self._render_boxes()

    def add_method(self, agent=None, model=None, name=None):
        from .config_panel.domain import add_method
        return add_method(self, agent=agent, model=model, name=name)


    def _on_experiment_env_changed(self, value):
        from .config_panel.domain import _on_experiment_env_changed
        return _on_experiment_env_changed(self, value=value)

    def remove_method(self, name):
        from .config_panel.domain import remove_method
        return remove_method(self, name=name)

    def rename_method(self, old_name: str, new_name: str) -> bool:
        from .config_panel.domain import rename_method
        return rename_method(self, old_name=old_name, new_name=new_name)

    def reorder_method(self, method_name: str, target_index: int) -> bool:
        from .config_panel.domain import reorder_method
        return reorder_method(self, method_name=method_name, target_index=target_index)

    def _on_paradigm_changed(self, value):
        from .config_panel.domain import _on_paradigm_changed
        return _on_paradigm_changed(self, value=value)

    def _on_nested_edited(self, parent_key, child_key, value):
        from .config_panel.domain import _on_nested_edited
        return _on_nested_edited(self, parent_key=parent_key, child_key=child_key, value=value)

    def _on_method_enabled_toggled(self, method_name: str, enabled: bool):
        from .config_panel.domain import _on_method_enabled_toggled
        return _on_method_enabled_toggled(self, method_name=method_name, enabled=enabled)

    def _on_method_param_edited(self, method_name, param_key, value):
        from .config_panel.domain import _on_method_param_edited
        return _on_method_param_edited(self, method_name=method_name, param_key=param_key, value=value)

    def _on_method_nested_param_edited(self, method_name: str, parent_key: str, child_key: str, value: Any):
        from .config_panel.domain import _on_method_nested_param_edited
        return _on_method_nested_param_edited(self, method_name=method_name, parent_key=parent_key, child_key=child_key, value=value)

    def _remove_method_nested_param(self, method_name: str, parent_key: str, child_key: str):
        from .config_panel.domain import _remove_method_nested_param
        return _remove_method_nested_param(self, method_name=method_name, parent_key=parent_key, child_key=child_key)

    def _method_component_parameter_root(self, method_name: str, kind: str, create: bool = False) -> dict:
        from .config_panel.domain import _method_component_parameter_root
        return _method_component_parameter_root(self, method_name=method_name, kind=kind, create=create)

    def _method_model_parameter_root(self, method_name: str, model_name: str, create: bool = False) -> dict:
        from .config_panel.domain import _method_model_parameter_root
        return _method_model_parameter_root(self, method_name=method_name, model_name=model_name, create=create)

    def _on_method_model_parameter_edited(
        self,
        method_name: str,
        model_name: str,
        path: tuple[str, ...],
        value: Any,
        schema_model_name: str | None = None,
        schema_path: tuple[str, ...] | None = None,
    ):
        from .config_panel.domain import _on_method_model_parameter_edited
        return _on_method_model_parameter_edited(self, method_name=method_name, model_name=model_name, path=path, value=value, schema_model_name=schema_model_name, schema_path=schema_path)

    def _remove_method_model_parameter(self, method_name: str, model_name: str, path: tuple[str, ...], schema_model_name=None, schema_path=None):
        from .config_panel.domain import _remove_method_model_parameter
        return _remove_method_model_parameter(self, method_name=method_name, model_name=model_name, path=path, schema_model_name=schema_model_name, schema_path=schema_path)

    def _add_method_model_parameter(
        self,
        method_name: str,
        model_name: str,
        path: tuple[str, ...],
        schema_model_name: str | None = None,
        schema_path: tuple[str, ...] | None = None,
    ):
        from .config_panel.domain import _add_method_model_parameter
        return _add_method_model_parameter(self, method_name=method_name, model_name=model_name, path=path, schema_model_name=schema_model_name, schema_path=schema_path)

    def _on_method_child_model_changed(
        self, method_name: str, parent_model_name: str, path: tuple[str, ...], child_model_name: str
    ):
        from .config_panel.domain import _on_method_child_model_changed
        return _on_method_child_model_changed(self, method_name=method_name, parent_model_name=parent_model_name, path=path, child_model_name=child_model_name)

    def _render_method_model_parameters(
        self,
        method_name: str,
        model_name: str,
        values: dict[str, Any],
        defaults: dict[str, Any],
        path: tuple[str, ...] = (),
        level: int = 1,
        show_defaults: bool = False,
        schema_model_name: str | None = None,
        schema_path: tuple[str, ...] = (),
    ) -> list[tuple[str, Any]]:
        from .config_panel.sections.model import _render_method_model_parameters
        return _render_method_model_parameters(self, method_name=method_name, model_name=model_name, values=values, defaults=defaults, path=path, level=level, show_defaults=show_defaults, schema_model_name=schema_model_name, schema_path=schema_path)

    def _browse_dataset_path(self):
        """Open file dialog to select a dataset file or directory."""
        workspace = self.tree.workspace if self.tree else _dataset_workspace()
        initial_dir = str(workspace.datasets if workspace.datasets.is_dir() else workspace.root)
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            initial_dir,
            "Dataset Files (*.pkl *.csv *.parquet *.h5 *.pt);;All Files (*)",
        )
        if selected:
            try:
                import os
                rel = Path(os.path.relpath(selected, workspace.root)).as_posix()
            except ValueError:  # Different Windows drives cannot share a relative path.
                rel = selected
            if hasattr(self, "combo_dataset_path"):
                self.combo_dataset_path.setEditText(rel)
            self._on_field_edited("dataset_path", rel)

    def remove_block(self, block_name: str):
        from .config_panel.domain import remove_block
        return remove_block(self, block_name=block_name)

    def _tuner_contributions(self, slot="configuration.tuning"):
        from .config_panel.domain import _tuner_contributions
        return _tuner_contributions(self, slot)

    def _tuner_ready(self):
        from .config_panel.domain import _tuner_ready
        return _tuner_ready(self)

    def _add_sweep_toggle(self, row, method, target, value):
        if not self._tuner_ready():
            return
        dimensions = self.raw_data.get("methods", {}).get(method, {}).get("tune", {})
        dimension = dimensions.get(target) if isinstance(dimensions, dict) else None
        pending = getattr(self, "_pending_sweep", None) == (method, target)
        if not pending and not isinstance(dimension, dict):
            button = _make_subtle_tool_button("sweep", f"Configure sweep for {target}", size=20, icon_size=14)
            button.setObjectName("sweepToggle_" + method + "_" + target)
            button.clicked.connect(lambda: self._toggle_method_sweep(method, target, value))
            delete_index = next(
                (i for i in range(row.count())
                 if isinstance(row.itemAt(i).widget(), QToolButton)
                 and row.itemAt(i).widget().toolTip().startswith("Remove")),
                row.count(),
            )
            row.insertWidget(delete_index, button)
            _reveal_remove_on_row_hover(row, button, reserve_space=False)
            return
        # The existing row X removes the sweep first; after rerender it resumes
        # its normal parameter-removal action.
        delete = None
        for i in range(row.count()):
            delete = row.itemAt(i).widget()
            if isinstance(delete, QToolButton) and delete.toolTip().startswith("Remove"):
                delete.clicked.disconnect()
                delete.setToolTip("Remove sweep")
                delete.setObjectName(f"sweepRemove_{method}_{target}")
                delete.clicked.connect(lambda _, m=method, t=target, v=value: self._cancel_sweep_setup(m, t) if pending else self._toggle_method_sweep(m, t, v))
                row.setAlignment(delete, Qt.AlignmentFlag.AlignVCenter)
                break
        original = None
        # Replace the value widget in this row, preserving its underlying config value.
        for index in range(row.count()):
            widget = row.itemAt(index).widget()
            if widget is not None and not isinstance(widget, (QLabel, QToolButton)):
                original = widget
                row.removeWidget(widget)
                # Hover handlers already reference this control; keep it alive
                # until the row's parent is destroyed on the next render.
                widget.hide()
                break
        else:
            return
        combo = _SweepTypePicker()
        combo.setObjectName("sweepType_" + method + "_" + target)
        combo.setProperty("parameterCompact", True)
        combo.setToolTip("Sweep type")
        combo.addItem("Select sweep type…", None)
        for title, key in (("List", "choice"), ("Integer range", "int"), ("Numeric range", "float"), ("Log numeric range", "log_float")):
            combo.addItem(title, key)
        combo.setCurrentIndex(max(0, combo.findData(dimension.get("type") if isinstance(dimension, dict) else None)))
        combo.activated.connect(lambda _, m=method, t=target, v=value, c=combo: self._select_sweep_type(m, t, v, c.currentData()))
        row._sweep_subtitle = combo
        if pending:
            combo.cancelled.connect(lambda m=method, t=target: self._cancel_sweep_setup(m, t))
            def open_picker(c=combo):
                if not sip.isdeleted(c):
                    c.setFocus()
                    c.showPopup()
            QTimer.singleShot(0, open_picker)

        content = WrappingFlowWidget(spacing=4, vertical_alignment=Qt.AlignmentFlag.AlignVCenter, fit_contents=True)
        content.setObjectName(f"sweepInline_{method}_{target}")
        content.setProperty("parameterInline", True)
        content.setStyleSheet('QWidget[parameterInline="true"] { background: transparent; }')
        if original is not None:
            original.setParent(content)
            original.hide()
        flow = content.layout()
        kind = dimension.get("type") if isinstance(dimension, dict) else None
        if kind == "choice":
            values = dimension.get("values", [])
            if isinstance(values, list):
                for i, item in enumerate(values):
                    entry = QWidget()
                    entry.setProperty("parameterInline", True)
                    line = QHBoxLayout(entry)
                    line.setContentsMargins(0, 0, 0, 0)
                    line.setSpacing(2)
                    if isinstance(original, QComboBox):
                        edit = ComboBox()
                        for n in range(original.count()):
                            edit.addItem(original.itemText(n), original.itemData(n))
                        if edit.findText(str(item)) < 0:
                            edit.addItem(str(item))
                        edit.setCurrentText(str(item))
                        edit.currentTextChanged.connect(lambda text, m=method, t=target, n=i: self._edit_sweep_choice(m, t, n, text))
                    else:
                        edit = QLineEdit(yaml.safe_dump(item, default_flow_style=True).strip().removesuffix("\n...") if item != "" else "")
                        edit.setPlaceholderText("Value")
                        edit.textChanged.connect(lambda text, m=method, t=target, n=i: self._edit_sweep_choice(m, t, n, text))
                    edit.setObjectName(f"sweep_choice_{method}_{target}_{i}")
                    edit.setAccessibleName(f"{target} value {i + 1}")
                    _fit_inline_control(edit)
                    line.addWidget(edit)
                    remove = _make_subtle_tool_button("close", "Remove value", size=20, icon_size=12)
                    remove.clicked.connect(lambda _, m=method, t=target, n=i: self._remove_sweep_choice(m, t, n))
                    line.addWidget(remove)
                    _reveal_remove_on_row_hover(line, remove, reserve_space=False)
                    if i < len(values) - 1:
                        line.insertWidget(1, label(","))
                    flow.addWidget(entry)
            add = _make_subtle_tool_button("plus", "Add value", size=20, icon_size=14)
            add.setObjectName(f"sweepAddChoice_{method}_{target}")
            add.clicked.connect(lambda _, m=method, t=target: self._add_sweep_choice(m, t))
            flow.addWidget(add)
        elif isinstance(dimension, dict):
            bounds = QWidget()
            bounds.setProperty("parameterInline", True)
            line = QHBoxLayout(bounds)
            line.setContentsMargins(0, 0, 0, 0)
            line.setSpacing(4)
            line.addWidget(label("("))
            for key, title in (("low", "Lower bound"), ("high", "Upper bound")):
                edit = self._sweep_dimension_field(method, target, key, dimension)
                edit.setObjectName(f"sweep_{key}_{method}_{target}")
                edit.setAccessibleName(f"{target} {title.lower()}")
                edit.setToolTip(title)
                _fit_inline_control(edit)
                line.addWidget(edit)
                if key == "low":
                    line.addWidget(label(","))
            line.addWidget(label(")"))
            flow.addWidget(bounds)
            if kind in ("int", "float"):
                if "step" in dimension:
                    step = QWidget()
                    step.setProperty("parameterInline", True)
                    line = QHBoxLayout(step)
                    line.setContentsMargins(0, 0, 0, 0)
                    line.setSpacing(4)
                    line.addWidget(label("Step", "muted"))
                    edit = self._sweep_dimension_field(method, target, "step", dimension)
                    edit.setObjectName(f"sweep_step_{method}_{target}")
                    edit.setAccessibleName(f"{target} step")
                    _fit_inline_control(edit)
                    line.addWidget(edit)
                    remove_step = _make_subtle_tool_button("close", "Remove step parameter", size=20, icon_size=12)
                    remove_step.clicked.connect(lambda _, m=method, t=target: self._remove_sweep_dimension(m, t, "step"))
                    line.addWidget(remove_step)
                    _reveal_remove_on_row_hover(line, remove_step)
                    flow.addWidget(step)
                else:
                    add_step = _make_subtle_tool_button("plus", "Add step parameter", size=20, icon_size=12)
                    add_step.setObjectName(f"sweepAddStep_{method}_{target}")
                    add_step.clicked.connect(lambda _, m=method, t=target: self._add_sweep_dimension(m, t, "step"))
                    flow.addWidget(add_step)
        row.insertWidget(index, content)
        if delete is not None:
            for hover in delete.findChildren(_RowHoverReveal):
                targets = [combo, content]
                hover._targets = [weakref.ref(w) for w in targets]
                for control in targets:
                    control.setMouseTracking(True)
                    control.installEventFilter(hover)
        self._rendered_sweeps.add((method, target))

    def _sweep_dimension_field(self, method, target, key, dimension):
        value = dimension.get(key)
        if dimension.get("type") == "int" and type(value) is int:
            edit = integer_field(value)
        elif type(value) in (int, float) and dimension.get("type") != "int":
            edit = ScientificDoubleSpinBox(float(value))
        else:
            # Keep incomplete or expression-based configurations visible.
            edit = QLineEdit(str(value) if value is not None else "")
        if hasattr(edit, "valueChanged"):
            edit.valueChanged.connect(lambda v: self._edit_sweep_dimension(method, target, key, str(v)))
        else:
            edit.textChanged.connect(lambda text: self._edit_sweep_dimension(method, target, key, text))
        return edit

    def _place_sweep_subtitles(self):
        """Attach the type selector below the caption in forms, grids and shared rows."""
        for row in self.container.findChildren(QHBoxLayout):
            subtitle = getattr(row, "_sweep_subtitle", None)
            if subtitle is None:
                continue
            caption = next((row.itemAt(i).widget() for i in range(row.count())
                            if isinstance(row.itemAt(i).widget(), QLabel)), None)
            owner = row if caption is not None else row.parent()
            if caption is None and isinstance(owner, QFormLayout):
                caption = owner.labelForField(row)
            if caption is None and isinstance(owner, QGridLayout):
                r, c, _, _ = owner.getItemPosition(owner.indexOf(row))
                item = owner.itemAtPosition(r, c - 1)
                caption = item.widget() if item else None
            if not isinstance(caption, QLabel):
                continue
            group = QWidget()
            group.setProperty("parameterLabel", True)
            group.setStyleSheet('QWidget[parameterLabel="true"] { background: transparent; }')
            layout = QVBoxLayout(group)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)
            if isinstance(owner, QFormLayout):
                r, role = owner.getWidgetPosition(caption)
                owner.removeWidget(caption)
                owner.setWidget(r, role, group)
            elif isinstance(owner, QGridLayout):
                r, c, rs, cs = owner.getItemPosition(owner.indexOf(caption))
                owner.removeWidget(caption)
                owner.addWidget(group, r, c, rs, cs)
            else:
                i = row.indexOf(caption)
                row.removeWidget(caption)
                row.insertWidget(i, group)
            layout.addWidget(caption)
            layout.addWidget(subtitle)
            subtitle.setStyleSheet(
                f"QComboBox {{ background: transparent; border: none; color: {theme_color('muted')}; "
                "font-size: 11px; padding: 0px; } "
                "QComboBox::drop-down { background: transparent; border: none; }")
            subtitle.setMaximumWidth(180)

    def _toggle_method_sweep(self, method, target, value):
        from .config_panel.domain import _toggle_method_sweep
        return _toggle_method_sweep(self, method=method, target=target, value=value)

    def _cancel_sweep_setup(self, method, target):
        from .config_panel.domain import _cancel_sweep_setup
        return _cancel_sweep_setup(self, method=method, target=target)

    def _select_sweep_type(self, method, target, value, kind):
        from .config_panel.domain import _select_sweep_type
        return _select_sweep_type(self, method=method, target=target, value=value, kind=kind)

    def _edit_sweep_choice(self, method, target, index, text):
        from .config_panel.domain import _edit_sweep_choice
        return _edit_sweep_choice(self, method=method, target=target, index=index, text=text)

    def _add_sweep_choice(self, method, target):
        from .config_panel.domain import _add_sweep_choice
        return _add_sweep_choice(self, method=method, target=target)

    def _remove_sweep_choice(self, method, target, index):
        from .config_panel.domain import _remove_sweep_choice
        return _remove_sweep_choice(self, method=method, target=target, index=index)

    def _render_method_sweeps(self, form, method, spec, prefix=None, exclude_prefix=None):
        # A configured range must remain visible even when its parameter's
        # normal row is absent (for example a component default or list index).
        for target, dimension in spec.get("tune", {}).items() if isinstance(spec.get("tune", {}), dict) else ():
            if (prefix and not target.startswith(prefix)) or (exclude_prefix and target.startswith(exclude_prefix)):
                continue
            if isinstance(dimension, dict):
                if (method, target) in self._rendered_sweeps:
                    continue
                if not self._tuner_ready():
                    form.add_parameter(("methods", method, "tune", target), "Sweep", label(f"{target}: {dimension}", "muted"))
                    continue
                row = QHBoxLayout()
                row.addWidget(QLineEdit())
                delete = _make_subtle_tool_button("close", "Remove sweep", size=20, icon_size=12)
                delete.clicked.connect(lambda: None)  # replaced by _add_sweep_toggle
                row.addWidget(delete)
                self._add_sweep_toggle(row, method, target, None)
                form.add_parameter(("methods", method, "tune", target), target, row,
                    children=getattr(row, "_sweep_block", None))
            else:
                form.add_parameter(("methods", method, "tune", target), "Sweep (legacy)", label(f"{target}: {dimension}", "muted"))

    def _edit_sweep_dimension(self, method, target, key, text):
        from .config_panel.domain import _edit_sweep_dimension
        return _edit_sweep_dimension(self, method=method, target=target, key=key, text=text)

    def _add_sweep_dimension(self, method, target, key):
        from .config_panel.domain import _add_sweep_dimension
        return _add_sweep_dimension(self, method=method, target=target, key=key)

    def _remove_sweep_dimension(self, method, target, key):
        from .config_panel.domain import _remove_sweep_dimension
        return _remove_sweep_dimension(self, method=method, target=target, key=key)

    def _on_tuner_selected(self, name):
        from .config_panel.domain import _on_tuner_selected
        return _on_tuner_selected(self, name=name)

    @provider_discovery_snapshot()
    def _refresh_sweep_box(self):
        """Update dependent sweep controls without replacing unrelated editors."""
        self._sweep_refresh_timer.stop()
        if self._block_updates:
            return
        old = next((box for box in self.container.findChildren(ConfigBox)
                    if box.property("configSection") == "sweep"), None)
        if old is None:
            self._render_boxes()
            return
        for controller in self._parameter_reorders:
            if controller.host is old and controller.dragged is not None:
                controller.finish(False)
        position = self.scroll.verticalScrollBar().value()
        index = self.boxes_layout.indexOf(old)
        self._block_updates = True
        self.setUpdatesEnabled(False)
        try:
            self._render_sweep_box(self.raw_data)
            new = self.boxes_layout.itemAt(self.boxes_layout.count() - 1).widget()
            self.boxes_layout.removeWidget(new)
            self.boxes_layout.removeWidget(old)
            self.boxes_layout.insertWidget(index, new)
            new.show()
            self._parameter_reorders = [c for c in self._parameter_reorders if c.host is not old]
            old.hide()
            old.setParent(None)
            old.deleteLater()
            self._setup_parameter_reordering(new)
            self.container.layout().activate()
            self.container.adjustSize()
            self.scroll.verticalScrollBar().setValue(position)
        finally:
            self._block_updates = False
            self.setUpdatesEnabled(True)

    def _on_tuner_option_edited(self, component, key, value):
        from .config_panel.domain import _on_tuner_option_edited
        return _on_tuner_option_edited(self, component=component, key=key, value=value)

    def _on_tuner_choice_edited(self, component, key, value):
        from .config_panel.domain import _on_tuner_choice_edited
        return _on_tuner_choice_edited(self, component=component, key=key, value=value)

    def _add_tuner_option(self, component, key, value):
        from .config_panel.domain import _add_tuner_option
        return _add_tuner_option(self, component=component, key=key, value=value)

    def _remove_tuner_option(self, component, key):
        from .config_panel.domain import _remove_tuner_option
        return _remove_tuner_option(self, component=component, key=key)

    @staticmethod
    def _tuner_scalar_input(text):
        try:
            return yaml.safe_load(text) if text.strip() else ""
        except yaml.YAMLError:
            return text

    def _render_sweep_box(self, data):
        from .config_panel.sections.sweep import _render_sweep_box
        return _render_sweep_box(self, data=data)

    def _add_standalone_monitor(self, component):
        from .config_panel.domain import _add_standalone_monitor
        return _add_standalone_monitor(self, component=component)

    def _remove_standalone_monitor(self, identity):
        from .config_panel.domain import _remove_standalone_monitor
        return _remove_standalone_monitor(self, identity=identity)

    def _render_monitors_box(self, data, resolved):
        from .config_panel.sections.monitors import _render_monitors_box
        return _render_monitors_box(self, data=data, resolved=resolved)

    def _render_metrics_box(self, data, resolved):
        from .config_panel.sections.monitors import _render_metrics_box
        return _render_metrics_box(self, data=data, resolved=resolved)

    def _on_recorded_metrics_changed(self, metrics):
        from .config_panel.domain import _on_recorded_metrics_changed
        return _on_recorded_metrics_changed(self, metrics=metrics)

    def add_block(self, block_name: str):
        from .config_panel.domain import add_block
        return add_block(self, block_name=block_name)

    def _on_universal_param_edited(self, key, value):
        from .config_panel.domain import _on_universal_param_edited
        return _on_universal_param_edited(self, key=key, value=value)

    def _remove_universal_param(self, key):
        from .config_panel.domain import _remove_universal_param
        return _remove_universal_param(self, key=key)

    def _remove_experiment_parameter(self, key):
        from .config_panel.domain import _remove_experiment_parameter
        return _remove_experiment_parameter(self, key=key)

    def _select_shared_component(self, kind, old, selected):
        from .config_panel.domain import _select_shared_component
        return _select_shared_component(self, kind=kind, old=old, selected=selected)

    def _add_universal_param(self, key, value=None):
        from .config_panel.domain import _add_universal_param
        return _add_universal_param(self, key=key, value=value)

    def _resolve_default_param_value(self, method_name: str, param_key: str):
        from .config_panel.domain import _resolve_default_param_value
        return _resolve_default_param_value(self, method_name=method_name, param_key=param_key)

    def _remove_method_param(self, method_name, param_key):
        from .config_panel.domain import _remove_method_param
        return _remove_method_param(self, method_name=method_name, param_key=param_key)

    def _add_method_param(self, method_name, param_key, value=None):
        from .config_panel.domain import _add_method_param
        return _add_method_param(self, method_name=method_name, param_key=param_key, value=value)

    def _checkpoint_row(self, data, resolved):
        """Checkpoints: every N timesteps, or a total count spread over the run (one key or the other).

        Switching the mode edits this row in place: rebuilding the boxes would delete the focused menu,
        and the scroll area would jump to wherever Qt moved the focus next.
        """
        source = data if any(k in data for k in CHECKPOINT_KEYS) else resolved if (self.view_mode == "resolved" or self._is_base_config()) else {}
        key = next((k for k in CHECKPOINT_KEYS if source.get(k) is not None), None)
        if key is None:
            return None
        state = {"key": key}
        mode = ComboBox()
        for k, text in zip(CHECKPOINT_KEYS, ("Every N timesteps", "Total count")):
            mode.addItem(text, k)
        mode.setCurrentIndex(CHECKPOINT_KEYS.index(key))
        mode.setToolTip("Save a checkpoint each time training passes another N timesteps, "
                        "or save this many, spread evenly over Total Timesteps")
        spin = integer_field(int(source[key]), minimum=1)
        spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
        spin.valueChanged.connect(lambda v: self._on_field_edited(state["key"], v))
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(mode)
        row.addWidget(_make_param_adder_button("Checkpoint schedule parameters", [], lambda _: None, allow_custom=False))
        block = IndentedHierarchyBlock()
        block.form_layout.add_parameter((key,), "Value", spin)
        row._parameter_block = block
        state["tail"] = (
            self._checkpoint_remove_button() if key in data
            else label(str(source[key]), "muted")
        )
        if state["tail"] is not None:
            row.addWidget(state["tail"])
        row.addStretch()
        mode.currentIndexChanged.connect(lambda _: self._switch_checkpoint_mode(state, spin, row, mode.currentData()))
        self._show_checkpoint_key(state, spin)
        return row

    def _checkpoint_remove_button(self):
        button = _make_subtle_tool_button("close", "Remove the checkpoint schedule", size=20, icon_size=12)
        button.clicked.connect(lambda: ([self.raw_data.pop(k, None) for k in CHECKPOINT_KEYS],
                                        self._mark_dirty(), self._render_boxes()))
        return button

    def _show_checkpoint_key(self, state, spin):
        for k in CHECKPOINT_KEYS:
            self.field_widgets.pop(k, None)
        self.field_widgets[state["key"]] = spin
        if hasattr(spin, "setSingleStep"):
            spin.setSingleStep(1_000 if state["key"] == "checkpoint_every_steps" else 1)

    def _switch_checkpoint_mode(self, state, spin, row, new_key):
        """Swap one checkpoint key for the other, converting the value so the schedule stays about the same."""
        old_key = state["key"]
        if self._block_updates or old_key == new_key:
            return
        resolved = self._get_resolved_defaults()
        total = int(self.raw_data.get("total_timesteps") or resolved.get("total_timesteps") or 0)
        value = int(self.raw_data.get(old_key) or resolved.get(old_key) or 1)
        for k in CHECKPOINT_KEYS:
            self.raw_data.pop(k, None)
        default = 10_000 if new_key == "checkpoint_every_steps" else DEFAULT_CHECKPOINTS
        self.raw_data[new_key] = max(1, -(-total // value)) if total else default
        state["key"] = new_key
        self._show_checkpoint_key(state, spin)
        spin.blockSignals(True)
        spin.setValue(self.raw_data[new_key])
        spin.blockSignals(False)
        if isinstance(state["tail"], QLabel):  # was inherited: it is this experiment's own setting now
            button = self._checkpoint_remove_button()
            row.replaceWidget(state["tail"], button)
            state["tail"].deleteLater()
            state["tail"] = button
        self._mark_dirty()

    def _add_experiment_param_from_menu(self, key: str, default_val: Any = None):
        """Apply a menu choice after its popup has closed.

        Rebuilding the entire pane while a popup still owns focus can cause Qt
        to re-parent the checkpoint row twice, leaving a phantom row on both
        sides of the add button.
        """
        self._add_experiment_param(key, default_val, render=False)
        QTimer.singleShot(0, self._render_boxes)

    def _add_experiment_param(self, key: str, default_val: Any = None, *, render: bool = True):
        from .config_panel.domain import _add_experiment_param
        return _add_experiment_param(self, key=key, default_val=default_val, render=render)

    def _prompt_add_custom_experiment_param(self):
        """Prompt user to add an arbitrary top-level experiment parameter."""
        key, ok = QInputDialog.getText(
            self,
            "Add Experiment Parameter",
            "Enter parameter name (e.g. eval_freq_steps, num_envs):"
        )
        if not ok or not key.strip():
            return
        key = key.strip()
        val_str, ok2 = QInputDialog.getText(
            self,
            "Initial Value",
            f"Enter initial value for '{key}':",
            text="1000" if key.endswith(("_steps", "_episodes", "_count")) else ""
        )
        if not ok2:
            return
        typed_val = _parse_yaml_value(val_str)
        self.raw_data[key] = typed_val
        self._mark_dirty()
        self._render_boxes()

    def flush_pending_save(self):
        """Write a pending auto-save now (before anything reads the file from disk, or on close)."""
        self._autosave_timer.stop()
        if getattr(self, "auto_save", False) and self.is_dirty:
            self.save_to_disk()

    def _typing_in_text_field(self) -> bool:
        """True while the user is typing into one of this viewer's text boxes."""
        focus = QApplication.focusWidget()
        return isinstance(focus, (QLineEdit, QPlainTextEdit, QTextEdit)) and self.isAncestorOf(focus)

    def _model_row_slot(self, method_name: str, current_model: str):
        """The extension point beside a method's model picker.

        Anything that edits a model architecture is a GUI concern and belongs
        in a plugin, not in this panel. The panel offers the row and the
        context describing it; a plugin decides whether to put anything there.

        The context carries `apply` rather than letting a contribution write
        to raw_data itself, so the panel stays the only thing that mutates
        its own document.
        """
        host = self.window()
        slots = getattr(host, "extension_slots", None)
        if slots is None:
            return None
        slot = slots.get_slot("experiment.method.model_row")
        if slot is None:
            return None

        from .plugins.slots import ExtensionSlotWidget

        return ExtensionSlotWidget(
            slot,
            parent=self,
            context={
                "method": method_name,
                "model": current_model,
                "paradigm": str(self.raw_data.get("paradigm", "") or ""),
                "apply": lambda name, mn=method_name: self._set_method_model(mn, name),
            },
        )

    def _set_method_model(self, method_name: str, model_name: str) -> None:
        from .config_panel.domain import _set_method_model
        return _set_method_model(self, method_name=method_name, model_name=model_name)

    def _mark_dirty(self):
        self.is_dirty = True
        if getattr(self, "auto_save", False) and self._typing_in_text_field():
            self._autosave_timer.start()  # restarts on each keystroke; saves once typing pauses
        elif getattr(self, "auto_save", False):
            self._autosave_timer.stop()  # this save covers any pending text edit
            self.save_to_disk()
        else:
            self._update_dirty_ui()
        self.config_changed.emit()

    def _update_dirty_ui(self):
        if self.is_dirty and not getattr(self, "auto_save", False):
            self.dirty_status.setText("● Modified (unsaved)")
            self.dirty_status.show()
            self.dirty_changed.emit(True)
        else:
            self.is_dirty = False
            self.dirty_status.setText("")
            self.dirty_status.hide()
            self.dirty_changed.emit(False)

    #: The only value each rollout field may take under a paradigm that forbids it.
    _FORBIDDEN_FIELD_VALUES = {"eval_episodes": 0}

    def _pin_fields_the_paradigm_forbids(self, rules):
        from .config_panel.domain import _pin_fields_the_paradigm_forbids
        return _pin_fields_the_paradigm_forbids(self, rules=rules)

    def _reconcile_group_base(self):
        from .config_panel.domain import _reconcile_group_base
        return _reconcile_group_base(self)

    def _drop_empty_methods(self):
        from .config_panel.domain import _drop_empty_methods
        return _drop_empty_methods(self)

    def save_to_disk(self, show_error: bool = False):
        """Save current YAML dictionary back to disk."""
        if not self.current_path:
            return False
        if self._hub_read_only():
            if show_error:
                QMessageBox.warning(self, "Read-only component", "Hub components are read-only. Make a component copy to edit it.")
            return False
        try:
            from src.app.components.configuration import ConfigurationFiles
            from src.app.workspace import WorkspacePaths
            tree = self.tree
            workspace = tree.workspace if tree else WorkspacePaths.for_process()
            self.document.save(
                lambda content: ConfigurationFiles(workspace).write(
                    self.current_path, content, allow_external=True),
                normalize=(self._clean_redundant_params, self._reconcile_group_base, self._drop_empty_methods),
                preamble=self._preamble,
            )
            self.is_dirty = False
            self._update_dirty_ui()
            if self.current_rel_path:
                self.dirty_state_changed.emit(self.current_rel_path, False)
            self.refresh_tree()
            self.save_requested.emit()
            return True
        except Exception as exc:
            if show_error:
                QMessageBox.critical(self, "Save Error", f"Could not save file to disk:\n{exc}")
            return False

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        if getattr(self, "auto_save", False) and self.is_dirty:
            self.save_to_disk()

    def get_overrides(self):
        """Build command-line Hydra overrides matching current UI values."""
        overrides = []
        data = self.raw_data

        exp_id = data.get("experiment_id") or (self.current_path.stem if self.current_path else None)
        if exp_id:
            overrides.append(f"++experiment_id='{exp_id}'")

        if "description" in data and data["description"]:
            overrides.append(f"++description='{data['description']}'")

        # Hydra config-group selections (defaults in in/config/config.yaml)
        for group in ("paradigm", "env", "site"):
            value = (self._get_explicit_env() if group == "env" else
                     self._site_selection(data) if group == "site" else data.get(group))
            if isinstance(value, str) and value and value != self.INHERIT:
                overrides.append(f"{group}={value}")

        if "seed" in data:
            overrides.append(f"seed={data['seed']}")
        if "total_timesteps" in data:
            overrides.append(f"total_timesteps={data['total_timesteps']}")
        if "epochs" in data:
            overrides.append(f"epochs={data['epochs']}")
        if "eval_every_n_epochs" in data:
            overrides.append(f"eval_every_n_epochs={data['eval_every_n_epochs']}")
        if "tensorboard" in data and self._plugin_enabled("tensorboard"):
            overrides.append(f"tensorboard={str(data['tensorboard']).lower()}")
        if "save_dataset" in data and data["save_dataset"]:
            overrides.append(f"save_dataset={str(data['save_dataset']).lower()}")
        if "recover" in data and data["recover"]:
            overrides.append(f"recover={str(data['recover']).lower()}")
        if "no_plot" in data:
            overrides.append(f"no_plot={str(data['no_plot']).lower()}")

        if "dataset_path" in data and data["dataset_path"]:
            overrides.append(f"dataset_path='{data['dataset_path']}'")

        # Universal params overrides
        if "methods" in data and isinstance(data["methods"], dict) and "params" in data["methods"]:
            for k, v in data["methods"]["params"].items():
                if isinstance(v, (int, float, str, bool)):
                    overrides.append(f"++methods.params.{k}={v}")

        # Methods overrides
        for m_name, m_spec in data.get("methods", {}).items():
            if m_name == "params" or not isinstance(m_spec, dict):
                continue
            for k, v in m_spec.items():
                if isinstance(v, (int, float, str, bool)):
                    overrides.append(f"++methods.{m_name}.{k}={v}")

        return overrides
