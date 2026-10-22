"""Full-screen, simplified, boxed configuration viewer for Theta-IDE."""
import copy
import inspect
import math
import random
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from PyQt6.QtCore import QByteArray, QEvent, QMimeData, QPoint, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QDrag, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_model import ConfigTree, default_method_model
from .config_model import add_method as add_method_to
from .config_model import remove_method as remove_method_from
from .sidetabs import svg_icon
from .theme import theme_color
from .widgets import (
    ComboBox, DoubleSpinBox, SpinBox, label, SmoothScrollArea, ToggleSlider,
    TagCloudWidget, LayerStackWidget,
)
from .metric_picker import MetricPicker, logged_metrics
from src.app.services import LOGS_DIR, PROJECT_ROOT
from src.app.telemetry.selection import SECTIONS as METRIC_SECTIONS

# Forms stay readable on wide windows: the column of boxes stops growing past CONTENT_WIDTH,
# and fields in label/field rows are capped by kind so a number doesn't get a 1,000px box.
CONTENT_WIDTH = 960
NUMBER_FIELD_WIDTH = 140
TEXT_FIELD_WIDTH = 220

CATEGORICAL_CHOICES = {
    "activation": ["relu", "gelu", "tanh", "elu", "silu", "sigmoid", "leaky_relu", "identity"],
    "activation_fn": ["relu", "gelu", "tanh", "elu", "silu", "sigmoid", "leaky_relu", "identity"],
    "act_fn": ["relu", "gelu", "tanh", "elu", "silu", "sigmoid", "leaky_relu", "identity"],
    "optimizer": ["adam", "adamw", "sgd", "rmsprop"],
    "opt": ["adam", "adamw", "sgd", "rmsprop"],
    "mode": ["min", "max"],
    "direction": ["min", "max"],
}


def _get_available_paradigms() -> list[str]:
    """Return all paradigm names defined in in/config/paradigms/*.yaml."""
    paradigms_dir = PROJECT_ROOT / "in" / "config" / "paradigms"
    if paradigms_dir.is_dir():
        names = [p.stem for p in paradigms_dir.glob("*.yaml") if not p.name.startswith(("_", "."))]
        if names:
            return sorted(names)
    return []


def _get_component_kind(rel_path: str) -> tuple[str, str]:
    """Return (Category Display Name, Pill Accent Role)."""
    p = (rel_path or "").lower()
    if p.startswith("agent/"):
        return ("Algorithm", "secondary")
    elif p.startswith("env/"):
        return ("Environment", "primary")
    elif p.startswith("model/"):
        return ("Model", "accent")
    elif p.startswith("data/"):
        return ("Data Loader", "number")
    elif p.startswith("eval/"):
        return ("Evaluator", "focus")
    elif p.startswith(("plot/", "plots/")):
        return ("Figure", "primary_hover")
    elif p.startswith(("paradigms/", "paradigm/")):
        return ("Paradigm", "accent")
    elif p.startswith("site/"):
        return ("Site Profile", "secondary")
    elif p.startswith("workflow/"):
        return ("Workflow", "primary")
    elif p.startswith(("hydra/", "sweeper/")):
        return ("Hydra / Sweeper", "muted")
    parts = Path(rel_path).parts
    return (parts[0].capitalize() if parts else "Component", "secondary")


def _make_categorical_combo(key: str, current_val: Any, on_change) -> ComboBox:
    combo = ComboBox()
    options = list(CATEGORICAL_CHOICES.get(key.lower(), []))
    str_val = str(current_val) if current_val is not None else ""
    if str_val and str_val not in options:
        options.insert(0, str_val)
    for opt in options:
        combo.addItem(opt)
    if str_val in options:
        combo.setCurrentText(str_val)
    combo.setEditable(True)
    combo.setMaximumWidth(TEXT_FIELD_WIDTH)
    combo.currentTextChanged.connect(on_change)
    return combo


SUBTLE_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "muted",
    (QIcon.Mode.Active, QIcon.State.Off): "text",
    (QIcon.Mode.Selected, QIcon.State.Off): "accent",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
}


def _make_subtle_tool_button(icon_name: str, tooltip: str, size: int = 24, icon_size: int = 14) -> QToolButton:
    """Create a subtle icon-only tool button (trash, close, plus) without prominent borders."""
    btn = QToolButton()
    btn.setIcon(svg_icon(icon_name, SUBTLE_ICON_COLORS))
    btn.setIconSize(QSize(icon_size, icon_size))
    btn.setFixedSize(size, size)
    btn.setToolTip(tooltip)
    btn.setStyleSheet("""
        QToolButton {
            background: transparent;
            border: none;
            border-radius: 3px;
            padding: 0px;
        }
        QToolButton:hover {
            background: #3c3836;
        }
    """)
    return btn


def _make_param_adder_button(
    tooltip: str,
    candidates: list[str],
    on_chosen,
    parent_widget: QWidget | None = None,
    allow_custom: bool = True,
) -> QToolButton:
    """Create a subtle plus button that opens a focused dropdown menu for selecting a parameter."""
    btn = _make_subtle_tool_button("plus", tooltip, size=22, icon_size=12)
    btn.setStyleSheet("""
        QToolButton {
            background: transparent;
            border: 1px dashed #504945;
            border-radius: 3px;
            padding: 0px;
        }
        QToolButton:hover {
            background: #3c3836;
            border-color: #a89984;
        }
        QToolButton::menu-indicator {
            image: none;
            width: 0px;
        }
    """)
    menu = QMenu(btn)
    menu.setStyleSheet("""
        QMenu {
            background-color: #282828;
            border: 1px solid #504945;
            border-radius: 4px;
            padding: 4px 0px;
        }
        QMenu::item {
            padding: 5px 18px 5px 12px;
            color: #ebdbb2;
            font-size: 12px;
        }
        QMenu::item:selected {
            background-color: #3c3836;
            color: #fbf1c7;
        }
        QMenu::separator {
            height: 1px;
            background: #3c3836;
            margin: 4px 6px;
        }
    """)
    for cand in candidates:
        act = menu.addAction(cand)
        act.triggered.connect(lambda _, c=cand: on_chosen(c))

    if allow_custom:
        if candidates:
            menu.addSeparator()
        act_custom = menu.addAction("+ Custom...")
        def _prompt_custom():
            val, ok = QInputDialog.getText(parent_widget or btn, "Add Parameter", "Parameter name:")
            if ok and val.strip():
                on_chosen(val.strip())
        act_custom.triggered.connect(_prompt_custom)

    btn.setMenu(menu)
    btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
    return btn



def _parse_yaml_value(text: str):
    """Convert text representation to typed YAML scalar, list, or dict."""
    if not isinstance(text, str):
        return text
    stripped = text.strip()
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    if stripped.lower() in ("null", "none", "~", ""):
        return None
    try:
        if "." in stripped or "e" in stripped.lower():
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass
    if (stripped.startswith("[") and stripped.endswith("]")) or (stripped.startswith("{") and stripped.endswith("}")):
        try:
            return yaml.safe_load(stripped)
        except Exception:
            pass
    return stripped


TRAINING_HYPERPARAMETERS = (
    "lr",
    "batch_size",
    "gamma",
    "epochs",
    "eval_every_n_epochs",
    "weight_decay",
    "from_study",
)


def get_agent_candidate_hyperparameters(
    agent_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Candidate hyperparameters specific to an agent algorithm."""
    if not agent_name:
        return []
    current_keys = current_keys or set()
    candidates: list[str] = []
    if tree:
        agent_params = tree.agent_hyperparameters(agent_name)
        agent_schema = tree.agents.get(agent_name).parameter_schema if agent_name in tree.agents else {}
        agent_keys = list(agent_params)
        for key in agent_schema.get("properties", {}):
            if key not in agent_keys:
                agent_keys.append(key)
        for k in agent_keys:
            if k not in ("algorithm", "name", "type") and k not in TRAINING_HYPERPARAMETERS:
                if k not in current_keys and k not in candidates:
                    candidates.append(k)
    return candidates


def get_model_candidate_hyperparameters(
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Candidate hyperparameters specific to a model architecture (or composite top-level)."""
    if not model_name:
        return []
    current_keys = current_keys or set()
    candidates: list[str] = []
    if tree:
        model_params = tree.model_parameters(model_name)
        for k in tree.model_parameter_keys(model_name):
            if k not in ("architecture", "type", "lightning_module", "name"):
                if k not in current_keys and k not in candidates:
                    candidates.append(k)
    return candidates


def get_training_candidate_hyperparameters(current_keys: set[str] | None = None) -> list[str]:
    """Candidate general training hyperparameters for a method."""
    current_keys = current_keys or set()
    return [k for k in TRAINING_HYPERPARAMETERS if k not in current_keys]


def get_method_candidate_hyperparameters(
    agent_name: str | None,
    model_name: str | None,
    tree: ConfigTree | None,
    current_keys: set[str] | None = None,
) -> list[str]:
    """Dynamically determine valid hyperparameters for a method based on its agent and model."""
    current_keys = current_keys or set()
    candidates: list[str] = []

    # 1. Base training parameters
    for p in get_training_candidate_hyperparameters(current_keys):
        if p not in candidates:
            candidates.append(p)

    # 2. Agent-specific parameters
    for k in get_agent_candidate_hyperparameters(agent_name, tree, current_keys):
        if k not in candidates:
            candidates.append(k)

    # 3. Model-specific parameters
    for k in get_model_candidate_hyperparameters(model_name, tree, current_keys):
        if k not in candidates:
            candidates.append(k)

    return candidates


def discover_datasets(env_name: str | None = None) -> list[str]:
    """Discover available dataset files and folders in in/datasets and results/datasets."""
    found = []
    base_dirs = [Path("in/datasets"), Path("results/datasets")]
    for b in base_dirs:
        try:
            if not b.is_dir():
                continue
            for p in b.glob("**/*"):
                try:
                    if p.name.startswith(".") or p.name.startswith("__"):
                        continue
                    if p.is_file() and p.suffix in (".pkl", ".csv", ".parquet", ".h5", ".pt"):
                        found.append(p.as_posix())
                    elif p.is_dir():
                        try:
                            has_files = any(p.iterdir())
                        except OSError:
                            has_files = False
                        if has_files:
                            found.append(p.as_posix())
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(list(set(found)))


class CompactDoubleSpinBox(DoubleSpinBox):
    """Keeps six decimals of precision but shows 0.95 rather than 0.950000."""

    def textFromValue(self, value):
        text = f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")
        return "0" if text in ("", "-0") else text


def number_field(value):
    """Spin box for a YAML number: floats step at one tenth of their magnitude, and values
    that start non-negative (counts, coefficients) can't be pushed below zero."""
    if isinstance(value, float):
        spin = CompactDoubleSpinBox()
        spin.setDecimals(6)
        magnitude = math.floor(math.log10(abs(value))) if value else -1
        spin.setSingleStep(10.0 ** (magnitude - 1))
    else:
        spin = SpinBox()
    spin.setRange(0 if value >= 0 else -1000000, 100000000)
    spin.setValue(value)
    return spin


class IndentedHierarchyBlock(QFrame):
    """Indented container with a subtle vertical guide line denoting hierarchy in Theta-IDE."""

    def __init__(self, parent=None, level: int = 1):
        super().__init__(parent)
        self.setObjectName(f"hierarchyBlock_lvl{level}")
        left_margin = 12 if level == 1 else 14
        border_color = theme_color("border")
        self.setStyleSheet(f"""
            QFrame#hierarchyBlock_lvl{level} {{
                border-left: 2px solid {border_color};
                margin-left: {left_margin}px;
                padding-left: 10px;
                background: transparent;
                border-top: none;
                border-right: none;
                border-bottom: none;
            }}
        """)
        self.form_layout = QFormLayout(self)
        self.form_layout.setContentsMargins(0, 4, 0, 6)
        self.form_layout.setVerticalSpacing(6)
        self.form_layout.setHorizontalSpacing(10)

    def addRow(self, label_or_widget, widget=None):
        if widget is not None:
            self.form_layout.addRow(label_or_widget, widget)
        else:
            self.form_layout.addRow(label_or_widget)


def _recorded_metrics(data):
    """The experiment's telemetry.metrics list, or None when it records every metric."""
    telemetry = data.get("telemetry") if isinstance(data, dict) else None
    metrics = telemetry.get("metrics") if isinstance(telemetry, dict) else None
    return [str(m) for m in metrics] if isinstance(metrics, (list, tuple)) else None


# Periodic checkpoints (src/app/core/checkpoints.py): an experiment sets one of these, not both.
CHECKPOINT_KEYS = ("checkpoint_every_steps", "num_checkpoints")
DEFAULT_CHECKPOINTS = 10

# Top-level experiment keys the fixed boxes already render. Everything else
# goes through _render_remaining_sections, so paradigm- and component-specific
# settings remain editable rather than being silently hidden.
FIXED_BOX_KEYS = frozenset(
    {
        "defaults",
        "paradigm",
        "experiment_id",
        "seed",
        "total_timesteps",
        "eval_episodes",
        "epochs",
        "eval_every_n_epochs",
        "methods",
        "env",
        "site",
        "trainer",
        "resources",
        "tensorboard",
        "no_plot",
        "save_dataset",
        "recover",
        # Dataset values are rendered in the inputs card.  Unsupported legacy
        # resource/sweeper blocks stay hidden rather than being surfaced as
        # generic editable settings.
        "hydra",
        "dataset_path",
        "offline_datasets",
        "description",
        "eval_freq_steps",
        "eval_freq",
        "checkpoint_every_steps",
        "num_checkpoints",
        "checkpoint",  # the Checkpoint Settings box
        "experiment_id",
    }
)


def parse_list_field(text):
    """Comma-separated text back to a YAML list, keeping each item's type.

    `plots` holds names while `eval_protocol.horizons` holds numbers, so items
    go through the YAML scalar parser instead of all becoming strings.
    """
    items = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            items.append(yaml.safe_load(part))
        except yaml.YAMLError:
            items.append(part)
    return items


_CONFIG_TREE = None
_CONFIG_TREE_LOADED = False


def invalidate_config_tree():
    """Clear cached in/config tree so subsequent lookups re-scan the filesystem."""
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    _CONFIG_TREE = None
    _CONFIG_TREE_LOADED = False


def set_config_tree(tree: ConfigTree | None):
    """Explicitly set or override the active ConfigTree (useful for tests and dependency injection)."""
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    _CONFIG_TREE = tree
    _CONFIG_TREE_LOADED = True


def config_tree(force_reload: bool = False):
    """The parsed in/config tree, or None if it cannot be found.

    Callers must handle None: the viewer falls back to free-text entry so it
    still works when run outside a checkout.
    """
    global _CONFIG_TREE, _CONFIG_TREE_LOADED
    if force_reload or not _CONFIG_TREE_LOADED:
        _CONFIG_TREE_LOADED = True
        try:
            _CONFIG_TREE = ConfigTree.discover()
        except (FileNotFoundError, OSError):
            _CONFIG_TREE = None
    return _CONFIG_TREE


def _leading_directives(text):
    """The comment block a config opens with, kept verbatim across a save.

    "# @package _global_" is a Hydra directive, not decoration: without it the
    file's keys are not applied at the global package level, so methods and the
    rest simply are not seen. Every one of the experiment configs starts with
    it, and yaml.safe_dump drops comments, so it has to be re-attached by hand.
    """
    kept = []
    for line in text.splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith("#"):
            break
        kept.append(line)
    return "".join(kept)


def _read_group_base(tree, group):
    """The group's _base.yaml as plain data, or {} when there is none."""
    if tree is None or not group:
        return {}
    path = tree.config_root / "experiment" / group / "_base.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _component_profiles(config_root: Path, directory: str, selector: str) -> dict[str, dict]:
    """Installed YAML profiles, keyed by the runtime component name."""
    profiles = {}
    profile_root = config_root / directory
    if profile_root.is_dir():
        for path in sorted(profile_root.rglob("*.yaml")):
            if path.name.startswith(("_", ".")):
                continue
            try:
                values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except (OSError, yaml.YAMLError):
                continue
            if isinstance(values, dict):
                name = str(values.get(selector) or path.stem)
                profiles.setdefault(name, values)
    return profiles


class ConfigBox(QFrame):
    """Themed card container representing a section of configuration."""
    def __init__(self, title=None, subtitle=None, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        self.title_label = None
        self.subtitle_label = None

    def add_widget(self, widget):
        self.layout.addWidget(widget)

    def add_layout(self, sub_layout):
        self.layout.addLayout(sub_layout)


class TitleLabel(QLabel):
    """Clickable heading label that emits clicked and double_clicked on click."""
    clicked = pyqtSignal()
    double_clicked = pyqtSignal()

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setObjectName("heading")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            self.double_clicked.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)



class InlineTitleEditor(QLineEdit):
    """Line editor supporting cancellation on Escape key press."""
    escape_pressed = pyqtSignal()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.escape_pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class MethodDragHandle(QToolButton):
    """Grip icon button to initiate dragging and reordering of method cards."""

    def __init__(self, method_name: str, parent=None):
        super().__init__(parent)
        self.method_name = method_name
        self.setIcon(svg_icon("grip", SUBTLE_ICON_COLORS))
        self.setIconSize(QSize(14, 14))
        self.setFixedSize(22, 22)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setToolTip("Drag to reorder method")
        self.setStyleSheet("""
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 3px;
                padding: 0;
            }
            QToolButton:hover {
                background: #3c3836;
            }
        """)
        self._drag_start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_start_pos is not None:
            distance = (event.pos() - self._drag_start_pos).manhattanLength()
            if distance >= QApplication.startDragDistance():
                self._start_drag()
                self._drag_start_pos = None
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_drag(self):
        drag = QDrag(self)
        mime = QMimeData()
        mime.setData("application/x-theta-method", QByteArray(self.method_name.encode("utf-8")))
        drag.setMimeData(mime)

        p = self.parent()
        while p and not isinstance(p, (ConfigBox, QFrame)):
            p = p.parent()
        target_widget = p or self
        pixmap = target_widget.grab()
        ghost = QPixmap(pixmap.size())
        ghost.fill(Qt.GlobalColor.transparent)
        painter = QPainter(ghost)
        painter.setOpacity(0.65)
        painter.drawPixmap(0, 0, pixmap)
        painter.end()

        drag.setPixmap(ghost)
        drag.setHotSpot(self._drag_start_pos or QPoint(pixmap.width() // 2, 15))
        drag.exec(Qt.DropAction.MoveAction)


class MethodCardsContainer(QWidget):
    """Container holding method cards arranged in 2 columns that accepts drops and displays drop indicator line."""

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(14)
        self.grid.setVerticalSpacing(14)
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.setAcceptDrops(True)
        self._drop_index = None
        self._cards: list[tuple[str, QWidget]] = []

    def add_method_card(self, method_name: str, card: QWidget):
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        idx = len(self._cards)
        row = idx // 2
        col = idx % 2
        self._cards.append((method_name, card))
        self.grid.addWidget(card, row, col)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            pos = event.position().toPoint()
            idx = self._calculate_drop_index(pos)
            if idx != self._drop_index:
                self._drop_index = idx
                self.update()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self._drop_index = None
        self.update()
        event.accept()

    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-theta-method"):
            method_name = bytes(event.mimeData().data("application/x-theta-method")).decode("utf-8")
            target_idx = self._drop_index
            self._drop_index = None
            self.update()
            if target_idx is not None:
                self.viewer.reorder_method(method_name, target_idx)
            event.acceptProposedAction()
        else:
            event.ignore()

    def _calculate_drop_index(self, pos: QPoint) -> int:
        if not self._cards:
            return 0
        for i, (_, card) in enumerate(self._cards):
            geom = card.geometry()
            if pos.y() < geom.top():
                return i
            if geom.top() <= pos.y() <= geom.bottom():
                if pos.x() < geom.center().x():
                    return i
                col = i % 2
                if col == 1 or i == len(self._cards) - 1:
                    if pos.x() >= geom.center().x():
                        return i + 1
        return len(self._cards)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drop_index is not None and self._cards:
            painter = QPainter(self)
            try:
                acc_color = QColor(theme_color("accent"))
            except Exception:
                acc_color = QColor("#fabd2f")
            pen = QPen(acc_color, 3)
            painter.setPen(pen)
            if self._drop_index < len(self._cards):
                card = self._cards[self._drop_index][1]
                g = card.geometry()
                painter.drawLine(g.left() - 4, g.top() + 4, g.left() - 4, g.bottom() - 4)
            else:
                card = self._cards[-1][1]
                g = card.geometry()
                painter.drawLine(g.right() + 4, g.top() + 4, g.right() + 4, g.bottom() - 4)
            painter.end()


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
        return config_tree()

    def __init__(self, settings_manager=None, parent=None):
        if isinstance(settings_manager, QWidget) and parent is None:
            parent = settings_manager
            settings_manager = None
        super().__init__(parent)
        self.settings_manager = settings_manager
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
        if self.settings_manager:
            self.settings_manager.changed.connect(self._on_settings_changed)

        self._init_ui()

    def _on_settings_changed(self):
        if self.settings_manager:
            new_mode = "resolved" if self.settings_manager.show_resolved_defaults else "overrides"
            if new_mode != self.view_mode:
                self.view_mode = new_mode
                self.chk_view_mode.blockSignals(True)
                self.chk_view_mode.setChecked(new_mode == "resolved")
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

        self.group_badge = TitleLabel("")
        self.group_badge.setObjectName("heading")
        self.group_badge.setToolTip("Group (click to rename)")
        self.group_badge.clicked.connect(self._start_group_edit)
        self.group_badge.double_clicked.connect(self._start_group_edit)
        self.group_badge.hide()
        hb_layout.addWidget(self.group_badge)

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
        hb_layout.addWidget(self.group_editor)

        self.breadcrumb_sep = QLabel("/")
        self.breadcrumb_sep.setObjectName("breadcrumbSep")
        self.breadcrumb_sep.setStyleSheet(
            "font-size: 20px; font-weight: 500; color: #888888; padding: 0 4px;"
        )
        self.breadcrumb_sep.hide()
        hb_layout.addWidget(self.breadcrumb_sep)

        self.file_title = TitleLabel("No config loaded")
        self.file_title.clicked.connect(self._start_title_edit)
        self.file_title.double_clicked.connect(self._start_title_edit)

        hb_layout.addWidget(self.file_title)

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
        hb_layout.addWidget(self.title_editor)

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
        self.chk_view_mode = QCheckBox("Show Resolved Defaults")
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
                tree = config_tree()
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
            invalidate_config_tree()
        if self.current_path and self.current_path.exists():
            if not self.is_dirty:
                self.load_file(self.current_path, self.current_rel_path or "")
            else:
                self._render_boxes()
        elif self.raw_data:
            self._render_boxes()

    def load_file(self, file_path: Path, rel_path: str):
        """Parse YAML file and render visual boxes."""
        if getattr(self, "auto_save", False) and self.is_dirty:
            self.save_to_disk()

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

    def _render_boxes(self):
        """Clear and build the configuration boxes."""
        self._block_updates = True
        self.field_widgets.clear()

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
        is_base = (self.current_path.stem == "_base") if self.current_path else False

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

        self._cap_form_fields()
        self.boxes_layout.addStretch()
        self._block_updates = False

    def _on_view_mode_toggled(self, checked: bool):
        self.view_mode = "resolved" if checked else "overrides"
        self._render_boxes()

    def _get_resolved_defaults(self) -> dict:
        tree = config_tree()
        group = self._group_name()
        resolved = {}
        if tree:
            root_cfg = getattr(tree, "config_root", None)
            if root_cfg and (root_cfg / "config.yaml").is_file():
                try:
                    resolved.update(yaml.safe_load((root_cfg / "config.yaml").read_text(encoding="utf-8")) or {})
                except Exception:
                    pass
            if group:
                base_data = _read_group_base(tree, group)
                if base_data:
                    resolved.update(base_data)
        return resolved

    def _group_name(self):
        """The experiment group this file sits in, from experiment/<group>/<file>."""
        parts = Path(str(self.current_rel_path or "")).parts
        return parts[1] if len(parts) > 2 and parts[0] == "experiment" else None

    def env_selection(self):
        """The chosen environment, or None when it is inherited from the base."""
        combo = getattr(self, "combo_env", None)
        if combo is None or combo.currentData() == self.INHERIT:
            return None
        return combo.currentText()

    def _inherited_paradigm(self):
        """The paradigm this experiment inherits from its group's _base.yaml."""
        tree, group = config_tree(), self._group_name()
        if tree is None or group is None:
            return None
        return tree.group(group).paradigm

    def _inherited_env(self):
        """The environment this experiment inherits from its group's _base.yaml."""
        tree, group = config_tree(), self._group_name()
        if tree is None or group is None or group not in tree.groups:
            return None
        return tree.group(group).env

    def _is_base_config(self) -> bool:
        """True if the currently open configuration is a group defaults (_base.yaml) file."""
        return bool(self.current_path and self.current_path.stem == "_base")

    def _get_explicit_env(self) -> str | None:
        """The environment explicitly overridden in defaults or raw_data."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return None
        for entry in self.raw_data.get("defaults") or []:
            if isinstance(entry, dict):
                if "override /env" in entry:
                    val = entry.get("override /env")
                    return str(val) if val is not None else None
                elif "/env" in entry:
                    val = entry.get("/env")
                    return str(val) if val is not None else None
        val = self.raw_data.get("env")
        return str(val) if isinstance(val, str) else None

    @staticmethod
    def _is_environment_choice(name: Any) -> bool:
        """Reject directory-like values; selectors should name environment configs."""
        return isinstance(name, str) and bool(name.strip()) and not name.strip().endswith(("/", "\\"))

    @staticmethod
    def _environment_for_name(tree: ConfigTree | None, name: str | None):
        if not tree or not name:
            return None
        if name in tree.environments:
            return tree.environments[name]
        return next((env for env in tree.environments.values() if env.name == name), None)

    def _environment_is_compatible(self, tree: ConfigTree | None, name: str | None, paradigm: str | None) -> bool:
        if not self._is_environment_choice(name):
            return False
        environment = self._environment_for_name(tree, name)
        rules = tree.paradigms.get(paradigm) if tree and paradigm else None
        if not environment:
            return False
        if not paradigm:
            return True
        return rules.permits_environment(environment) if rules else False

    def _add_unavailable_environment_option(
        self, combo: ComboBox, name: str, paradigm: str | None, tree: ConfigTree | None = None
    ) -> int | None:
        """Keep an existing incompatible or missing choice visible but impossible to select."""
        if not self._is_environment_choice(name):
            return None
        exists = self._environment_for_name(tree or config_tree(), name) is not None
        reason = f"incompatible with {paradigm}" if exists and paradigm else "missing"
        combo.addItem(f"{name} ({reason})", name)
        index = combo.count() - 1
        item = combo.model().item(index)
        if item is not None:
            item.setEnabled(False)
        return index

    def _set_experiment_env(self, env_name: str | None):
        """Update or insert override /env in defaults, removing legacy scalar env."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return
        self.raw_data.pop("env", None)
        defaults = self.raw_data.get("defaults")

        if env_name is None or env_name == self.INHERIT or env_name in ("", "— Select Environment —"):
            if isinstance(defaults, list):
                self.raw_data["defaults"] = [
                    d for d in defaults
                    if not (isinstance(d, dict) and ("override /env" in d or "/env" in d))
                ]
            self._mark_dirty()
            return

        if not isinstance(defaults, list):
            defaults = []
            self.raw_data["defaults"] = defaults

        found = False
        for entry in defaults:
            if isinstance(entry, dict) and ("override /env" in entry or "/env" in entry):
                entry.clear()
                entry["override /env"] = env_name
                found = True
                break
        if not found:
            defaults.append({"override /env": env_name})
        self._mark_dirty()

    def _get_base_env(self) -> str | None:
        """The environment configured in a group's _base.yaml defaults."""
        return self._get_explicit_env()

    def _set_base_env(self, env_name: str | None):
        """Update or insert override /env in the group's _base.yaml defaults."""
        self._set_experiment_env(env_name)

    def _clean_redundant_params(self):
        """Strip redundant parameters inherited from defaults or matching filename stem."""
        if not self.raw_data or not isinstance(self.raw_data, dict):
            return
        is_base = self._is_base_config()
        if is_base:
            self.raw_data.pop("group", None)
            self.raw_data.pop("experiment_id", None)
            # A group base may declare one default method, which new experiments inherit, so its
            # concrete methods are kept. (They used to be stripped here, on load and on save, which
            # deleted the default method from disk the first time the base was opened and saved.)
            # The one-method limit is enforced where methods are added.
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                if not self.raw_data["methods"]:
                    self.raw_data.pop("methods", None)
            return

        is_experiment = (
            (self.current_rel_path and self.current_rel_path.startswith("experiment/"))
            or "methods" in self.raw_data
            or any("_base" in str(d) for d in self.raw_data.get("defaults", []))
        )
        if is_experiment:
            self.raw_data.pop("group", None)
            stem = self.current_path.stem if self.current_path else ""
            if self.raw_data.get("experiment_id") == stem:
                self.raw_data.pop("experiment_id", None)
            inh_paradigm = self._inherited_paradigm()
            if inh_paradigm is not None and self.raw_data.get("paradigm") == inh_paradigm:
                self.raw_data.pop("paradigm", None)
            inh_env = self._inherited_env()
            explicit_env = self._get_explicit_env()
            if explicit_env is not None and (
                (inh_env is not None and explicit_env == inh_env) or explicit_env == self.INHERIT
            ):
                self._set_experiment_env(None)
            elif isinstance(self.raw_data.get("env"), str):
                self._set_experiment_env(self.raw_data.get("env"))
            if "description" in self.raw_data and not self.raw_data["description"]:
                self.raw_data.pop("description", None)
            if "resources" in self.raw_data and not self.raw_data["resources"]:
                self.raw_data.pop("resources", None)
            if "trainer" in self.raw_data and not self.raw_data["trainer"]:
                self.raw_data.pop("trainer", None)
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                if "params" in self.raw_data["methods"] and not self.raw_data["methods"]["params"]:
                    self.raw_data["methods"].pop("params", None)
            # Strip redundant default flags matching root/base defaults
            if self.raw_data.get("tensorboard") is False:
                self.raw_data.pop("tensorboard", None)
            if self.raw_data.get("save_dataset") is False:
                self.raw_data.pop("save_dataset", None)
            if self.raw_data.get("recover") is False:
                self.raw_data.pop("recover", None)
            if self.raw_data.get("no_plot") is False:
                self.raw_data.pop("no_plot", None)

    def _cap_form_fields(self):
        """Limit field widths in every label/field form and two-column grid."""
        for grid in self.container.findChildren(QGridLayout):
            if isinstance(grid.parentWidget(), MethodCardsContainer):
                continue
            for index in range(grid.count()):
                widget = grid.itemAt(index).widget()
                if isinstance(widget, QAbstractSpinBox):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
                elif isinstance(widget, (QComboBox, QLineEdit)):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
            # An empty last column takes the slack, so fields stay next to their labels
            grid.setColumnStretch(grid.columnCount(), 1)
        for row_layout in self.container.findChildren(QHBoxLayout):
            for index in range(row_layout.count()):
                widget = row_layout.itemAt(index).widget()
                if isinstance(widget, QAbstractSpinBox):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
        for form in self.container.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                widget = item.widget() if item else None
                label_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if widget is not None and label_item and isinstance(label_item.widget(), QLabel):
                    # Labels sit at the top of their row; match the field height so the text lines up
                    label_item.widget().setMinimumHeight(widget.sizeHint().height())
                    label_item.widget().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
                if isinstance(widget, QAbstractSpinBox):
                    widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
                elif isinstance(widget, (QComboBox, QLineEdit)):
                    widget.setMaximumWidth(TEXT_FIELD_WIDTH)

    def _render_experiment_boxes(self):
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

        # Paradigm + Environment, driven by in/config/paradigms constraints.
        tree = config_tree()
        top_box = None
        if is_base:
            box_base = ConfigBox("1. Group Paradigm & Environment", "Default paradigm and environment applied to all experiments in this group")
            self._add_experiment_section_header(box_base, data, resolved)
            self._add_experiment_description(box_base, data)
            form_base = QGridLayout()
            form_base.setVerticalSpacing(10)
            form_base.setHorizontalSpacing(16)
            form_base.setColumnMinimumWidth(0, 145)
            form_base.setColumnMinimumWidth(2, 145)

            paradigm_label = label("Paradigm")
            paradigm_label.setFixedWidth(145)
            form_base.addWidget(paradigm_label, 0, 0)
            self.combo_paradigm = ComboBox()
            self.combo_paradigm.addItem("— Select Paradigm —", None)
            if tree and tree.paradigms:
                self.combo_paradigm.addItems(sorted(tree.paradigms))
            base_p = data.get("paradigm")
            if base_p and tree and base_p in tree.paradigms:
                self.combo_paradigm.setCurrentText(base_p)
            else:
                self.combo_paradigm.setCurrentIndex(0)
            form_base.addWidget(self.combo_paradigm, 0, 1)

            env_label = label("Environment")
            env_label.setFixedWidth(145)
            form_base.addWidget(env_label, 0, 2)
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
                    allowed = sorted(set(e.name for e in tree.environments.values() if self._is_environment_choice(e.name)))
                else:
                    allowed = []
                self.combo_env.addItems(allowed)
                candidate = self._get_base_env() or (curr_data if isinstance(curr_data, str) else None)
                if candidate in allowed:
                    self.combo_env.setCurrentText(candidate)
                elif self._is_environment_choice(candidate):
                    unavailable_index = self._add_unavailable_environment_option(self.combo_env, candidate, selected_p, tree)
                    self.combo_env.setCurrentIndex(unavailable_index if unavailable_index is not None else 0)
                else:
                    self.combo_env.setCurrentIndex(0)
                self.combo_env.blockSignals(False)

            base_env = self._get_base_env()
            if base_p and tree and base_p in tree.paradigms:
                allowed_init = [e.name for e in tree.environments_for(base_p) if self._is_environment_choice(e.name)]
            elif tree:
                allowed_init = sorted(set(e.name for e in tree.environments.values() if self._is_environment_choice(e.name)))
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
            form_base.addWidget(self.combo_env, 0, 3)

            def _on_base_p_changed(p_text):
                val = None if p_text in ("", "— Select Paradigm —") else p_text
                self.raw_data["paradigm"] = val
                _refresh_base_env_choices(val)
                self._mark_dirty()

            def _on_base_e_changed(e_text):
                val = None if e_text in ("", "— Select Environment —") else e_text
                self._set_base_env(val)

            self.combo_paradigm.currentTextChanged.connect(_on_base_p_changed)
            self.combo_env.currentTextChanged.connect(_on_base_e_changed)

            self.field_widgets["paradigm"] = self.combo_paradigm
            self.field_widgets["env"] = self.combo_env

            box_base.add_layout(form_base)
            if self._has_experiment_lower_params(data, resolved):
                self._add_experiment_separator(box_base)
            self._add_experiment_context_fields(box_base, data, resolved)
            top_box = box_base
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
                    allowed_envs = [e.name for e in tree.environments_for(selected_paradigm) if self._is_environment_choice(e.name)]
                else:
                    allowed_envs = sorted(set(e.name for e in tree.environments.values() if self._is_environment_choice(e.name)))
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
                self.combo_env.setToolTip(
                    f"Environments for {self.combo_paradigm.currentText()}"
                )
                self.combo_env.currentTextChanged.connect(
                    lambda v: self._on_experiment_env_changed(v)
                )
                self.field_widgets["env"] = self.combo_env

            # An experiment starts with its research environment. For static
            # paradigms that environment is realized by a dataset; the data
            # module below remains an adapter, not a second kind of dataset.
            box_inputs = ConfigBox(
                "1. Experiment Inputs",
                "Paradigm, research environment, and its live or static source",
            )
            self._add_experiment_section_header(box_inputs, data, resolved)
            self._add_experiment_description(box_inputs, data)
            inputs_grid = QGridLayout()
            inputs_grid.setVerticalSpacing(10)
            inputs_grid.setHorizontalSpacing(16)
            inputs_grid.setColumnMinimumWidth(0, 145)
            inputs_grid.setColumnMinimumWidth(2, 145)
            if hasattr(self, "combo_paradigm"):
                paradigm_label = label("Paradigm")
                paradigm_label.setFixedWidth(145)
                inputs_grid.addWidget(paradigm_label, 0, 0)
                inputs_grid.addWidget(self.combo_paradigm, 0, 1)
            if hasattr(self, "combo_env"):
                env_label = label("Environment / Task")
                env_label.setFixedWidth(145)
                inputs_grid.addWidget(env_label, 0, 2)
                inputs_grid.addWidget(self.combo_env, 0, 3)
            inputs_grid.setColumnStretch(1, 1)
            inputs_grid.setColumnStretch(3, 1)
            box_inputs.add_layout(inputs_grid)
            if self._has_experiment_lower_params(data, resolved):
                self._add_experiment_separator(box_inputs)
            self._add_experiment_context_fields(box_inputs, data, resolved)

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

        form_budget = QGridLayout()
        form_budget.setVerticalSpacing(10)
        form_budget.setHorizontalSpacing(16)
        form_budget.setColumnMinimumWidth(0, 145)
        form_budget.setColumnMinimumWidth(2, 145)

        # Baseline budget widget instantiation
        self.spin_timesteps = SpinBox()
        self.spin_timesteps.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_timesteps.setRange(100, 100_000_000)
        self.spin_timesteps.setSingleStep(1_000)
        self.spin_timesteps.setValue(int(data.get("total_timesteps") or resolved.get("total_timesteps") or 10000))
        self.spin_timesteps.valueChanged.connect(lambda v: self._on_field_edited("total_timesteps", v))
        self.field_widgets["total_timesteps"] = self.spin_timesteps

        self.spin_seed = SpinBox()
        self.spin_seed.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_seed.setRange(0, 2147483647)
        self.spin_seed.setValue(int(data.get("seed") if data.get("seed") is not None else resolved.get("seed", 42)))
        self.spin_seed.valueChanged.connect(lambda v: self._on_field_edited("seed", v))
        self.field_widgets["seed"] = self.spin_seed

        self.spin_eval_ep = SpinBox()
        self.spin_eval_ep.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_eval_ep.setRange(0, 1000)
        self.spin_eval_ep.setValue(int(data.get("eval_episodes") if data.get("eval_episodes") is not None else resolved.get("eval_episodes", 100)))
        self.spin_eval_ep.valueChanged.connect(lambda v: self._on_field_edited("eval_episodes", v))
        self.field_widgets["eval_episodes"] = self.spin_eval_ep

        eval_ep_enabled = True
        is_timestep_budget = False
        if tree and paradigm in tree.paradigms:
            rules = tree.paradigms[paradigm]
            is_timestep_budget = rules.budget_unit == "timesteps"
            base = _read_group_base(tree, self._group_name())
            if not rules.field_enabled("eval_episodes"):
                eval_ep_enabled = False
                self.spin_eval_ep.setEnabled(False)
                if "eval_episodes" in base:
                    self.spin_eval_ep.setValue(int(base["eval_episodes"]))
                    self.spin_eval_ep.setSuffix(f"   (fixed by {paradigm})")
                else:
                    self.spin_eval_ep.setSuffix(f"   (not used by {paradigm})")
                self.spin_eval_ep.setToolTip(rules.disabled_reason("eval_episodes"))

        show_timesteps = is_timestep_budget and ("total_timesteps" in data or show_resolved)
        show_seed = ("seed" in data or show_resolved)
        show_eval_ep = is_timestep_budget and ("eval_episodes" in data or show_resolved)
        show_eval_freq = is_timestep_budget and ("eval_freq_steps" in data or "eval_freq" in data or (show_resolved and ("eval_freq_steps" in resolved or "eval_freq" in resolved)))
        show_epochs = not is_timestep_budget and ("epochs" in data or show_resolved)
        show_eval_every = not is_timestep_budget and ("eval_every_n_epochs" in data or show_resolved)

        # Row layout for epochs
        self.spin_epochs = SpinBox()
        self.spin_epochs.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_epochs.setRange(1, 100000)
        self.spin_epochs.setValue(int(data.get("epochs") or resolved.get("epochs") or 10))
        self.spin_epochs.valueChanged.connect(lambda v: self._on_field_edited("epochs", v))
        self.field_widgets["epochs"] = self.spin_epochs

        row_epochs = QHBoxLayout()
        row_epochs.setSpacing(6)
        row_epochs.addWidget(self.spin_epochs)
        if "epochs" in data and not is_base:
            btn_del_ep = _make_subtle_tool_button("close", "Remove epochs override to rely on group default", size=20, icon_size=12)
            btn_del_ep.clicked.connect(lambda: (self.raw_data.pop("epochs", None), self._mark_dirty(), self._render_boxes()))
            row_epochs.addWidget(btn_del_ep)
        elif not is_base:
            row_epochs.addWidget(label(str(self.spin_epochs.value()), "muted"))
        row_epochs.addStretch()

        # Row layout for eval_every_n_epochs
        self.spin_eval_every = SpinBox()
        self.spin_eval_every.setMaximumWidth(NUMBER_FIELD_WIDTH)
        self.spin_eval_every.setRange(1, 1000)
        self.spin_eval_every.setValue(int(data.get("eval_every_n_epochs") or resolved.get("eval_every_n_epochs") or 1))
        self.spin_eval_every.valueChanged.connect(lambda v: self._on_field_edited("eval_every_n_epochs", v))
        self.field_widgets["eval_every_n_epochs"] = self.spin_eval_every

        row_eval_every = QHBoxLayout()
        row_eval_every.setSpacing(6)
        row_eval_every.addWidget(self.spin_eval_every)
        if "eval_every_n_epochs" in data and not is_base:
            btn_del_ee = _make_subtle_tool_button("close", "Remove eval_every_n_epochs override to rely on group default", size=20, icon_size=12)
            btn_del_ee.clicked.connect(lambda: (self.raw_data.pop("eval_every_n_epochs", None), self._mark_dirty(), self._render_boxes()))
            row_eval_every.addWidget(btn_del_ee)
        elif not is_base:
            row_eval_every.addWidget(label(str(self.spin_eval_every.value()), "muted"))
        row_eval_every.addStretch()

        # Row layout for timesteps
        row_timesteps = QHBoxLayout()
        row_timesteps.setSpacing(6)
        row_timesteps.addWidget(self.spin_timesteps)
        if "total_timesteps" in data and not is_base:
            btn_del_tt = _make_subtle_tool_button("close", "Remove total_timesteps override to rely on group default", size=20, icon_size=12)
            btn_del_tt.clicked.connect(lambda: (self.raw_data.pop("total_timesteps", None), self._mark_dirty(), self._render_boxes()))
            row_timesteps.addWidget(btn_del_tt)
        elif not is_base:
            row_timesteps.addWidget(label(str(self.spin_timesteps.value()), "muted"))
        row_timesteps.addStretch()

        # Row layout for seed
        row_seed = QHBoxLayout()
        row_seed.setSpacing(6)
        row_seed.addWidget(self.spin_seed)
        btn_rand_seed = QToolButton()
        btn_rand_seed.setText("Random")
        btn_rand_seed.setToolTip("Pick a random seed")
        btn_rand_seed.clicked.connect(lambda: self.spin_seed.setValue(random.randint(1, 9999)))
        row_seed.addWidget(btn_rand_seed)
        if "seed" in data and not is_base:
            btn_del_seed = _make_subtle_tool_button("close", "Remove seed override to rely on group default", size=20, icon_size=12)
            btn_del_seed.clicked.connect(lambda: (self.raw_data.pop("seed", None), self._mark_dirty(), self._render_boxes()))
            row_seed.addWidget(btn_del_seed)
        elif not is_base:
            row_seed.addWidget(label(str(self.spin_seed.value()), "muted"))
        row_seed.addStretch()

        # Row layout for eval_episodes
        row_eval_ep = QHBoxLayout()
        row_eval_ep.setSpacing(6)
        row_eval_ep.addWidget(self.spin_eval_ep)
        if "eval_episodes" in data and not is_base:
            btn_del_ee = _make_subtle_tool_button("close", "Remove eval_episodes override to rely on group default", size=20, icon_size=12)
            btn_del_ee.clicked.connect(lambda: (self.raw_data.pop("eval_episodes", None), self._mark_dirty(), self._render_boxes()))
            row_eval_ep.addWidget(btn_del_ee)
        elif eval_ep_enabled and not is_base:
            row_eval_ep.addWidget(label(str(self.spin_eval_ep.value()), "muted"))
        row_eval_ep.addStretch()

        # Row layout for eval frequency (steps)
        row_eval_freq = None
        if show_eval_freq:
            freq_key = "eval_freq_steps" if ("eval_freq_steps" in data or "eval_freq_steps" in resolved) else "eval_freq"
            spin_eval_freq = SpinBox()
            spin_eval_freq.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_eval_freq.setRange(10, 100_000_000)
            spin_eval_freq.setSingleStep(1_000)
            spin_eval_freq.setValue(int(data.get(freq_key) or resolved.get(freq_key) or 10000))
            spin_eval_freq.valueChanged.connect(lambda v, fk=freq_key: self._on_field_edited(fk, v))
            self.field_widgets[freq_key] = spin_eval_freq
            row_eval_freq = QHBoxLayout()
            row_eval_freq.setSpacing(6)
            row_eval_freq.addWidget(spin_eval_freq)
            if freq_key in data and not is_base:
                btn_del_ef = _make_subtle_tool_button("close", f"Remove '{freq_key}' override to rely on group default", size=20, icon_size=12)
                btn_del_ef.clicked.connect(lambda _, fk=freq_key: (self.raw_data.pop(fk, None), self._mark_dirty(), self._render_boxes()))
                row_eval_freq.addWidget(btn_del_ef)
            elif not is_base:
                row_eval_freq.addWidget(label(str(spin_eval_freq.value()), "muted"))
            row_eval_freq.addStretch()

        row_checkpoints = self._checkpoint_row(data, resolved)

        # Extra custom experiment parameters (top-level scalars)
        standard_top_level_keys = {
            "defaults", "paradigm", "experiment_id", "description",
            "seed", "total_timesteps", "eval_episodes", "eval_freq_steps", "eval_freq",
            "epochs", "eval_every_n_epochs", "checkpoint_every_steps", "num_checkpoints",
            "methods", "env", "site", "trainer", "resources", "dataset", "dataset_path",
            "tensorboard", "no_plot", "save_dataset", "recover",
            "hydra", "plots", "tuning", "sweeper"
        }
        extra_experiment_items = []
        for extra_k, extra_v in list(data.items()):
            if extra_k in standard_top_level_keys:
                continue
            if isinstance(extra_v, (int, float, str, bool)):
                row_extra = QHBoxLayout()
                row_extra.setSpacing(6)
                if isinstance(extra_v, bool):
                    chk_extra = ToggleSlider(checked=extra_v)
                    chk_extra.toggled.connect(lambda v, ek=extra_k: self._on_field_edited(ek, v))
                    row_extra.addWidget(chk_extra)
                elif isinstance(extra_v, (int, float)):
                    spin_extra = number_field(extra_v)
                    spin_extra.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_extra.valueChanged.connect(lambda v, ek=extra_k: self._on_field_edited(ek, v))
                    row_extra.addWidget(spin_extra)
                else:
                    line_extra = QLineEdit(str(extra_v if extra_v is not None else ""))
                    line_extra.setMaximumWidth(TEXT_FIELD_WIDTH)
                    line_extra.textChanged.connect(lambda v, ek=extra_k: self._on_field_edited(ek, v))
                    row_extra.addWidget(line_extra)
                if not is_base:
                    btn_del_extra = _make_subtle_tool_button("close", f"Remove '{extra_k}' parameter", size=20, icon_size=12)
                    btn_del_extra.clicked.connect(lambda _, ek=extra_k: (self.raw_data.pop(ek, None), self._mark_dirty(), self._render_boxes()))
                    row_extra.addWidget(btn_del_extra)
                row_extra.addStretch()
                extra_experiment_items.append((label(str(extra_k)), row_extra))

        def budget_label(text):
            widget = label(text)
            widget.setFixedWidth(145)
            return widget

        grid_items = []
        if is_timestep_budget:
            if show_timesteps:
                grid_items.append((budget_label("Total Timesteps"), row_timesteps))
            if show_seed:
                grid_items.append((budget_label("Random Seed"), row_seed))
            if show_eval_ep:
                grid_items.append((budget_label("Eval Episodes"), row_eval_ep))
            if show_eval_freq and row_eval_freq:
                grid_items.append((budget_label("Eval Frequency"), row_eval_freq))
            if row_checkpoints:  # counted in timesteps: online RL only (validation rejects it elsewhere)
                grid_items.append((budget_label("Checkpoints"), row_checkpoints))
        else:
            if show_epochs:
                grid_items.append((budget_label("Epochs"), row_epochs))
            if show_seed:
                grid_items.append((budget_label("Random Seed"), row_seed))
            if show_eval_every:
                grid_items.append((budget_label("Eval Frequency (Epochs)"), row_eval_every))
        for extra_lbl, extra_row in extra_experiment_items:
            extra_lbl.setFixedWidth(145)
            grid_items.append((extra_lbl, extra_row))

        curr_row = 0
        for idx, (lbl_widget, val_layout) in enumerate(grid_items):
            c_col = 0 if (idx % 2 == 0) else 2
            c_row = idx // 2
            form_budget.addWidget(lbl_widget, c_row, c_col)
            form_budget.addLayout(val_layout, c_row, c_col + 1)
            curr_row = max(curr_row, c_row + 1)
        field_column_min = max(
            (val_layout.sizeHint().width() for _, val_layout in grid_items),
            default=0,
        )
        form_budget.setColumnMinimumWidth(1, field_column_min)
        form_budget.setColumnMinimumWidth(3, field_column_min)
        form_budget.setColumnStretch(1, 1)
        form_budget.setColumnStretch(3, 1)

        # Runtime & Logging Options (TensorBoard, Save Dataset, Recover, No Plots)
        OPTIONAL_RUNTIME_FLAGS = [
            ("tensorboard", "Log to TensorBoard"),
            ("save_dataset", "Save Dataset"),
            ("recover", "Recover Checkpoint"),
            ("no_plot", "No Plots"),
        ]

        chk_row = QHBoxLayout()
        chk_row.setSpacing(14)
        chk_widgets = {}

        # Show checkboxes only for flags explicitly present in the experiment config (or when showing resolved defaults)
        active_flags = [
            (k, lbl) for k, lbl in OPTIONAL_RUNTIME_FLAGS
            if k in data or show_resolved
        ]

        for flag_key, flag_label in active_flags:
            row_flag = QHBoxLayout()
            row_flag.setSpacing(6)
            chk = ToggleSlider(checked=bool(data.get(flag_key, resolved.get(flag_key, False))))
            chk.toggled.connect(
                lambda v, fk=flag_key: (
                    self._on_field_edited(fk, True)
                    if v
                    else (self.raw_data.pop(fk, None), self._mark_dirty(), self._render_boxes())
                )
            )
            row_flag.addWidget(chk)
            row_flag.addWidget(label(flag_label))
            chk_widgets[flag_key] = chk

            if flag_key in data and not is_base:
                btn_del_flag = _make_subtle_tool_button("close", f"Remove '{flag_key}' override", size=18, icon_size=11)
                btn_del_flag.clicked.connect(
                    lambda _, fk=flag_key: (self.raw_data.pop(fk, None), self._mark_dirty(), self._render_boxes())
                )
                row_flag.addWidget(btn_del_flag)
            elif not is_base:
                row_flag.addWidget(label(str(chk.isChecked()).lower(), "muted"))

            chk_row.addLayout(row_flag)

        self.chk_tb = chk_widgets.get("tensorboard")
        self.chk_save_ds = chk_widgets.get("save_dataset")
        self.chk_recover = chk_widgets.get("recover")
        self.chk_no_plot = chk_widgets.get("no_plot")

        chk_row.addStretch()
        form_budget.addLayout(chk_row, curr_row, 0, 1, 4)

        box_budget.add_layout(form_budget)
        self.boxes_layout.addWidget(box_budget)

        # --- BOX 2 / 3: METHODS & AGENT ARCHITECTURES ---
        methods_dict = data.get("methods", {})
        methods_num = "3" if is_base else "2"
        box_methods = ConfigBox("Methods", "Configured agents, policies, and hyperparameter overrides")
        m_layout = QVBoxLayout()
        m_layout.setSpacing(10)

        # Universal / Shared Parameters (methods.params)
        has_params = "params" in methods_dict and isinstance(methods_dict["params"], dict)
        resolved_params = resolved.get("methods", {}).get("params", {}) if isinstance(resolved.get("methods"), dict) else {}
        show_params = has_params or (self.view_mode == "resolved" and bool(resolved_params))

        # The method-plus creates a blank method; shared parameters have their
        # own explicit action at the opposite edge of this header.
        methods_header = QHBoxLayout()
        methods_header.setSpacing(8)
        lbl_methods_title = label("Methods", "heading")
        lbl_methods_title.setStyleSheet("font-size: 23px; font-weight: 600;")
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

            p_header.addStretch()
            if has_params:
                btn_rm_params = _make_subtle_tool_button("trash", "Remove Shared Parameters block", size=24, icon_size=14)
                btn_rm_params.clicked.connect(lambda: self.remove_block("params"))
                p_header.addWidget(btn_rm_params)
            p_layout.addLayout(p_header)

            curr_params = methods_dict.get("params", {}) if has_params else resolved_params

            common_universal = ["lr", "gamma", "epochs", "eval_every_n_epochs", "weight_decay", "batch_size", "ent_coef"]
            if tree:
                for a in sorted(tree.agents):
                    common_universal.append(f"agent.{a}")
                for m in tree.model_names:
                    common_universal.append(f"model.{m}")
            cand_u = [u for u in common_universal if u not in curr_params]
            btn_add_u = _make_param_adder_button("Add shared parameter", cand_u, lambda pk: self._add_universal_param(pk), parent_widget=self)

            rendered_u_rows = []

            def _render_param_group(parent_layout, values: dict, parent_path: tuple[str, ...] = (), level: int = 0):
                for key, val in list(values.items()):
                    key_path = parent_path + (str(key),)
                    dotted_path = ".".join(key_path)
                    if isinstance(val, dict):
                        group = IndentedHierarchyBlock(level=max(1, level + 1))
                        group_header = QHBoxLayout()
                        group_header.setSpacing(6)
                        group_header.addWidget(label(f"[{key}]", "muted"))
                        group_header.addStretch()
                        if has_params:
                            btn_del_group = _make_subtle_tool_button(
                                "close", f"Remove parameter group '{dotted_path}'", size=20, icon_size=12
                            )
                            btn_del_group.clicked.connect(lambda _, kp=dotted_path: self._remove_universal_param(kp))
                            group_header.addWidget(btn_del_group)
                        group.addRow("", group_header)
                        _render_param_group(group.form_layout, val, key_path, level + 1)
                        if isinstance(parent_layout, QFormLayout):
                            parent_layout.addRow("", group)
                        else:
                            parent_layout.addWidget(group)
                        continue

                    row = QHBoxLayout()
                    row.setSpacing(8)
                    field_label = label(str(key))
                    field_label.setMinimumWidth(160)
                    row.addWidget(field_label)
                    if isinstance(val, bool):
                        widget = ToggleSlider(checked=val)
                        widget.toggled.connect(lambda value, kp=dotted_path: self._on_universal_param_edited(kp, value))
                    elif isinstance(val, (int, float)):
                        widget = number_field(val)
                        widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
                        widget.valueChanged.connect(lambda value, kp=dotted_path: self._on_universal_param_edited(kp, value))
                    else:
                        widget = QLineEdit(str(val if val is not None else ""))
                        widget.setMaximumWidth(TEXT_FIELD_WIDTH)
                        widget.textChanged.connect(lambda value, kp=dotted_path: self._on_universal_param_edited(kp, value))
                    row.addWidget(widget)
                    if has_params:
                        btn_del_param = _make_subtle_tool_button(
                            "close", f"Remove parameter '{dotted_path}'", size=20, icon_size=12
                        )
                        btn_del_param.clicked.connect(lambda _, kp=dotted_path: self._remove_universal_param(kp))
                        row.addWidget(btn_del_param)
                    row.addStretch()
                    rendered_u_rows.append(row)
                    if isinstance(parent_layout, QFormLayout):
                        parent_layout.addRow("", row)
                    else:
                        parent_layout.addLayout(row)

            _render_param_group(p_layout, curr_params)

            if rendered_u_rows:
                rendered_u_rows[-1].insertWidget(rendered_u_rows[-1].count() - 1, btn_add_u)
            else:
                row_empty_p = QHBoxLayout()
                row_empty_p.addWidget(label("No shared parameters configured.", "muted"))
                row_empty_p.addWidget(btn_add_u)
                row_empty_p.addStretch()
                p_layout.addLayout(row_empty_p)

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

            show_ag_defs = bool(self.settings_manager and (self.settings_manager.show_method_agent_defaults or self.settings_manager.show_all_default_params)) or (self.view_mode == "resolved")
            show_mo_defs = bool(self.settings_manager and (self.settings_manager.show_method_model_defaults or self.settings_manager.show_all_default_params)) or (self.view_mode == "resolved")
            show_all_defs = bool(self.settings_manager and self.settings_manager.show_all_default_params) or (self.view_mode == "resolved")

            for m_name, m_spec in methods_dict.items():
                if m_name == "params" or not isinstance(m_spec, dict):
                    continue
                sub_card = QFrame()
                sub_card.setObjectName("card")
                card_vbox = QVBoxLayout(sub_card)
                card_vbox.setContentsMargins(10, 10, 10, 10)
                card_vbox.setSpacing(8)

                sc_layout = QFormLayout()
                sc_layout.setContentsMargins(0, 0, 0, 0)
                sc_layout.setVerticalSpacing(8)

                title_row = QHBoxLayout()
                title_row.setSpacing(6)

                btn_grip = MethodDragHandle(m_name, sub_card)
                title_row.addWidget(btn_grip)

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
                btn_remove = _make_subtle_tool_button("trash", f"Remove method '{m_name}' from this experiment", size=24, icon_size=14)
                btn_remove.clicked.connect(lambda _, mn=m_name: self.remove_method(mn))
                title_row.addWidget(btn_remove)
                sc_layout.addRow(title_row)

                # --- AGENT SECTION ---
                agent_raw = m_spec.get("agent")
                if isinstance(agent_raw, dict):
                    agent_val = (agent_raw.get("algorithm") or agent_raw.get("name") or next((k for k in agent_raw if k != "name"), ""))
                else:
                    agent_val = str(agent_raw or "")
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
                                tree and (
                                    agent_val in tree.agents
                                    or any(a.name == agent_val or a.algorithm == agent_val for a in tree.agents.values())
                                    or (tree.config_root / "agent" / f"{agent_val}.yaml").is_file()
                                )
                            )
                            if is_missing:
                                txt_agent.setToolTip(f"Missing component: Algorithm '{agent_val}' is not installed in in/config/agent/. Install it via Component Hub.")
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
                        methods = self.raw_data.setdefault("methods", {})
                        ms = methods.setdefault(mn, {})
                        if v:
                            ms["agent"] = v
                        else:
                            ms.pop("agent", None)
                        self._mark_dirty()
                        self._render_boxes()

                    txt_agent.currentTextChanged.connect(_on_agent_changed)
                else:
                    txt_agent = QLineEdit(agent_val)
                    txt_agent.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt_agent.setPlaceholderText("Algorithm...")
                    txt_agent.textChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "agent", v)
                    )

                agent_keys_present = set()
                if txt_agent is not None:
                    agent_rows = []
                    btn_add_ap = None
                    if agent_val:
                        cand_agent = get_agent_candidate_hyperparameters(agent_val, tree, set(m_spec.keys()))
                        btn_add_ap = _make_param_adder_button(
                            f"Add {agent_val or 'agent'} parameter",
                            cand_agent,
                            lambda p, mn=m_name: self._add_method_param(mn, p),
                            parent_widget=sub_card
                        )

                        agent_defaults = dict(tree.agent_hyperparameters(agent_val)) if (tree and agent_val) else {}
                        agent_param_keys = set(agent_defaults.keys()) - set(TRAINING_HYPERPARAMETERS) - {"algorithm", "name", "type"}

                        # 1. Overridden keys in m_spec
                        _m_raw = m_spec.get("model", "")
                        _m_name = next(iter(_m_raw)) if isinstance(_m_raw, dict) and _m_raw else str(_m_raw or "")
                        known_model_params = set(tree.model_parameter_keys(_m_name)) if (tree and _m_name in tree.models) else set()
                        for ak, av in list(m_spec.items()):
                            if ak in ("agent", "model", "base") or ak in known_model_params:
                                continue
                            if ak in agent_param_keys or (agent_val and ak.startswith(f"{agent_val}_")):
                                agent_keys_present.add(ak)
                                row_ap = QHBoxLayout()
                                if isinstance(av, bool):
                                    cb_ap = ToggleSlider(checked=av)
                                    cb_ap.toggled.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                    row_ap.addWidget(cb_ap)
                                elif isinstance(av, (int, float)):
                                    spin_ap = number_field(av)
                                    spin_ap.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                    spin_ap.valueChanged.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                    row_ap.addWidget(spin_ap)
                                else:
                                    param_schema = tree.agent_parameter_schema(agent_val, (ak,)) if tree else {}
                                    row_ap.addWidget(self._schema_widget(
                                        av, param_schema,
                                        lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v),
                                    ))
                                btn_del_ap = _make_subtle_tool_button("close", f"Remove '{ak}'", size=20, icon_size=12)
                                btn_del_ap.clicked.connect(lambda _, mn=m_name, pk=ak: self._remove_method_param(mn, pk))
                                row_ap.addWidget(btn_del_ap)
                                agent_rows.append((str(ak), row_ap))

                        # 2. Defaults if show_ag_defs enabled
                        if show_ag_defs:
                            for ak in sorted(agent_param_keys):
                                if ak not in agent_keys_present:
                                    def_val = agent_defaults.get(ak)
                                    if def_val is None:
                                        continue
                                    row_ap = QHBoxLayout()
                                    if isinstance(def_val, bool):
                                        cb_ap = ToggleSlider(checked=def_val)
                                        cb_ap.toggled.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                        row_ap.addWidget(cb_ap)
                                    elif isinstance(def_val, (int, float)):
                                        spin_ap = number_field(def_val)
                                        spin_ap.setMaximumWidth(NUMBER_FIELD_WIDTH)
                                        spin_ap.valueChanged.connect(lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v))
                                        row_ap.addWidget(spin_ap)
                                    else:
                                        param_schema = tree.agent_parameter_schema(agent_val, (ak,)) if tree else {}
                                        row_ap.addWidget(self._schema_widget(
                                            def_val, param_schema,
                                            lambda v, mn=m_name, pk=ak: self._on_method_param_edited(mn, pk, v),
                                        ))
                                    lbl_def = QLabel("(default)")
                                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                                    row_ap.addWidget(lbl_def)
                                    agent_rows.append((str(ak), row_ap))

                    row_ag = QHBoxLayout()
                    row_ag.addWidget(txt_agent)
                    if btn_add_ap:
                        row_ag.addWidget(btn_add_ap)
                    row_ag.addStretch()
                    sc_layout.addRow("Algorithm", row_ag)

                    if agent_rows:
                        agent_block = IndentedHierarchyBlock(level=1)
                        for r_lbl, r_box in agent_rows:
                            r_box.addStretch()
                            agent_block.addRow(r_lbl, r_box)
                        sc_layout.addRow(agent_block)

                # --- MODEL SECTION ---
                model_raw = m_spec.get("model", "")
                if isinstance(model_raw, dict):
                    model_name = next(iter(model_raw))
                    model_inner = model_raw.get(model_name) if isinstance(model_raw.get(model_name), dict) else {}
                else:
                    model_name = str(model_raw or "")
                    model_inner = {}

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
                                tree and (
                                    model_name in tree.models
                                    or mo_stem in tree.models
                                    or (tree.config_root / "model" / f"{model_name}.yaml").is_file()
                                    or (tree.config_root / "model" / "basics" / f"{mo_stem}.yaml").is_file()
                                )
                            )
                            if is_missing:
                                txt_model.setToolTip(f"Missing component: Model '{model_name}' is not installed in in/config/model/. Install it via Component Hub.")
                            else:
                                txt_model.setToolTip("Architectures defined in in/config/model/")
                        else:
                            txt_model.setToolTip("Architectures defined in in/config/model/")
                        txt_model.setCurrentText(model_name)

                    def _on_model_changed(v, mn=m_name, cur_mn=model_name):
                        if self._block_updates or v == cur_mn:
                            return
                        methods = self.raw_data.setdefault("methods", {})
                        ms = methods.setdefault(mn, {})
                        if not v:
                            ms.pop("model", None)
                        else:
                            ms["model"] = v
                            old_params = tree.model_parameter_keys(cur_mn) if (tree and cur_mn in tree.models) else []
                            for key in old_params:
                                ms.pop(key, None)
                        self._mark_dirty()
                        self._render_boxes()

                    txt_model.currentTextChanged.connect(_on_model_changed)
                else:
                    txt_model = QLineEdit(model_name)
                    txt_model.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt_model.setPlaceholderText("Model...")
                    txt_model.textChanged.connect(
                        lambda v, mn=m_name: self._on_method_param_edited(mn, "model", v)
                    )

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
                    candidate_keys = get_model_candidate_hyperparameters(
                        model_name, tree, model_keys_present
                    )
                    if candidate_keys:
                        btn_add_mp = _make_param_adder_button(
                            f"Add {model_name} parameter",
                            candidate_keys,
                            lambda key, mn=m_name, model=model_name: self._add_method_model_parameter(
                                mn, model, (key,)
                            ),
                            parent_widget=sub_card,
                        )

                    row_mo = QHBoxLayout()
                    row_mo.addWidget(txt_model)
                    if btn_add_mp:
                        row_mo.addWidget(btn_add_mp)
                    slot_widget = self._model_row_slot(m_name, model_name)
                    if slot_widget is not None:
                        row_mo.addWidget(slot_widget)
                    row_mo.addStretch()
                    sc_layout.addRow("Model", row_mo)

                    if model_rows:
                        model_block = IndentedHierarchyBlock(level=1)
                        for r_lbl, r_box in model_rows:
                            if isinstance(r_box, QWidget):
                                model_block.addRow(r_box)
                            else:
                                r_box.addStretch()
                                model_block.addRow(r_lbl, r_box)
                        sc_layout.addRow(model_block)
                else:
                    sc_layout.addRow("Model", txt_model)

                # --- TRAINING PARAMETERS SECTION (UNINDENTED, NO HEADER) ---
                train_rows = []
                u_params = self.raw_data.get("methods", {}).get("params", {})
                base_def_name, base_def_spec = self._get_group_default_method()
                base_def_spec = base_def_spec or {}

                if "lr" in m_spec:
                    row_lr = QHBoxLayout()
                    spin_lr = CompactDoubleSpinBox()
                    spin_lr.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_lr.setDecimals(6)
                    spin_lr.setRange(0.000001, 1.0)
                    spin_lr.setSingleStep(0.0001)
                    spin_lr.setValue(float(m_spec["lr"]))
                    spin_lr.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "lr", v))
                    row_lr.addWidget(spin_lr)
                    btn_del = _make_subtle_tool_button("close", "Remove lr parameter", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "lr"))
                    row_lr.addWidget(btn_del)
                    train_rows.append(("Learning Rate (lr)", row_lr))
                elif show_all_defs:
                    def_lr = float(u_params.get("lr") or base_def_spec.get("lr") or 0.0003)
                    row_lr = QHBoxLayout()
                    spin_lr = CompactDoubleSpinBox()
                    spin_lr.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_lr.setDecimals(6)
                    spin_lr.setRange(0.000001, 1.0)
                    spin_lr.setSingleStep(0.0001)
                    spin_lr.setValue(def_lr)
                    spin_lr.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "lr", v))
                    row_lr.addWidget(spin_lr)
                    lbl_def = QLabel("(default)")
                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                    row_lr.addWidget(lbl_def)
                    train_rows.append(("Learning Rate (lr)", row_lr))

                if "batch_size" in m_spec:
                    row_bs = QHBoxLayout()
                    spin_bs = SpinBox()
                    spin_bs.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_bs.setRange(1, 131072)
                    spin_bs.setValue(int(m_spec["batch_size"]))
                    spin_bs.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "batch_size", v))
                    row_bs.addWidget(spin_bs)
                    btn_del = _make_subtle_tool_button("close", "Remove batch_size parameter", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "batch_size"))
                    row_bs.addWidget(btn_del)
                    train_rows.append(("Batch Size", row_bs))
                elif show_all_defs:
                    def_bs = int(u_params.get("batch_size") or base_def_spec.get("batch_size") or 64)
                    row_bs = QHBoxLayout()
                    spin_bs = SpinBox()
                    spin_bs.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_bs.setRange(1, 131072)
                    spin_bs.setValue(def_bs)
                    spin_bs.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "batch_size", v))
                    row_bs.addWidget(spin_bs)
                    lbl_def = QLabel("(default)")
                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                    row_bs.addWidget(lbl_def)
                    train_rows.append(("Batch Size", row_bs))

                if "gamma" in m_spec:
                    row_g = QHBoxLayout()
                    spin_g = DoubleSpinBox()
                    spin_g.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_g.setDecimals(4)
                    spin_g.setRange(0.0, 1.0)
                    spin_g.setValue(float(m_spec["gamma"]))
                    spin_g.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "gamma", v))
                    row_g.addWidget(spin_g)
                    btn_del = _make_subtle_tool_button("close", "Remove gamma parameter", size=20, icon_size=12)
                    btn_del.clicked.connect(lambda _, mn=m_name: self._remove_method_param(mn, "gamma"))
                    row_g.addWidget(btn_del)
                    train_rows.append(("Discount Factor (γ)", row_g))
                elif show_all_defs:
                    def_gamma = float(u_params.get("gamma") or base_def_spec.get("gamma") or 0.99)
                    row_g = QHBoxLayout()
                    spin_g = DoubleSpinBox()
                    spin_g.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin_g.setDecimals(4)
                    spin_g.setRange(0.0, 1.0)
                    spin_g.setValue(def_gamma)
                    spin_g.valueChanged.connect(lambda v, mn=m_name: self._on_method_param_edited(mn, "gamma", v))
                    row_g.addWidget(spin_g)
                    lbl_def = QLabel("(default)")
                    lbl_def.setStyleSheet("color: #928374; font-size: 11px; font-style: italic;")
                    row_g.addWidget(lbl_def)
                    train_rows.append(("Discount Factor (γ)", row_g))

                consumed_keys = {"agent", "model", "base", "lr", "batch_size", "gamma"}
                consumed_keys.update(agent_keys_present)
                consumed_keys.update(model_keys_present)

                for extra_k, extra_v in list(m_spec.items()):
                    if extra_k in consumed_keys:
                        continue
                    if isinstance(extra_v, (int, float, str, bool)):
                        row_extra = QHBoxLayout()
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
                        train_rows.append((str(extra_k), row_extra))

                if train_rows:
                    for t_lbl, t_row in train_rows:
                        t_row.addStretch()
                        sc_layout.addRow(t_lbl, t_row)

                card_vbox.addLayout(sc_layout)
                card_vbox.addStretch()

                method_container.add_method_card(m_name, sub_card)

            if pending_new:
                pending_card = QFrame()
                pending_card.setObjectName("card")
                pc_vbox = QVBoxLayout(pending_card)
                pc_vbox.setContentsMargins(10, 10, 10, 10)
                pc_vbox.setSpacing(8)

                pc_layout = QFormLayout()
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
                            lambda v: self._pending_new_method_spec.__setitem__("agent", v.strip()) if v.strip() else self._pending_new_method_spec.pop("agent", None)
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
                            lambda v: self._pending_new_method_spec.__setitem__("agent", v.strip()) if v.strip() else self._pending_new_method_spec.pop("agent", None)
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
                        lambda v: self._pending_new_method_spec.__setitem__("model", v.strip()) if v.strip() else self._pending_new_method_spec.pop("model", None)
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
                        lambda v: self._pending_new_method_spec.__setitem__("model", v.strip()) if v.strip() else self._pending_new_method_spec.pop("model", None)
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
                    methods = self.raw_data.setdefault("methods", {})
                    methods[name] = dict(getattr(self, "_pending_new_method_spec", {}))
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

        # --- MODULAR BOX: CHECKPOINT SETTINGS ---
        has_checkpoint = "checkpoint" in data and isinstance(data["checkpoint"], dict)
        show_checkpoint = has_checkpoint or (self.view_mode == "resolved" and "checkpoint" in resolved and isinstance(resolved["checkpoint"], dict))

        if show_checkpoint:
            box_ckpt = ConfigBox("Checkpoint Settings", "Model checkpointing and validation metric monitoring")
            ckpt_header = QHBoxLayout()
            ckpt_header.setSpacing(8)
            lbl_ckpt_title = label("Checkpoint Settings", "heading")  # ConfigBox drops its title argument
            lbl_ckpt_title.setStyleSheet("font-size: 23px; font-weight: 600;")
            ckpt_header.addWidget(lbl_ckpt_title)
            ckpt_header.addStretch()
            if has_checkpoint:
                btn_rm_ckpt = _make_subtle_tool_button("trash", "Remove checkpoint overrides", size=24, icon_size=14)
                btn_rm_ckpt.clicked.connect(lambda: self.remove_block("checkpoint"))
                ckpt_header.addWidget(btn_rm_ckpt)
            box_ckpt.add_layout(ckpt_header)

            grid_c = QGridLayout()
            grid_c.setVerticalSpacing(10)
            grid_c.setHorizontalSpacing(16)
            ckpt_source = data.get("checkpoint", {}) if has_checkpoint else resolved.get("checkpoint", {})
            if not isinstance(ckpt_source, dict):
                ckpt_source = {}

            grid_c.addWidget(label("Monitor Metric"), 0, 0)
            edit_mon = QLineEdit()
            edit_mon.setMaximumWidth(NUMBER_FIELD_WIDTH)
            edit_mon.setPlaceholderText("e.g. val/loss")
            edit_mon.setText(str(ckpt_source.get("monitor") or ""))
            edit_mon.textChanged.connect(lambda v: self._on_nested_edited("checkpoint", "monitor", v.strip() or None))
            grid_c.addWidget(edit_mon, 0, 1)

            grid_c.addWidget(label("Mode"), 0, 2)
            combo_mode = ComboBox()
            combo_mode.setMaximumWidth(NUMBER_FIELD_WIDTH)
            combo_mode.addItems(["min", "max"])
            combo_mode.setCurrentText(str(ckpt_source.get("mode") or "min"))
            combo_mode.currentTextChanged.connect(lambda v: self._on_nested_edited("checkpoint", "mode", v))
            grid_c.addWidget(combo_mode, 0, 3)

            grid_c.addWidget(label("Save Top K"), 1, 0)
            spin_topk = SpinBox()
            spin_topk.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_topk.setRange(0, 50)
            spin_topk.setValue(int(ckpt_source.get("save_top_k") if ckpt_source.get("save_top_k") is not None else 1))
            spin_topk.valueChanged.connect(lambda v: self._on_nested_edited("checkpoint", "save_top_k", v))
            grid_c.addWidget(spin_topk, 1, 1)

            grid_c.addWidget(label("Every N Epochs"), 1, 2)
            spin_every = SpinBox()
            spin_every.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin_every.setRange(1, 1000)
            spin_every.setValue(int(ckpt_source.get("every_n_epochs") or 1))
            spin_every.valueChanged.connect(lambda v: self._on_nested_edited("checkpoint", "every_n_epochs", v))
            grid_c.addWidget(spin_every, 1, 3)

            box_ckpt.add_layout(grid_c)
            self.boxes_layout.addWidget(box_ckpt)

        self._render_component_overrides(data, resolved)
        self._render_remaining_sections()

        # --- ADD CONFIGURATION BLOCK DROPDOWN BUTTON ---
        self._render_metrics_box(data, resolved)

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
        for kind, title in (("data_module", "Data Loaders"), ("evaluator", "Evaluators"), ("plots", "Figures")):
            action = add_menu.addAction(title)
            action.triggered.connect(lambda _, k=kind: self.add_component_override(k, resolved))
            action.setEnabled(not self._has_component_override(kind, data))
        act_ckpt = add_menu.addAction("Checkpoint Settings")
        act_ckpt.triggered.connect(lambda: self.add_block("checkpoint"))
        act_ckpt.setEnabled("checkpoint" not in data)

        act_env = add_menu.addAction("Environment Overrides")
        act_env.triggered.connect(lambda: self.add_block("env"))
        act_env.setEnabled(not isinstance(data.get("env"), dict))

        act_ds = add_menu.addAction("Dataset Configuration")
        act_ds.triggered.connect(lambda: self.add_block("dataset"))
        act_ds.setEnabled("dataset_path" not in data)

        has_sweeper = (
            (isinstance(data.get("hydra"), dict) and "sweeper" in data["hydra"])
            or "sweeper" in data
        )
        act_sw = add_menu.addAction("Hyperparameter Sweeper (Optuna)")
        act_sw.triggered.connect(lambda: self.add_block("sweeper"))
        act_sw.setEnabled(not has_sweeper)
        act_metrics = add_menu.addAction("Recorded Metrics")
        act_metrics.setToolTip("Record only some metrics (and skip the work behind the others) to train faster")
        act_metrics.triggered.connect(lambda: self.add_block("metrics"))
        act_metrics.setEnabled(_recorded_metrics(data) is None)
        self.btn_add_block.setMenu(add_menu)
        btn_add_block_row.addWidget(self.btn_add_block)

        btn_add_block_row.addStretch()
        self.boxes_layout.addLayout(btn_add_block_row)

    @staticmethod
    def _has_component_override(kind: str, data: dict) -> bool:
        return ("plots" in data or "plot_overrides" in data) if kind == "plots" else kind in data

    def add_component_override(self, kind: str, resolved: dict | None = None) -> None:
        resolved = resolved or self._get_resolved_defaults()
        if kind in ("data_module", "evaluator"):
            inherited = self._component_name(self._inherited_component(kind, resolved))
            choices = self._component_choices(kind)
            name = inherited or (sorted(choices)[0] if choices else "")
            if name:
                self.raw_data[kind] = {"name": name, "params": self._component_default_params(kind, name)}
        elif kind == "plots":
            inherited = resolved.get("plots")
            if isinstance(inherited, (list, dict)):
                plots = copy.deepcopy(inherited)
            else:
                plots = self._inherited_plot_names(resolved)
            self.raw_data["plots"] = plots
            selected = ([str(item) for item in plots] if isinstance(plots, list) else
                        [str(key) for key, value in plots.items() if value is not None and key != "style"] if isinstance(plots, dict) else [])
            self.raw_data["plot_overrides"] = {
                name: self._component_default_params("plots", name)
                for name in selected if name in self._component_choices("plots")
            }
            if not self.raw_data["plot_overrides"]:
                self.raw_data.pop("plot_overrides", None)
        else:
            return
        self._mark_dirty()
        self._render_boxes()

    def _component_choices(self, kind: str) -> dict[str, dict]:
        tree = config_tree()
        root = tree.config_root if tree else PROJECT_ROOT / "in" / "config"
        directory, selector = {
            "data_module": ("data", "data_module"),
            "evaluator": ("eval", "evaluator"),
            "plots": ("plot", "plotter"),
        }[kind]
        profiles = _component_profiles(root, directory, selector)
        try:
            if kind == "data_module":
                from src.app.core.datamodule_registry import get_datamodule_class, list_registered_datamodules
                names = list_registered_datamodules()
                resolve = get_datamodule_class
            elif kind == "evaluator":
                from src.app.core.eval_registry import get_evaluator_class, list_registered_evaluators
                names = list_registered_evaluators()
                resolve = get_evaluator_class
            else:
                from src.app.core.plotter_registry import get_plotter_class, list_registered_plotters
                names = list_registered_plotters()
                resolve = get_plotter_class
            for name in names:
                try:
                    component_class = resolve(str(name))
                    source = Path(inspect.getfile(component_class)) if component_class else None
                    if source is not None and not source.exists():
                        continue
                except (OSError, TypeError):
                    pass
                profiles.setdefault(str(name), {})
            paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
            if paradigm:
                allowed = {}
                for name, values in profiles.items():
                    supported = values.get("compatible_paradigms") or values.get("paradigms")
                    if not supported:
                        try:
                            cls = resolve(name)
                        except Exception:
                            cls = None
                        declared = getattr(cls, "get_supported_paradigms", None)
                        if callable(declared):
                            try:
                                supported = declared()
                            except Exception:
                                supported = None
                    if not supported or paradigm in supported:
                        allowed[name] = values
                profiles = allowed
        except Exception:
            pass
        return profiles

    @staticmethod
    def _component_name(value: Any) -> str:
        if isinstance(value, Mapping):
            return str(value.get("name") or value.get("class") or value.get("type") or "")
        return str(value or "")

    def _component_default_params(self, kind: str, name: str) -> dict:
        profile = self._component_choices(kind).get(name, {})
        reserved = {
            "data_module": {"name", "class", "type", "data_module", "compatible_paradigms", "paradigms"},
            "evaluator": {"name", "class", "type", "evaluator", "compatible_paradigms", "paradigms"},
        }.get(kind, {"name", "class", "type", "plotter", "compatible_paradigms", "paradigms"})
        return {key: copy.deepcopy(value) for key, value in profile.items() if key not in reserved}

    def _inherited_component(self, kind: str, resolved: dict) -> Any:
        value = resolved.get(kind)
        if value is not None:
            return value
        tree = config_tree()
        env_name = self._get_explicit_env() or self._inherited_env()
        env = tree.environments.get(env_name) if tree and env_name else None
        if env and env.raw.get(kind) is not None:
            return env.raw[kind]
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        rules = tree.paradigms.get(paradigm) if tree and paradigm else None
        if rules:
            key = "default_data_module" if kind == "data_module" else "eval_protocol"
            return rules.raw.get(key)
        return None

    def _set_component_choice(self, kind: str, name: str) -> None:
        if not name:
            self.raw_data.pop(kind, None)
        else:
            old = self.raw_data.get(kind)
            if isinstance(old, dict) and self._component_name(old) == name:
                return
            self.raw_data[kind] = {"name": name, "params": self._component_default_params(kind, name)}
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _add_component_parameter(self, kind: str, key: str, default: Any) -> None:
        if not key:
            return
        value = self.raw_data.get(kind)
        if not isinstance(value, dict):
            value = {"name": self._component_name(value)}
            self.raw_data[kind] = value
        value.setdefault("params", {})[key] = copy.deepcopy(default)
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _inherited_plot_names(self, resolved: dict) -> list[str]:
        plots = resolved.get("plots")
        if isinstance(plots, list):
            return [str(item) for item in plots]
        if isinstance(plots, dict):
            choices = self._component_choices("plots")
            return [name for name, value in plots.items() if name in choices or isinstance(value, dict) and name != "style"]
        tree = config_tree()
        env_name = self._get_explicit_env() or self._inherited_env()
        env = tree.environments.get(env_name) if tree and env_name else None
        defaults = env.raw.get("default_plots") if env else None
        return [str(item) for item in defaults] if isinstance(defaults, list) else []

    def _set_plot_override(self, name: str, enabled: bool, resolved: dict | None = None) -> None:
        plots = self.raw_data.get("plots")
        inherited = (resolved or self._get_resolved_defaults()).get("plots")
        if isinstance(plots, list):
            plots = [str(item) for item in plots]
        elif plots is None and isinstance(inherited, list):
            plots = [str(item) for item in inherited]
        elif not isinstance(plots, dict):
            plots = {item: {} for item in self._inherited_plot_names(resolved or self._get_resolved_defaults())}
        if isinstance(plots, list):
            if enabled and name not in plots:
                plots.append(name)
            elif not enabled:
                plots = [item for item in plots if item != name]
        elif enabled:
            plots[name] = plots.get(name) or {}
        else:
            if isinstance(inherited, dict) and name in inherited:
                plots[name] = None  # tombstone: Hydra deep-merges group dictionaries
            else:
                plots.pop(name, None)
        if enabled:
            params = self.raw_data.setdefault("plot_overrides", {})
            params.setdefault(name, self._component_default_params("plots", name))
        self.raw_data["plots"] = plots
        if not enabled and isinstance(self.raw_data.get("plot_overrides"), dict):
            self.raw_data["plot_overrides"].pop(name, None)
            if not self.raw_data["plot_overrides"]:
                self.raw_data.pop("plot_overrides", None)
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _add_plot_parameter(self, name: str, key: str, default: Any) -> None:
        plot = self.raw_data.setdefault("plot_overrides", {}).setdefault(name, {})
        plot[key] = copy.deepcopy(default)
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _render_component_overrides(self, data: dict, resolved: dict) -> None:
        """Expose component selection and per-component YAML parameters in recipes."""
        for kind, title in (("data_module", "Data Loaders"), ("evaluator", "Evaluators")):
            if kind not in data:
                continue
            explicit = data.get(kind)
            inherited = self._inherited_component(kind, resolved)
            name = self._component_name(explicit)
            inherited_name = self._component_name(inherited)
            choices = self._component_choices(kind)
            box = ConfigBox()
            heading = QHBoxLayout()
            heading.addWidget(label(title, "cardTitle"))
            heading.addStretch()
            trash = _make_subtle_tool_button("trash", f"Remove {title.lower()} override", size=24, icon_size=14)
            trash.clicked.connect(lambda _, k=kind: self.remove_block(k))
            heading.addWidget(trash)
            box.add_layout(heading)
            row = QHBoxLayout()
            row.addWidget(label("Component"))
            combo = ComboBox()
            combo.setEditable(True)
            inherited_label = inherited_name or "None"
            combo.addItem(inherited_label, "")
            for choice in sorted((set(choices) | ({name} if name else set())) - ({inherited_name} if inherited_name else set())):
                combo.addItem(choice, choice)
            if name:
                explicit_index = combo.findData(name)
                if explicit_index < 0 and name == inherited_name:
                    combo.insertItem(1, name, name)
                    explicit_index = 1
                combo.setCurrentIndex(explicit_index if explicit_index >= 0 else 0)
            else:
                combo.setCurrentIndex(0)
            combo.currentTextChanged.connect(
                lambda selected, k=kind, c=combo: self._set_component_choice(
                    k, str(c.currentData() or selected) if c.currentIndex() != 0 else ""
                )
            )
            row.addWidget(combo)
            if explicit:
                reset = _make_subtle_tool_button("close", f"Use inherited {title.lower()} component", size=20, icon_size=12)
                reset.clicked.connect(lambda _, k=kind: self._set_component_choice(k, ""))
                row.addWidget(reset)
            elif inherited_name:
                customize = QPushButton("Use with defaults")
                customize.setToolTip("Create an experiment override using this component and its profile defaults")
                customize.clicked.connect(lambda _, k=kind, n=inherited_name: self._set_component_choice(k, n))
                row.addWidget(customize)
            row.addStretch()
            box.add_layout(row)
            self.field_widgets[kind] = combo

            if name:
                params = explicit.get("params", {}) if isinstance(explicit, dict) else {}
                if not isinstance(params, dict):
                    params = {}
                defaults = self._component_default_params(kind, name)
                candidates = {key: val for key, val in defaults.items() if key not in params}
                param_row = QHBoxLayout()
                param_row.addWidget(label("Parameters", "muted"))
                add = _make_param_adder_button(
                    f"Add {title.lower()} parameter", list(candidates),
                    lambda key, k=kind, options=candidates: self._add_component_parameter(k, key, options.get(key)),
                    parent_widget=self,
                )
                param_row.addWidget(add)
                param_row.addStretch()
                box.add_layout(param_row)
                form = QFormLayout()
                self._render_generic_dict_contents(form, params, (kind, "params"))
                if params:
                    box.add_layout(form)
                if isinstance(explicit, dict):
                    extra = {key: value for key, value in explicit.items()
                             if key not in ("name", "class", "type", "params")}
                    if extra:
                        extra_form = QFormLayout()
                        self._render_generic_dict_contents(extra_form, extra, (kind,))
                        box.add_layout(extra_form)
            self.boxes_layout.addWidget(box)

        if not ("plots" in data or "plot_overrides" in data):
            return
        plot_choices = self._component_choices("plots")
        explicit_plots = data.get("plots")
        per_plot_overrides = data.get("plot_overrides") if isinstance(data.get("plot_overrides"), dict) else {}
        if isinstance(explicit_plots, list):
            selected = [str(item) for item in explicit_plots]
        else:
            selected = self._inherited_plot_names(resolved)
            if isinstance(explicit_plots, dict):
                for key, value in explicit_plots.items():
                    if value is None:
                        selected = [item for item in selected if item != key]
                    elif key in plot_choices or isinstance(value, dict) and key != "style":
                        if key not in selected:
                            selected.append(key)
        box = ConfigBox()
        heading = QHBoxLayout()
        heading.addWidget(label("Figures", "cardTitle"))
        available = [name for name in plot_choices if name not in selected]
        heading.addWidget(_make_param_adder_button(
            "Add compatible figure", available,
            lambda name: self._set_plot_override(name, True, resolved), parent_widget=self,
            allow_custom=False,
        ))
        heading.addStretch()
        trash = _make_subtle_tool_button("trash", "Remove figures override", size=24, icon_size=14)
        trash.clicked.connect(lambda: self.remove_block("plots"))
        heading.addWidget(trash)
        box.add_layout(heading)
        if explicit_plots is not None:
            box.add_widget(label("Selected Figures", "muted"))
        if isinstance(explicit_plots, list):
            list_form = QFormLayout()
            list_form.addRow("items", self._list_field(
                explicit_plots, lambda value: self._on_field_edited("plots", value)
            ))
            box.add_layout(list_form)
        if explicit_plots is not None:
            reset = _make_subtle_tool_button("close", "Inherit figures from group or environment", size=20, icon_size=12)
            reset.clicked.connect(lambda: self._remove_generic_field("plots"))
            heading.insertWidget(heading.count() - 2, reset)
        for name in selected:
            row = QHBoxLayout()
            row.addWidget(label(name))
            profile = plot_choices.get(name, {})
            inline_values = explicit_plots.get(name, {}) if isinstance(explicit_plots, dict) else {}
            values = per_plot_overrides.get(name, inline_values)
            if not isinstance(values, dict):
                values = {}
            candidates = {key: val for key, val in profile.items() if key not in values}
            if name not in per_plot_overrides and not values:
                customize = QPushButton("Use with defaults")
                customize.clicked.connect(lambda _, n=name: self._add_plot_parameter_defaults(n))
                row.addWidget(customize)
            row.addWidget(_make_param_adder_button(
                f"Override {name} parameter", list(candidates),
                lambda key, n=name, options=candidates: self._add_plot_parameter(n, key, options.get(key)),
                parent_widget=self,
            ))
            row.addWidget(_make_subtle_tool_button("close", f"Remove {name} figure", size=20, icon_size=12))
            row.itemAt(row.count() - 1).widget().clicked.connect(lambda _, n=name: self._set_plot_override(n, False, resolved))
            row.addStretch()
            box.add_layout(row)
            if values:
                form = QFormLayout()
                path = ("plot_overrides", name) if name in per_plot_overrides else ("plots", name)
                self._render_generic_dict_contents(form, values, path)
                box.add_layout(form)
        if isinstance(explicit_plots, dict):
            general = {key: value for key, value in explicit_plots.items() if key not in selected and value is not None}
            if general:
                box.add_widget(label("Figure settings", "muted"))
                form = QFormLayout()
                self._render_generic_dict_contents(form, general, ("plots",))
                box.add_layout(form)
        self.boxes_layout.addWidget(box)

    def _add_plot_parameter_defaults(self, name: str) -> None:
        """Materialize all first-level defaults for a selected inherited figure."""
        params = self._component_default_params("plots", name)
        self.raw_data.setdefault("plot_overrides", {})[name] = params
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _add_experiment_section_header(self, box: ConfigBox, data: dict, resolved: dict) -> None:
        """Render the Experiment/Group Defaults heading and its override picker."""
        section_header = QHBoxLayout()
        section_header.setSpacing(8)
        is_base = self._is_base_config()
        section_title = label("Group Defaults" if is_base else "Experiment", "heading")
        section_title.setStyleSheet("font-size: 23px; font-weight: 600;")
        section_header.addWidget(section_title)

        tree = config_tree()
        paradigm = data.get("paradigm") or self._inherited_paradigm()
        rules = tree.paradigms.get(paradigm) if tree and paradigm else None
        context_candidates = [
            ("trainer.accelerator", "Accelerator", None),
            ("trainer.max_epochs", "Max Epochs", None),
        ]
        if rules and rules.uses_static_dataset:
            context_candidates.append(("dataset_path", "Dataset", None))
        present_context = set()
        env_data = data.get("env")
        if isinstance(env_data, dict):
            present_context.update(f"env.{key}" for key in env_data)
        trainer_data = data.get("trainer")
        if isinstance(trainer_data, dict):
            present_context.update(f"trainer.{key}" for key in trainer_data)
        if "dataset_path" in data:
            present_context.add("dataset_path")
        if not is_base and self.view_mode != "resolved":
            available_context = [(key, title, default) for key, title, default in context_candidates if key not in present_context]
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
        desc_row = QHBoxLayout()
        desc_row.setSpacing(10)
        desc_label = label("Description")
        desc_label.setFixedWidth(145)
        desc_row.addWidget(desc_label)
        self.txt_description = QLineEdit(str(data.get("description") or ""))
        self.txt_description.textChanged.connect(
            lambda v: self._on_field_edited("description", v.strip()) if v.strip()
            else (self.raw_data.pop("description", None), self._mark_dirty())
        )
        desc_row.addWidget(self.txt_description)
        btn_del_desc = _make_subtle_tool_button("close", "Clear description", size=20, icon_size=12)
        btn_del_desc.clicked.connect(lambda: self.txt_description.setText(""))
        desc_row.addWidget(btn_del_desc)
        self.field_widgets["description"] = self.txt_description
        box.add_layout(desc_row)

    def _add_experiment_separator(self, box: ConfigBox) -> None:
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {theme_color('border')}; border: none;")
        box.add_widget(divider)

    def _has_experiment_lower_params(self, data: dict, resolved: dict) -> bool:
        """Return whether the lower parameter area has anything visible."""
        if self._is_base_config():
            return True  # group defaults always expose runtime defaults below the divider
        env_data = data.get("env") if isinstance(data.get("env"), dict) else {}
        trainer_data = data.get("trainer") if isinstance(data.get("trainer"), dict) else {}
        if any(key in env_data for key in ("num_envs", "num_steps")) or any(
            key in trainer_data for key in ("accelerator", "max_epochs")
        ):
            return True
        tree = config_tree()
        paradigm = data.get("paradigm") or self._inherited_paradigm()
        rules = tree.paradigms.get(paradigm) if tree and paradigm else None
        if rules and rules.uses_static_dataset and ("dataset_path" in data or self.view_mode == "resolved"):
            return True
        lower_keys = {
            "total_timesteps", "epochs", "seed", "eval_episodes", "eval_freq_steps",
            "eval_freq", "eval_every_n_epochs", "num_checkpoints", "checkpoint_every_steps",
            "tensorboard", "save_dataset", "recover", "no_plot",
        }
        if any(key in data for key in lower_keys):
            return True
        if self.view_mode == "resolved" and any(key in resolved for key in lower_keys):
            return True
        standard_keys = lower_keys | {
            "defaults", "paradigm", "experiment_id", "description", "methods", "env",
            "site", "trainer", "resources", "dataset", "dataset_path", "hydra", "plots",
            "tuning", "sweeper",
        }
        return any(key not in standard_keys and isinstance(value, (int, float, str, bool)) for key, value in data.items())

    def _add_experiment_context_fields(self, box: ConfigBox, data: dict, resolved: dict) -> None:
        """Render explicit experiment overrides, leaving inherited values hidden."""
        is_base = self._is_base_config()

        context_rows = []
        env_data = data.get("env") if isinstance(data.get("env"), dict) else {}
        trainer_data = data.get("trainer") if isinstance(data.get("trainer"), dict) else {}
        resolved_env = resolved.get("env") if isinstance(resolved.get("env"), dict) else {}
        resolved_trainer = resolved.get("trainer") if isinstance(resolved.get("trainer"), dict) else {}
        if "num_envs" in env_data or (is_base and "num_envs" in resolved_env):
            spin = SpinBox()
            spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin.setRange(1, 1024)
            spin.setValue(int(env_data.get("num_envs", resolved_env.get("num_envs", 4))))
            spin.valueChanged.connect(lambda value: self._on_nested_edited("env", "num_envs", value))
            context_rows.append(("Parallel Envs", "env.num_envs", spin))
        if "num_steps" in env_data or (is_base and "num_steps" in resolved_env):
            spin = SpinBox()
            spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin.setRange(1, 100000)
            spin.setValue(int(env_data.get("num_steps", resolved_env.get("num_steps", 128))))
            spin.valueChanged.connect(lambda value: self._on_nested_edited("env", "num_steps", value))
            context_rows.append(("Steps per Env", "env.num_steps", spin))
        if "accelerator" in trainer_data or is_base:
            combo = ComboBox()
            combo.setMaximumWidth(NUMBER_FIELD_WIDTH)
            combo.addItems(["cpu", "gpu", "mps", "auto"])
            combo.setCurrentText(str(trainer_data.get("accelerator", resolved_trainer.get("accelerator", "cpu"))))
            combo.currentTextChanged.connect(lambda value: self._on_nested_edited("trainer", "accelerator", value))
            context_rows.append(("Accelerator", "trainer.accelerator", combo))
        if "max_epochs" in trainer_data or is_base:
            spin = SpinBox()
            spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
            spin.setRange(1, 1000)
            spin.setValue(int(trainer_data.get("max_epochs", resolved_trainer.get("max_epochs", 25))))
            spin.valueChanged.connect(lambda value: self._on_nested_edited("trainer", "max_epochs", value))
            context_rows.append(("Max Epochs", "trainer.max_epochs", spin))

        if context_rows:
            runtime_grid = QGridLayout()
            runtime_grid.setVerticalSpacing(8)
            runtime_grid.setHorizontalSpacing(16)
            runtime_grid.setColumnMinimumWidth(0, 145)
            runtime_grid.setColumnMinimumWidth(2, 145)
            for index, (title, path, widget) in enumerate(context_rows):
                row, pair = divmod(index, 2)
                label_column = pair * 2
                field_label = label(title)
                field_label.setFixedWidth(145)
                runtime_grid.addWidget(field_label, row, label_column)
                widget.setFixedWidth(NUMBER_FIELD_WIDTH)
                field_row = QHBoxLayout()
                field_row.setSpacing(6)
                field_row.addWidget(widget)
                if not is_base:
                    btn_remove = _make_subtle_tool_button("close", f"Remove {title} override", size=20, icon_size=12)
                    btn_remove.clicked.connect(lambda _, p=path: self._remove_experiment_context_param(p))
                    field_row.addWidget(btn_remove)
                field_row.addStretch()
                runtime_grid.addLayout(field_row, row, label_column + 1)
            runtime_grid.setColumnStretch(1, 1)
            runtime_grid.setColumnStretch(3, 1)
            box.add_layout(runtime_grid)

    def _add_experiment_context_param(self, path: str) -> None:
        """Add a context override, initialized from the group's resolved value."""
        resolved = self._get_resolved_defaults()
        parts = path.split(".")
        default_value = resolved
        for part in parts:
            default_value = default_value.get(part) if isinstance(default_value, dict) else None
        if path == "dataset_path":
            self._set_experiment_dataset_override()
            return
        if default_value is None:
            default_value = {
                "trainer.accelerator": "cpu",
                "trainer.max_epochs": 25,
            }.get(path)
        if path.startswith("env."):
            self.raw_data.setdefault("env", {})[parts[-1]] = default_value
        elif path.startswith("trainer."):
            self.raw_data.setdefault("trainer", {})[parts[-1]] = default_value
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _set_experiment_dataset_override(self) -> None:
        """Add a static dataset override without rebuilding under the open menu."""
        tree = config_tree()
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        rules = tree.paradigms.get(paradigm) if tree else None
        if not rules or not rules.uses_static_dataset:
            return
        env_name = self.env_selection() or self._get_explicit_env() or self._inherited_env() or self._group_name() or "dataset"
        self.raw_data["dataset_path"] = self._get_resolved_defaults().get("dataset_path") or f"in/datasets/{env_name}"
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

    def _experiment_budget_candidates(self, data: dict, resolved: dict, rules) -> list[tuple]:
        """Return budget and runtime options available from the Experiment header."""
        candidates = []
        if rules and rules.budget_unit == "timesteps":
            possible = [
                ("total_timesteps", "Total Timesteps", int(resolved.get("total_timesteps") or 10000)),
                ("seed", "Random Seed", int(resolved.get("seed") if resolved.get("seed") is not None else 42)),
            ]
            if rules.field_enabled("eval_episodes"):
                possible.append(("eval_episodes", "Eval Episodes", int(resolved.get("eval_episodes") or 100)))
            possible.extend([
                ("eval_freq_steps", "Eval Frequency (Steps)", int(resolved.get("eval_freq_steps") or 10000)),
                ("num_checkpoints", "Checkpoints", DEFAULT_CHECKPOINTS),
            ])
        else:
            possible = [
                ("epochs", "Epochs", int(resolved.get("epochs") or 10)),
                ("seed", "Random Seed", int(resolved.get("seed") if resolved.get("seed") is not None else 42)),
                ("eval_every_n_epochs", "Eval Frequency (Epochs)", int(resolved.get("eval_every_n_epochs") or 1)),
            ]
        for key, title, default in possible:
            if key not in data and not (key == "eval_freq_steps" and "eval_freq" in data):
                if key == "num_checkpoints" and any(k in data for k in CHECKPOINT_KEYS):
                    continue
                candidates.append((key, title, default))
        for key, title in (
            ("tensorboard", "Log to TensorBoard"),
            ("save_dataset", "Save Dataset"),
            ("recover", "Recover Checkpoint"),
            ("no_plot", "No Plots"),
        ):
            if key not in data:
                candidates.append((key, title, True))
        candidates.append(("custom", "+ Custom Parameter...", None))
        return candidates

    def _add_selected_experiment_param(self, item: tuple) -> None:
        key, _title, default = item
        if key == "custom":
            self._prompt_add_custom_experiment_param()
        elif key == "dataset_path":
            self._set_experiment_dataset_override()
        elif "." in key:
            self._add_experiment_context_param(key)
        else:
            self._add_experiment_param_from_menu(key, default)

    def _remove_experiment_context_param(self, path: str) -> None:
        parts = path.split(".")
        section, key = parts
        values = self.raw_data.get(section)
        if isinstance(values, dict):
            values.pop(key, None)
            if not values:
                self.raw_data.pop(section, None)
        self._mark_dirty()
        QTimer.singleShot(0, self._render_boxes)

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

        ds_form = QFormLayout()
        ds_form.setVerticalSpacing(10)
        source_row = QHBoxLayout()
        source_row.setSpacing(6)
        self.combo_dataset_path = ComboBox()
        self.combo_dataset_path.setMaximumWidth(320)
        self.combo_dataset_path.setEditable(True)
        for dataset_path in discover_datasets():
            self.combo_dataset_path.addItem(dataset_path)
        current_path = str(data.get("dataset_path") or resolved.get("dataset_path") or "")
        self.combo_dataset_path.setEditText(current_path)

        status = QLabel("")

        def _update_source_status(path_text):
            path = Path(path_text.strip())
            if path_text.strip() and path.exists():
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
        ds_form.addRow("Dataset / Static Environment", source_row)

        box.add_layout(ds_form)

    def _scalar_widget(self, value, write):
        """A widget matching the YAML type, so an int survives a round trip.

        The bool test comes first: bool is a subclass of int, and a checkbox
        rendered as a spin box would write 0/1 back over true/false.
        """
        if isinstance(value, bool):
            widget = ToggleSlider(checked=value)
            widget.toggled.connect(write)
        elif isinstance(value, (int, float)):
            widget = number_field(value)
            widget.valueChanged.connect(write)
        else:
            widget = QLineEdit("" if value is None else str(value))
            widget.textChanged.connect(write)
        return widget

    def _schema_widget(self, value, schema, write):
        """Create an editor from a parameter's declared JSON Schema domain."""
        schema = schema if isinstance(schema, dict) else {}
        choices = schema.get("enum")
        if schema.get("x-theta-domain") == "model":
            tree = config_tree()
            choices = tree.model_names if tree else []
        if isinstance(choices, list) and choices:
            combo = ComboBox()
            combo.setMaximumWidth(TEXT_FIELD_WIDTH)
            combo.addItem("", None)
            for choice in choices:
                combo.addItem(str(choice), choice)
            index = next((i for i in range(combo.count()) if combo.itemData(i) == value), -1)
            if index < 0 and value is not None:
                combo.insertItem(1, str(value), value)
                index = 1
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.currentIndexChanged.connect(lambda i: write(combo.itemData(i)))
            return combo

        schema_type = schema.get("type")
        if schema_type == "string":
            widget = QLineEdit("" if value is None else str(value))
            widget.setMaximumWidth(TEXT_FIELD_WIDTH)
            widget.textChanged.connect(write)
            return widget
        if schema_type == "boolean" or isinstance(value, bool):
            widget = ToggleSlider(checked=bool(value))
            widget.toggled.connect(write)
            return widget
        if schema_type == "integer":
            widget = SpinBox()
            widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
            widget.setRange(int(schema.get("minimum", -2147483648)), int(schema.get("maximum", 2147483647)))
            widget.setValue(int(value if value is not None else schema.get("default", 0)))
            widget.valueChanged.connect(write)
            return widget
        if schema_type == "number" or isinstance(value, float):
            widget = DoubleSpinBox()
            widget.setMaximumWidth(NUMBER_FIELD_WIDTH)
            widget.setRange(float(schema.get("minimum", -1e12)), float(schema.get("maximum", 1e12)))
            widget.setValue(float(value if value is not None else schema.get("default", 0.0)))
            widget.valueChanged.connect(write)
            return widget
        if schema_type == "array" or isinstance(value, list):
            items = value if isinstance(value, list) else schema.get("default", [])
            widget = QLineEdit(str(list(items)) if isinstance(items, list) else "[]")
            widget.setMaximumWidth(TEXT_FIELD_WIDTH + 80)
            widget.setToolTip("YAML list, for example [64, 64]")
            widget.textChanged.connect(lambda text: write(_parse_yaml_value(text)))
            return widget

        return self._scalar_widget(value if value is not None else schema.get("default", ""), write)

    def _component_parameter_schema(self, path: tuple[str, ...]) -> dict[str, Any]:
        """Look up the active component's declared schema node for a YAML path."""
        tree = config_tree()
        schema = tree.parameter_schema_for_config(str(self.current_rel_path or "")) if tree else {}
        for key in path:
            schema = schema.get("properties", {}).get(key, {}) if isinstance(schema, dict) else {}
        return schema if isinstance(schema, dict) else {}

    @staticmethod
    def _schema_has_choices(schema: dict[str, Any]) -> bool:
        return bool(schema.get("enum")) or schema.get("x-theta-domain") == "model"

    def _list_field(self, items, write):
        """Comma-separated editor for a YAML list, item types preserved."""
        field = QLineEdit(", ".join(str(item) for item in items))
        field.setToolTip("Comma-separated list")
        field.textChanged.connect(lambda text: write(parse_list_field(text)))
        return field

    def _render_remaining_sections(self):
        """Surface every top-level key the fixed boxes do not cover.

        Without this an experiment's paradigm- or component-specific configuration
        is absent from the panel even though saving preserves it. Nested
        dictionaries are rendered recursively so every value round-trips from
        the panel, regardless of how a component structures its parameters.
        """
        leftovers = {
            k: v for k, v in self.raw_data.items()
            if k not in FIXED_BOX_KEYS and k not in ("data_module", "evaluator", "plots", "plot_overrides")
        }
        if isinstance(leftovers.get("telemetry"), dict):  # telemetry.metrics has its own box (Recorded Metrics)
            rest = {k: v for k, v in leftovers["telemetry"].items() if k != "metrics"}
            if rest:
                leftovers["telemetry"] = rest
            else:
                leftovers.pop("telemetry")
        if not leftovers:
            return

        scalars = {k: v for k, v in leftovers.items() if not isinstance(v, (dict, list))}
        sections = {k: v for k, v in leftovers.items() if isinstance(v, (dict, list))}

        if scalars:
            box = ConfigBox()
            # ConfigBox drops its title argument since the param box headers were
            # removed, and an unlabelled box of tau_min/tau_max tells you nothing,
            # so these sections carry their own heading the way method cards do.
            box.add_widget(label("Additional Settings", "cardTitle"))
            form = QFormLayout()
            form.setVerticalSpacing(8)
            for key, value in scalars.items():
                form.addRow(key, self._scalar_widget(value, lambda v, k=key: self._on_field_edited(k, v)))
            box.add_layout(form)
            self.boxes_layout.addWidget(box)

        for name, value in sections.items():
            box = ConfigBox()
            box.add_widget(label(name, "cardTitle"))
            form = QFormLayout()
            form.setVerticalSpacing(8)
            if isinstance(value, list):
                form.addRow("items", self._list_field(value, lambda v, k=name: self._on_field_edited(k, v)))
            else:
                self._render_experiment_dict_contents(form, value, (name,))
            box.add_layout(form)
            self.boxes_layout.addWidget(box)

    def _render_experiment_dict_contents(self, parent_layout, values: Mapping[str, Any], path: tuple[str, ...], level: int = 1):
        """Render an experiment override dictionary at arbitrary depth."""
        for key, value in values.items():
            value_path = path + (key,)
            if isinstance(value, dict):
                sub_block = IndentedHierarchyBlock(level=level)
                sub_header = QHBoxLayout()
                sub_header.addWidget(label(f"[{key}]", "muted"))
                sub_header.addStretch()
                sub_block.addRow("", sub_header)
                self._render_experiment_dict_contents(sub_block, value, value_path, level + 1)
                parent_layout.addRow("", sub_block)
            elif isinstance(value, list):
                parent_layout.addRow(
                    key,
                    self._list_field(value, lambda updated, p=value_path: self._on_generic_path_edited(p, updated)),
                )
            else:
                parent_layout.addRow(
                    key,
                    self._scalar_widget(value, lambda updated, p=value_path: self._on_generic_path_edited(p, updated)),
                )

    def _render_generic_boxes(self):
        """Render clean boxes for modular YAML configs (agent/*.yaml, env/*.yaml, model/*.yaml, etc.)."""
        self.paradigm_badge.hide()
        data = self.raw_data

        stem = self.current_path.stem if self.current_path else ""
        rel_path = str(self.current_rel_path or "")

        # --- Component Overview Header Card ---
        overview_box = ConfigBox()
        ov_layout = QVBoxLayout()
        ov_layout.setContentsMargins(14, 12, 14, 12)
        ov_layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        title_lbl = QLabel(stem)
        title_lbl.setStyleSheet(f"font-size: 15px; font-weight: bold; color: {theme_color('text')};")
        top_row.addWidget(title_lbl)

        kind_name, pill_color_role = _get_component_kind(rel_path)
        kind_badge = QLabel(f" {kind_name} ")
        kind_badge.setStyleSheet(f"""
            background-color: {theme_color('surface')};
            color: {theme_color(pill_color_role)};
            border: 1px solid {theme_color('border')};
            border-radius: 10px;
            padding: 2px 8px;
            font-size: 11px;
            font-weight: 600;
        """)
        top_row.addWidget(kind_badge)
        top_row.addStretch()
        ov_layout.addLayout(top_row)

        desc = str(self.raw_data.get("description", "") or "").strip()
        desc_edit = QLineEdit(desc)
        desc_edit.setPlaceholderText("Add a brief description for this component...")
        desc_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {theme_color('base')};
                border: 1px dashed {theme_color('border')};
                border-radius: 4px;
                color: {theme_color('text')};
                font-size: 12px;
                padding: 4px 8px;
            }}
            QLineEdit:focus {{
                border: 1px solid {theme_color('accent')};
                background-color: {theme_color('surface')};
            }}
        """)
        desc_edit.textChanged.connect(lambda val: self._on_field_edited("description", val.strip()))
        ov_layout.addWidget(desc_edit)

        overview_box.add_layout(ov_layout)
        self.boxes_layout.addWidget(overview_box)

        leaf_params = {}
        nested_sections = {}

        for k, v in data.items():
            if k in ("name", "algorithm", "architecture", "experiment_id", "id", "description"):
                continue
            if k in ("type", "env_id") and str(v).lower() == stem.lower():
                continue
            if isinstance(v, str) and v.lower() == stem.lower() and k in ("model", "agent"):
                continue
            if isinstance(v, dict):
                nested_sections[k] = v
            else:
                leaf_params[k] = v

        # --- Parameters Card (Scalars and Lists) ---
        box_props = ConfigBox()
        hdr_layout = QHBoxLayout()
        hdr_layout.setSpacing(8)
        lbl_card_title = label("Parameters", "cardTitle")
        hdr_layout.addWidget(lbl_card_title)
        hdr_layout.addWidget(label(f"Settings in {stem}", "muted"))
        hdr_layout.addStretch()

        btn_add_p = _make_subtle_tool_button("plus", "Add parameter", size=22, icon_size=12)
        btn_add_p.clicked.connect(self._prompt_add_generic_leaf_param)
        hdr_layout.addWidget(btn_add_p)
        box_props.add_layout(hdr_layout)

        if leaf_params:
            form = QFormLayout()
            form.setVerticalSpacing(8)
            for k, v in leaf_params.items():
                row = QHBoxLayout()
                row.setSpacing(8)
                k_lower = k.lower()
                schema = self._component_parameter_schema((k,))
                if self._schema_has_choices(schema):
                    row.addWidget(self._schema_widget(v, schema, lambda val, key=k: self._on_field_edited(key, val)))
                elif k in ("compatible_paradigms", "paradigms"):
                    tag_cloud = TagCloudWidget(
                        tags=v if isinstance(v, list) else ([v] if v else []),
                        available_options=_get_available_paradigms(),
                    )
                    tag_cloud.tagsChanged.connect(lambda tags, key=k: self._on_field_edited(key, tags))
                    row.addWidget(tag_cloud)
                elif (
                    k in ("hidden_sizes", "hidden_dims", "latent_sizes")
                    and isinstance(v, list)
                    and getattr(self.settings_manager, "visual_layer_editor", True)
                ):
                    layer_stack = LayerStackWidget(layers=v)
                    layer_stack.layersChanged.connect(lambda layers, key=k: self._on_field_edited(key, layers))
                    row.addWidget(layer_stack)
                elif k_lower in CATEGORICAL_CHOICES:
                    combo = _make_categorical_combo(k, v, lambda val, key=k: self._on_field_edited(key, val))
                    row.addWidget(combo)
                elif isinstance(v, bool):
                    chk = ToggleSlider(checked=v)
                    chk.toggled.connect(lambda val, key=k: self._on_field_edited(key, val))
                    row.addWidget(chk)
                elif isinstance(v, (int, float)):
                    spin = number_field(v)
                    spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin.valueChanged.connect(lambda val, key=k: self._on_field_edited(key, val))
                    row.addWidget(spin)
                elif isinstance(v, list):
                    txt = QLineEdit(str(list(v)))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH + 80)
                    txt.setToolTip("YAML list, e.g. [64, 64]")
                    txt.textChanged.connect(lambda val, key=k: self._on_field_edited(key, _parse_yaml_value(val)))
                    row.addWidget(txt)
                else:
                    txt = QLineEdit(str(v if v is not None else ""))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH)
                    txt.textChanged.connect(lambda val, key=k: self._on_field_edited(key, _parse_yaml_value(val)))
                    row.addWidget(txt)
                btn_del = _make_subtle_tool_button("close", f"Remove '{k}'", size=20, icon_size=12)
                btn_del.clicked.connect(lambda _, key=k: self._remove_generic_field(key))
                row.addWidget(btn_del)
                row.addStretch()
                form.addRow(k, row)
            box_props.add_layout(form)
        else:
            empty_row = QHBoxLayout()
            empty_row.addWidget(label("No parameters configured. Click '+' to add one.", "muted"))
            empty_row.addStretch()
            box_props.add_layout(empty_row)

        self.boxes_layout.addWidget(box_props)

        # --- Nested Sections (Hierarchical Parameters) ---
        for section_name, section_val in nested_sections.items():
            box_sec = ConfigBox()
            sec_hdr = QHBoxLayout()
            sec_hdr.setSpacing(8)
            sec_hdr.addWidget(label(section_name, "cardTitle"))
            sec_hdr.addStretch()

            btn_add_sec_p = _make_subtle_tool_button("plus", f"Add parameter to '{section_name}'", size=22, icon_size=12)
            btn_add_sec_p.clicked.connect(lambda _, sn=section_name: self._prompt_add_nested_generic_param((sn,)))
            sec_hdr.addWidget(btn_add_sec_p)

            btn_del_sec = _make_subtle_tool_button("close", f"Remove section '{section_name}'", size=22, icon_size=12)
            btn_del_sec.clicked.connect(lambda _, sn=section_name: self._remove_generic_field(sn))
            sec_hdr.addWidget(btn_del_sec)
            box_sec.add_layout(sec_hdr)

            if isinstance(section_val, dict):
                form_sec = QFormLayout()
                form_sec.setVerticalSpacing(8)
                self._render_generic_dict_contents(form_sec, section_val, (section_name,), level=1)
                box_sec.add_layout(form_sec)
            self.boxes_layout.addWidget(box_sec)

    def _render_generic_dict_contents(self, parent_layout, d: dict, path: tuple[str, ...], level: int = 1):
        """Recursively render hierarchical nested dictionary contents in the generic component viewer."""
        for k, v in d.items():
            if isinstance(v, dict):
                sub_block = IndentedHierarchyBlock(level=level)
                sub_hdr = QHBoxLayout()
                sub_hdr.addWidget(label(f"[{k}]", "muted"))
                sub_hdr.addStretch()
                btn_add_sub = _make_subtle_tool_button("plus", f"Add parameter to '{k}'", size=20, icon_size=12)
                btn_add_sub.clicked.connect(lambda _, p=path + (k,): self._prompt_add_nested_generic_param(p))
                sub_hdr.addWidget(btn_add_sub)
                btn_del_sub = _make_subtle_tool_button("close", f"Remove '{k}'", size=20, icon_size=12)
                btn_del_sub.clicked.connect(lambda _, p=path + (k,): self._remove_generic_path(p))
                sub_hdr.addWidget(btn_del_sub)

                sub_block.addRow("", sub_hdr)
                self._render_generic_dict_contents(sub_block, v, path + (k,), level=level + 1)
                parent_layout.addRow("", sub_block)
            else:
                row = QHBoxLayout()
                row.setSpacing(8)
                k_lower = k.lower()
                schema = self._component_parameter_schema(path + (k,))
                if self._schema_has_choices(schema):
                    row.addWidget(self._schema_widget(
                        v, schema,
                        lambda val, p=path + (k,): self._on_generic_path_edited(p, val),
                    ))
                elif k in ("compatible_paradigms", "paradigms"):
                    tag_cloud = TagCloudWidget(
                        tags=v if isinstance(v, list) else ([v] if v else []),
                        available_options=_get_available_paradigms(),
                    )
                    tag_cloud.tagsChanged.connect(lambda tags, p=path + (k,): self._on_generic_path_edited(p, tags))
                    row.addWidget(tag_cloud)
                elif (
                    k in ("hidden_sizes", "hidden_dims", "latent_sizes")
                    and isinstance(v, list)
                    and getattr(self.settings_manager, "visual_layer_editor", True)
                ):
                    layer_stack = LayerStackWidget(layers=v)
                    layer_stack.layersChanged.connect(lambda layers, p=path + (k,): self._on_generic_path_edited(p, layers))
                    row.addWidget(layer_stack)
                elif k_lower in CATEGORICAL_CHOICES:
                    combo = _make_categorical_combo(k, v, lambda val, p=path + (k,): self._on_generic_path_edited(p, val))
                    row.addWidget(combo)
                elif isinstance(v, bool):
                    chk = ToggleSlider(checked=v)
                    chk.toggled.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, val))
                    row.addWidget(chk)
                elif isinstance(v, (int, float)):
                    spin = number_field(v)
                    spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
                    spin.valueChanged.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, val))
                    row.addWidget(spin)
                elif isinstance(v, list):
                    txt = QLineEdit(str(list(v)))
                    txt.setMaximumWidth(TEXT_FIELD_WIDTH + 80)
                    txt.setToolTip("YAML list, e.g. [64, 64]")
                    txt.textChanged.connect(lambda val, p=path + (k,): self._on_generic_path_edited(p, _parse_yaml_value(val)))
                    row.addWidget(txt)
                else:
                    schema = self._component_parameter_schema(path + (k,))
                    row.addWidget(self._schema_widget(
                        v, schema,
                        lambda val, p=path + (k,): self._on_generic_path_edited(p, val),
                    ))
                btn_del = _make_subtle_tool_button("close", f"Remove '{k}'", size=20, icon_size=12)
                btn_del.clicked.connect(lambda _, p=path + (k,): self._remove_generic_path(p))
                row.addWidget(btn_del)
                row.addStretch()
                parent_layout.addRow(k, row)

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
            tree = config_tree()
            choices = tree.model_names if tree else []
        if isinstance(choices, list) and choices:
            labels = [str(value) for value in choices]
            selected, ok = QInputDialog.getItem(self, "Parameter Value", f"Value for '{key}':", labels, 0, False)
            return choices[labels.index(selected)] if ok else choices[0]
        if isinstance(schema, dict) and "default" in schema:
            return schema["default"]
        val_str, ok = QInputDialog.getText(self, "Parameter Value", f"Initial value for '{key}':")
        return _parse_yaml_value(val_str) if ok else None

    def _on_generic_path_edited(self, path: tuple[str, ...], value: Any):
        if self._block_updates or not path:
            return
        cur = self.raw_data
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[path[-1]] = _parse_yaml_value(value) if isinstance(value, str) else value
        self._mark_dirty()

    def _remove_generic_field(self, key: str):
        self.raw_data.pop(key, None)
        self._mark_dirty()
        self._render_boxes()

    def _remove_generic_path(self, path: tuple[str, ...]):
        if not path:
            return
        cur = self.raw_data
        for p in path[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                return
            cur = cur[p]
        cur.pop(path[-1], None)
        self._mark_dirty()
        self._render_boxes()

    # ── Edit Handlers ──────────────────────────────────────────────────────────

    def _on_field_edited(self, key, value):
        if self._block_updates:
            return
        self.raw_data[key] = _parse_yaml_value(value) if isinstance(value, str) else value
        self._mark_dirty()

    def _validate_method_name(self, name: str, existing_names: set[str]) -> str | None:
        """Return an error message if invalid, or None if valid."""
        name = name.strip()
        if not name:
            return "Method name cannot be empty."
        if name == "params":
            return "The method name 'params' is reserved for universal parameters."
        if name in existing_names:
            return f"A method named '{name}' already exists in this experiment."
        if not re.match(r"^[a-zA-Z0-9_\-]+$", name):
            return "Method name contains invalid characters. Use only letters, digits, hyphens, and underscores."
        return None

    def duplicate_method(self, name: str) -> str | None:
        """Duplicate an existing method card with all its configuration."""
        if self._is_base_config():
            return None
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict) or name not in methods:
            return None
        source_spec = methods[name]

        existing_keys = set(methods.keys())
        base = f"{name}_copy"
        if base not in existing_keys:
            dup_name = base
        else:
            n = 2
            while f"{base}_{n}" in existing_keys:
                n += 1
            dup_name = f"{base}_{n}"

        new_methods = {}
        for k, v in methods.items():
            new_methods[k] = v
            if k == name:
                new_methods[dup_name] = copy.deepcopy(source_spec)

        self.raw_data["methods"] = new_methods
        self._mark_dirty()
        self._render_boxes()
        return dup_name

    def _get_group_default_method(self) -> tuple[str | None, dict | None]:
        """Return (method_name, method_spec) for the default method declared in group's _base.yaml."""
        base_data = None
        if self.current_path:
            local_base = self.current_path.parent / "_base.yaml"
            if local_base.is_file():
                try:
                    base_data = yaml.safe_load(local_base.read_text(encoding="utf-8")) or {}
                except Exception:
                    pass
        if not base_data:
            tree = config_tree()
            group = self._group_name()
            if tree and group:
                base_data = _read_group_base(tree, group)
        if base_data:
            base_methods = base_data.get("methods", {})
            if isinstance(base_methods, dict):
                for k, v in base_methods.items():
                    if k != "params" and isinstance(v, dict):
                        return k, copy.deepcopy(v)
        return None, None

    def _start_add_new_method(self, *, empty: bool = False):
        """Initiate adding a new method card with focus on the name editor.

        ``empty`` is used by the Methods header action: it deliberately does
        not clone a group's default method into a new experiment variant.
        """
        tree = config_tree()
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
        """Add a method to the open experiment or base defaults. Returns its name, or None."""
        if self._is_base_config():
            methods = self.raw_data.setdefault("methods", {})
            if any(k != "params" and isinstance(v, dict) for k, v in methods.items()):
                return None
            meth_name = name or "default_method"
            spec = {}
            if agent:
                spec["agent"] = agent
            if model:
                spec["model"] = model
            methods[meth_name] = spec
            self._mark_dirty()
            self._render_boxes()
            return meth_name

        def_name, def_spec = self._get_group_default_method()
        paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
        tree = config_tree()
        if not tree or paradigm not in tree.paradigms:
            return None
        p_obj = tree.paradigms.get(paradigm) if tree and paradigm in tree.paradigms else None
        default_model = (
            (def_spec.get("model") if def_spec else None)
            or (p_obj.raw.get("default_model") if p_obj else None)
        )

        methods = self.raw_data.setdefault("methods", {})
        if name:
            meth_name = name
        elif agent or model:
            from .config_model import unique_method_name
            meth_name = unique_method_name(methods, agent, model or default_model)
        elif def_name and def_name not in methods:
            meth_name = def_name
        else:
            base = "method"
            if base not in methods:
                meth_name = base
            else:
                n = 2
                while f"{base}_{n}" in methods:
                    n += 1
                meth_name = f"{base}_{n}"

        spec = copy.deepcopy(def_spec) if def_spec else {}
        if agent:
            spec["agent"] = agent
        if model:
            spec["model"] = model
        methods[meth_name] = spec
        self._mark_dirty()
        self._render_boxes()
        return meth_name


    def _on_experiment_env_changed(self, value):
        if self._block_updates:
            return
        selected = self.env_selection()
        self._set_experiment_env(selected)
        self._mark_dirty()
        self._render_boxes()

    def remove_method(self, name):
        """Remove a method from the open experiment. Returns True if it went."""
        if not remove_method_from(self.raw_data, name):
            return False
        self._mark_dirty()
        self._render_boxes()
        return True

    def rename_method(self, old_name: str, new_name: str) -> bool:
        """Rename a method in the open experiment while preserving dict ordering."""
        if not old_name or not new_name or old_name == new_name:
            return False
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict) or old_name not in methods:
            return False
        new_methods = {}
        for k, v in methods.items():
            if k == old_name:
                new_methods[new_name] = v
            else:
                new_methods[k] = v
        self.raw_data["methods"] = new_methods
        self._mark_dirty()
        self._render_boxes()
        return True

    def reorder_method(self, method_name: str, target_index: int) -> bool:
        """Reorder method cards by moving method_name to target_index."""
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict) or method_name not in methods:
            return False
        has_params = "params" in methods
        params_val = methods.get("params")

        method_keys = [k for k in methods.keys() if k != "params"]
        if method_name not in method_keys:
            return False
        method_keys.remove(method_name)
        target_index = max(0, min(target_index, len(method_keys)))
        method_keys.insert(target_index, method_name)

        new_methods = {}
        if has_params:
            new_methods["params"] = params_val
        for k in method_keys:
            new_methods[k] = methods[k]

        self.raw_data["methods"] = new_methods
        self._mark_dirty()
        self._render_boxes()
        return True

    def _on_paradigm_changed(self, value):
        if self._block_updates:
            return
        self.raw_data["paradigm"] = value
        # An env or agent valid under the old paradigm may be forbidden under the
        # new one, so drop stale choices and rebuild against the new constraints.
        tree = config_tree()
        if tree and value in tree.paradigms:
            rules = tree.paradigms[value]
            env_name = self._get_explicit_env()
            environment = self._environment_for_name(tree, env_name)
            if environment and not rules.permits_environment(environment):
                self._set_experiment_env(None)
            self._pin_fields_the_paradigm_forbids(rules)
        self._mark_dirty()
        self._render_boxes()

    def _on_nested_edited(self, parent_key, child_key, value):
        if self._block_updates:
            return
        if parent_key not in self.raw_data or not isinstance(self.raw_data[parent_key], dict):
            self.raw_data[parent_key] = {}
        self.raw_data[parent_key][child_key] = value
        self._mark_dirty()

    def _on_method_param_edited(self, method_name, param_key, value):
        if self._block_updates:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        parsed_val = _parse_yaml_value(value) if isinstance(value, str) else value
        if isinstance(m_spec.get("model"), dict):
            model_name = next(iter(m_spec["model"]))
            if isinstance(m_spec["model"].get(model_name), dict) and param_key in m_spec["model"][model_name]:
                m_spec["model"][model_name][param_key] = parsed_val
                self._mark_dirty()
                return
        m_spec[param_key] = parsed_val
        self._mark_dirty()

    def _on_method_nested_param_edited(self, method_name: str, parent_key: str, child_key: str, value: Any):
        """Update a nested method parameter, including parameters inside a model config."""
        if self._block_updates:
            return
        m_spec = self.raw_data.setdefault("methods", {}).setdefault(method_name, {})
        parsed_val = _parse_yaml_value(value) if isinstance(value, str) else value
        model_spec = m_spec.get("model")
        if isinstance(model_spec, dict):
            for model_config in model_spec.values():
                if isinstance(model_config, dict) and isinstance(model_config.get(parent_key), dict):
                    model_config[parent_key][child_key] = parsed_val
                    self._mark_dirty()
                    return
        nested = m_spec.setdefault(parent_key, {})
        if not isinstance(nested, dict):
            nested = m_spec[parent_key] = {}
        nested[child_key] = parsed_val
        self._mark_dirty()

    def _remove_method_nested_param(self, method_name: str, parent_key: str, child_key: str):
        """Remove a nested method parameter, including parameters inside a model config."""
        methods = self.raw_data.get("methods", {})
        m_spec = methods.get(method_name, {}) if isinstance(methods, dict) else {}
        if isinstance(m_spec, dict):
            model_spec = m_spec.get("model")
            if isinstance(model_spec, dict):
                for model_config in model_spec.values():
                    if isinstance(model_config, dict) and isinstance(model_config.get(parent_key), dict):
                        model_config[parent_key].pop(child_key, None)
            nested = m_spec.get(parent_key)
            if isinstance(nested, dict):
                nested.pop(child_key, None)
        self._mark_dirty()
        self._render_boxes()

    def _method_model_parameter_root(self, method_name: str, model_name: str, create: bool = False) -> dict:
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        model_spec = m_spec.get("model")
        if isinstance(model_spec, dict) and model_name in model_spec:
            value = model_spec.get(model_name)
            if isinstance(value, dict):
                return value
            if create:
                model_spec[model_name] = {}
                return model_spec[model_name]
        return m_spec

    def _on_method_model_parameter_edited(
        self,
        method_name: str,
        model_name: str,
        path: tuple[str, ...],
        value: Any,
        schema_model_name: str | None = None,
        schema_path: tuple[str, ...] | None = None,
    ):
        if self._block_updates or not path:
            return
        target = self._method_model_parameter_root(method_name, model_name, create=True)
        for key in path[:-1]:
            if not isinstance(target.get(key), dict):
                target[key] = {"name": schema_model_name} if schema_model_name and schema_model_name != model_name else {}
            target = target[key]
        tree = config_tree()
        schema_owner = schema_model_name or model_name
        schema = tree.model_parameter_schema(
            schema_owner, schema_path if schema_path is not None else path
        ) if tree else {}
        target[path[-1]] = (
            value
            if isinstance(value, str) and schema.get("type") == "string"
            else _parse_yaml_value(value) if isinstance(value, str) else value
        )
        self._mark_dirty()

    def _remove_method_model_parameter(self, method_name: str, model_name: str, path: tuple[str, ...]):
        if not path:
            return
        target = self._method_model_parameter_root(method_name, model_name)
        for key in path[:-1]:
            target = target.get(key, {}) if isinstance(target, dict) else {}
        if isinstance(target, dict):
            target.pop(path[-1], None)
            self._mark_dirty()
            self._render_boxes()

    def _add_method_model_parameter(
        self,
        method_name: str,
        model_name: str,
        path: tuple[str, ...],
        schema_model_name: str | None = None,
        schema_path: tuple[str, ...] | None = None,
    ):
        if not path:
            return
        tree = config_tree()
        schema_owner = schema_model_name or model_name
        default = tree.model_parameters(schema_owner) if tree else {}
        lookup_path = schema_path if schema_path is not None else path
        found = True
        for key in lookup_path:
            if isinstance(default, dict) and key in default:
                default = default[key]
            else:
                found = False
                break
        if not found:
            schema = tree.model_parameter_schema(schema_owner, lookup_path) if tree else {}
            default = schema.get("default")
        self._on_method_model_parameter_edited(
            method_name, model_name, path, default, schema_owner, lookup_path
        )
        self._render_boxes()

    def _on_method_child_model_changed(
        self, method_name: str, parent_model_name: str, path: tuple[str, ...], child_model_name: str
    ):
        if self._block_updates:
            return
        value = {"name": child_model_name} if child_model_name else None
        target = self._method_model_parameter_root(method_name, parent_model_name, create=True)
        for key in path[:-1]:
            if not isinstance(target.get(key), dict):
                target[key] = {}
            target = target[key]
        if child_model_name:
            target[path[-1]] = value
        else:
            target.pop(path[-1], None)
        self._mark_dirty()
        self._render_boxes()

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
        """Render typed, recursively nested model params from the component schema."""
        tree = config_tree()
        schema_owner = schema_model_name or model_name
        schema = tree.model_parameter_schema(schema_owner, schema_path) if tree else {}
        schema_properties = schema.get("properties", {}) if isinstance(schema, dict) else {}
        keys = list(dict.fromkeys([*values.keys(), *defaults.keys(), *schema_properties.keys()]))
        rows = []

        for key in keys:
            present = key in values
            param_path = path + (key,)
            param_schema_path = schema_path + (key,)
            param_schema = tree.model_parameter_schema(schema_owner, param_schema_path) if tree else {}
            if not present and not show_defaults and key not in set(schema.get("required", [])):
                continue
            value = values.get(key, defaults.get(key, schema_properties.get(key, {}).get("default")))
            default_value = defaults.get(key, param_schema.get("default")) if isinstance(param_schema, dict) else defaults.get(key)

            if param_schema.get("x-theta-domain") == "model":
                slot_values = value if isinstance(value, dict) else {}
                selected_model = value if isinstance(value, str) else (
                    slot_values.get("name") or slot_values.get("architecture") or slot_values.get("type")
                )
                if tree and selected_model in tree.models:
                    selected_model = tree.models[selected_model].name
                row = QHBoxLayout()
                picker = ComboBox()
                picker.setMaximumWidth(TEXT_FIELD_WIDTH)
                picker.setPlaceholderText("Select model...")
                picker.addItem("", "")
                for choice in (tree.model_names if tree else []):
                    picker.addItem(choice, choice)
                if selected_model:
                    index = next(
                        (i for i in range(picker.count()) if picker.itemData(i) == selected_model), -1
                    )
                    if index < 0:
                        picker.insertItem(1, str(selected_model), selected_model)
                        index = 1
                    picker.setCurrentIndex(index)
                picker.currentIndexChanged.connect(
                    lambda i, mn=method_name, parent=model_name, p=param_path, combo=picker:
                    self._on_method_child_model_changed(mn, parent, p, combo.itemData(i) or "")
                )
                row.addWidget(picker)

                child_model = selected_model
                if child_model and tree and child_model in tree.models:
                    child_values = {
                        child_key: child_value for child_key, child_value in slot_values.items()
                        if child_key not in {"name", "architecture", "type", "model"}
                    }
                    child_defaults = tree.model_parameters(child_model)
                    child_candidates = [
                        candidate for candidate in tree.model_parameter_keys(child_model)
                        if candidate not in child_values and candidate not in {"name", "architecture", "type"}
                    ]
                    if child_candidates:
                        row.addWidget(_make_param_adder_button(
                            f"Add {child_model} parameter",
                            child_candidates,
                            lambda child, mn=method_name, parent=model_name, p=param_path, cm=child_model:
                            self._add_method_model_parameter(
                                mn, parent, p + (child,), cm, (child,)
                            ),
                            parent_widget=self,
                        ))
                    delete_slot = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                    delete_slot.clicked.connect(
                        lambda _, mn=method_name, parent=model_name, p=param_path:
                        self._remove_method_model_parameter(mn, parent, p)
                    )
                    row.addWidget(delete_slot)
                    rows.append((str(key), row))
                    child_rows = self._render_method_model_parameters(
                        method_name,
                        model_name,
                        child_values,
                        child_defaults,
                        param_path,
                        level + 1,
                        show_defaults,
                        child_model,
                        (),
                    )
                    if child_rows:
                        child_block = IndentedHierarchyBlock(level=level + 1)
                        for child_label, child_row in child_rows:
                            if isinstance(child_row, QWidget):
                                child_block.addRow(child_row)
                            else:
                                child_row.addStretch()
                                child_block.addRow(child_label, child_row)
                        rows.append(("", child_block))
                else:
                    delete_slot = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                    delete_slot.clicked.connect(
                        lambda _, mn=method_name, parent=model_name, p=param_path:
                        self._remove_method_model_parameter(mn, parent, p)
                    )
                    row.addWidget(delete_slot)
                    rows.append((str(key), row))
                continue

            enum_properties = param_schema.get("properties", {}) if isinstance(param_schema, dict) else {}
            if (
                isinstance(value, dict)
                and enum_properties
                and all(
                    isinstance(child_schema, dict)
                    and (
                        (isinstance(child_schema.get("enum"), list) and child_schema["enum"])
                        or child_schema.get("x-theta-domain") == "model"
                    )
                    for child_schema in enum_properties.values()
                )
            ):
                # Compact option groups (such as BlendRL's blender settings) into
                # one labeled row instead of nesting each setting under a generic
                # object editor. Model slots stay selectable here and expose their
                # model parameters only through the adjacent add button.
                row = QHBoxLayout()
                model_block = IndentedHierarchyBlock(level=level + 1)
                has_model_slot = False
                for child_key, child_schema in enum_properties.items():
                    if child_key not in value and not show_defaults:
                        continue
                    child_path = param_path + (child_key,)
                    child_value = value.get(
                        child_key,
                        defaults.get(key, {}).get(child_key)
                        if isinstance(defaults.get(key), dict)
                        else child_schema.get("default"),
                    )
                    if child_schema.get("x-theta-domain") == "model":
                        slot_values = child_value if isinstance(child_value, dict) else {}
                        selected_model = child_value if isinstance(child_value, str) else (
                            slot_values.get("name") or slot_values.get("architecture") or slot_values.get("type")
                        )
                        if tree and selected_model in tree.models:
                            selected_model = tree.models[selected_model].name
                        slot_row = QHBoxLayout()
                        picker = ComboBox()
                        picker.setMaximumWidth(TEXT_FIELD_WIDTH)
                        picker.setPlaceholderText("Select model...")
                        picker.addItem("", "")
                        for choice in (tree.model_names if tree else []):
                            picker.addItem(choice, choice)
                        if selected_model:
                            index = next(
                                (i for i in range(picker.count()) if picker.itemData(i) == selected_model), -1
                            )
                            if index < 0:
                                picker.insertItem(1, str(selected_model), selected_model)
                                index = 1
                            picker.setCurrentIndex(index)
                        picker.currentIndexChanged.connect(
                            lambda i, mn=method_name, parent=model_name, p=child_path, combo=picker:
                            self._on_method_child_model_changed(mn, parent, p, combo.itemData(i) or "")
                        )
                        slot_row.addWidget(picker)

                        nested_block = None
                        if selected_model and tree and selected_model in tree.models:
                            child_values = {
                                name: child_value for name, child_value in slot_values.items()
                                if name not in {"name", "architecture", "type", "model"}
                            }
                            child_defaults = tree.model_parameters(selected_model)
                            candidates = [
                                name for name in tree.model_parameter_keys(selected_model)
                                if name not in child_values and name not in {"name", "architecture", "type"}
                            ]
                            if candidates:
                                slot_row.addWidget(_make_param_adder_button(
                                    f"Add {selected_model} parameter",
                                    candidates,
                                    lambda child, mn=method_name, parent=model_name, p=child_path, cm=selected_model:
                                    self._add_method_model_parameter(mn, parent, p + (child,), cm, (child,)),
                                    parent_widget=self,
                                ))
                            nested_rows = self._render_method_model_parameters(
                                method_name, model_name, child_values, child_defaults,
                                child_path, level + 1, False, selected_model, (),
                            )
                            if nested_rows:
                                nested_block = IndentedHierarchyBlock(level=level + 2)
                                for nested_label, nested_row in nested_rows:
                                    if isinstance(nested_row, QWidget):
                                        nested_block.addRow(nested_row)
                                    else:
                                        nested_row.addStretch()
                                        nested_block.addRow(nested_label, nested_row)
                        remove_model = _make_subtle_tool_button(
                            "close", f"Remove '{child_key}' parameter", size=20, icon_size=12
                        )
                        remove_model.clicked.connect(
                            lambda _, mn=method_name, parent=model_name, p=child_path:
                            self._remove_method_model_parameter(mn, parent, p)
                        )
                        slot_row.addWidget(remove_model)
                        model_block.addRow(str(child_key), slot_row)
                        if nested_block is not None:
                            model_block.addRow("", nested_block)
                        has_model_slot = True
                    else:
                        if not child_schema.get("x-theta-hide-label"):
                            child_title = child_schema.get("title") or child_key.replace("_", " ").title()
                            row.addWidget(label(child_title, "muted"))
                        row.addWidget(self._schema_widget(
                            child_value,
                            child_schema,
                            lambda edited, mn=method_name, parent=model_name, p=child_path,
                            sm=schema_owner, sp=param_schema_path + (child_key,):
                            self._on_method_model_parameter_edited(mn, parent, p, edited, sm, sp),
                        ))
                missing_model_slots = [
                    child_key for child_key, child_schema in enum_properties.items()
                    if isinstance(child_schema, dict)
                    and child_schema.get("x-theta-domain") == "model"
                    and child_key not in value
                ]
                if missing_model_slots:
                    slot_key = missing_model_slots[0]
                    add_model = _make_subtle_tool_button("plus", "Add model parameter", size=20, icon_size=12)
                    add_model.clicked.connect(
                        lambda _, mn=method_name, parent=model_name, p=param_path + (slot_key,),
                        sm=schema_owner, sp=param_schema_path + (slot_key,):
                        self._add_method_model_parameter(mn, parent, p, sm, sp)
                    )
                    row.addWidget(add_model)
                row.addStretch()
                delete_group = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                delete_group.clicked.connect(
                    lambda _, mn=method_name, parent=model_name, p=param_path:
                    self._remove_method_model_parameter(mn, parent, p)
                )
                row.addWidget(delete_group)
                rows.append((str(key), row))
                if has_model_slot:
                    rows.append(("", model_block))
                continue

            if isinstance(value, dict) or (isinstance(param_schema, dict) and param_schema.get("type") == "object"):
                child_values = value if isinstance(value, dict) else {}
                child_defaults = defaults.get(key, {}) if isinstance(defaults.get(key), dict) else {}
                block = IndentedHierarchyBlock(level=level + 1)
                child_candidates = list(dict.fromkeys([*child_defaults.keys(), *param_schema.get("properties", {}).keys()]))
                child_candidates = [candidate for candidate in child_candidates if candidate not in child_values]
                header = QHBoxLayout()
                header.addWidget(label(f"[{key}]", "muted"))
                if child_candidates:
                    header.addWidget(_make_param_adder_button(
                        f"Add {key} parameter",
                        child_candidates,
                        lambda child, mn=method_name, model=model_name, p=param_path: self._add_method_model_parameter(mn, model, p + (child,)),
                        parent_widget=self,
                    ))
                header.addStretch()
                delete_group = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
                delete_group.clicked.connect(
                    lambda _, mn=method_name, model=model_name, p=param_path: self._remove_method_model_parameter(mn, model, p)
                )
                header.addWidget(delete_group)
                block.addRow("", header)
                for child_label, child_row in self._render_method_model_parameters(
                    method_name, model_name, child_values, child_defaults, param_path, level + 1, show_defaults,
                    schema_owner, param_schema_path
                ):
                    if isinstance(child_row, QWidget):
                        block.addRow(child_row)
                    else:
                        child_row.addStretch()
                        block.addRow(child_label, child_row)
                rows.append(("", block))
                continue

            row = QHBoxLayout()
            widget = self._schema_widget(
                value,
                param_schema,
                lambda edited, mn=method_name, model=model_name, p=param_path, sm=schema_owner, sp=param_schema_path: self._on_method_model_parameter_edited(
                    mn, model, p, edited, sm, sp
                ),
            )
            row.addWidget(widget)
            delete_value = _make_subtle_tool_button("close", f"Remove '{key}'", size=20, icon_size=12)
            delete_value.clicked.connect(
                lambda _, mn=method_name, model=model_name, p=param_path: self._remove_method_model_parameter(mn, model, p)
            )
            row.addWidget(delete_value)
            if not present:
                row.addWidget(QLabel("(default)"))
            row.addStretch()
            rows.append((str(key), row))

        return rows

    def _browse_dataset_path(self):
        """Open file dialog to select a dataset file or directory."""
        initial_dir = str(Path.cwd() / "in" / "datasets")
        if not Path(initial_dir).is_dir():
            initial_dir = str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            initial_dir,
            "Dataset Files (*.pkl *.csv *.parquet *.h5 *.pt);;All Files (*)",
        )
        if selected:
            try:
                rel = Path(selected).relative_to(Path.cwd()).as_posix()
            except ValueError:
                rel = selected
            if hasattr(self, "combo_dataset_path"):
                self.combo_dataset_path.setEditText(rel)
            self._on_field_edited("dataset_path", rel)

    def remove_block(self, block_name: str):
        """Remove a configuration block from the open experiment."""
        if block_name == "trainer":
            self.raw_data.pop("trainer", None)
        elif block_name == "checkpoint":
            self.raw_data.pop("checkpoint", None)
        elif block_name == "env":
            if isinstance(self.raw_data.get("env"), dict):
                self.raw_data.pop("env", None)
        elif block_name == "dataset":
            self.raw_data.pop("dataset_path", None)
            self.raw_data.pop("offline_datasets", None)
            self.raw_data.pop("data_module", None)
        elif block_name == "params":
            if "methods" in self.raw_data and isinstance(self.raw_data["methods"], dict):
                self.raw_data["methods"].pop("params", None)
        elif block_name in ("data_module", "evaluator"):
            self.raw_data.pop(block_name, None)
        elif block_name == "plots":
            self.raw_data.pop("plots", None)
            self.raw_data.pop("plot_overrides", None)
        elif block_name == "metrics":
            telemetry = self.raw_data.get("telemetry")
            if isinstance(telemetry, dict):
                telemetry.pop("metrics", None)
                if not telemetry:
                    self.raw_data.pop("telemetry", None)
        self._mark_dirty()
        self._render_boxes()

    def _render_metrics_box(self, data, resolved):
        """Recorded metrics (telemetry.metrics): shown once the experiment sets the list."""
        own = _recorded_metrics(data)
        source = own if own is not None else (_recorded_metrics(resolved) if self.view_mode == "resolved" else None)
        if source is None:
            return
        box = ConfigBox("Recorded Metrics", "Which metrics the run records")
        header = QHBoxLayout()
        header.addWidget(label("Recorded Metrics", "heading"))
        header.addStretch()
        if own is not None:
            btn_rm = _make_subtle_tool_button("trash", "Record every metric again", size=24, icon_size=14)
            btn_rm.clicked.connect(lambda: self.remove_block("metrics"))
            header.addWidget(btn_rm)
        else:
            header.addWidget(label("(inherited defaults)", "muted"))
        box.add_layout(header)
        experiment = self.current_path.stem if self.current_path else ""
        known = logged_metrics(LOGS_DIR, str(self._group_name() or ""), experiment)
        known += [name for name in source if not name.endswith("/") and name not in known]  # listed, not yet seen
        picker = MetricPicker(source, sorted(known))
        picker.changed.connect(self._on_recorded_metrics_changed)
        self.field_widgets["telemetry.metrics"] = picker
        box.add_widget(picker)
        self.boxes_layout.addWidget(box)

    def _on_recorded_metrics_changed(self, metrics):
        if self._block_updates:
            return
        self.raw_data.setdefault("telemetry", {})["metrics"] = list(metrics)
        self._mark_dirty()

    def add_block(self, block_name: str):
        """Add an optional configuration block to the open experiment."""
        if block_name == "trainer":
            self.raw_data["trainer"] = {"accelerator": "cpu", "max_epochs": 25}
        elif block_name == "checkpoint":
            self.raw_data["checkpoint"] = {"monitor": "val/loss", "mode": "min", "save_top_k": 1, "every_n_epochs": 1}
        elif block_name == "env":
            self.raw_data["env"] = {}
        elif block_name == "dataset":
            tree = config_tree()
            paradigm = self.raw_data.get("paradigm") or self._inherited_paradigm()
            rules = tree.paradigms.get(paradigm) if tree else None
            if not rules or not rules.uses_static_dataset:
                return
            env_name = self.env_selection() or self._get_explicit_env() or self._inherited_env() or self._group_name() or "dataset"
            self.raw_data["dataset_path"] = self._get_resolved_defaults().get("dataset_path") or f"in/datasets/{env_name}"
        elif block_name == "params":
            self.raw_data.setdefault("methods", {})["params"] = {"lr": 0.0003}
        elif block_name == "metrics":
            # Start from what the run records anyway: every section ticked, so nothing changes until unticked.
            telemetry = self.raw_data.setdefault("telemetry", {})
            telemetry["metrics"] = [f"{section}/" for section in METRIC_SECTIONS]
        self._mark_dirty()
        self._render_boxes()

    def _on_universal_param_edited(self, key, value):
        if self._block_updates:
            return
        methods = self.raw_data.setdefault("methods", {})
        params = methods.setdefault("params", {})
        if "." in key:
            parts = key.split(".")
            target = params
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = _parse_yaml_value(value)
        else:
            params[key] = _parse_yaml_value(value)
        self._mark_dirty()

    def _remove_universal_param(self, key):
        if "methods" in self.raw_data and "params" in self.raw_data["methods"]:
            params = self.raw_data["methods"]["params"]
            if "." in key:
                parts = key.split(".")
                target = params
                for p in parts[:-1]:
                    if isinstance(target, dict) and p in target:
                        target = target[p]
                    else:
                        target = None
                        break
                if isinstance(target, dict):
                    target.pop(parts[-1], None)
                if len(parts) > 1 and parts[0] in params and isinstance(params[parts[0]], dict) and not params[parts[0]]:
                    params.pop(parts[0], None)
            else:
                params.pop(key, None)
            if not self.raw_data["methods"]["params"]:
                self.raw_data["methods"].pop("params", None)
            self._mark_dirty()
            self._render_boxes()

    def _add_universal_param(self, key, value=None):
        if not key:
            return
        if value is None or value == "":
            value = self._resolve_default_param_value("", key)
        methods = self.raw_data.setdefault("methods", {})
        params = methods.setdefault("params", {})
        if "." in key:
            parts = key.split(".")
            target = params
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = _parse_yaml_value(value)
        else:
            params[key] = _parse_yaml_value(value)
        self._mark_dirty()
        self._render_boxes()

    def _resolve_default_param_value(self, method_name: str, param_key: str):
        """Look up hierarchical default for param_key:
        1. methods.params in current file
        2. Group _base.yaml (methods.params or methods[method_name])
        3. in/config/agent/<agent>.yaml
        4. in/config/model/<model>.yaml
        5. Known defaults fallback
        """
        methods = self.raw_data.get("methods")
        if isinstance(methods, dict):
            univ_params = methods.get("params")
            if isinstance(univ_params, dict) and param_key in univ_params:
                return univ_params[param_key]

        m_spec = (
            methods.get(method_name)
            if isinstance(methods, dict) and isinstance(methods.get(method_name), dict)
            else {}
        )
        agent_raw = m_spec.get("agent")
        agent_name = (
            (agent_raw.get("name") or agent_raw.get("algorithm") or next((k for k in agent_raw if k != "name"), None))
            if isinstance(agent_raw, dict)
            else agent_raw
        )
        model_raw = m_spec.get("model")
        model_name = next(iter(model_raw)) if isinstance(model_raw, dict) else model_raw

        tree = config_tree()
        group = self._group_name()

        # 2. Group _base.yaml
        if tree and group:
            base_data = _read_group_base(tree, group)
            base_methods = base_data.get("methods")
            if isinstance(base_methods, dict):
                base_params = base_methods.get("params")
                if isinstance(base_params, dict) and param_key in base_params:
                    return base_params[param_key]
                if method_name in base_methods and isinstance(base_methods[method_name], dict):
                    if param_key in base_methods[method_name]:
                        return base_methods[method_name][param_key]

        # 3. Agent YAML: in/config/agent/<agent>.yaml
        if tree and agent_name:
            agent_params = tree.agent_hyperparameters(agent_name)
            if param_key in agent_params:
                return agent_params[param_key]

        # 4. Model YAML: in/config/model/<model>.yaml
        if tree and model_name:
            model_params = tree.model_parameters(model_name)
            if param_key in model_params:
                return model_params[param_key]

        # 5. Generic fallback
        known = {
            "lr": 0.0003,
            "learning_rate": 0.0003,
            "batch_size": 64,
            "gamma": 0.99,
            "ent_coef": 0.01,
            "clip_range": 0.2,
            "epochs": 25,
            "eval_every_n_epochs": 5,
            "weight_decay": 0.0001,
        }
        if param_key in known:
            return known[param_key]

        if "lr" in param_key:
            return 0.0003
        if "coef" in param_key or "alpha" in param_key:
            return 0.01
        return 0.0

    def _remove_method_param(self, method_name, param_key):
        if "methods" in self.raw_data and method_name in self.raw_data["methods"]:
            m_spec = self.raw_data["methods"][method_name]
            if isinstance(m_spec.get("model"), dict):
                model_name = next(iter(m_spec["model"]))
                if isinstance(m_spec["model"].get(model_name), dict):
                    m_spec["model"][model_name].pop(param_key, None)
            m_spec.pop(param_key, None)
            self._mark_dirty()
            self._render_boxes()

    def _add_method_param(self, method_name, param_key, value=None):
        if not param_key:
            return
        methods = self.raw_data.setdefault("methods", {})
        m_spec = methods.setdefault(method_name, {})
        model_name = None
        if isinstance(m_spec.get("model"), str):
            model_name = m_spec["model"]
        elif isinstance(m_spec.get("model"), dict):
            model_name = next(iter(m_spec["model"]), None)

        if self.tree and model_name and param_key in self.tree.model_parameter_keys(model_name):
            self._add_method_model_parameter(method_name, model_name, (param_key,))
            return

        if value is None or value == "":
            value = self._resolve_default_param_value(method_name, param_key)
        parsed_val = _parse_yaml_value(value)
        if isinstance(m_spec.get("model"), dict) and model_name:
            if self.tree and model_name in self.tree.models and param_key in self.tree.model_parameters(model_name):
                m_spec["model"].setdefault(model_name, {})[param_key] = parsed_val
                self._mark_dirty()
                self._render_boxes()
                return
        m_spec[param_key] = parsed_val
        self._mark_dirty()
        self._render_boxes()

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
        spin = SpinBox()
        spin.setMaximumWidth(NUMBER_FIELD_WIDTH)
        spin.setRange(1, 100_000_000)
        spin.setValue(int(source[key]))
        spin.valueChanged.connect(lambda v: self._on_field_edited(state["key"], v))
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(mode)
        row.addWidget(spin)
        state["tail"] = (
            None if self._is_base_config()
            else self._checkpoint_remove_button() if key in data
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
        """Add an experiment-level parameter override to the configuration."""
        if not key:
            return
        if default_val is not None:
            self.raw_data[key] = default_val
        elif key == "total_timesteps":
            self.raw_data[key] = int(self._get_resolved_defaults().get("total_timesteps") or 10000)
        elif key == "seed":
            res_seed = self._get_resolved_defaults().get("seed")
            self.raw_data[key] = int(res_seed if res_seed is not None else 42)
        elif key == "eval_episodes":
            self.raw_data[key] = int(self._get_resolved_defaults().get("eval_episodes") or 100)
        elif key == "eval_freq_steps":
            self.raw_data[key] = int(self._get_resolved_defaults().get("eval_freq_steps") or 10000)
        elif key in ("tensorboard", "save_dataset", "recover", "no_plot"):
            self.raw_data[key] = True
        else:
            self.raw_data[key] = ""
        self._mark_dirty()
        if render:
            self._render_boxes()

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
        """Point a method at a model, and re-read the configs from disk.

        The model combo is filled from in/config/model/, so a model written
        while the app is open is invisible until the cached tree re-scans.
        """
        if not model_name:
            return
        methods = self.raw_data.setdefault("methods", {})
        methods.setdefault(method_name, {})["model"] = model_name
        self._mark_dirty()
        config_tree(force_reload=True)
        self._render_boxes()

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
        """Write legal values for the fields the chosen paradigm disables.

        in/config/config.yaml defaults eval_episodes to 100 for the online case.
        A config that simply omits it inherits that and is rejected before
        training starts, which is why the shipped offline and supervised bases pin it.
        Marking the field "not used" in the panel is not enough on its own:
        the inherited value still reaches the backend.

        Switching back to a paradigm that allows a field clears the pin, but only
        while it still holds the value set here, so a deliberate choice survives.
        """
        for name, forced in self._FORBIDDEN_FIELD_VALUES.items():
            if not rules.field_enabled(name):
                self.raw_data[name] = forced
            elif self.raw_data.get(name) == forced:
                self.raw_data.pop(name, None)

    def _reconcile_group_base(self):
        """Re-pin a group base against whatever paradigm it currently declares.

        _on_paradigm_changed covers picking a paradigm, but not a base that
        already had one - an older group, or one where only the environment was
        changed. Those keep inheriting eval_episodes: 100 and stay unrunnable,
        so the base is reconciled whenever it is saved.

        Only _base.yaml: the pin belongs on the file that binds the paradigm,
        and experiments inherit it.
        """
        if not self.current_path or self.current_path.name != "_base.yaml":
            return
        paradigm = self.raw_data.get("paradigm")
        tree = config_tree()
        if tree and isinstance(paradigm, str) and paradigm in tree.paradigms:
            self._pin_fields_the_paradigm_forbids(tree.paradigms[paradigm])

    def _drop_empty_methods(self):
        """Remove a methods block with nothing in it.

        Preserves methods if it has non-empty universal params or concrete methods.
        """
        methods = self.raw_data.get("methods")
        if not isinstance(methods, dict):
            return
        if "params" in methods and not methods["params"]:
            methods.pop("params", None)
        if not methods:
            self.raw_data.pop("methods", None)

    def save_to_disk(self, show_error: bool = False):
        """Save current YAML dictionary back to disk."""
        if not self.current_path:
            return False
        try:
            self._clean_redundant_params()
            self._reconcile_group_base()
            self._drop_empty_methods()
            yaml_str = yaml.safe_dump(self.raw_data, sort_keys=False)
            self.current_path.write_text(self._preamble + yaml_str, encoding="utf-8")
            self.is_dirty = False
            self._update_dirty_ui()
            if self.current_rel_path:
                self.dirty_state_changed.emit(self.current_rel_path, False)
            invalidate_config_tree()
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
            value = self._get_explicit_env() if group == "env" else data.get(group)
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
        if "tensorboard" in data:
            overrides.append(f"tensorboard={str(data['tensorboard']).lower()}")
        if "save_dataset" in data and data["save_dataset"]:
            overrides.append(f"save_dataset={str(data['save_dataset']).lower()}")
        if "recover" in data and data["recover"]:
            overrides.append(f"recover={str(data['recover']).lower()}")
        if "no_plot" in data and data["no_plot"]:
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
