"""The model builder window: palette, canvas, parameters, save.

Saving writes an ordinary `in/config/model/<name>.yaml`. That directory is
already scanned to fill the method's model combo, so a design shows up as a
selectable architecture without the panel knowing this dialog exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from src.plugins.components.bundled.sequential_spec.sequential_spec import LAYER_KINDS
from frontend.theme import theme_color
from frontend.widgets import SpinBox as QSpinBox, DoubleSpinBox as QDoubleSpinBox
from .canvas import GraphCanvas, LayerPalette
from .graph import GraphNode, ModelGraph, model_config_yaml


def _column(graph: ModelGraph) -> list[list[int]]:
    """What this graph's layout would be if it were tidied into a column."""
    saved = [(n.x, n.y) for n in graph.nodes.values()]
    graph.tidy()
    column = graph.layout()
    for node, (x, y) in zip(graph.nodes.values(), saved):
        node.x, node.y = x, y
    return column


def _parse_scalar(text: str) -> Any:
    """Read a field back as a number, a pair, a bool, or None when blank."""
    raw = text.strip()
    if not raw:
        return None
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        parts = [p.strip() for p in raw[1:-1].split(",") if p.strip()]
        return [_parse_scalar(p) for p in parts]
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


class ModelBuilderDialog(QDialog):
    """Design a network by wiring blocks, then save it as a model config."""

    def __init__(
        self,
        model_dir: Path,
        input_shape: tuple[int, ...] = (30,),
        lightning_module: str | None = None,
        initial_layers: list[dict[str, Any]] | None = None,
        initial_layout: Any = None,
        initial_name: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Model Builder")
        self.resize(1040, 720)
        self.setMinimumSize(820, 560)

        self.model_dir = Path(model_dir)
        self.saved_name: str | None = None
        self.graph = (
            ModelGraph.from_layers(initial_layers, input_shape, initial_layout)
            if initial_layers
            else ModelGraph(input_shape)
        )
        # from_layers falls back to a column when the saved layout does not fit,
        # so compare against that column to tell a restored layout from a
        # defaulted one. Only a restored one should survive the opening tidy.
        self._restored_layout = bool(initial_layers) and self.graph.layout() != _column(self.graph)

        self.canvas = GraphCanvas(self.graph)
        self.canvas.refuse = self._show_refusal  # type: ignore[method-assign]
        self.canvas.graph_changed.connect(self._refresh_status)
        self.canvas.selection_changed.connect(self._show_params)

        self.palette = LayerPalette()
        self.palette.setMaximumWidth(190)

        self._build_ui(lightning_module, initial_name)
        if self._restored_layout:
            self.canvas.fit()
        else:
            self.canvas.tidy()
        self._refresh_status()
        self._show_params(None)

    # ── Layout ───────────────────────────────────────────────────────────

    def _build_ui(self, lightning_module: str | None, initial_name: str) -> None:
        root = QVBoxLayout(self)

        hint = QLabel(
            "Drag a layer onto the canvas. Drag from the dot at the bottom of a block "
            "to the dot on top of another to connect them; click a connected input dot "
            "to detach it. Delete removes the selected block."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {theme_color('muted')};")
        root.addWidget(hint)

        self.params_host = QWidget()
        self.params_form = QFormLayout(self.params_host)
        self.params_form.setContentsMargins(10, 10, 10, 10)
        self.params_host.setMaximumWidth(260)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.palette)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.params_host)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, 1)

        # Input shape, so shapes preview against the dataset being targeted.
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("Input shape:"))
        self.txt_shape = QLineEdit(", ".join(str(d) for d in self.graph.input_shape))
        self.txt_shape.setMaximumWidth(140)
        self.txt_shape.setToolTip(
            "The shape of one sample, without the batch dimension.\n"
            "30 for breast_cancer's features; 1, 8, 8 to treat digits as images."
        )
        self.txt_shape.editingFinished.connect(self._on_shape_edited)
        shape_row.addWidget(self.txt_shape)

        shape_row.addSpacing(16)
        shape_row.addWidget(QLabel("Trainer module:"))
        self.txt_module = QLineEdit(lightning_module or "")
        self.txt_module.setPlaceholderText("package.TrainingModule")
        self.txt_module.setMaximumWidth(280)
        self.txt_module.setToolTip("The Lightning module the paradigm should wrap this model in.")
        shape_row.addWidget(self.txt_module)

        shape_row.addStretch()
        btn_tidy = QPushButton("Tidy layout")
        btn_tidy.clicked.connect(lambda: self.canvas.tidy())
        shape_row.addWidget(btn_tidy)
        root.addLayout(shape_row)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Save as:"))
        self.txt_name = QLineEdit(initial_name)
        self.txt_name.setPlaceholderText("my_model")
        self.txt_name.setMaximumWidth(260)
        self.txt_name.setToolTip("Written to in/config/model/<name>.yaml")
        name_row.addWidget(self.txt_name)
        name_row.addStretch()
        root.addLayout(name_row)

        self.lbl_status = QLabel()
        self.lbl_status.setWordWrap(True)
        root.addWidget(self.lbl_status)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self._on_save)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

    # ── Status ───────────────────────────────────────────────────────────

    def _refresh_status(self) -> None:
        problems = self.graph.problems()
        save = self.buttons.button(QDialogButtonBox.StandardButton.Save)
        if problems:
            self.lbl_status.setText("· " + "\n· ".join(problems[:4]))
            self.lbl_status.setStyleSheet("color: #fb4934;")
            if save:
                save.setEnabled(False)
            return

        out = self.graph.output_shape()
        shape = "(B, " + ", ".join(str(d) for d in (out or ())) + ")"
        squeezed = " squeezed to (B,) for a binary task" if out == (1,) else ""
        self.lbl_status.setText(f"{len(self.graph.layers())} layers, output {shape}{squeezed}.")
        self.lbl_status.setStyleSheet(f"color: {theme_color('primary')};")
        if save:
            save.setEnabled(True)

    def _show_refusal(self, reason: str) -> None:
        self.lbl_status.setText(f"· {reason}")
        self.lbl_status.setStyleSheet("color: #fb4934;")

    def _on_shape_edited(self) -> None:
        parts = [p.strip() for p in self.txt_shape.text().replace("(", "").replace(")", "").split(",")]
        try:
            shape = tuple(int(p) for p in parts if p)
        except ValueError:
            shape = ()
        if not shape or any(d <= 0 for d in shape):
            self._show_refusal("input shape must be one or more positive whole numbers, for example 1, 8, 8")
            return
        self.graph.input_shape = shape
        self.canvas.refresh_labels()

    # ── Parameter editor ─────────────────────────────────────────────────

    def _show_params(self, node: GraphNode | None) -> None:
        while self.params_form.rowCount():
            self.params_form.removeRow(0)

        if node is None:
            label = QLabel("Select a block to edit its parameters.")
            label.setWordWrap(True)
            label.setStyleSheet(f"color: {theme_color('muted')};")
            self.params_form.addRow(label)
            return

        title = QLabel("input" if node.is_input else node.kind)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        self.params_form.addRow(title)

        if node.is_input:
            note = QLabel("Set the dataset shape in the Input shape field below.")
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {theme_color('muted')};")
            self.params_form.addRow(note)
            return

        kind = LAYER_KINDS[node.kind]
        doc = QLabel(kind.doc)
        doc.setWordWrap(True)
        doc.setStyleSheet(f"color: {theme_color('muted')};")
        self.params_form.addRow(doc)

        if not kind.params:
            none = QLabel("No parameters.")
            none.setStyleSheet(f"color: {theme_color('comment')};")
            self.params_form.addRow(none)
            return

        for key, default in kind.params.items():
            current = node.params.get(key, default)
            self.params_form.addRow(key, self._param_widget(node, key, current, default))

    def _param_widget(self, node: GraphNode, key: str, current: Any, default: Any) -> QWidget:
        """A typed editor for one parameter, chosen from its default's type."""

        def commit(value: Any) -> None:
            if value is None:
                node.params.pop(key, None)
            else:
                node.params[key] = value
            self.canvas.refresh_labels()

        # bool before int: bool is a subclass of int, so the order matters.
        if isinstance(default, bool):
            box = QCheckBox()
            box.setChecked(bool(current))
            box.toggled.connect(commit)
            return box

        if isinstance(default, int):
            spin = QSpinBox()
            spin.setRange(1, 16384)
            spin.setValue(int(current) if isinstance(current, (int, float)) else int(default))
            spin.valueChanged.connect(commit)
            return spin

        if isinstance(default, float):
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 1000.0)
            spin.setDecimals(3)
            spin.setSingleStep(0.05)
            spin.setValue(float(current) if isinstance(current, (int, float)) else float(default))
            spin.valueChanged.connect(commit)
            return spin

        # Optional or compound values (a blank stride, a [3, 5] kernel).
        edit = QLineEdit("" if current is None else str(current))
        edit.setPlaceholderText("same as kernel_size" if key == "stride" else "")
        edit.textChanged.connect(lambda text: commit(_parse_scalar(text)))
        return edit

    # ── Saving ───────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        # Reported on the status line rather than in a modal: it sits directly
        # under the field being complained about, and a popup over a canvas
        # hides the thing the message is about.
        name = self.txt_name.text().strip()
        if not name:
            self._show_refusal("give the model a name to save it under")
            self.txt_name.setFocus()
            return
        if not all(ch.isalnum() or ch in "_-" for ch in name):
            self._show_refusal("the name becomes a file name, so use letters, numbers, dashes and underscores")
            self.txt_name.setFocus()
            return
        if not self.txt_module.text().strip():
            self._show_refusal("enter the Lightning module exported by the model component")
            self.txt_module.setFocus()
            return
        if self.graph.problems():
            self._refresh_status()
            return

        target = self.model_dir / f"{name}.yaml"
        if target.exists():
            answer = QMessageBox.question(
                self,
                "Replace model?",
                f"{name}.yaml already exists. Replace it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        text = model_config_yaml(
            self.graph.layers(),
            self.txt_module.text().strip(),
            self.graph.input_shape,
            self.graph.layout(),
        )
        try:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Could not save", f"Writing {target} failed:\n{exc}")
            return

        self.saved_name = name
        self.accept()
