"""Visual model builder: an app plugin, because it changes the UI.

The architecture it writes, `sequential_spec`, is a separate thing: a model
component the trainer instantiates, with no Qt in it. This half is the
editor, and it reaches the Experiment panel through the
`experiment.method.model_row` extension slot rather than by being wired into
that panel directly.

The slot hands over the row's context, including an `apply` callback, so the
panel stays the only thing that edits its own document.
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtWidgets import QDialog, QPushButton

from frontend.plugins.base import Plugin

SLOT_ID = "experiment.method.model_row"
ACTION_ID = "design_model"


class ModelBuilderPlugin(Plugin):
    """Contributes the Design... button to each method's model row."""

    def __init__(self, manifest):
        super().__init__(manifest)
        self.context = None

    def activate(self, context) -> None:
        self.context = context
        context.contribute_to_slot(
            slot_id=SLOT_ID,
            action_id=ACTION_ID,
            factory_or_widget=self._make_button,
            priority=100,
        )
        context.log("Visual Model Builder activated.")

    def deactivate(self) -> None:
        if self.context is not None:
            self.context.remove_from_slot(SLOT_ID, ACTION_ID)
            self.context = None

    # ── The contribution ─────────────────────────────────────────────────

    def _make_button(self, row: dict) -> QPushButton:
        """One button per model row, bound to that row's method."""
        button = QPushButton("Design...")
        button.setToolTip(
            "Design this architecture by wiring layer blocks together.\nSaves a new architecture into in/config/model/."
        )
        # clicked passes a checked flag, which would land in `row` if the
        # slot took it positionally.
        button.clicked.connect(lambda _checked=False, r=row: self._open(r))
        return button

    def _open(self, row: dict) -> None:
        from .dialog import ModelBuilderDialog

        window = self.context.window if self.context else None
        tree = _config_tree()
        if tree is None:
            if self.context:
                self.context.show_status_message("The config tree is not loaded yet.")
            return

        current = str(row.get("model") or "")
        params = dict(tree.models[current].parameters) if current in tree.models else {}
        layers = params.get("layers") if isinstance(params.get("layers"), list) else None
        shape = params.get("input_shape")
        input_shape = tuple(int(d) for d in shape) if isinstance(shape, (list, tuple)) and shape else (30,)

        declared = params.get("lightning_module")
        if not isinstance(declared, str):
            declared = None

        dialog = ModelBuilderDialog(
            model_dir=tree.config_root / "model",
            input_shape=input_shape,
            lightning_module=declared,
            initial_layers=layers,
            initial_layout=params.get("builder_layout"),
            initial_name=current if layers else "",
            parent=window,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted or not dialog.saved_name:
            return

        apply: Optional[Any] = row.get("apply")
        if callable(apply):
            apply(dialog.saved_name)


def _config_tree():
    """The parsed in/config tree, or None outside a checkout."""
    from frontend.config_viewer import config_tree

    return config_tree()
