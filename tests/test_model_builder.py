"""Tests for the visual model builder.

The graph rules and the saved format are tested headlessly, since graph.py
imports no Qt. The dialog tests cover the parts that are easy to get wrong in
a canvas: that save is gated on a valid stack, that blocks carry the running
shape, and that reopening a saved design rebuilds the same chain.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

from src.plugins.app.bundled.model_builder.graph import ModelGraph, model_config_yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtWidgets import QApplication, QDialogButtonBox

    _app = QApplication.instance() or QApplication(sys.argv[:1])
    from src.plugins.app.bundled.model_builder.dialog import ModelBuilderDialog

    HAS_QT = True
except ImportError:  # pragma: no cover - exercised only without a GUI build
    HAS_QT = False


def chain(graph: ModelGraph, *specs: tuple[str, dict]) -> ModelGraph:
    """Wire the given layers into a line after the input block."""
    previous = graph.input_node
    assert previous is not None
    for kind, params in specs:
        node = graph.add_node(kind)
        node.params.update(params)
        graph.connect(previous.node_id, node.node_id)
        previous = node
    return graph


DENSE = (("linear", {"out_features": 64}), ("relu", {}), ("linear", {"out_features": 1}))


# ── Graph rules ───────────────────────────────────────────────────────────


def test_a_new_graph_has_only_an_input_block():
    graph = ModelGraph((30,))
    assert len(graph.nodes) == 1
    assert graph.input_node is not None
    assert not graph.is_valid()
    assert "nothing is connected" in graph.problems()[0]


def test_a_wired_chain_is_valid_and_ordered():
    graph = chain(ModelGraph((30,)), *DENSE)
    assert graph.is_valid()
    assert [layer["type"] for layer in graph.layers()] == ["linear", "relu", "linear"]
    assert graph.output_shape() == (1,)


def test_new_nodes_carry_the_layer_defaults():
    graph = ModelGraph((30,))
    assert graph.add_node("dropout").params == {"p": 0.1}
    # stride defaults to None, which is dropped rather than written out.
    assert "stride" not in graph.add_node("maxpool2d").params


def test_a_second_input_block_is_refused():
    with pytest.raises(ValueError, match="already has an input"):
        ModelGraph((30,)).add_node("input")


def test_unknown_layer_kinds_are_refused():
    with pytest.raises(ValueError, match="unknown layer kind"):
        ModelGraph((30,)).add_node("attention")


def test_a_layer_cannot_feed_itself():
    graph = ModelGraph((30,))
    node = graph.add_node("relu")
    assert "cannot feed itself" in (graph.can_connect(node.node_id, node.node_id) or "")


def test_the_input_block_takes_no_incoming_edge():
    graph = ModelGraph((30,))
    node = graph.add_node("relu")
    assert "no incoming" in (graph.can_connect(node.node_id, graph.input_node.node_id) or "")


def test_a_fork_is_refused_because_the_spec_is_a_chain():
    graph = ModelGraph((30,))
    a, b = graph.add_node("relu"), graph.add_node("tanh")
    graph.connect(graph.input_node.node_id, a.node_id)
    assert "single line" in (graph.can_connect(graph.input_node.node_id, b.node_id) or "")


def test_a_second_incoming_edge_is_refused():
    graph = ModelGraph((30,))
    a, b = graph.add_node("relu"), graph.add_node("tanh")
    graph.connect(graph.input_node.node_id, a.node_id)
    assert "single line" in (graph.can_connect(b.node_id, a.node_id) or "")


def test_a_loop_is_refused():
    graph = ModelGraph((30,))
    a, b = graph.add_node("relu"), graph.add_node("tanh")
    graph.connect(a.node_id, b.node_id)
    assert "loop" in (graph.can_connect(b.node_id, a.node_id) or "")


def test_removing_a_node_drops_its_edges():
    graph = chain(ModelGraph((30,)), *DENSE)
    victim = graph.chain().nodes[1]
    graph.remove_node(victim.node_id)
    assert victim.node_id not in graph.nodes
    assert all(victim.node_id not in edge for edge in graph.edges)


def test_the_input_block_cannot_be_removed():
    graph = ModelGraph((30,))
    graph.remove_node(graph.input_node.node_id)
    assert graph.input_node is not None


def test_detached_blocks_are_reported_and_left_out():
    graph = chain(ModelGraph((30,)), *DENSE)
    graph.add_node("dropout")
    assert len(graph.layers()) == 3
    assert any("not connected to the stack" in p for p in graph.problems())


def test_shape_errors_name_the_layer():
    graph = chain(ModelGraph((1, 8, 8)), ("linear", {"out_features": 10}))
    assert any("flatten" in p for p in graph.problems())
    assert graph.output_shape() is None


def test_plans_are_keyed_by_node():
    graph = chain(ModelGraph((30,)), *DENSE)
    by_node = graph.plan_by_node()
    first = graph.chain().nodes[0]
    assert by_node[first.node_id].out_shape == (64,)


def test_tidy_stacks_the_chain_into_a_column():
    graph = chain(ModelGraph((30,)), *DENSE)
    for node in graph.nodes.values():
        node.x, node.y = 500.0, 500.0
    graph.tidy()
    column = [graph.input_node, *graph.chain().nodes]
    assert [n.x for n in column] == [0.0] * 4
    assert [n.y for n in column] == sorted(n.y for n in column)


# ── Saved format ──────────────────────────────────────────────────────────


def test_round_trip_through_layers():
    graph = chain(ModelGraph((30,)), *DENSE)
    assert ModelGraph.from_layers(graph.layers(), (30,)).layers() == graph.layers()


def test_layout_covers_the_input_block_and_every_layer():
    graph = chain(ModelGraph((30,)), *DENSE)
    assert len(graph.layout()) == len(graph.layers()) + 1


def test_layout_round_trips_through_from_layers():
    graph = chain(ModelGraph((30,)), *DENSE)
    for index, node in enumerate(graph.chain().nodes):
        node.x, node.y = 300.0 + index * 40, 500.0 - index * 70
    placed = graph.layout()
    reopened = ModelGraph.from_layers(graph.layers(), (30,), placed)
    assert reopened.layout() == placed


def test_a_missing_layout_falls_back_to_a_column():
    graph = chain(ModelGraph((30,)), *DENSE)
    reopened = ModelGraph.from_layers(graph.layers(), (30,), None)
    assert [position[0] for position in reopened.layout()] == [0, 0, 0, 0]


@pytest.mark.parametrize(
    "stale",
    [
        [[0, 0], [9, 9]],  # too few, as if a layer were added by hand
        [[0, 0]] * 9,  # too many, as if layers were deleted by hand
        [[0, 0], "nonsense", [1, 1], [2, 2]],
        [[0], [1], [2], [3]],
        [[0, 0], [1, 1], [2, 2], [True, False]],
    ],
)
def test_a_layout_that_does_not_fit_is_ignored(stale):
    # Pairing blocks with stale positions is worse than losing the layout.
    graph = chain(ModelGraph((30,)), *DENSE)
    reopened = ModelGraph.from_layers(graph.layers(), (30,), stale)
    assert [position[0] for position in reopened.layout()] == [0, 0, 0, 0]


def test_yaml_carries_the_layout_and_parses_back():
    import yaml

    graph = chain(ModelGraph((30,)), *DENSE)
    graph.chain().nodes[0].x = 275.0
    parsed = yaml.safe_load(model_config_yaml(graph.layers(), "X", (30,), graph.layout()))
    assert parsed["builder_layout"] == graph.layout()


def test_yaml_omits_the_layout_when_there_is_none():
    assert "builder_layout" not in model_config_yaml([{"type": "relu"}], "X", (30,), None)


def test_from_layers_skips_entries_it_cannot_place():
    graph = ModelGraph.from_layers([{"type": "relu"}, "nonsense", {"type": "attention"}], (30,))
    assert [layer["type"] for layer in graph.layers()] == ["relu"]


def test_yaml_declares_the_spec_architecture():
    text = model_config_yaml([{"type": "relu"}], "example_package.ClassifierModule")
    assert "architecture: sequential_spec" in text
    assert "lightning_module: example_package.ClassifierModule" in text


def test_yaml_states_input_shape_only_when_unflattened():
    flat = model_config_yaml([{"type": "relu"}], None, (30,))
    image = model_config_yaml([{"type": "relu"}], None, (1, 8, 8))
    assert "input_shape" not in flat
    assert "input_shape: [1, 8, 8]" in image


def test_yaml_omits_unset_optional_parameters():
    text = model_config_yaml([{"type": "maxpool2d", "kernel_size": 2, "stride": None}])
    assert "kernel_size: 2" in text
    assert "stride" not in text


def test_yaml_writes_booleans_lowercase():
    text = model_config_yaml([{"type": "linear", "out_features": 8, "bias": False}])
    assert "bias: false" in text


def test_saved_yaml_parses_back_to_the_same_layers():
    import yaml

    graph = chain(ModelGraph((30,)), *DENSE)
    parsed = yaml.safe_load(model_config_yaml(graph.layers(), "X", (30,)))
    assert parsed["layers"] == graph.layers()
    assert parsed["architecture"] == "sequential_spec"


# ── The dialog ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not installed")
class TestModelBuilderDialog:
    def _dialog(self, shape=(30,), layers=None):
        self.tmp = Path(tempfile.mkdtemp(prefix="theta_builder_"))
        return ModelBuilderDialog(model_dir=self.tmp, input_shape=shape, initial_layers=layers)

    def _save_button(self, dialog):
        return dialog.buttons.button(QDialogButtonBox.StandardButton.Save)

    def test_save_is_disabled_until_the_stack_is_valid(self):
        dialog = self._dialog()
        assert not self._save_button(dialog).isEnabled()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        assert self._save_button(dialog).isEnabled()

    def test_blocks_are_labelled_with_the_running_shape(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        labels = {i.node.kind: i.shape_label for i in dialog.canvas._items.values()}
        assert labels["input"] == "(B, 30)"
        assert "(B, 64)" in labels["relu"]

    def test_a_broken_block_is_flagged_rather_than_hidden(self):
        dialog = self._dialog(shape=(1, 8, 8))
        chain(dialog.graph, ("linear", {"out_features": 10}))
        dialog.canvas.rebuild()
        broken = [i for i in dialog.canvas._items.values() if i.error]
        assert len(broken) == 1
        assert "flatten" in broken[0].error
        assert not self._save_button(dialog).isEnabled()

    def test_editing_the_input_shape_repropagates(self):
        dialog = self._dialog()
        chain(dialog.graph, ("flatten", {}), ("linear", {"out_features": 4}))
        dialog.canvas.rebuild()
        dialog.txt_shape.setText("1, 8, 8")
        dialog._on_shape_edited()
        assert dialog.graph.input_shape == (1, 8, 8)
        assert dialog.graph.output_shape() == (4,)

    def test_a_nonsense_input_shape_is_refused_not_applied(self):
        dialog = self._dialog()
        dialog.txt_shape.setText("banana")
        dialog._on_shape_edited()
        assert dialog.graph.input_shape == (30,)

    def test_saving_writes_a_loadable_config(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        dialog.txt_name.setText("saved_net")
        dialog.txt_module.setText("example_package.TrainingModule")
        dialog._on_save()
        assert dialog.saved_name == "saved_net"
        assert "architecture: sequential_spec" in (self.tmp / "saved_net.yaml").read_text()

    def test_an_invalid_name_is_refused(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        dialog.txt_name.setText("bad name/slash")
        dialog.txt_module.setText("example_package.TrainingModule")
        dialog._on_save()
        assert dialog.saved_name is None

    def test_reopening_a_saved_design_rebuilds_the_chain(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        reopened = self._dialog(layers=dialog.graph.layers())
        assert reopened.graph.layers() == dialog.graph.layers()
        assert self._save_button(reopened).isEnabled()

    def test_selecting_a_block_shows_its_parameters(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        target = dialog.graph.chain().nodes[0]
        dialog._show_params(target)
        assert dialog.params_form.rowCount() > 1

    def test_saving_records_where_the_blocks_were_put(self):
        import yaml

        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        moved = dialog.graph.chain().nodes[0]
        moved.x, moved.y = 320.0, 480.0
        dialog.txt_name.setText("placed")
        dialog.txt_module.setText("example_package.TrainingModule")
        dialog._on_save()
        saved = yaml.safe_load((self.tmp / "placed.yaml").read_text())
        assert [320, 480] in saved["builder_layout"]

    def test_reopening_keeps_the_saved_positions(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        for index, node in enumerate(dialog.graph.chain().nodes):
            node.x, node.y = 300.0 + index * 40, 500.0 - index * 70
        placed = dialog.graph.layout()

        reopened = ModelBuilderDialog(
            model_dir=self.tmp,
            input_shape=(30,),
            initial_layers=dialog.graph.layers(),
            initial_layout=placed,
        )
        # The opening tidy must not run, or the restored positions are lost.
        assert reopened.graph.layout() == placed
        assert reopened._restored_layout

    def test_reopening_without_a_layout_still_tidies(self):
        dialog = self._dialog()
        chain(dialog.graph, *DENSE)
        dialog.canvas.rebuild()
        reopened = ModelBuilderDialog(model_dir=self.tmp, input_shape=(30,), initial_layers=dialog.graph.layers())
        assert not reopened._restored_layout
        assert [position[0] for position in reopened.graph.layout()] == [0, 0, 0, 0]


# ── The plugin and the slot it fills ──────────────────────────────────────


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not installed")
class TestModelBuilderPlugin:
    """The GUI half is an app plugin; the architecture it writes is a component."""

    def test_the_slot_contribution_knows_which_method_its_row_serves(self):
        from PyQt6.QtWidgets import QPushButton

        from src.plugins.app.bundled.model_builder import ACTION_ID, SLOT_ID, ModelBuilderPlugin
        from frontend.plugins.slots import ExtensionSlot, ExtensionSlotWidget

        slot = ExtensionSlot(SLOT_ID)
        plugin = ModelBuilderPlugin(manifest=None)

        class Ctx:
            def contribute_to_slot(self, slot_id, action_id, factory_or_widget, priority=100):
                slot.add_action(action_id, factory_or_widget, priority)

            def log(self, message):
                pass

        plugin.activate(Ctx())
        assert [a for a, _ in slot.get_items()] == [ACTION_ID]

        # One row per method, each carrying its own context.
        seen = []
        for method in ("clf", "ae"):
            widget = ExtensionSlotWidget(
                slot, context={"method": method, "model": "", "paradigm": "supervised", "apply": seen.append}
            )
            buttons = widget.findChildren(QPushButton)
            assert [b.text() for b in buttons] == ["Design..."]

    def test_a_factory_taking_no_context_still_works(self):
        # Existing plugins contribute zero-argument factories.
        from PyQt6.QtWidgets import QPushButton

        from frontend.plugins.slots import ExtensionSlot, ExtensionSlotWidget

        slot = ExtensionSlot("legacy")
        slot.add_action("plain", lambda: QPushButton("Plain"))
        widget = ExtensionSlotWidget(slot, context={"method": "clf"})
        assert [b.text() for b in widget.findChildren(QPushButton)] == ["Plain"]

    def test_deactivate_removes_the_contribution(self):
        from src.plugins.app.bundled.model_builder import ACTION_ID, SLOT_ID, ModelBuilderPlugin
        from frontend.plugins.slots import ExtensionSlot

        slot = ExtensionSlot(SLOT_ID)
        removed = []

        class Ctx:
            def contribute_to_slot(self, slot_id, action_id, factory_or_widget, priority=100):
                slot.add_action(action_id, factory_or_widget, priority)

            def remove_from_slot(self, slot_id, action_id):
                removed.append((slot_id, action_id))
                slot.remove_action(action_id)

            def log(self, message):
                pass

        plugin = ModelBuilderPlugin(manifest=None)
        plugin.activate(Ctx())
        plugin.deactivate()
        assert removed == [(SLOT_ID, ACTION_ID)]
        assert slot.get_items() == []
