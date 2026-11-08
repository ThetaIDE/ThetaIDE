"""The node graph behind the visual model builder, with no Qt in it.

Kept free of Qt imports for the same reason config_model.py is: the rules
about what connects to what, which order the layers end up in, and what the
saved YAML looks like are the parts worth testing, and they run headless in
CI. The canvas in canvas.py is a view over this.

The spec this feeds, `sequential_spec`, is a straight chain, so the graph
enforces one incoming and one outgoing edge per node. That is the only reason
branching is rejected; the structure here already carries arbitrary edges, so
lifting the restriction later is a change to `chain()`, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.plugins.components.bundled.sequential_spec.sequential_spec import LAYER_KINDS, LayerPlan, plan_shapes

INPUT_KIND = "input"

# Laying the chain out in a column, in scene units.
NODE_SPACING_Y = 110.0
NODE_X = 0.0


@dataclass
class GraphNode:
    """One block on the canvas: a layer, or the dataset input."""

    node_id: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    x: float = NODE_X
    y: float = 0.0

    @property
    def is_input(self) -> bool:
        return self.kind == INPUT_KIND

    def spec(self) -> dict[str, Any]:
        """This node as a `layers:` entry."""
        return {"type": self.kind, **self.params}


class ConnectionError_(ValueError):
    """A proposed edge would break the chain."""


@dataclass
class ChainResult:
    """The ordered layer nodes, plus anything wrong with the graph.

    `problems` are structural (no input node, a fork, a node wired to
    nothing); per-layer shape errors come back separately through `plans`,
    because a stack can be correctly wired and still shaped wrong.
    """

    nodes: list[GraphNode]
    problems: list[str] = field(default_factory=list)
    orphans: list[GraphNode] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _layout_fits(layout: Any, count: int) -> bool:
    """Whether *layout* is a usable set of positions for *count* blocks."""
    if not isinstance(layout, (list, tuple)) or len(layout) != count:
        return False
    return all(
        isinstance(item, (list, tuple))
        and len(item) == 2
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in item)
        for item in layout
    )


class ModelGraph:
    """Nodes, edges, and the rules that keep them a single chain."""

    def __init__(self, input_shape: tuple[int, ...] = (30,)):
        self.input_shape: tuple[int, ...] = tuple(int(d) for d in input_shape)
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[tuple[str, str]] = []
        self._next_id = 0
        self.add_node(INPUT_KIND)

    # ── Nodes ────────────────────────────────────────────────────────────

    def _new_id(self, kind: str) -> str:
        self._next_id += 1
        return f"{kind}_{self._next_id}"

    def add_node(self, kind: str, x: float | None = None, y: float | None = None) -> GraphNode:
        """Add a node, seeded with the layer kind's default parameters."""
        if kind != INPUT_KIND and kind not in LAYER_KINDS:
            raise ValueError(f"unknown layer kind {kind!r}")
        if kind == INPUT_KIND and self.input_node is not None:
            raise ValueError("the graph already has an input node")

        defaults = dict(LAYER_KINDS[kind].params) if kind != INPUT_KIND else {}
        node = GraphNode(
            node_id=self._new_id(kind),
            kind=kind,
            params={k: v for k, v in defaults.items() if v is not None},
            x=NODE_X if x is None else x,
            y=(len(self.nodes) * NODE_SPACING_Y) if y is None else y,
        )
        self.nodes[node.node_id] = node
        return node

    def remove_node(self, node_id: str) -> None:
        """Remove a node and every edge touching it. The input node stays."""
        node = self.nodes.get(node_id)
        if node is None or node.is_input:
            return
        del self.nodes[node_id]
        self.edges = [(a, b) for a, b in self.edges if a != node_id and b != node_id]

    @property
    def input_node(self) -> GraphNode | None:
        return next((n for n in self.nodes.values() if n.is_input), None)

    # ── Edges ────────────────────────────────────────────────────────────

    def can_connect(self, src_id: str, dst_id: str) -> str | None:
        """Why this edge is not allowed, or None when it is."""
        if src_id == dst_id:
            return "a layer cannot feed itself"
        if src_id not in self.nodes or dst_id not in self.nodes:
            return "that node is gone"
        if self.nodes[dst_id].is_input:
            return "the input block takes no incoming connection"
        if (src_id, dst_id) in self.edges:
            return "those are already connected"
        if any(a == src_id for a, _ in self.edges):
            return "that output is already connected: a stack runs in a single line"
        if any(b == dst_id for _, b in self.edges):
            return "that input is already connected: a stack runs in a single line"
        if self._reaches(dst_id, src_id):
            return "that would make a loop"
        return None

    def _reaches(self, start: str, target: str) -> bool:
        seen: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            stack.extend(b for a, b in self.edges if a == current)
        return False

    def connect(self, src_id: str, dst_id: str) -> None:
        reason = self.can_connect(src_id, dst_id)
        if reason:
            raise ConnectionError_(reason)
        self.edges.append((src_id, dst_id))

    def disconnect(self, src_id: str, dst_id: str) -> None:
        self.edges = [e for e in self.edges if e != (src_id, dst_id)]

    # ── Reading the chain ────────────────────────────────────────────────

    def chain(self) -> ChainResult:
        """Walk the graph from the input node and return the layers in order."""
        start = self.input_node
        if start is None:
            return ChainResult([], ["the graph has no input block"])

        ordered: list[GraphNode] = []
        seen = {start.node_id}
        current = start.node_id
        while True:
            nxt = [b for a, b in self.edges if a == current]
            if not nxt:
                break
            if len(nxt) > 1:
                return ChainResult(ordered, ["a layer feeds more than one place: a stack runs in a single line"])
            current = nxt[0]
            if current in seen:
                return ChainResult(ordered, ["the connections form a loop"])
            seen.add(current)
            ordered.append(self.nodes[current])

        orphans = [n for n in self.nodes.values() if n.node_id not in seen]
        problems: list[str] = []
        if not ordered:
            problems.append("nothing is connected to the input block yet")
        if orphans:
            names = ", ".join(n.kind for n in orphans[:3])
            more = f" and {len(orphans) - 3} more" if len(orphans) > 3 else ""
            problems.append(f"not connected to the stack, so it will not be saved: {names}{more}")
        return ChainResult(ordered, problems, orphans)

    def layers(self) -> list[dict[str, Any]]:
        """The `layers:` list the chain describes, ready for YAML."""
        return [node.spec() for node in self.chain().nodes]

    def plans(self) -> list[LayerPlan]:
        """Per-layer shapes for the chain, for the canvas to label nodes with."""
        return plan_shapes(self.layers(), self.input_shape)

    def plan_by_node(self) -> dict[str, LayerPlan]:
        """Shape plans keyed by node id, so a node can show its own shape."""
        return {node.node_id: plan for node, plan in zip(self.chain().nodes, self.plans())}

    def output_shape(self) -> tuple[int, ...] | None:
        plans = self.plans()
        if not plans:
            return self.input_shape
        last = plans[-1]
        return last.out_shape if last.ok else None

    def is_valid(self) -> bool:
        """True when the graph is wired correctly and every shape lines up."""
        return self.chain().ok and all(p.ok for p in self.plans())

    def problems(self) -> list[str]:
        """Everything wrong with the graph, structural first then shapes."""
        found = list(self.chain().problems)
        for plan in self.plans():
            if not plan.ok:
                found.append(f"layer {plan.index + 1} ({plan.type}): {plan.error}")
        return found

    # ── Layout and persistence ───────────────────────────────────────────

    def layout(self) -> list[list[int]]:
        """Scene positions for the input block and each layer, in chain order.

        Positional rather than keyed by node id: ids are generated per session
        and mean nothing in a saved file, while the chain order is something
        both sides already agree on.
        """
        nodes = ([self.input_node] if self.input_node else []) + self.chain().nodes
        return [[round(n.x), round(n.y)] for n in nodes]

    def tidy(self) -> None:
        """Stack the chain into a single column, orphans off to the side."""
        chain = self.chain()
        start = self.input_node
        column = ([start] if start else []) + chain.nodes
        for row, node in enumerate(column):
            node.x, node.y = NODE_X, row * NODE_SPACING_Y
        for row, node in enumerate(chain.orphans):
            node.x, node.y = NODE_X + 280.0, row * NODE_SPACING_Y

    @classmethod
    def from_layers(
        cls,
        layers: list[dict[str, Any]],
        input_shape: tuple[int, ...] = (30,),
        layout: Any = None,
    ) -> ModelGraph:
        """Build a chain from a saved `layers:` list, restoring its layout.

        A *layout* that does not describe exactly this many blocks is ignored
        and the stack is laid out in a column instead. That covers a
        hand-written config, which has no layout at all, and keeps a
        hand-edited `layers:` list from pairing blocks with stale positions.
        """
        graph = cls(input_shape)
        previous = graph.input_node
        assert previous is not None
        for spec in layers:
            if not isinstance(spec, dict):
                continue
            kind = str(spec.get("type", "")).lower().strip()
            if kind not in LAYER_KINDS:
                continue
            node = graph.add_node(kind)
            node.params = {k: v for k, v in spec.items() if k != "type"}
            graph.connect(previous.node_id, node.node_id)
            previous = node

        ordered = ([graph.input_node] if graph.input_node else []) + graph.chain().nodes
        if _layout_fits(layout, len(ordered)):
            for node, position in zip(ordered, layout):
                node.x, node.y = float(position[0]), float(position[1])
        else:
            graph.tidy()
        return graph


# ── Saving ────────────────────────────────────────────────────────────────


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_scalar(v) for v in value) + "]"
    return str(value)


def model_config_yaml(
    layers: list[dict[str, Any]],
    lightning_module: str | None = None,
    input_shape: tuple[int, ...] | None = None,
    layout: list[list[int]] | None = None,
) -> str:
    """The text of an `in/config/model/<name>.yaml` for this stack.

    Written by hand rather than dumped so the file reads like the
    hand-written model configs beside it, keeps `layers` in order, and can
    carry the comment explaining where it came from.
    """
    lines = [
        "# Written by the visual model builder. The architecture is the `layers:`",
        "# list below rather than a Python class, so build_model resolves it",
        "# through the one spec-driven module instead of a named architecture.",
        "architecture: sequential_spec",
    ]
    if lightning_module:
        lines += [
            "# Resolved through the component registry. The model package exports this module.",
            f"lightning_module: {lightning_module}",
        ]
    # Only a non-flat input needs stating: a flat width arrives as input_dim.
    if input_shape is not None and len(input_shape) > 1:
        lines += [
            "# A convolutional stack needs its unflattened shape; loaders hand over",
            "# flat features, which the model reshapes to this before the first layer.",
            f"input_shape: {list(input_shape)}",
        ]

    lines.append("layers:")
    if not layers:
        lines.append("  []")
    for spec in layers:
        kind = spec.get("type")
        lines.append(f"  - type: {kind}")
        for key, value in spec.items():
            if key == "type" or value is None:
                continue
            lines.append(f"    {key}: {_scalar(value)}")

    if layout:
        lines += [
            "",
            "# Where the blocks sit on the builder canvas, one [x, y] per block",
            "# starting with the input. Training ignores it; delete it and the",
            "# builder lays the stack out in a column again.",
            "builder_layout:",
        ]
        lines += [f"  - {_scalar(list(position))}" for position in layout]
    return "\n".join(lines) + "\n"
