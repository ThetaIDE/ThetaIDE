"""String diagram workflow model for Theta-IDE.

Models workflows as monoidal category string diagrams:
- Objects are typed strings/wires carrying artifacts (Dataset, Checkpoint, ValueEstimator, etc.)
- Morphisms are nodes (experiments, paradigms, transformations, evaluators)
- Connections represent typed artifact dataflow and parameter bindings
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class PortType(str, Enum):
    """Artifact types carried by strings (wires) in the workflow diagram."""

    DATASET = "dataset"
    CHECKPOINT = "checkpoint"
    VALUE_ESTIMATOR = "value_estimator"
    METRICS = "metrics"
    GENERIC = "generic"

    @property
    def color(self) -> str:
        """Hex color for rendering ports and bezier strings."""
        colors = {
            PortType.DATASET: "#06b6d4",  # Cyan
            PortType.CHECKPOINT: "#a855f7",  # Purple
            PortType.VALUE_ESTIMATOR: "#f59e0b",  # Amber
            PortType.METRICS: "#10b981",  # Emerald
            PortType.GENERIC: "#ebdbb2",  # Warm Text
        }
        return colors.get(self, "#ebdbb2")

    @property
    def display_name(self) -> str:
        names = {
            PortType.DATASET: "Dataset [Transitions]",
            PortType.CHECKPOINT: "Model Checkpoint",
            PortType.VALUE_ESTIMATOR: "Value Estimator",
            PortType.METRICS: "Evaluation Metrics",
            PortType.GENERIC: "Generic Artifact",
        }
        return names.get(self, self.value)


class PortDirection(str, Enum):
    INPUT = "input"
    OUTPUT = "output"


@dataclass
class Port:
    """A typed communication port on an experiment or task node."""

    name: str
    direction: PortDirection
    port_type: PortType
    description: str = ""
    param_binding: str = ""
    required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "direction": self.direction.value,
            "port_type": self.port_type.value,
            "description": self.description,
            "param_binding": self.param_binding,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Port:
        return cls(
            name=data["name"],
            direction=PortDirection(data.get("direction", "input")),
            port_type=PortType(data.get("port_type", "generic")),
            description=data.get("description", ""),
            param_binding=data.get("param_binding", ""),
            required=bool(data.get("required", False)),
        )


@dataclass
class WorkflowNode:
    """A computation node (morphism) in the workflow diagram."""

    id: str
    label: str
    category: str = "paradigm"  # "paradigm" | "task" | "experiment" | "transform"
    paradigm: str = "online_rl"  # online_rl, offline_rl, supervised, eval, etc.
    experiment_ref: str = ""  # e.g. "group/experiment"
    overrides: list[str] = field(default_factory=list)
    pos_x: float = 0.0
    pos_y: float = 0.0
    inputs: dict[str, Port] = field(default_factory=dict)
    outputs: dict[str, Port] = field(default_factory=dict)

    def add_input(self, port: Port) -> None:
        port.direction = PortDirection.INPUT
        self.inputs[port.name] = port

    def add_output(self, port: Port) -> None:
        port.direction = PortDirection.OUTPUT
        self.outputs[port.name] = port

    def get_port(self, name: str) -> Port | None:
        return self.inputs.get(name) or self.outputs.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "paradigm": self.paradigm,
            "experiment_ref": self.experiment_ref,
            "overrides": list(self.overrides),
            "pos": [round(self.pos_x, 1), round(self.pos_y, 1)],
            "inputs": [p.to_dict() for p in self.inputs.values()],
            "outputs": [p.to_dict() for p in self.outputs.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowNode:
        pos = data.get("pos", [0.0, 0.0])
        node = cls(
            id=data["id"],
            label=data.get("label", data["id"]),
            category=data.get("category", "paradigm"),
            paradigm=data.get("paradigm", "online_rl"),
            experiment_ref=data.get("experiment_ref", ""),
            overrides=list(data.get("overrides", [])),
            pos_x=float(pos[0]) if len(pos) > 0 else 0.0,
            pos_y=float(pos[1]) if len(pos) > 1 else 0.0,
        )
        for inp in data.get("inputs", []):
            node.add_input(Port.from_dict(inp))
        for out in data.get("outputs", []):
            node.add_output(Port.from_dict(out))
        return node


@dataclass
class WorkflowString:
    """A typed connection wire representing artifact transfer between ports."""

    id: str
    source_node_id: str
    source_port_name: str
    target_node_id: str
    target_port_name: str
    port_type: PortType
    param_binding: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "from": f"{self.source_node_id}.{self.source_port_name}",
            "to": f"{self.target_node_id}.{self.target_port_name}",
            "port_type": self.port_type.value,
            "binding": self.param_binding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowString:
        # Supports both verbose and 'from'/'to' compact schema
        wire_id = data.get("id")
        if "from" in data and "to" in data:
            src_parts = data["from"].split(".", 1)
            tgt_parts = data["to"].split(".", 1)
            src_node = src_parts[0]
            src_port = src_parts[1] if len(src_parts) > 1 else "output"
            tgt_node = tgt_parts[0]
            tgt_port = tgt_parts[1] if len(tgt_parts) > 1 else "input"
        else:
            src_node = data["source_node_id"]
            src_port = data["source_port_name"]
            tgt_node = data["target_node_id"]
            tgt_port = data["target_port_name"]

        if not wire_id:
            wire_id = f"wire_{src_node}_{src_port}_to_{tgt_node}_{tgt_port}"

        return cls(
            id=wire_id,
            source_node_id=src_node,
            source_port_name=src_port,
            target_node_id=tgt_node,
            target_port_name=tgt_port,
            port_type=PortType(data.get("port_type", "generic")),
            param_binding=data.get("binding", data.get("param_binding", "")),
        )


@dataclass
class WorkflowGraph:
    """Complete string diagram workflow graph with nodes and typed strings."""

    id: str
    name: str
    description: str = ""
    nodes: dict[str, WorkflowNode] = field(default_factory=dict)
    strings: list[WorkflowString] = field(default_factory=list)

    def add_node(self, node: WorkflowNode) -> None:
        self.nodes[node.id] = node

    def remove_node(self, node_id: str) -> None:
        if node_id in self.nodes:
            del self.nodes[node_id]
        # Remove any strings connected to this node
        self.strings = [s for s in self.strings if s.source_node_id != node_id and s.target_node_id != node_id]

    def connect(
        self,
        source_node_id: str,
        source_port_name: str,
        target_node_id: str,
        target_port_name: str,
        param_binding: str = "",
    ) -> WorkflowString:
        """Create a typed string wire between two node ports."""
        src_node = self.nodes.get(source_node_id)
        if not src_node:
            raise ValueError(f"Source node '{source_node_id}' does not exist.")
        tgt_node = self.nodes.get(target_node_id)
        if not tgt_node:
            raise ValueError(f"Target node '{target_node_id}' does not exist.")

        src_port = src_node.outputs.get(source_port_name)
        if not src_port:
            raise ValueError(f"Output port '{source_port_name}' on node '{source_node_id}' not found.")
        tgt_port = tgt_node.inputs.get(target_port_name)
        if not tgt_port:
            raise ValueError(f"Input port '{target_port_name}' on node '{target_node_id}' not found.")

        # Type checking
        if not self.is_type_compatible(src_port.port_type, tgt_port.port_type):
            raise TypeError(
                f"Cannot connect {src_port.port_type.value} to {tgt_port.port_type.value}: port types are incompatible."
            )

        # Use port's default param_binding if not provided
        binding = param_binding or tgt_port.param_binding or src_port.param_binding

        wire_id = f"wire_{source_node_id}_{source_port_name}_to_{target_node_id}_{target_port_name}"
        # Remove existing wire if it connects the same target port
        self.strings = [
            s
            for s in self.strings
            if not (s.target_node_id == target_node_id and s.target_port_name == target_port_name)
        ]

        new_wire = WorkflowString(
            id=wire_id,
            source_node_id=source_node_id,
            source_port_name=source_port_name,
            target_node_id=target_node_id,
            target_port_name=target_port_name,
            port_type=src_port.port_type,
            param_binding=binding,
        )
        self.strings.append(new_wire)
        return new_wire

    def disconnect(self, wire_id: str) -> None:
        self.strings = [s for s in self.strings if s.id != wire_id]

    @staticmethod
    def is_type_compatible(src_type: PortType, tgt_type: PortType) -> bool:
        """Determines if two port types can be connected."""
        if src_type == PortType.GENERIC or tgt_type == PortType.GENERIC:
            return True
        return src_type == tgt_type

    def validate(self) -> list[str]:
        """Validates graph structure, port existence, and type constraints."""
        issues = []
        for wire in self.strings:
            src = self.nodes.get(wire.source_node_id)
            tgt = self.nodes.get(wire.target_node_id)
            if not src:
                issues.append(f"Wire '{wire.id}': source node '{wire.source_node_id}' missing.")
                continue
            if not tgt:
                issues.append(f"Wire '{wire.id}': target node '{wire.target_node_id}' missing.")
                continue

            src_port = src.outputs.get(wire.source_port_name)
            tgt_port = tgt.inputs.get(wire.target_port_name)
            if not src_port:
                issues.append(f"Wire '{wire.id}': source port '{wire.source_port_name}' missing on '{src.id}'.")
                continue
            if not tgt_port:
                issues.append(f"Wire '{wire.id}': target port '{wire.target_port_name}' missing on '{tgt.id}'.")
                continue

            if not self.is_type_compatible(src_port.port_type, tgt_port.port_type):
                issues.append(
                    f"Type mismatch on wire '{wire.id}': "
                    f"'{src.id}.{src_port.name}' ({src_port.port_type.value}) cannot flow into "
                    f"'{tgt.id}.{tgt_port.name}' ({tgt_port.port_type.value})."
                )

        # Check required ports
        for node in self.nodes.values():
            for port in node.inputs.values():
                if port.required:
                    connected = any(
                        s.target_node_id == node.id and s.target_port_name == port.name for s in self.strings
                    )
                    if not connected:
                        issues.append(
                            f"Required input '{node.id}.{port.name}' ({port.port_type.value}) is not connected."
                        )

        return issues

    def topological_levels(self) -> list[list[str]]:
        """Groups node IDs into parallel execution levels according to dependencies.

        Handles DAGs and breaks simple feedback loops gracefully.
        """
        in_degree = {nid: 0 for nid in self.nodes}
        adj: dict[str, set[str]] = {nid: set() for nid in self.nodes}

        for wire in self.strings:
            if wire.source_node_id in self.nodes and wire.target_node_id in self.nodes:
                adj[wire.source_node_id].add(wire.target_node_id)
                in_degree[wire.target_node_id] += 1

        levels = []
        current_zero = [nid for nid, deg in in_degree.items() if deg == 0]

        visited = set()
        while current_zero:
            levels.append(current_zero)
            visited.update(current_zero)
            next_zero = []
            for u in current_zero:
                for v in adj[u]:
                    in_degree[v] -= 1
                    if in_degree[v] == 0:
                        next_zero.append(v)
            current_zero = next_zero

        # If there are remaining unvisited nodes (e.g. feedback cycles), append them in final level
        unvisited = [nid for nid in self.nodes if nid not in visited]
        if unvisited:
            levels.append(unvisited)

        return levels

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "strings": [s.to_dict() for s in self.strings],
        }

    def to_yaml(self) -> str:
        return str(yaml.dump(self.to_dict(), sort_keys=False))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowGraph:
        graph = cls(
            id=data.get("id", "workflow_graph"),
            name=data.get("name", "Untitled Workflow"),
            description=data.get("description", ""),
        )
        nodes_dict = data.get("nodes", {})
        if isinstance(nodes_dict, list):
            for n in nodes_dict:
                graph.add_node(WorkflowNode.from_dict(n))
        elif isinstance(nodes_dict, dict):
            for nid, n in nodes_dict.items():
                if "id" not in n:
                    n["id"] = nid
                graph.add_node(WorkflowNode.from_dict(n))

        for s in data.get("strings", []):
            try:
                graph.strings.append(WorkflowString.from_dict(s))
            except Exception:
                pass
        return graph

    @classmethod
    def from_yaml(cls, text: str) -> WorkflowGraph:
        data = yaml.safe_load(text) or {}
        return cls.from_dict(data)

    def save_yaml(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_yaml(), encoding="utf-8")

    @classmethod
    def load_yaml(cls, path: Path | str) -> WorkflowGraph:
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_yaml(content)
