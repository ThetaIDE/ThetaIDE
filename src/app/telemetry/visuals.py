"""Versioned visual declarations. Discovery needs no NumPy, Torch or Qt."""
from __future__ import annotations

VISUAL_VERSION = "theta.visual-output/v1"
KINDS = frozenset({"scalar_series", "matrix", "distribution", "network_graph", "intervention_graph"})


def validate_visual(item):
    kind = item.get("kind", "scalar_series")
    if kind not in KINDS:
        raise ValueError(f"Unsupported visual output kind: {kind}")
    if kind == "scalar_series":
        return
    if item.get("visual_schema") != VISUAL_VERSION:
        raise ValueError("Rich outputs require visual_schema theta.visual-output/v1")
    for field in ("artifact_schema", "value_semantics", "label"):
        if not isinstance(item.get(field), str) or not item[field].strip():
            raise ValueError(f"Rich outputs require {field}")
    axes = item.get("axes")
    if not isinstance(axes, list) or not axes or not all(isinstance(axis, str) and axis.strip() for axis in axes):
        raise ValueError("Rich outputs require named axes")
    if item.get("retention") != "snapshots":
        raise ValueError("Rich output retention must be snapshots (bounded by max_snapshots)")
    if item["name"].endswith("/"):
        raise ValueError("Rich outputs require an exact stable name")


def validate_visual_data(declaration, payload, arrays):
    """Validate references before publishing the snapshot completion marker."""
    if not isinstance(payload, dict):
        raise ValueError("Visual payload must be a mapping")
    graph = payload.get("graph")
    schema = graph.get("schema") if isinstance(graph, dict) else payload.get("schema")
    if schema != declaration["artifact_schema"]:
        raise ValueError("Visual payload schema differs from its declaration")
    kind = declaration["kind"]
    if kind == "matrix":
        layers = payload.get("layers", [])
        if not layers or any(row.get("array") not in arrays or arrays[row["array"]].ndim != 2 for row in layers):
            raise ValueError("Matrix outputs require referenced two-dimensional arrays")
    elif kind == "distribution":
        if payload.get("array") not in arrays:
            raise ValueError("Distribution output requires an array reference")
    elif kind == "network_graph":
        if not isinstance(graph, dict) or not graph.get("layers"):
            raise ValueError("Network output requires layers")
        for layer in graph["layers"]:
            if layer.get("activation") not in arrays:
                raise ValueError("Network layer requires an activation array")
    elif kind == "intervention_graph":
        if not isinstance(graph, dict) or not isinstance(graph.get("nodes"), list) or not isinstance(graph.get("edges"), list):
            raise ValueError("Intervention output requires nodes and edges")


def completed_metadata(path):
    """Read only a committed JSON/NPZ pair; never load arrays while listing steps."""
    import json
    from pathlib import Path
    path = Path(path)
    meta = json.loads(path.read_text(encoding="utf-8"))
    if meta.get("schema") != "theta.monitor-snapshot/v1":
        raise ValueError("Unsupported diagnostic snapshot schema")
    arrays = path.parent / meta["arrays"]
    if arrays.resolve().parent != path.resolve().parent or not arrays.is_file():
        raise ValueError("Snapshot requires completed arrays beside its metadata")
    return meta
