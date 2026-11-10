"""Artifact references and lazy arrays for live and completed diagnostics (no ML imports)."""
from __future__ import annotations

from pathlib import Path

from src.app.telemetry.visuals import completed_metadata


def snapshots(lives):
    result = []
    for run, live in lives.items():
        for artifact in live.artifacts:
            if artifact.get("kind") != "monitor_snapshot":
                continue
            path = Path(artifact["path"])
            if not path.is_absolute() and path.parts[:1] == ("results",):
                from src.app.workspace import WorkspacePaths
                path = WorkspacePaths.for_process().root / path
            # New telemetry carries only declarations, never arrays or graph payloads.
            data = artifact.get("data") or {}
            for visual in data.get("visuals", []):
                result.append({"run": run, "instance": data.get("instance", ""),
                               "path": path, "updates": data.get("optimizer_updates", 0),
                               "transitions": data.get("transitions", 0), **visual})
    # Replayed telemetry may contain duplicate references.
    unique = {(row["run"], str(row["path"]), row["name"]): row for row in result}
    return sorted(unique.values(), key=lambda row: (row["updates"], row["transitions"], str(row["path"])))


class VisualSnapshot:
    def __init__(self, reference, *, metadata=None, array_bytes=None):
        from io import BytesIO
        import numpy as np
        self.metadata = completed_metadata(reference["path"]) if metadata is None else metadata
        self.visual = next((value for value in self.metadata.get("visuals", {}).values()
                            if value["name"] == reference["name"]), None)
        if self.visual is None:
            raise ValueError("Selected visual is unavailable in this snapshot")
        source = (reference["path"].parent / self.metadata["arrays"] if array_bytes is None
                  else BytesIO(array_bytes))
        self.archive = np.load(source, allow_pickle=False)
        self.cache = {}
        schema = self.visual["data"].get("schema", self.visual["data"].get("graph", {}).get("schema"))
        if schema != self.visual["artifact_schema"]:
            self.close()
            raise ValueError("Visual artifact schema differs from its declaration")

    def __getitem__(self, key):
        if key not in self.cache:
            self.cache[key] = self.archive[key]
        return self.cache[key]

    def close(self):
        self.archive.close()


def comparable(first, second):
    """Never align different initializations, topologies, targets or probe banks."""
    a, b = first.metadata, second.metadata
    if not a.get("lineage") or any(a.get(key) != b.get(key) for key in
                                  ("lineage", "target", "probe_set_sha256")):
        return False
    ga = first.visual["data"].get("graph", {})
    gb = second.visual["data"].get("graph", {})
    return ga.get("topology_sha256") == gb.get("topology_sha256")
