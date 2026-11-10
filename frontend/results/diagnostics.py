"""Reopen saved visual snapshots through the Results backend, without training telemetry."""
from __future__ import annotations

import json
import re
from pathlib import PurePosixPath
from types import SimpleNamespace

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from ..monitor.visual_data import VisualSnapshot
from ..monitor.visual_view import VisualsView
from ..widgets import label


class SavedDiagnostics(QWidget):
    def __init__(self, loader, parent=None):
        super().__init__(parent)
        self.loader = loader
        self.view = VisualsView(snapshot_loader=self._snapshot)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.note = label("Select a completed run to inspect its saved diagnostics.", "muted")
        self.note.setWordWrap(True)
        layout.addWidget(self.note)
        layout.addWidget(self.view, 1)
        self._candidates = {}
        self._metadata = {}
        self._errors = {}
        self._locations = {}
        self._scheduled = False
        self._active = False
        self.loader.file_loaded.connect(self._file_loaded)
        self.loader.file_failed.connect(self._file_failed)

    def reset(self):
        self._metadata.clear()
        self._errors.clear()
        self._candidates.clear()
        self._locations.clear()
        self.view.show_lives({})

    def set_sources(self, sources, manifests, *, active=True):
        """Read only selected methods/versions and only while this tab is open."""
        self._active = active
        candidates = {}
        for manifest in manifests:
            selected = [source for source in sources if source.run_key == manifest.run_key]
            selected_versions = set()
            for source in selected:
                versions = manifest.agents.get(source.agent, [])
                selected_versions.add((source.agent, source.version or (versions[-1] if versions else None)))
            files = {artifact["path"] for artifact in manifest.artifacts}
            prefix = f"logs/{manifest.run_key}/"
            for path in sorted(files):
                if not path.startswith(prefix):
                    continue
                parts = PurePosixPath(path.removeprefix(prefix)).parts
                if (len(parts) != 5 or parts[2] != "monitors"
                        or not re.fullmatch(r"snapshot_\d+\.json", parts[-1])):
                    continue
                agent, version = parts[:2]
                if (agent, version) not in selected_versions and (agent, None) not in selected_versions:
                    continue
                key = manifest.run_key, path
                candidates[key] = {"run": f"{manifest.run_key} / {agent} ({version})", "files": files}
        self._candidates = candidates
        self._metadata = {key: value for key, value in self._metadata.items() if key in candidates}
        if active:
            self._update()

    def _schedule(self):
        if self._active and not self._scheduled:
            self._scheduled = True
            QTimer.singleShot(0, self._update)

    def _file_loaded(self, run_key, path):
        key = run_key, path
        if key in self._candidates:
            try:
                metadata = json.loads(self.loader.files[key])
                if not isinstance(metadata, dict) or metadata.get("schema") != "theta.monitor-snapshot/v1":
                    raise ValueError("Unsupported diagnostic snapshot schema")
                arrays = metadata.get("arrays")
                if (not isinstance(arrays, str) or not arrays or "\\" in arrays
                        or PurePosixPath(arrays).name != arrays or arrays in (".", "..")):
                    raise ValueError("Snapshot arrays must be beside their metadata")
                if not isinstance(metadata.get("visuals", {}), dict):
                    raise ValueError("Invalid visual output declarations")
                if not isinstance(metadata.get("instance"), str) or any(
                        type(metadata.get(field)) is not int or metadata[field] < 0
                        for field in ("optimizer_updates", "transitions")):
                    raise ValueError("Snapshot requires an instance and training counters")
                from src.app.telemetry.monitors import validate_monitoring_metadata
                validate_monitoring_metadata({"provides": list(metadata.get("visuals", {}).values())})
                self._metadata[key] = metadata
            except (TypeError, ValueError, KeyError) as exc:
                self._errors[key] = str(exc)
        if key in self._candidates or any(key == (k[0], str(PurePosixPath(k[1]).parent / meta["arrays"]))
                                          for k, meta in self._metadata.items()):
            self._schedule()

    def _file_failed(self, run_key, path, error):
        self._errors[run_key, path] = error
        self._schedule()

    def _update(self):
        self._scheduled = False
        if not self._active:
            return
        for key in self._candidates:
            if key not in self._metadata and key not in self._errors:
                self.loader.load_file(*key)
        lives, locations = {}, {}
        incomplete = 0
        for key, metadata in self._metadata.items():
            candidate = self._candidates[key]
            array_path = str(PurePosixPath(key[1]).parent / metadata["arrays"])
            if array_path not in candidate["files"]:
                incomplete += 1
                continue
            visuals = [{field: value for field, value in visual.items() if field != "data"}
                       for visual in metadata.get("visuals", {}).values() if isinstance(visual, dict)]
            if not visuals:
                continue
            live = lives.setdefault(candidate["run"], SimpleNamespace(artifacts=[]))
            live.artifacts.append({"kind": "monitor_snapshot", "path": key[1], "data": {
                "instance": metadata["instance"], "optimizer_updates": metadata["optimizer_updates"],
                "transitions": metadata["transitions"], "visuals": visuals}})
            locations[key[1]] = key
        self._locations = locations
        # Array bytes may arrive without changing the available steps.
        self.view.show_lives(lives, refresh=True)
        errors = [message for key, message in self._errors.items()
                  if key in self._candidates or key[1] in {str(PurePosixPath(path).parent / meta["arrays"])
                                                        for (_, path), meta in self._metadata.items()}]
        pending = len(self._candidates.keys() - self._metadata.keys() - self._errors.keys())
        note = ("Loading saved diagnostic metadata…" if pending else
                "Saved diagnostics for the selected runs, methods and versions." if lives else
                "No saved rich diagnostics for this selection. Older HTML reports remain available in Files.")
        if incomplete:
            note += f" {incomplete} incomplete snapshot(s) omitted."
        if errors:
            note += f" Could not load: {errors[0]}. Use Refresh to retry."
        self.note.setText(note)

    def _snapshot(self, reference):
        key = self._locations[str(reference["path"])]
        metadata = self._metadata[key]
        array_key = key[0], str(PurePosixPath(key[1]).parent / metadata["arrays"])
        if array_key in self._errors:
            raise ValueError(self._errors[array_key])
        if array_key not in self.loader.files:
            self.loader.load_file(*array_key)
        if array_key not in self.loader.files:
            raise ValueError(self._errors.get(array_key, "Loading snapshot arrays…"))
        return VisualSnapshot(reference, metadata=metadata, array_bytes=self.loader.files[array_key])
