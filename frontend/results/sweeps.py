"""Saved sweep dashboards, read through the Results artifact loader."""
from __future__ import annotations

import json
from pathlib import PurePosixPath

from PyQt6.QtCore import Qt

from ..monitor.sweep_view import SweepView


class SweepResultsView(SweepView):
    def __init__(self, loader, open_metrics):
        super().__init__()
        self.loader = loader
        self.open_metrics = open_metrics
        self._artifacts = []
        self._trial_files = {}
        self.open_button.setText("Open trial metrics")
        self.trial_selected.connect(self._open_metrics)
        loader.file_loaded.connect(self._file_loaded)

    def set_manifests(self, manifests):
        artifacts = []
        for manifest in manifests:
            prefix = f"checkpoints/{manifest.run_key}/"
            for artifact in manifest.artifacts:
                path = artifact.get("path", "")
                if not path.startswith(prefix):
                    continue
                parts = PurePosixPath(path.removeprefix(prefix)).parts
                if (len(parts) == 2 or (len(parts) == 4 and parts[1] == "sweeps")) and parts[-1] == "tuning_summary.json":
                    artifacts.append((manifest.run_key, parts[0], path))
        # The per-study snapshots take precedence over the method's latest snapshot.
        self._artifacts = sorted(artifacts, key=lambda item: (len(PurePosixPath(item[2]).parts), item[2]))
        self._reload_studies()
        for run_key, _method, path in self._artifacts:
            if (run_key, path) not in self.loader.files:
                self.loader.load_file(run_key, path)

    def _file_loaded(self, run_key, path):
        if any(key == run_key and file == path for key, _method, file in self._artifacts):
            self._reload_studies()

    def _reload_studies(self):
        found = {}
        self._trial_files = {}
        for run_key, method, path in self._artifacts:
            raw = self.loader.files.get((run_key, path))
            if raw is None:
                continue
            try:
                study = json.loads(raw)
            except (ValueError, UnicodeError):
                continue
            if not isinstance(study, dict) or not study.get("study") or not isinstance(study.get("trials"), list):
                continue
            found[(run_key, method, study["study"])] = study
        for (run_key, method, name), study in found.items():
            for trial in study["trials"]:
                version = trial.get("log_version")
                if version:
                    label = f"{run_key} / {method} · {name} · trial {trial['id']}"
                    log_method = study.get("log_method") or method
                    self._trial_files[label] = (run_key, f"logs/{run_key}/{log_method}/{version}/metrics.csv")
        self.set_studies([(f"{run_key} / {method}", study)
                          for (run_key, method, _name), study in found.items()])

    def _show_trial_detail(self):
        super()._show_trial_detail()
        index = self.selector.currentIndex()
        item = self.table.item(self.table.currentRow(), 0)
        if item is not None and 0 <= index < len(self.studies):
            method, study = self.studies[index]
            label = f"{method} · {study['study']} · trial {item.data(Qt.ItemDataRole.UserRole)}"
            available = label in self._trial_files
            self.open_button.setEnabled(available)
            self.open_button.setToolTip("Open the recorded metrics file for this trial" if available
                                       else "This trial has no recorded metrics version.")

    def _open_metrics(self, label):
        target = self._trial_files.get(label)
        if target:
            self.open_metrics(*target)
