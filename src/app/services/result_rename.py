"""Rename Results tree folders together across artifact roots and execution records."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath

from src.app.pipeline.result_paths import check_result_path, iter_run_dirs

from . import catalog

_RUN_SUFFIX = re.compile(r"_\d{8}-\d{6}(?:-\d+)?$")


def rename_results(kind: str, group: str, experiment_id: str, new_name: str) -> dict:
    catalog.check_path_segment(group)
    catalog.check_path_segment(new_name)
    if kind not in ("group", "experiment", "run"):
        raise ValueError("Only groups, experiments and runs can be renamed")
    if kind != "group":
        check_result_path(experiment_id)
    base = catalog.results_dir().resolve()
    roots = (*catalog.RESULT_ROOTS, "executions")
    known = [path.relative_to(catalog.LOGS_DIR / group).as_posix()
             for path in iter_run_dirs(catalog.LOGS_DIR) if path.is_relative_to(catalog.LOGS_DIR / group)]
    new_group = new_name if kind == "group" else group
    if kind == "group":
        identities = {run: run for run in known}
        folders = [(Path(group), Path(new_group))]
    elif kind == "run":
        renamed = PurePosixPath(experiment_id).with_name(new_name).as_posix()
        identities = {experiment_id: renamed}
        folders = [(Path(group) / experiment_id, Path(group) / renamed)]
    else:
        renamed = PurePosixPath(experiment_id).with_name(new_name).as_posix()
        identities = {run: renamed + run[len(experiment_id):] for run in known
                      if _RUN_SUFFIX.sub("", run) == experiment_id}
        folders = [(Path(group) / run, Path(group) / replacement) for run, replacement in identities.items()]
    result = {"runs": {f"{group}/{old}": f"{new_group}/{new}" for old, new in identities.items()}}
    if (kind == "group" and new_name == group) or (kind != "group" and renamed == experiment_id):
        return result

    moves = []
    for root in roots:
        for old, new in folders:
            source, target = base / root / old, base / root / new
            for path in (source, target):
                if any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(base)):
                    raise ValueError(f"Cannot rename through symbolic links: {path}")
            if target.exists():
                raise FileExistsError(f"Results already exist at {target.relative_to(base)}")
            if source.is_dir():
                moves.append((source, target))
    if not moves:
        raise FileNotFoundError("No results found for the selected folder")

    def moved(path):
        for old, new in moves:
            if path.is_relative_to(old):
                return new / path.relative_to(old)
        return path

    # Executions can be filed under the original recipe, separately from their
    # result run. Update those records too, without changing saved config snapshots.
    records = set((base / "executions").rglob("manifest.json"))
    records.update(path for old, _ in moves for path in old.rglob("execution.json"))
    edits = []
    for path in sorted(records):
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent.is_relative_to(base)):
            continue
        original = path.read_bytes()
        try:
            document = json.loads(original)
        except ValueError:
            continue
        if not isinstance(document, dict):
            continue
        updated = dict(document)
        if document.get("group") == group:
            affected = kind == "group" or any(
                record.get(key) in identities
                for record in [document, *document.get("runs", [])]
                for key in ("experiment_id", "result_experiment_id")
            )
            if affected:
                if any(run.get("status") in ("running", "queued", "pending") for run in document.get("runs", [])):
                    raise ValueError("Wait for this execution to finish before renaming its results")
                updated["group"] = new_group
                for key in ("experiment_id", "result_experiment_id"):
                    if document.get(key) in identities:
                        updated[key] = identities[document[key]]
                updated["runs"] = [dict(run, experiment_id=identities.get(run.get("experiment_id"), run.get("experiment_id")))
                                   for run in document.get("runs", [])]

        def paths(value, key=""):
            if key == "config":
                return value  # original experiment configuration is a historical snapshot
            if isinstance(value, dict):
                return {k: paths(v, k) for k, v in value.items()}
            if isinstance(value, list):
                return [paths(v, key) for v in value]
            if isinstance(value, str) and key in ("logs", "checkpoints", "event_path", "output_log", "manifest"):
                candidate = Path(value)
                absolute = (path.parent / candidate if key == "manifest" else candidate).resolve()
                replacement = moved(absolute)
                if replacement != absolute or (key == "manifest" and moved(path) != path):
                    if candidate.is_absolute():
                        return str(replacement)
                    start = moved(path).parent if key == "manifest" else Path.cwd()
                    return os.path.relpath(replacement, start)
            return value

        updated = paths(updated)
        if updated != document:
            edits.append((path, original, (json.dumps(updated, indent=2) + "\n").encode()))

    completed, written = [], []
    try:
        for source, target in moves:
            source.rename(target)
            completed.append((source, target))
        for path, original, content in edits:
            destination = moved(path)
            written.append((destination, original))
            destination.write_bytes(content)
    except Exception:
        for path, original in reversed(written):
            path.write_bytes(original)
        for source, target in reversed(completed):
            target.rename(source)
        raise
    return result
