"""A launch composes every method, trial and replicate from the config tree as it was at launch.

Hydra re-composes the experiment for each sweep trial, so without a snapshot an edit to
``src/usr/config`` while a launch runs would reach its later trials. The launch copies the
workspace config tree once and exports ``THETA_CONFIG_SNAPSHOT``; ``WorkspacePaths.config``
resolves to the copy in the launch's process tree, while the IDE keeps editing the live files.
Provider packages, their profiles, and declared workspace resources are copied too.
"""
from __future__ import annotations

import os
import json
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from src.app.workspace import CONFIG_SNAPSHOT_ENV

STALE_SECONDS = 7 * 24 * 3600
"""Snapshots left by launches that were killed outright are removed after this long."""


def _snapshots(paths):
    return paths.data_dir / "launches"


@contextmanager
def frozen_config(paths):
    """Snapshot the workspace config for this launch, unless an enclosing launch already did."""
    if paths.launch_config_snapshot() is not None:
        yield paths.config
        return
    root = _snapshots(paths)
    _remove_stale(root)
    launch = root / uuid.uuid4().hex
    target = launch / "config"
    from src.app.pipeline.launch_inputs import INPUTS_SNAPSHOT_ENV, capture_inputs
    previous = {key: os.environ.get(key) for key in (CONFIG_SNAPSHOT_ENV, INPUTS_SNAPSHOT_ENV)}
    try:
        from src.app.components.transactions import workspace_lock, write_json, machine_id
        with workspace_lock(paths, wait=True):
            write_json(launch / "owner.json", {"pid": os.getpid(), "machine": machine_id()})
            source = paths.research / "config"
            if source.exists():
                shutil.copytree(source, target)
            else:
                target.mkdir()
            inputs = capture_inputs(paths, launch)
        from src.app.core.provider_loader import invalidate
        invalidate(paths, include_bundled=True)
        os.environ[CONFIG_SNAPSHOT_ENV] = str(target)
        os.environ[INPUTS_SNAPSHOT_ENV] = str(inputs)
        from src.app.core.provider_discovery import invalidate_discovery_snapshot
        invalidate_discovery_snapshot()
        yield target
    finally:
        from src.app.core.provider_loader import invalidate
        invalidate(paths, include_bundled=True)
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from src.app.core.provider_discovery import invalidate_discovery_snapshot
        invalidate_discovery_snapshot()
        shutil.rmtree(launch, ignore_errors=True)


def _remove_stale(root: Path):
    if not root.is_dir():
        return
    cutoff = time.time() - STALE_SECONDS
    for launch in root.iterdir():
        try:
            if launch.stat().st_mtime < cutoff:
                owner = launch / "owner.json"
                if owner.is_file():
                    from src.app.components.transactions import alive, machine_id
                    record = json.loads(owner.read_text())
                    if record.get("machine") != machine_id() or alive(record["pid"]):
                        continue  # A long-running or foreign-host launch still owns its inputs.
                shutil.rmtree(launch, ignore_errors=True)
        except (OSError, ValueError, KeyError):
            continue
