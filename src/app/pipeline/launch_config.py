"""A launch composes every method, trial and replicate from the config tree as it was at launch.

Hydra re-composes the experiment for each sweep trial, so without a snapshot an edit to
``src/usr/config`` while a launch runs would reach its later trials. The launch copies the
workspace config tree once and exports ``THETA_CONFIG_SNAPSHOT``; ``WorkspacePaths.config``
resolves to the copy in the launch's process tree, while the IDE keeps editing the live files.
Package profiles are versioned components and are not copied.
"""
from __future__ import annotations

import os
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
    shutil.copytree(paths.research / "config", target, symlinks=True)
    os.environ[CONFIG_SNAPSHOT_ENV] = str(target)
    try:
        yield target
    finally:
        os.environ.pop(CONFIG_SNAPSHOT_ENV, None)
        shutil.rmtree(launch, ignore_errors=True)


def _remove_stale(root: Path):
    if not root.is_dir():
        return
    cutoff = time.time() - STALE_SECONDS
    for launch in root.iterdir():
        try:
            if launch.stat().st_mtime < cutoff:
                shutil.rmtree(launch, ignore_errors=True)
        except OSError:
            continue
