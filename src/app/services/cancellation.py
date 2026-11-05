"""Durable, per-launch cancellation requests shared by runner and workers."""
import hashlib
import os
from pathlib import Path


def request_path(directory, target):
    return Path(directory) / hashlib.sha256(target.encode()).hexdigest()


def requested(targets, directory=None):
    directory = directory or os.environ.get("THETA_CANCEL_DIR")
    return bool(directory) and any(request_path(directory, target).exists() for target in targets)


class TrainingCancelled(Exception):
    pass
