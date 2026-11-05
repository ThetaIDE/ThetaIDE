"""Stable paths for artifacts produced by an individual tuning trial."""

import hashlib
from pathlib import Path
from urllib.parse import quote


def trial_log_version(cfg):
    """A stable, collision-resistant log directory for a study's trial."""
    runtime = cfg.get("theta_tuning") if hasattr(cfg, "get") else None
    if not hasattr(runtime, "get") or runtime.get("trial_number") is None or not runtime.get("study_name"):
        return None
    explicit = runtime.get("log_version")
    if explicit:
        version = str(explicit)
        if not version.startswith("version_sweep_") or Path(version).name != version:
            raise ValueError("Invalid trial log version")
        return version
    digest = hashlib.sha256(str(runtime["study_name"]).encode("utf-8")).hexdigest()[:16]
    return f"version_sweep_{digest}_trial_{int(runtime['trial_number'])}"


def trial_checkpoint_dir(root, study_name: str, trial_number: int) -> Path:
    """Keep study and trial identity separate from Hydra's launch job number."""
    if not study_name or trial_number < 0:
        raise ValueError("A study name and a nonnegative trial number are required")
    # urllib leaves dots unescaped even with safe=""; escape them as well so
    # study names such as ".." cannot address a parent directory.
    study_dir = quote(str(study_name), safe="").replace(".", "%2E")
    return Path(root) / "sweeps" / study_dir / f"trial_{trial_number}"


def trial_checkpoint_dir_from_config(root, cfg):
    runtime = cfg.get("theta_tuning") if hasattr(cfg, "get") else None
    if not hasattr(runtime, "get") or runtime.get("trial_number") is None or not runtime.get("study_name"):
        return None
    return trial_checkpoint_dir(root, str(runtime["study_name"]), int(runtime["trial_number"]))
