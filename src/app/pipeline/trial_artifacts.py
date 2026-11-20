"""Stable paths for artifacts produced by an individual tuning trial."""

import hashlib
from pathlib import Path
from urllib.parse import quote

from src.app.pipeline.result_paths import SAVED_RUN_NAME, SEED_NAME, SIBLING_EXECUTION_NAME, trial_result_version


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
    root = Path(root)
    archive = root / "sweeps" / study_dir
    if SAVED_RUN_NAME.fullmatch(root.parent.name) and not archive.is_dir():
        return root / trial_result_version(trial_number)
    return archive / f"trial_{trial_number}"


def trial_checkpoint_dir_from_config(root, cfg):
    runtime = cfg.get("theta_tuning") if hasattr(cfg, "get") else None
    if not hasattr(runtime, "get") or runtime.get("trial_number") is None or not runtime.get("study_name"):
        return None
    return trial_checkpoint_dir(root, str(runtime["study_name"]), int(runtime["trial_number"]))


def trial_logs(method_dir: Path, trial: dict) -> Path:
    """Resolve a tuner record to the current trial directory or its legacy version."""
    version = str(trial.get("log_version") or "")
    number = trial.get("id")
    candidate = method_dir / trial_result_version(number) if type(number) is int and number >= 0 else None
    if version.startswith("version_") and Path(version).name == version:
        legacy = method_dir / version
        run_name = method_dir.parent.name
        if (legacy.is_dir() or candidate is None or
                (not candidate.is_dir() and not (SAVED_RUN_NAME.fullmatch(run_name)
                                                 or SIBLING_EXECUTION_NAME.fullmatch(run_name)))):
            return legacy
    return candidate if candidate is not None else method_dir


def trial_metrics_version(method_dir: Path, trial: dict) -> str:
    directory = trial_logs(method_dir, trial)
    if directory.name.startswith("trial_") and not (directory / "metrics.csv").is_file():
        seed = next((path for path in sorted(directory.glob("seed_*"))
                     if SEED_NAME.fullmatch(path.name) and path.is_dir()), None)
        if seed is not None:
            directory = seed
    return directory.relative_to(method_dir).as_posix() if directory != method_dir else ""
