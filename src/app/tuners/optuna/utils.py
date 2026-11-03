"""Optuna-related utilities for the pipeline.

Provides functions for querying best trials, managing studies,
and launching the Optuna dashboard.
"""

import logging
import os
import subprocess
import sys
import threading
import time
import webbrowser
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_OPTUNA_DB_URL = "sqlite:///results/optuna/optuna.db"


from src.app.pipeline.runtime import get_python_executable


def is_valid_storage_url(storage_url) -> bool:
    """Checks whether the given storage URL is non-empty and not in-memory (None/null)."""
    if not storage_url:
        return False
    s = str(storage_url).strip().lower()
    return s not in ("none", "null", "", "false")


def get_best_trial_id(storage_url, study_name):
    """Queries the Optuna database to find the best trial ID for a given study.

    Uses the Optuna Python API instead of raw SQL for robustness across
    schema versions.
    """
    if not is_valid_storage_url(storage_url):
        return "0"

    try:
        import optuna

        try:
            study = optuna.load_study(study_name=study_name, storage=storage_url)
            return str(study.best_trial.number)
        except KeyError as e:
            logger.debug("Study '%s' not found directly, checking versioned studies: %s", study_name, e)
            all_studies = optuna.get_all_study_summaries(storage=storage_url)
            matches = [
                s for s in all_studies if s.study_name == study_name or s.study_name.startswith(f"{study_name}_v")
            ]
            if matches:
                matches.sort(key=lambda s: s.study_name, reverse=True)
                target_study = optuna.load_study(study_name=matches[0].study_name, storage=storage_url)
                return str(target_study.best_trial.number)
            raise
        except Exception as e:
            logger.debug("Could not load study '%s' directly: %s", study_name, e)
            all_studies = optuna.get_all_study_summaries(storage=storage_url)
            matches = [
                s for s in all_studies if s.study_name == study_name or s.study_name.startswith(f"{study_name}_v")
            ]
            if matches:
                matches.sort(key=lambda s: s.study_name, reverse=True)
                target_study = optuna.load_study(study_name=matches[0].study_name, storage=storage_url)
                return str(target_study.best_trial.number)
            raise
    except Exception as e:
        print(f"Warning: Could not query best trial from Optuna: {e}")
        return "0"


def launch_optuna_dashboard(storage_url):
    """Launches the optuna-dashboard in the background and opens the browser."""
    if not storage_url:
        return

    # Ensure directory exists if using sqlite
    if storage_url.startswith("sqlite:///"):
        db_path = storage_url.replace("sqlite:///", "")
        if "/" in db_path:
            os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Check if a dashboard is already running for this storage_url
    try:
        ps_output = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
        if storage_url in ps_output and "optuna_dashboard" in ps_output:
            print(f"Optuna Dashboard already running for: {storage_url}. Skipping launch.")
            return
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("Could not check running processes via ps aux: %s", e)
    except Exception as e:
        logger.debug("Unexpected error checking running processes: %s", e)

    print(f"Launching Optuna Dashboard for: {storage_url}")

    # Use managed python to ensure dashboard is available
    venv_python = get_python_executable()
    venv_bin_dir = os.path.dirname(venv_python)
    optuna_dashboard_bin = os.path.join(venv_bin_dir, "optuna-dashboard")

    # Try to launch optuna-dashboard
    try:
        # Launch in background
        port = 8080
        # Try different ports if 8080 is taken
        for p in range(8080, 8090):
            try:
                # Use a dummy check to see if port is in use
                subprocess.run(["nc", "-z", "localhost", str(p)], capture_output=True, check=True)
                continue  # Port is in use
            except (subprocess.CalledProcessError, FileNotFoundError):
                # CalledProcessError means port is not in use; FileNotFoundError means nc is missing
                port = p
                break
            except subprocess.SubprocessError as e:
                logger.debug("Subprocess error checking port %s: %s; selecting port", p, e)
                port = p
                break

        # Ensure logging directory exists
        log_file_path = "results/optuna/dashboard.log"
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        log_file = open(log_file_path, "w")

        cmd = [optuna_dashboard_bin, storage_url, "--port", str(port)]
        subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

        print(f"Optuna Dashboard started at http://127.0.0.1:{port}")

        # Wait a moment for server to start then open browser in a separate thread
        def open_browser():
            time.sleep(2)
            url = f"http://127.0.0.1:{port}"
            if sys.platform == "darwin":
                # On macOS, -g opens in background without focusing
                subprocess.run(["open", "-g", url])
            else:
                webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: optuna-dashboard could not be started: {e}")
        print("Please ensure 'optuna-dashboard' is installed in your environment.")


def get_optuna_storage(storage_url: str):
    """Safely return an Optuna RDBStorage object with timeout and clean schema handling."""
    if not storage_url:
        return None
    import optuna

    if storage_url.startswith("sqlite:///"):
        import sqlite3

        db_raw = storage_url.replace("sqlite:///", "").split("?")[0]
        if db_raw and os.path.dirname(db_raw):
            os.makedirs(os.path.dirname(os.path.abspath(db_raw)), exist_ok=True)
        # If DB file exists but is 0 bytes (empty file created by touch/connect),
        # remove it so Optuna creates the full table schema cleanly without AssertionError
        if os.path.exists(db_raw) and os.path.getsize(db_raw) == 0:
            try:
                os.remove(db_raw)
            except OSError as e:
                logger.debug("Could not remove 0-byte SQLite DB %s: %s", db_raw, e)
        # Pre-configure SQLite database with WAL journal mode and busy timeout via Python
        try:
            conn = sqlite3.connect(db_raw, timeout=120.0)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=120000;")
            conn.close()
        except sqlite3.Error as e:
            logger.debug("Could not pre-configure SQLite WAL/busy timeout on %s: %s", db_raw, e)
        except Exception as e:
            logger.debug("Unexpected error configuring SQLite DB %s: %s", db_raw, e)
        return optuna.storages.RDBStorage(url=f"sqlite:///{db_raw}", engine_kwargs={"connect_args": {"timeout": 120}})
    return storage_url


def get_next_study_name(group: str, experiment_id: str, agent_name: str, storage_url: str | None = None) -> str:
    """Generate clean [experiment]_[method]_v[number] study name using existing run versions.

    Uses filesystem inspection to avoid database locking during batch submission.
    """
    base_prefix = f"{experiment_id}_{agent_name}"
    from pathlib import Path

    log_dir = Path("results/logs") / group / experiment_id / agent_name
    ckpt_dir = Path("results/checkpoints") / group / experiment_id / agent_name

    existing_versions = [0]
    for d in [log_dir, ckpt_dir]:
        if d.exists():
            for v_path in d.glob("version_*"):
                if v_path.is_dir():
                    try:
                        existing_versions.append(int(v_path.name.split("_")[-1]) + 1)
                    except ValueError:
                        pass
    version = max(existing_versions)
    return f"{base_prefix}_v{version}"


def delete_optuna_study(storage_url, study_name):
    """Deletes an existing study from the Optuna database to start fresh."""
    if not is_valid_storage_url(storage_url):
        return
    try:
        import optuna

        storage = get_optuna_storage(storage_url)
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f"Reset existing Optuna study: {study_name}")
    except KeyError:
        logger.debug("Optuna study '%s' does not exist to delete.", study_name)
    except Exception as e:
        logger.warning("Could not delete Optuna study '%s': %s", study_name, e)


def create_optuna_study(storage_url, study_name, direction="minimize"):
    """Pre-creates/initializes an Optuna study to avoid schema initialization race conditions on cluster nodes."""
    if not is_valid_storage_url(storage_url):
        return
    try:
        import optuna

        storage = get_optuna_storage(storage_url)
        optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, direction=direction)
        print(f"Pre-initialized Optuna study: {study_name}")
    except Exception as e:
        print(f"Warning: Failed to pre-initialize Optuna study '{study_name}': {e}")


def find_best_trial_from_logs(group: str, experiment_id: str, agent_name: str, direction: str = "minimize"):
    """Scan CSV logs to find the winning trial number and score for in-memory sweeps."""
    from pathlib import Path

    import pandas as pd

    log_dir = Path("results/logs") / group / experiment_id / agent_name
    best_id = "0"
    best_val = float("inf") if direction == "minimize" else float("-inf")

    if not log_dir.exists():
        return best_id, None

    for v_dir in sorted(
        log_dir.glob("version_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else 0
    ):
        metrics_file = v_dir / "metrics.csv"
        if not metrics_file.exists():
            continue
        try:
            df = pd.read_csv(metrics_file)
            trial_num = v_dir.name.split("_")[-1]
            val = None
            for col in ["val/loss", "val/robust_loss", "losses/bellman_loss", "losses/total_loss", "eval/reward"]:
                if col in df.columns:
                    valid_series = df[col].dropna()
                    if len(valid_series) > 0:
                        val = float(valid_series.iloc[-1])
                        break
            if val is not None:
                if (direction == "minimize" and val < best_val) or (direction == "maximize" and val > best_val):
                    best_val = val
                    best_id = trial_num
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.debug("Skipping unparseable metrics file %s: %s", metrics_file, e)
        except (KeyError, ValueError, OSError) as e:
            logger.warning("Error reading metrics from %s: %s", metrics_file, e)
        except Exception as e:
            logger.warning("Unexpected error reading metrics from %s: %s", metrics_file, e)

    return best_id, (best_val if best_val != float("inf") and best_val != float("-inf") else None)


def write_study_summary(study, ckpt_root):
    """Export a backend-neutral snapshot, including incomplete and failed trials."""
    import json
    import math
    import tempfile
    from pathlib import Path

    from src.app.pipeline.trial_artifacts import trial_checkpoint_dir

    ckpt_root = Path(ckpt_root)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    try:
        best_trial_id = study.best_trial.number
    except ValueError:
        best_trial_id = None
    trials = []
    for item in study.trials:
        directory = trial_checkpoint_dir(ckpt_root, study.study_name, item.number)
        available = [str(path) for path in (directory / "best_model.ckpt", directory / "best_model.zip") if path.is_file()]
        trials.append({
            "id": item.number, "state": item.state.name.lower(),
            "value": item.value if item.value is not None and math.isfinite(item.value) else None,
            "parameters": item.params,
            "checkpoint": available[0] if item.state.name == "COMPLETE" and len(available) == 1 else None,
            "intermediate_values": {str(step): value for step, value in item.intermediate_values.items()
                                    if value is not None and math.isfinite(value)},
        })
    summary = {
        "schema_version": 1, "component": "optuna", "study": study.study_name,
        "direction": study.direction.name.lower(), "objective_metric": study.user_attrs.get("theta_objective_metric"),
        "stop_reason": study.user_attrs.get("theta_last_stop_reason"), "best_trial_id": best_trial_id, "trials": trials,
    }
    study_root = trial_checkpoint_dir(ckpt_root, study.study_name, 0).parent
    study_root.mkdir(parents=True, exist_ok=True)
    for directory in (study_root, ckpt_root):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as stream:
                temporary = stream.name
                json.dump(summary, stream, indent=2, allow_nan=False)
            os.replace(temporary, directory / "tuning_summary.json")
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)


def promote_best_trial_checkpoint(
    group: str, experiment_id: str, agent_name: str, storage_url: str | None = None, study_name: str | None = None
):
    """After an Optuna sweep, queries the winning trial, copies its checkpoint
    and hyperparameters to the main agent checkpoint root, and writes a summary."""
    import json
    import math
    import shutil
    from pathlib import Path

    import yaml

    from src.app.pipeline.trial_artifacts import trial_checkpoint_dir
    import optuna

    if not is_valid_storage_url(storage_url) or not study_name:
        raise ValueError("Winner promotion requires persistent storage and an exact study name")
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    try:
        trial = study.best_trial
    except ValueError as exc:
        raise ValueError(f"Study {study_name!r} has no completed trial to promote") from exc
    if trial.value is None or not math.isfinite(trial.value):
        raise ValueError(f"Best trial in study {study_name!r} does not have a finite objective")
    ckpt_root = Path("results/checkpoints") / group / experiment_id / agent_name
    trial_dir = trial_checkpoint_dir(ckpt_root, study_name, trial.number)
    candidates = [trial_dir / "best_model.ckpt", trial_dir / "best_model.zip"]
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint for study {study_name!r}, trial {trial.number} in {trial_dir}; "
            "legacy sweeps need an explicit trial-to-checkpoint mapping."
        )
    source = matches[0]
    suffix = source.suffix
    ckpt_root.mkdir(parents=True, exist_ok=True)
    for destination in (ckpt_root / f"best_model{suffix}", ckpt_root / f"{agent_name}{suffix}",
                        ckpt_root.parent / f"{agent_name}{suffix}"):
        shutil.copy2(source, destination)
    best_info = {
        "study_name": study_name, "best_trial_id": trial.number,
        "best_value": trial.value, "best_params": trial.params,
        "direction": study.direction.name, "checkpoint": str(source),
    }
    with (ckpt_root / "best_params.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(trial.params, f)
    with (ckpt_root / "best_trial_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=2)
    write_study_summary(study, ckpt_root)
    print(f"\n[Optuna Winner] Promoted trial #{trial.number} from {source}")
    return str(trial.number)
