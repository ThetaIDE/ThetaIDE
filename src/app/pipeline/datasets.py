"""Dataset resolution and experiment execution utilities.

Provides functions for finding datasets, running experiments, and triggering plots.
"""

import functools
import logging
import os
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from src.app.pipeline.runtime import get_python_executable

logger = logging.getLogger(__name__)


def fast_purge_dir(path: Path):
    """Instantly remove a directory on network storage (NFS) via atomic rename + background purge."""
    if not path.exists():
        return
    trash_path = path.parent / f".trash_{path.name}_{uuid.uuid4().hex[:8]}"
    try:
        path.rename(trash_path)
        threading.Thread(target=shutil.rmtree, args=(trash_path, True), daemon=True).start()
    except OSError as e:
        logger.debug("Fast purge rename failed for %s (%s), falling back to synchronous rmtree", path, e)
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        logger.debug("Unexpected error in fast_purge_dir for %s (%s), falling back to rmtree", path, e)
        shutil.rmtree(path, ignore_errors=True)


def ensure_online_dataset_path(group: str, experiment_id: str, agent_name_internal: str, is_sweep: bool = False):
    """Determine expected online dataset path and ensure it exists.

    Returns:
        tuple[str, bool]: (dataset_path, has_existing_dataset)
    """
    if is_sweep:
        dataset_path = f"in/datasets/{experiment_id}/{agent_name_internal}"
    else:
        dataset_path = f"in/datasets/{group}/{experiment_id}/{agent_name_internal}"

    has_pkl = False
    if os.path.exists(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            if any(f.endswith(".pkl") for f in files):
                has_pkl = True
                break

    return dataset_path, has_pkl


@functools.lru_cache(maxsize=128)
def resolve_dataset_path(
    dataset_id: str, group: str = "", experiment_id: str = "", yaml_ds_path: str | None = None
) -> Path:
    """Robustly resolve the filesystem path for an offline dataset.

    Checks in order:
    1. in/datasets/[group]/[experiment_id]/[dataset_name]
    2. in/datasets/[group]/[dataset_name]
    3. in/datasets/[dataset_name]
    4. in/datasets/[dataset_id]
    5. yaml_ds_path (if explicitly configured and valid)
    6. Shallowest global match under in/datasets/
    """
    dataset_name_internal = dataset_id.replace("/", "_")
    alt_ids = [dataset_id, dataset_name_internal]
    if not dataset_id.endswith("(w)"):
        alt_ids.extend([f"{dataset_id}(w)", dataset_id.replace("_w", "(w)")])

    candidates = []
    for aid in alt_ids:
        aid_clean = aid.replace("/", "_")
        if group and experiment_id:
            candidates.append(Path("in/datasets") / group / experiment_id / aid_clean)
            candidates.append(Path("in/datasets") / group / experiment_id / aid)
        if group:
            candidates.append(Path("in/datasets") / group / "per_problem" / aid)
            candidates.append(Path("in/datasets") / group / "per_problem" / aid_clean)
            candidates.append(Path("in/datasets") / group / aid_clean)
            candidates.append(Path("in/datasets") / group / aid)
        candidates.extend(
            [
                Path("in/datasets") / aid_clean,
                Path("in/datasets") / aid,
            ]
        )
    if yaml_ds_path:
        candidates.append(Path(yaml_ds_path))

    for cand in candidates:
        if cand.exists() and any(cand.glob("*.pkl")):
            return cand
        if cand.exists() and cand.is_file():
            return cand
        if cand.with_suffix(".npz").exists():
            return cand.with_suffix(".npz")
        if (cand.parent / f"{cand.name}.npz").exists():
            return cand.parent / f"{cand.name}.npz"

    if yaml_ds_path and Path(yaml_ds_path).exists():
        return Path(yaml_ds_path)

    if group:
        return Path("in/datasets") / group / dataset_name_internal

    return Path("in/datasets") / dataset_name_internal


def resolve_dataset_for_method(method_name: str, method_cfg: dict, cfg: Any) -> Path:
    """Resolve the dataset path for an offline method from method_cfg or env config."""
    explicit_ds = method_cfg.get("dataset_path") or (cfg.get("dataset_path") if hasattr(cfg, "get") else None)
    if explicit_ds and Path(explicit_ds).exists():
        return Path(explicit_ds)

    # For offline paradigms, dataset comes from the environment config
    env_cfg = cfg.get("env") if hasattr(cfg, "get") else getattr(cfg, "env", None)
    if env_cfg and hasattr(env_cfg, "get"):
        env_dataset = env_cfg.get("dataset_name", None)
        env_name = env_cfg.get("name", None)
        if env_dataset:
            try:
                group_val = env_name or (cfg.get("group", "") if hasattr(cfg, "get") else getattr(cfg, "group", ""))
                exp_val = cfg.get("experiment_id", "") if hasattr(cfg, "get") else getattr(cfg, "experiment_id", "")
                return resolve_dataset_path(
                    dataset_id=str(env_dataset).replace(".npz", ""),
                    group=group_val,
                    experiment_id=exp_val,
                    yaml_ds_path=str(explicit_ds) if explicit_ds else None,
                )
            except FileNotFoundError:
                pass

    # Fallback: look in standard dataset directories
    group = str(cfg.get("group", "") if hasattr(cfg, "get") else getattr(cfg, "group", ""))
    exp_id = str(cfg.get("experiment_id", "") if hasattr(cfg, "get") else getattr(cfg, "experiment_id", ""))
    ds_root = Path("in/datasets") / group / exp_id
    if ds_root.exists():
        return ds_root

    raise FileNotFoundError(
        f"Cannot resolve dataset for method '{method_name}'. "
        f"No dataset_name in env config and no datasets found at {ds_root}."
    )


def run_experiment(
    overrides, site_cfg=None, script_entrypoint: str | None = None, python_executable: str | None = None,
    output_log: str | Path | None = None, cancel_targets=(), on_start=None, on_wait=None,
    task_config=None, coordinator=False, resource_ticket=None,
):
    """Run target training script as a subprocess with the given Hydra overrides."""
    from src.app.pipeline.runtime import get_subprocess_env

    env = get_subprocess_env(site_cfg)
    import json
    env["THETA_CANCEL_TARGETS"] = json.dumps(cancel_targets)
    from src.app.services.cancellation import requested, TrainingCancelled
    if requested(cancel_targets, env.get("THETA_CANCEL_DIR")):
        raise TrainingCancelled()

    venv_python = python_executable or get_python_executable(site_cfg)
    target_script = script_entrypoint or "src/app/train.py"
    cmd = ([venv_python, "-u", "-m"] + list(overrides) if target_script == "-m" else
           [venv_python, "-u", "-m", "src.app.pipeline.provider_worker", str(target_script)] + list(overrides))
    print(f"Running: {' '.join(cmd)}")
    if env.get('THETA_RESOURCE_POOL') and not coordinator and not any(arg in ('-m', '--multirun') for arg in overrides):
        import json
        from src.app.services.resource_budget import ResourcePool, workload_estimate, uses_accelerator
        from omegaconf import OmegaConf
        pool = ResourcePool(env['THETA_RESOURCE_POOL'], env.get('THETAIDE_SETTINGS_FILE'))
        inherited = json.loads(env.get('THETA_RESOURCE_SPEC', '{}'))
        nested = [arg.lstrip('+') for arg in overrides if '=' in arg and
                  arg.lstrip('+').startswith(('model.', 'agent.'))]
        if task_config is None:
            task_config = OmegaConf.to_container(OmegaConf.from_dotlist(nested), resolve=False)
            task_config['entrypoint'] = str(target_script)
            for name in ('agent', 'model'):
                group = next((arg.split('=', 1)[1] for arg in overrides if arg.lstrip('+').startswith(name + '=')), None)
                if group:
                    task_config.setdefault(name, {})['name'] = group
        spec = workload_estimate(task_config)
        # Keep dataset/architecture hints discovered by the experiment preflight.
        spec['memory'] = max(spec['memory'], inherited.get('memory', 0))
        spec['minimum_memory'] = max(spec['minimum_memory'], inherited.get('minimum_memory', 0))
        def check_cancel():
            if requested(cancel_targets, env.get('THETA_CANCEL_DIR')):
                raise TrainingCancelled()
        with pool.slot(spec, accelerator=uses_accelerator(task_config), check_cancel=check_cancel,
                       ticket=resource_ticket, **({'on_wait': on_wait} if on_wait else {})) as token:
            def started(pid):
                pool.attach(token, pid)
                if on_start:
                    on_start(pid)
            return _execute_training(cmd, env, output_log, on_start=started)
    if resource_ticket:
        env['THETA_RESOURCE_QUEUE_PARENT'] = resource_ticket
    return _execute_training(cmd, env, output_log, on_start=on_start)


def _execute_training(cmd, env, output_log, on_start=None):
    import json
    from src.app.services.cancellation import requested, TrainingCancelled
    from src.app.services.processes import kill_process_tree
    targets = json.loads(env.get("THETA_CANCEL_TARGETS", "[]"))
    stopped = threading.Event()
    cancelled = threading.Event()

    def watch():
        while not stopped.wait(.2):
            if requested(targets, env.get("THETA_CANCEL_DIR")):
                cancelled.set()
                kill_process_tree(process.pid)
                return

    if requested(targets, env.get("THETA_CANCEL_DIR")):
        raise TrainingCancelled()
    # A watcher also handles silent workers whose output pipe is blocked.
    if output_log is None:
        process = subprocess.Popen(cmd, env=env)
        if on_start:
            on_start(process.pid)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            code = process.wait()
        finally:
            stopped.set()
            watcher.join()
        if cancelled.is_set():
            raise TrainingCancelled()
        if code:
            raise subprocess.CalledProcessError(code, cmd)
        return
    # Preserve live console output while retaining a per-run copy for the
    # execution manifest and later debugging.
    output_path = Path(output_log)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", errors="replace") as log_file:
        process = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        if on_start:
            on_start(process.pid)
        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        assert process.stdout is not None
        try:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            return_code = process.wait()
        finally:
            stopped.set()
            watcher.join()
    if cancelled.is_set():
        raise TrainingCancelled()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def run_plotting(experiment, style=None, base_experiment=None, site_cfg=None, wipe=False, use_cache=False):
    """Run plot/manager.py for the given experiment."""
    from src.app.pipeline.runtime import get_subprocess_env

    env = get_subprocess_env(site_cfg)

    venv_python = get_python_executable()
    cmd = [venv_python, "plot/manager.py", str(experiment)]
    if base_experiment and str(base_experiment) != str(experiment):
        cmd.extend(["--experiment", str(base_experiment)])
    if style:
        cmd.extend(["--style", str(style)])
    if wipe:
        cmd.append("--wipe")
    if use_cache:
        cmd.append("--use-cache")

    print(f"\n=== Auto-Generating Modular Plots for experiment: {experiment} ===")
    subprocess.run(cmd, check=True, env=env)
