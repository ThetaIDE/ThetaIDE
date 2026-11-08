"""Experiment builder contract: the form's fields, composing a config exactly as run_pipeline.py
would, and checking a launch before any process starts."""

from __future__ import annotations

import copy
import functools
import shutil
import sys
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.app.pipeline.compose import (
    _HYDRA_LOCK,
    CONFIG_DIR,
    compose_experiment,
    effective_config,
    method_plans,
    validate_composed,
)
from src.app.pipeline.config import normalize_agent_name
from src.app.workspace import WorkspacePaths

from . import LOGS_DIR

# A Hydra compose costs ~130-200 ms, and the GUI asks for the same one again and again: at startup,
# on every Settings visit, on reopening an experiment, after an undo. The result depends only on its
# arguments, the YAML under in/config, and the agent registry (auto-discovered from src/usr), so
# results are reused while all of those are byte-for-byte unchanged (~3 ms to check).
_USER_CODE_DIR = CONFIG_DIR.parents[1] / "src" / "usr"
_COMPOSE_CACHE: OrderedDict[tuple, tuple[bool, Any]] = OrderedDict()
_COMPOSE_CACHE_SIZE = 64
_COMPOSE_CACHE_LOCK = threading.Lock()


def _inputs_fingerprint() -> int:
    """Hash of everything a compose can depend on: the files (paths and exact contents) and the
    agents registered in memory (hot reload and tests register agents without touching files)."""
    registry_module = sys.modules.get("src.app.core.agent_registry")  # don't import it just to look
    registry = getattr(registry_module, "AGENT_REGISTRY", None) or {}
    parts: list[Any] = [tuple(sorted((name, id(cls)) for name, cls in registry.items()))]
    for root, pattern in ((CONFIG_DIR, "*.yaml"), (CONFIG_DIR, "*.yml"), (_USER_CODE_DIR, "*.py"),
                          (_USER_CODE_DIR, ".theta_component.json"), (_USER_CODE_DIR, "plugin.json")):
        for path in sorted(root.rglob(pattern)):
            try:
                parts.append((str(path), path.read_bytes()))
            except OSError:
                continue
    paths = WorkspacePaths.from_config_root(CONFIG_DIR)
    for path in [*paths.manifests(), paths.data_dir / "provider_overrides.json"]:
        if path.is_file(): parts.append((str(path), path.read_bytes()))
    return hash(tuple(parts))


def _cached_on_inputs(fn):
    """Memoize `fn` on its arguments plus _inputs_fingerprint(). Errors are cached too: with the
    same inputs a failed compose fails again. Callers get copies, so they can't alter the cache."""

    @functools.wraps(fn)
    def wrapper(*args):
        key = (fn.__name__, repr(args), _inputs_fingerprint())  # args are strings and lists of strings
        with _COMPOSE_CACHE_LOCK:
            hit = _COMPOSE_CACHE.get(key)
            if hit is not None:
                _COMPOSE_CACHE.move_to_end(key)
        if hit is None:
            try:
                hit = (True, fn(*args))
            except Exception as exc:  # noqa: BLE001 - re-raised below, for this and later calls
                hit = (False, exc)
            with _COMPOSE_CACHE_LOCK:
                _COMPOSE_CACHE[key] = hit
                while len(_COMPOSE_CACHE) > _COMPOSE_CACHE_SIZE:
                    _COMPOSE_CACHE.popitem(last=False)
        ok, value = hit
        if not ok:
            raise value.with_traceback(None)
        return copy.deepcopy(value)

    wrapper.uncached = fn
    return wrapper


def clear_compose_cache() -> None:
    with _COMPOSE_CACHE_LOCK:
        _COMPOSE_CACHE.clear()


class LaunchError(Exception):
    """The experiment can't be launched; `conflict` marks a clash with existing results or jobs."""

    def __init__(self, message: str, conflict: bool = False):
        super().__init__(message)
        self.conflict = conflict


def effective_timesteps(plans: dict[str, dict], total_timesteps):
    """Return an engine-reported effective step budget, or the configured budget."""
    budgets = [plan.get("effective_timesteps") for plan in plans.values()]
    budgets = [int(value) for value in budgets if value is not None]
    return max(budgets, default=total_timesteps)


@_cached_on_inputs
def config_schema() -> dict[str, Any]:
    """Describe one real workspace recipe for a backend-health check.

    The experiment editor owns its schema and reads paradigm/component metadata
    directly.  This service intentionally does not synthesize an algorithm/model form:
    doing so made a health check quietly dictate an unrelated experiment.
    """
    exp_dir = CONFIG_DIR / "experiment"
    candidates = (
        sorted(
            path
            for path in exp_dir.rglob("*.yaml")
            if path.is_file() and not path.name.startswith("_") and "/methods/" not in path.as_posix()
        )
        if exp_dir.is_dir()
        else []
    )
    for candidate in candidates:
        try:
            composed = compose_experiment(str(candidate.relative_to(exp_dir).with_suffix("")))
            validate_composed(composed)
            method_plans(composed)
        except Exception:
            continue
        env = composed.cfg.get("env") or {}
        return {
            "base_experiment": composed.experiment,
            "environment": {
                "name": env.get("name") if hasattr(env, "get") else None,
                "env_id": env.get("env_id") if hasattr(env, "get") else None,
            },
            "fixed_overrides": [],
            "fields": [],
        }

    # Clean setup fallback: verify Hydra composition without requiring recipes.
    with _HYDRA_LOCK:
        GlobalHydra.instance().clear()
        initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR))
        compose(config_name="config", return_hydra_config=True)

    return {
        "base_experiment": None,
        "environment": {"name": None, "env_id": None},
        "fixed_overrides": [],
        "fields": [],
    }


@_cached_on_inputs
def compose_report(experiment: str, overrides: list[str] | None = None, draft_yaml: str | None = None) -> dict[str, Any]:
    """Compose and validate an experiment exactly as `run_pipeline.py` would, without running it."""
    overrides = list(overrides or [])
    if draft_yaml is not None:
        from src.app.pipeline.config import resolve_experiment_config_name
        resolved = resolve_experiment_config_name(experiment)
        with tempfile.TemporaryDirectory(prefix="theta-config-preview-") as directory:
            draft_dir = Path(directory) / "config"
            shutil.copytree(CONFIG_DIR, draft_dir)
            (draft_dir / "experiment" / f"{resolved}.yaml").write_text(draft_yaml, encoding="utf-8")
            return _compose_report(experiment, overrides, draft_dir)
    return _compose_report(experiment, overrides, None)


def _compose_report(experiment, overrides, config_dir):
    argv = ["python", "run_pipeline.py", experiment, *overrides]
    try:
        composed = compose_experiment(experiment, overrides, config_dir=config_dir,
                                      workspace=WorkspacePaths.from_config_root(CONFIG_DIR))
    except Exception as e:
        return {
            "valid": False,
            "experiment": experiment,
            "argv": argv,
            "config": None,
            "config_yaml": "",
            "notices": [],
            "errors": [{"stage": "compose", "message": str(e)}],
        }

    errors, notices, methods = [], [], {}
    try:
        notices = validate_composed(composed)
    except Exception as e:
        errors.append({"stage": "validation", "message": str(e).strip()})
    try:
        methods = method_plans(composed)
    except Exception as e:
        errors.append({"stage": "methods", "message": str(e).strip()})
    config = effective_config(composed.cfg)
    return {
        "valid": not errors,
        "experiment": composed.experiment,
        "argv": composed.argv,
        "config": config,
        "config_yaml": yaml.safe_dump(config, sort_keys=False),
        "methods": methods,
        "methods_yaml": yaml.safe_dump({name: plan["settings"] for name, plan in methods.items()}, sort_keys=False),
        "notices": notices,
        "errors": errors,
    }


def prepare_launch(
    experiment: str, overrides: list[str] | None = None, overwrite: bool = False, active_jobs: list[dict] | None = None
) -> dict[str, Any]:
    """Validate a launch exactly as the pipeline will and describe the job to create.

    Raises LaunchError for an invalid config, for existing results that launching would delete
    (unless `overwrite`), or for a queued/running job already using the same results folder.
    """
    overrides = list(overrides or [])
    try:
        composed = compose_experiment(experiment, overrides)
        validate_composed(composed)
        plans = method_plans(composed)
    except Exception as e:
        raise LaunchError(str(e).strip()) from e

    cfg = composed.cfg
    local = str((cfg.get('hydra') or {}).get('runtime', {}).get('choices', {}).get('site')
                or (cfg.get('site') or {}).get('name', 'local')) == 'local'
    trainer = cfg.get('trainer') or {}
    devices = trainer.get('devices', 1)
    single_device = (devices == 1 or (isinstance(devices, (list, tuple)) and len(devices) == 1)
                     or (OmegaConf.is_list(devices) and len(devices) == 1))
    if local and (not single_device or trainer.get('num_nodes', 1) != 1):
        raise LaunchError('Resource-managed local launches support one training device and one node per job. '
                          'Use trainer.devices=1 and trainer.num_nodes=1, or submit through a cluster site.')
    log_dir = LOGS_DIR / cfg.group / cfg.experiment_id
    if cfg.get("variability") is None and log_dir.exists() and any(log_dir.iterdir()) and not overwrite and not cfg.get("recover", False):
        raise LaunchError(
            f"Results already exist for {cfg.group}/{cfg.experiment_id}; launching would delete them. "
            "Choose another experiment_id.",
            conflict=True,
        )
    clash = next(
        (j for j in (active_jobs or []) if (j.get("group"), j.get("experiment_id")) == (cfg.group, cfg.experiment_id)),
        None,
    )
    if clash and cfg.get("variability") is None:
        raise LaunchError(
            f"Job {clash['job_id']} ({clash['status']}) already uses {cfg.group}/{cfg.experiment_id}; "
            "running both would overwrite one run's results with the other's.",
            conflict=True,
        )
    agents = [plan["settings"].get("name", normalize_agent_name(name)) for name, plan in plans.items()]
    from src.app.services.resource_budget import uses_accelerator, workload_estimate
    resource_config = effective_config(cfg)
    return {
        "experiment": composed.experiment,
        "overrides": overrides,
        "group": cfg.group,
        "experiment_id": cfg.experiment_id,
        "agents": agents,
        "total_timesteps": cfg.get("total_timesteps"),
        "effective_timesteps": effective_timesteps(plans, cfg.get("total_timesteps")),
        "run_in_parallel": bool(cfg.get("run_in_parallel", False)),
        "local": local,
        "accelerator": uses_accelerator(resource_config),
        "resource_estimate": workload_estimate(resource_config),
    }
