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
    compose_experiment,
    effective_config,
    method_plans,
    one_discovery,
    validate_composed,
)
from src.app.pipeline.config import normalize_agent_name
from src.app.workspace import WorkspacePaths


# A full check (compose, validate, method plans) costs 0.2-0.4 s, and the GUI asks for the same one
# again and again: at startup, on reopening an experiment, after an undo. The result depends only on
# its arguments, the effective config profiles, provider sources and manifests, and the agent registry,
# so results are reused while all of those are byte-for-byte unchanged (tens of ms to check).
def _workspace(workspace=None):
    return workspace if workspace is not None else WorkspacePaths.for_process()
_COMPOSE_CACHE: OrderedDict[tuple, tuple[bool, Any]] = OrderedDict()
_COMPOSE_CACHE_SIZE = 64
_COMPOSE_CACHE_LOCK = threading.Lock()


def _authored_sources(paths):
    """Python sources and manifests under src/usr; run output in `paths.results` is never walked."""
    import fnmatch
    import os
    found = []
    for directory, children, names in os.walk(paths.research):
        if Path(directory) == paths.results.parent:
            children[:] = [name for name in children if Path(directory, name) != paths.results]
        found.extend(Path(directory, name) for name in names
                     if fnmatch.fnmatch(name, "*.py") or name in (".theta_component.json", "plugin.json"))
    return sorted(found)


def provider_fingerprint(workspace=None) -> int:
    """Hash of what imported provider code depends on: authored Python and manifests under src/usr,
    every package manifest, and provider selection/enablement. Config YAML is not part of it: a
    process that imported providers only has to re-import them when this changes."""
    paths = _workspace(workspace)
    parts: list[Any] = [(str(paths.root), str(paths.application), str(paths.data_dir))]
    for path in [*_authored_sources(paths), *paths.manifests(),
                 paths.data_dir / "provider_overrides.json", paths.data_dir / "extensions.json"]:
        try:
            parts.append((str(path), path.read_bytes()))
        except OSError:
            continue
    return hash(tuple(parts))


def _inputs_fingerprint(workspace=None) -> int:
    """Hash of everything a compose can depend on: provider inputs, every effective config file
    (paths and exact contents), and the agents registered in memory (hot reload and tests register
    agents without touching files)."""
    registry_module = sys.modules.get("src.app.core.agent_registry")  # don't import it just to look
    registry = getattr(registry_module, "AGENT_REGISTRY", None) or {}
    paths = _workspace(workspace)
    parts: list[Any] = [tuple(sorted((name, id(cls)) for name, cls in registry.items())), provider_fingerprint(paths)]
    for path in (path for root in paths.config_roots() for pattern in ("*.yaml", "*.yml") for path in sorted(root.rglob(pattern))):
        try:
            parts.append((str(path), path.read_bytes()))
        except OSError:
            continue
    return hash(tuple(parts))


def _cached_on_inputs(fn):
    """Memoize `fn` on its arguments plus _inputs_fingerprint(). Errors are cached too: with the
    same inputs a failed compose fails again. Callers get copies, so they can't alter the cache."""

    @functools.wraps(fn)
    @one_discovery
    def wrapper(*args, **kwargs):
        key = (fn.__name__, repr(args), repr(sorted(kwargs.items())), _inputs_fingerprint(kwargs.get("workspace")))
        with _COMPOSE_CACHE_LOCK:
            hit = _COMPOSE_CACHE.get(key)
            if hit is not None:
                _COMPOSE_CACHE.move_to_end(key)
        if hit is None:
            try:
                hit = (True, fn(*args, **kwargs))
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

    wrapper.uncached = one_discovery(fn)
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
def config_schema(*, workspace=None) -> dict[str, Any]:
    """Describe one real workspace recipe for a backend-health check.

    The experiment editor owns its schema and reads paradigm/component metadata
    directly.  This service intentionally does not synthesize an algorithm/model form:
    doing so made a health check quietly dictate an unrelated experiment.
    """
    paths = _workspace(workspace)
    candidates = [Path(selector).relative_to("experiment") for selector in paths.config_resolver().selectors("experiment")
                  if not Path(selector).name.startswith("_") and "/methods/" not in selector]
    for candidate in candidates:
        try:
            composed = compose_experiment(candidate.with_suffix("").as_posix(), workspace=paths)
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
    from src.app.config_resolution import configuration_context
    with _HYDRA_LOCK, configuration_context(paths):
        GlobalHydra.instance().clear()
        initialize_config_dir(version_base=None, config_dir=str(paths.primary_config))
        compose(config_name="config", return_hydra_config=True)

    return {
        "base_experiment": None,
        "environment": {"name": None, "env_id": None},
        "fixed_overrides": [],
        "fields": [],
    }


@_cached_on_inputs
def compose_report(experiment: str, overrides: list[str] | None = None, draft_yaml: str | None = None, *, workspace=None) -> dict[str, Any]:
    """Compose and validate an experiment exactly as `run_pipeline.py` would, without running it."""
    overrides = list(overrides or [])
    paths = _workspace(workspace)
    if draft_yaml is not None:
        from src.app.pipeline.config import resolve_experiment_config_name
        resolved = resolve_experiment_config_name(experiment, paths=paths)
        with tempfile.TemporaryDirectory(prefix="theta-config-preview-") as directory:
            draft_dir = Path(directory) / "config"
            draft_dir.mkdir()
            # Copy effective files, including provider defaults, without writing
            # into their package or inferring a workspace from the draft path.
            resolver = paths.config_resolver()
            for selector in resolver.selectors():
                target = draft_dir / selector
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(resolver.resolve(selector, required=True).file.read_bytes())
            (draft_dir / "experiment" / f"{resolved}.yaml").parent.mkdir(parents=True, exist_ok=True)
            (draft_dir / "experiment" / f"{resolved}.yaml").write_text(draft_yaml, encoding="utf-8")
            return _compose_report(experiment, overrides, draft_dir, workspace=paths)
    return _compose_report(experiment, overrides, None, workspace=paths)


def _compose_report(experiment, overrides, config_dir, *, workspace=None):
    argv = ["python", "run_pipeline.py", experiment, *overrides]
    try:
        composed = compose_experiment(experiment, overrides, config_dir=config_dir,
                                      workspace=_workspace(workspace))
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
    from src.app.sites import scheduler, site_name
    config.setdefault('site', {}).setdefault('scheduler', scheduler(composed.cfg))
    config['site']['name'] = site_name(composed.cfg)
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


@one_discovery
def prepare_launch(
    experiment: str, overrides: list[str] | None = None, overwrite: bool = False, active_jobs: list[dict] | None = None,
    *, workspace=None,
) -> dict[str, Any]:
    """Validate a launch exactly as the pipeline will and describe the job to create.

    Raises LaunchError for an invalid config, for existing results that launching would delete
    (unless `overwrite`), or for a queued/running job already using the same results folder.
    """
    overrides = list(overrides or [])
    try:
        composed = compose_experiment(experiment, overrides, workspace=_workspace(workspace))
        validate_composed(composed)
        plans = method_plans(composed)
    except Exception as e:
        raise LaunchError(str(e).strip()) from e

    cfg = composed.cfg
    from src.app.sites import scheduler
    if scheduler(cfg) != 'local':
        raise LaunchError('This experiment targets a remote site. Use Run to upload project code and open the cluster terminal, '
                          'then submit run_pipeline.py there. Sync Metrics/Results pulls its outputs back.')
    trainer = cfg.get('trainer') or {}
    devices = trainer.get('devices', 1)
    single_device = (devices == 1 or (isinstance(devices, (list, tuple)) and len(devices) == 1)
                     or (OmegaConf.is_list(devices) and len(devices) == 1))
    if not single_device or trainer.get('num_nodes', 1) != 1:
        raise LaunchError('Resource-managed local launches support one training device and one node per job. '
                          'Use trainer.devices=1 and trainer.num_nodes=1, or submit through a cluster site.')
    log_dir = composed.workspace.results / "logs" / cfg.group / cfg.experiment_id
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
    return {**describe_launch(composed, plans), "overrides": overrides}


def describe_launch(composed, plans) -> dict[str, Any]:
    """The job record fields of a composed local experiment, shared by desktop and terminal launches."""
    cfg = composed.cfg
    agents = [plan["settings"].get("name", normalize_agent_name(name)) for name, plan in plans.items()]
    from src.app.services.resource_budget import uses_accelerator, workload_estimate
    resource_config = effective_config(cfg)
    return {
        "experiment": composed.experiment,
        "group": cfg.group,
        "experiment_id": cfg.experiment_id,
        "agents": agents,
        "total_timesteps": cfg.get("total_timesteps"),
        "effective_timesteps": effective_timesteps(plans, cfg.get("total_timesteps")),
        "run_in_parallel": bool(cfg.get("run_in_parallel", False)),
        "local": True,  # only local launches are described; remote sites submit elsewhere
        "accelerator": uses_accelerator(resource_config, paths=composed.workspace),
        "resource_estimate": workload_estimate(resource_config, paths=composed.workspace),
    }
