"""Backend extension point for optional training-run loggers.

Run logger integrations are discovered in the interpreter that runs training,
so they work from both the IDE and ``run_pipeline.py``. Third-party packages
may expose a provider through the ``thetaide.run_loggers`` entry-point group.
An entry point named ``wandb`` should load a callable with the signature
``create_logger(settings, context)``.
"""
from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class RunLoggerContext:
    """Stable run information shared with logger providers."""

    base_root: Path
    group_name: str
    experiment_id: str
    method_name: str


def _tensorboard_logger(settings: Mapping[str, Any], context: RunLoggerContext):
    """Create TensorBoard lazily, keeping its optional dependency off the default path."""
    from lightning.pytorch.loggers import TensorBoardLogger

    log_dir = context.base_root / "results" / "tensorboard" / context.group_name / context.experiment_id
    logger = TensorBoardLogger(str(log_dir), name=context.method_name, default_hp_metric=False)
    _ = logger.experiment
    return logger


_BUILTIN_PROVIDERS: dict[str, Callable] = {"tensorboard": _tensorboard_logger}


def _configured_backends(cfg) -> dict[str, dict[str, Any]]:
    """Return enabled backend settings, translating the historical TensorBoard flag."""
    logging_config = cfg.get("logging", {}) if hasattr(cfg, "get") else {}
    backends = logging_config.get("backends", {}) if hasattr(logging_config, "get") else {}
    result: dict[str, dict[str, Any]] = {}
    configured_ids: set[str] = set()
    if hasattr(backends, "items"):
        for backend_id, raw_settings in backends.items():
            backend_id = str(backend_id)
            configured_ids.add(backend_id)
            settings = dict(raw_settings) if hasattr(raw_settings, "items") else {}
            if settings.pop("enabled", False):
                result[backend_id] = settings

    # Compatibility for existing experiment YAML and CLI overrides. New configs
    # should use logging.backends.tensorboard.enabled.
    legacy_tensorboard = cfg.get("tensorboard", None) if hasattr(cfg, "get") else None
    if legacy_tensorboard is None:
        agent = cfg.get("agent", {}) if hasattr(cfg, "get") else {}
        legacy_tensorboard = agent.get("tensorboard", False) if hasattr(agent, "get") else False
    if bool(legacy_tensorboard) and "tensorboard" not in configured_ids:
        result.setdefault("tensorboard", {})
    return result


def _entry_points():
    try:
        return list(entry_points(group="thetaide.run_loggers"))
    except TypeError:  # Python 3.10 compatibility for older importlib.metadata APIs
        return list(entry_points().get("thetaide.run_loggers", ()))


def _provider_for(backend_id: str, discovered) -> Callable | None:
    builtin = _BUILTIN_PROVIDERS.get(backend_id)
    matches = [point for point in discovered if point.name == backend_id]
    if builtin is not None:
        if matches:
            raise RuntimeError(f"Run logger provider id '{backend_id}' conflicts with a built-in provider")
        return builtin
    if len(matches) > 1:
        raise RuntimeError(f"Duplicate run logger provider id: {backend_id}")
    if not matches:
        return None
    provider = matches[0].load()
    if not callable(provider):
        raise TypeError(f"Run logger provider '{backend_id}' must load a callable")
    return provider


def _tensorboard_plugin_enabled() -> bool:
    """Honor the IDE's plugin toggle in child runs and standalone CLI runs."""
    env_state = os.environ.get("THETA_ENABLED_PLUGINS")
    if env_state is not None:
        try:
            enabled = json.loads(env_state)
        except (TypeError, json.JSONDecodeError):
            enabled = []
        return isinstance(enabled, list) and "tensorboard" in enabled

    paths = [Path.home() / ".config" / "thetaide" / "settings.toml"]
    workspace_settings = os.environ.get("THETAIDE_SETTINGS_FILE")
    paths.append(Path(workspace_settings) if workspace_settings else Path.cwd() / ".thetaide" / "settings.toml")
    plugins: dict[str, Any] = {}
    for path in paths:
        try:
            with path.open("rb") as settings_file:
                file_plugins = tomllib.load(settings_file).get("plugins", {})
        except (OSError, tomllib.TOMLDecodeError):
            continue
        if isinstance(file_plugins, dict):
            plugins.update(file_plugins)

    enabled = plugins.get("enabled", [])
    disabled = plugins.get("disabled", [])
    if "tensorboard" in disabled:
        return False
    if "tensorboard" in enabled:
        return True
    # Match the TensorBoard core plugin's manifest default on a fresh workspace.
    return True


def create_run_loggers(cfg, context: RunLoggerContext) -> list:
    """Instantiate configured run logger providers in their declared order."""
    configured = _configured_backends(cfg)
    if not configured:
        return []
    discovered = _entry_points()
    created = []
    for backend_id, settings in configured.items():
        if backend_id == "tensorboard" and not _tensorboard_plugin_enabled():
            continue
        provider = _provider_for(backend_id, discovered)
        if provider is None:
            raise RuntimeError(
                f"Run logger '{backend_id}' is enabled but no provider is installed. "
                "Install its plugin package or disable it under logging.backends."
            )
        result = provider(settings, context)
        if result is None:
            continue
        if isinstance(result, (list, tuple)):
            created.extend(logger for logger in result if logger is not None)
        else:
            created.append(result)
    return created
