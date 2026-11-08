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
    """Compatibility wrapper for the historical default logger factory."""
    from src.app.core.provider_loader import resolve_provider
    return resolve_provider("run_logger", "tensorboard")(settings, context)


_BUILTIN_PROVIDERS: dict[str, Callable] = {}


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
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_capability
    catalog = get_catalog()
    local = [(p, c) for p, c in catalog.candidates("run_logger", backend_id) if p.origin != "python_distribution"]
    if local:
        if len(local) != 1 or any(point.name == backend_id for point in discovered) or backend_id in _BUILTIN_PROVIDERS:
            raise RuntimeError(f"Duplicate run logger provider id: {backend_id}")
        return load_capability(*local[0], paths=catalog.paths)
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


def _plugin_enabled(plugin_id: str) -> bool:
    """Honor the IDE's plugin toggle in child runs and standalone CLI runs."""
    env_state = os.environ.get("THETA_ENABLED_PLUGINS")
    if env_state is not None:
        try:
            enabled = json.loads(env_state)
        except (TypeError, json.JSONDecodeError):
            enabled = []
        return isinstance(enabled, list) and plugin_id in enabled

    paths = [Path.home() / ".config" / "thetaide" / "settings.toml"]
    workspace_settings = os.environ.get("THETAIDE_SETTINGS_FILE")
    from src.app.workspace import WorkspacePaths
    paths.append(Path(workspace_settings) if workspace_settings else WorkspacePaths.for_process().data_dir / "settings.toml")
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
    if plugin_id in disabled:
        return False
    if plugin_id in enabled:
        return True
    from src.app.core.provider_catalog import get_catalog
    selected = get_catalog().resolve("plugin", plugin_id)
    return bool(selected and selected[1].metadata.get("default_enabled", False))


def create_run_loggers(cfg, context: RunLoggerContext) -> list:
    """Instantiate configured run logger providers in their declared order."""
    configured = _configured_backends(cfg)
    if not configured:
        return []
    discovered = _entry_points()
    created = []
    for backend_id, settings in configured.items():
        from src.app.core.provider_catalog import get_catalog
        selected = get_catalog().resolve("run_logger", backend_id)
        activation = selected[1].metadata.get("activation_plugin") if selected else None
        if activation and not _plugin_enabled(activation):
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
