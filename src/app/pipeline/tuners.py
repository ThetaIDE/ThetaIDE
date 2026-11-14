"""Discovery and the execution contract for tuning components.

The IDE provides job execution and artifacts; the selected component decides
how to compile, prepare and finalize a tuning run. Components are factories
registered in the ``theta.tuners`` Python entry-point group.
"""

from __future__ import annotations

from typing import Any, Protocol

from src.app.pipeline.exceptions import ConfigurationError

TUNER_API_VERSION = 1


class Tuner(Protocol):
    api_version: int
    def settings(self) -> list[dict[str, Any]]: ...
    def validate_config(self, cfg: Any) -> None: ...
    def validate_dimension(self, target: str, dimension: Any) -> None: ...
    def compile(self, overrides: list[str], **context: Any) -> list[str]: ...
    def job_entrypoint(self, cfg: Any, method_cfg: Any, script: str, python: str | None) -> tuple[str, str | None]: ...
    def study_name(self, cfg: Any, method_name: str) -> str: ...
    def prepare(self, cfg: Any, storage: str | None, study_name: str) -> None: ...
    def finalize(self, group: str, experiment: str, method: str, storage: str | None, study_name: str) -> Any: ...
    def storage(self, cfg: Any) -> str | None: ...


class PruningTuner(Tuner, Protocol):
    """Optional query required by the capability's tuning_features.pruning flag."""
    def pruning_enabled(self, cfg: Any) -> bool: ...


class PortableTuner(Tuner, Protocol):
    """Optional remote contract; local-only tuner API v1 remains supported."""
    remote_api_version: int
    def execution_plan(self, cfg: Any, *, execution_id: str) -> Any: ...
    def launch_bindings(self, cfg: Any, *, plan: Any, runtime: Any, continuation: Any = None) -> Any: ...
    def snapshot_artifact(self, source: Any, target: Any, *, artifact: Any) -> None: ...
    def restore_continuation(self, cfg: Any, *, plan: Any, runtime: Any, lineage: Any) -> None: ...
    def continuation_config(self, tuning: Any) -> Any: ...


def tuner_name(cfg: Any) -> str:
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    selected = tuning.get("component") if hasattr(tuning, "get") else None
    return _selector(selected)


def _selector(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ConfigurationError(
            "Select a tuning provider explicitly with tuning.component (a nonempty capability selector). "
            "Existing recipes without a selector must choose their intended installed tuner; "
            "no default tuner is selected."
        )
    return value


def available_tuners(*, paths=None) -> list[str]:
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.contributions import capability_status
    return sorted({c.id for p in get_catalog(paths).packages for c in p.capabilities
                   if c.kind == "tuner" and capability_status(p, c, paths=paths).enabled})


def training_callbacks(cfg: Any) -> list[Any]:
    """Ask the active component for engine callbacks, if it provides any."""
    runtime = cfg.get("theta_tuning") if hasattr(cfg, "get") else None
    name = runtime.get("component") if hasattr(runtime, "get") else None
    if not name:
        return []
    component = get_tuner(name=str(name))
    factory = getattr(component, "training_callbacks", None)
    return list(factory(cfg)) if callable(factory) else []


def trial_monitor(cfg: Any) -> Any:
    """Return the tuner's engine-independent intermediate-value monitor, if enabled."""
    runtime = cfg.get("theta_tuning") if hasattr(cfg, "get") else None
    name = runtime.get("component") if hasattr(runtime, "get") else None
    if not name:
        return None
    factory = getattr(get_tuner(name=str(name)), "trial_monitor", None)
    return factory(cfg) if callable(factory) else None


def get_tuner(cfg: Any = None, *, name: str | None = None, paths=None) -> Tuner:
    selected = _selector(name) if name is not None else tuner_name(cfg)
    if cfg is not None:
        from src.app.core.provider_catalog import get_catalog
        from src.app.core.contributions import capability_status
        metadata = get_catalog(paths).resolve('tuner', selected)
        if metadata is not None:
            status = capability_status(*metadata, paths=paths, config=cfg)
            if status.state != 'ready':
                raise ConfigurationError(status.reason)
    from src.app.core.provider_loader import resolve_provider
    factory = resolve_provider("tuner", selected, **({"paths": paths} if paths is not None else {}))
    if factory is None:
        raise ConfigurationError(f"Tuning component {selected!r} is not installed (entry point: theta.tuners).")
    component = factory() if callable(factory) else factory
    if getattr(component, "api_version", None) != TUNER_API_VERSION:
        raise ConfigurationError(
            f"Tuning component {selected!r} must declare api_version = {TUNER_API_VERSION}."
        )
    for method in ("settings", "validate_config", "validate_dimension", "compile", "job_entrypoint", "study_name", "prepare", "finalize", "storage"):
        if not callable(getattr(component, method, None)):
            raise ConfigurationError(f"Tuning component {selected!r} is missing {method}().")
    return component
