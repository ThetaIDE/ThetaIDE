"""Version 1 execution contract: one resolved method, owned by one worker.

Provider ``entry_point`` remains import metadata. No script selectors are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

EXECUTION_CONTRACT = 1
REMOVED_SELECTORS = ("entrypoint", "runner", "script", "engine")


@dataclass
class RunContext:
    run_id: str
    workspace: Any
    seed: int
    artifacts: Path
    checkpoint: str | None = None
    cancelled: Callable[[], bool] = lambda: False
    contract_version: int = EXECUTION_CONTRACT
    metadata: dict = field(default_factory=dict)
    _cleanup: list[Callable] = field(default_factory=list, repr=False)

    def check_cancelled(self):
        if self.cancelled():
            from src.app.services.cancellation import TrainingCancelled
            raise TrainingCancelled()

    def on_cleanup(self, callback):
        self._cleanup.append(callback)

    def remove_cleanup(self, callback):
        self._cleanup.remove(callback)

    def close(self):
        errors = []
        for callback in reversed(self._cleanup):
            try:
                callback()
            except BaseException as exc:
                errors.append(exc)
        self._cleanup.clear()
        if errors:
            raise RuntimeError("Backend cleanup failed: " + "; ".join(str(error) for error in errors)) from errors[0]


@dataclass
class RunResult:
    status: str = "completed"
    metrics: dict[str, float] = field(default_factory=dict)
    metric_definitions: dict[str, dict] = field(default_factory=dict)
    artifacts: list[dict] = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    objective: float | None = None
    contract_version: int = EXECUTION_CONTRACT

    def __post_init__(self):
        import math
        if self.contract_version != EXECUTION_CONTRACT:
            raise ValueError(f"Unsupported result contract: {self.contract_version}")
        if any(not math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("RunResult metrics must be finite")
        if self.objective is not None and not math.isfinite(float(self.objective)):
            raise ValueError("RunResult objective must be finite")
        if self.status not in ("completed", "cancelled", "pruned"):
            raise ValueError(f"Invalid successful runner result status: {self.status}")


def reject_selectors(value, location):
    if not hasattr(value, "keys"):
        return
    removed = [key for key in REMOVED_SELECTORS if key in value]
    aliases = [key for key in ("venv", "venv_python") if key in value]
    if removed or aliases:
        from src.app.pipeline.exceptions import ConfigurationError
        fields = ", ".join(f"{location}.{key}" for key in removed + aliases)
        raise ConfigurationError(
            f"Removed execution fields: {fields}. Conflicting declarations require manual resolution. "
            "Select a registered paradigm runner and a migrated dev algorithm; use python for the interpreter. "
            "Preview explicitly selected editable configs with python -m src.app.pipeline.migrate_execution."
        )


def algorithm_selection(cfg, *, paths=None):
    """Resolve an agent profile and explicit overrides once, without importing it."""
    from src.app.workspace import WorkspacePaths
    from src.app.core.provider_catalog import get_catalog
    import yaml
    paths = paths or WorkspacePaths.for_process()
    agent = cfg.get("agent") or {}
    selector = agent if isinstance(agent, str) else agent.get("algorithm") or agent.get("name")
    if not selector:
        return {}, None
    profile_path = paths.resolve_config(f"agent/{selector}.yaml")
    profile = (yaml.safe_load(profile_path.read_text()) or {}) if profile_path.is_file() else {}
    selected = get_catalog(paths).resolve("algorithm", str(selector))
    owner = f" ({selected[0].id}@{selected[0].version})" if selected else ""
    reject_selectors(profile, f"agent profile {selector}{owner}")
    if hasattr(agent, "items"):
        profile.update(dict(agent.items()))
    reject_selectors(profile, "agent")
    catalog = get_catalog(paths)
    selected = catalog.resolve("algorithm", str(selector))
    if selected is None and profile.get("algorithm"):
        selected = catalog.resolve("algorithm", str(profile["algorithm"]))
    return profile, selected


def execution_backend(cfg, *, paths=None):
    """Component metadata owns backend choice. A supplied backend must match it."""
    from src.app.workspace import WorkspacePaths
    paths = paths or WorkspacePaths.for_process()
    reject_selectors(cfg, "execution")
    profile, selected = algorithm_selection(cfg, paths=paths)
    if selected:
        package, capability = selected
        version = capability.metadata.get("execution_contract")
        if type(version) is not int or version != EXECUTION_CONTRACT:
            raise ValueError(f"Provider {package.id}@{package.version} ({capability.id}) declares execution_contract={version!r}; "
                             f"requires {EXECUTION_CONTRACT}. Select a migrated component "
                             "or install a future compatible release.")
        backend = capability.metadata.get("backend")
        if not isinstance(backend, str) or not backend.strip():
            raise ValueError(f"Provider {package.id}@{package.version}: a nonempty backend is required")
        from src.app.core.execution_features import selected_features, provider_hook
        selected_features(selected, backend).validate(cfg, owner=f"{package.id}@{package.version}")
    elif cfg.get("agent"):
        raise ValueError(f"Unresolved selected algorithm: {cfg.get('agent')}")
    else:
        backend = "lightning"
    explicit = cfg.get("backend") or profile.get("backend")
    if explicit and explicit != backend:
        raise ValueError(f"Selected backend {explicit!r} conflicts with component backend {backend!r}")
    if selected:
        provider_hook(selected, "execution_preflight", cfg, paths=paths)
    return backend, selected


def execute_backend(cfg, context):
    backend, selected = execution_backend(cfg, paths=context.workspace)
    from src.app.pipeline.tuning_features import validate_tuning_compatibility, validate_pruning_runner
    if validate_tuning_compatibility(cfg, paths=context.workspace):
        validate_pruning_runner(cfg, paths=context.workspace)
    context.metadata["backend"] = backend
    if selected:
        context.metadata["algorithm"] = {"id": selected[1].id, "package": selected[0].id, "version": selected[0].version}
    if backend == "lightning":
        from src.app.core.lightning_execution import execute
        return execute(cfg, context)
    from src.app.core.provider_loader import load_capability
    adapter = load_capability(*selected, paths=context.workspace)
    return adapter(cfg, context)
