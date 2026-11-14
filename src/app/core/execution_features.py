"""Provider-neutral execution features and selected-provider hooks.

Catalog inspection reads declarations only. Hooks execute only for the selected
provider during preflight or resource planning and must be read-only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Mapping
import math


@dataclass(frozen=True)
class ExecutionFeatures:
    input_types: tuple[str, ...] = ()
    evaluation: str = "provider"
    resume_modes: tuple[str, ...] = ()
    checkpoint_formats: tuple[str, ...] = ()
    tuning: bool = False
    winner_promotion: bool = False
    episode_outcomes: bool = False
    intermediate_values: bool = False

    @classmethod
    def from_metadata(cls, metadata):
        values = metadata.get("execution_features", {})
        if not isinstance(values, Mapping):
            raise ValueError("execution_features must be a mapping")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown execution features: {sorted(unknown)}")
        values = dict(values)
        # Read the original v1 declaration until downstream manifests migrate.
        if "episode_outcomes" not in values and "episode_outcomes" in metadata:
            values["episode_outcomes"] = metadata["episode_outcomes"]
        for name in ("input_types", "resume_modes", "checkpoint_formats"):
            if name in values:
                items = values[name]
                if not isinstance(items, (list, tuple)) or any(not isinstance(x, str) or not x for x in items):
                    raise ValueError(f"execution_features.{name} must be a list of nonempty strings")
                values[name] = tuple(items)
        for name in ("tuning", "winner_promotion", "episode_outcomes", "intermediate_values"):
            if name in values and type(values[name]) is not bool:
                raise ValueError(f"execution_features.{name} must be a boolean")
        result = cls(**values)
        if result.evaluation not in ("provider", "configured", "both"):
            raise ValueError("execution_features.evaluation must be provider, configured or both")
        if set(result.resume_modes) - {"training", "inference"}:
            raise ValueError("execution_features.resume_modes supports training and inference")
        return result

    def validate(self, cfg, *, owner):
        from src.app.pipeline.tuning_config import tuning_enabled
        if cfg.get("evaluator") and self.evaluation == "provider":
            raise ValueError(f"{owner}: runner owns native evaluation; configured evaluator providers are unsupported")
        if any(cfg.get(key) for key in ("resume", "recover", "resume_from")) and "training" not in self.resume_modes:
            raise ValueError(f"{owner}: exact training resume is unsupported; consult the provider's inference reload modes")
        if cfg.get("data_module") and "data_module" not in self.input_types:
            raise ValueError(f"{owner}: custom data_module selection is unsupported; supported inputs: {self.input_types}")
        if cfg.get("dataset_path") and "dataset_path" not in self.input_types:
            raise ValueError(f"{owner}: dataset_path input is unsupported; supported inputs: {self.input_types}")
        if tuning_enabled(cfg) and not (self.tuning and self.winner_promotion):
            raise ValueError(f"{owner}: tuning/winner-promotion is unsupported for this execution provider")


LIGHTNING_FEATURES = ExecutionFeatures(
    input_types=("data_module", "dataset_path", "live_environment"),
    evaluation="configured", resume_modes=("training", "inference"),
    checkpoint_formats=("lightning.ckpt",), tuning=True, winner_promotion=True,
    intermediate_values=True,
)


def selected_features(selected, backend):
    if selected is None:
        return LIGHTNING_FEATURES
    metadata = selected[1].metadata
    # Lightning is a retained core implementation, including its legacy providers.
    if backend == "lightning" and "execution_features" not in metadata:
        return LIGHTNING_FEATURES
    return ExecutionFeatures.from_metadata(metadata)


def provider_hook(selected, name, cfg, *, paths):
    """Load only the explicitly selected package's declared hook."""
    package, capability = selected
    reference = capability.metadata.get(name)
    if reference is None:
        return None
    if not isinstance(reference, str) or ":" not in reference:
        raise ValueError(f"{package.id}: {name} must name a package-relative file.py:callable")
    if package.entrypoint is not None:
        raise ValueError(f"{package.id}: package-relative {name} requires a native package")
    from src.app.core.provider_loader import load_capability
    hook = load_capability(package, replace(capability, entry_point=reference), paths=paths)
    if not callable(hook):
        raise ValueError(f"{package.id}: {name} is not callable")
    return hook(cfg, paths=paths)


def resource_hints(cfg, *, paths=None):
    from src.app.core.execution import algorithm_selection
    from src.app.workspace import WorkspacePaths
    paths = paths or WorkspacePaths.for_process()
    _, selected = algorithm_selection(cfg, paths=paths)
    if selected is None:
        agent = cfg.get("agent")
        selector = agent if isinstance(agent, str) else (agent or {}).get("algorithm") or (agent or {}).get("name")
        if selector:
            from src.app.core.provider_catalog import get_catalog
            diagnostics = "; ".join(get_catalog(paths).diagnostics)
            raise ValueError(f"Unresolved selected algorithm for resource hints: {selector}. Install or enable its provider. {diagnostics}")
        return {}
    hints = selected[1].metadata.get("resource_hints", {})
    if not isinstance(hints, Mapping):
        raise ValueError("resource_hints must be a mapping")
    hints = dict(hints)
    if selected[1].metadata.get("resource_hint_hook") is not None:
        extra = provider_hook(selected, "resource_hint_hook", cfg, paths=paths)
        if not isinstance(extra, Mapping):
            raise ValueError("resource_hint_hook must return a mapping")
        hints.update(extra)
    if set(hints) - {"base_memory_gib", "cpu", "accelerator"}:
        raise ValueError(f"Unknown resource hints: {sorted(set(hints) - {'base_memory_gib', 'cpu', 'accelerator'})}")
    for key in ("base_memory_gib", "cpu"):
        if key in hints and (type(hints[key]) not in (int, float) or not math.isfinite(hints[key]) or hints[key] <= 0):
            raise ValueError(f"resource_hints.{key} must be finite and positive")
    if "accelerator" in hints and type(hints["accelerator"]) is not bool:
        raise ValueError("resource_hints.accelerator must be a boolean")
    return hints
