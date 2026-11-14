"""Metadata-only tuner capabilities and selected-provider compatibility checks."""
from dataclasses import dataclass
from collections.abc import Mapping


@dataclass(frozen=True)
class TuningFeatures:
    objective_modes: tuple[str, ...] = ("metric",)
    pruning: bool = False
    pruning_objectives: bool = False
    pruning_repeated_runs: bool = False

    @classmethod
    def from_metadata(cls, metadata):
        values = metadata.get("tuning_features", {})
        if not isinstance(values, Mapping):
            raise ValueError("tuning_features must be a mapping")
        values = dict(values)
        if set(values) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown tuning_features declaration")
        if "objective_modes" in values:
            modes = values["objective_modes"]
            if not isinstance(modes, (list, tuple)) or any(mode not in ("metric", "component") for mode in modes):
                raise ValueError("tuning_features.objective_modes supports metric and component")
            values["objective_modes"] = tuple(modes)
        for key in ("pruning", "pruning_objectives", "pruning_repeated_runs"):
            if key in values and type(values[key]) is not bool:
                raise ValueError(f"tuning_features.{key} must be a boolean")
        result = cls(**values)
        if not result.pruning and (result.pruning_objectives or result.pruning_repeated_runs):
            raise ValueError("Pruning compatibility requires tuning_features.pruning")
        return result


def validate_tuning_compatibility(cfg, *, paths=None, tuner=None):
    """Validate declarations without interpreting a provider's option vocabulary.

    Returns whether pruning is active. Providers declaring pruning must supply a
    read-only ``pruning_enabled(cfg) -> bool`` query, including legacy settings.
    """
    from src.app.core.provider_catalog import get_catalog
    from src.app.pipeline.tuners import get_tuner, tuner_name
    from src.app.workspace import WorkspacePaths

    if cfg.get("tuning") is None and not (cfg.get("theta_tuning") or {}).get("component"):
        return False
    paths = paths or WorkspacePaths.for_process()
    runtime = cfg.get("theta_tuning") or {}
    name = runtime.get("component") or tuner_name(cfg)
    selected = get_catalog(paths).resolve("tuner", str(name))
    tuner = tuner if tuner is not None else get_tuner(cfg, name=str(name), paths=paths)
    features = TuningFeatures.from_metadata(selected[1].metadata if selected else {})
    mode = "component" if cfg.get("objective") is not None else "metric"
    if mode not in features.objective_modes:
        raise ValueError(f"Tuner {name!r} does not support {mode} objectives")
    query = getattr(tuner, "pruning_enabled", None)
    if features.pruning and not callable(query):
        raise ValueError(f"Tuner {name!r} declares pruning but is missing pruning_enabled()")
    active = query(cfg) if callable(query) else False
    if type(active) is not bool:
        raise ValueError(f"Tuner {name!r} pruning_enabled() must return a boolean")
    if not active:
        return False
    if not features.pruning:
        raise ValueError(f"Tuner {name!r} does not declare pruning support")
    if mode == "component" and not features.pruning_objectives:
        raise ValueError(f"Tuner {name!r}: custom objectives require pruning disabled")
    if cfg.get("variability") is not None and not features.pruning_repeated_runs:
        raise ValueError(f"Tuner {name!r}: repeated tuning requires pruning disabled")
    return True


def validate_pruning_runner(cfg, *, paths=None):
    from src.app.core.execution import algorithm_selection
    from src.app.core.execution_features import selected_features
    _, selected = algorithm_selection(cfg, paths=paths)
    backend = selected[1].metadata.get("backend") if selected else "lightning"
    if not selected_features(selected, backend).intermediate_values:
        raise ValueError("Execution provider does not declare intermediate-value reporting required for pruning")
