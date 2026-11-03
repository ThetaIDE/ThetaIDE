"""Interpret tuning activation without importing a tuning backend."""


def tuning_disabled(cfg) -> bool:
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    return hasattr(tuning, "get") and tuning.get("enabled") is False


def tuning_enabled(cfg) -> bool:
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    return tuning is not None and not tuning_disabled(cfg)


def require_runnable_tuning(cfg, methods) -> None:
    """Never silently replace a configured search space with a fixed run."""
    if not tuning_disabled(cfg):
        return
    affected = [str(name) for name, method in methods.items()
                if hasattr(method, "get") and (method.get("tune") or method.get("search_space"))]
    if affected:
        from src.app.pipeline.exceptions import ConfigurationError

        raise ConfigurationError(
            "Tuning is disabled, but these methods declare search spaces: " + ", ".join(affected)
            + ". Enable tuning or remove their search spaces before running."
        )
