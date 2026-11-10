"""Interpret tuning activation without importing a tuning backend."""


def tuning_disabled(cfg) -> bool:
    return not tuning_enabled(cfg)


def tuning_enabled(cfg) -> bool:
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    # The composed component configuration is authoritative, including inherited
    # settings and explicit nulls. Legacy sweep/enabled flags have no effect.
    return tuning is not None


def require_runnable_tuning(cfg, methods) -> None:
    """Never silently replace a configured search space with a fixed run."""
    if not tuning_disabled(cfg):
        return
    affected = [str(name) for name, method in methods.items()
                if hasattr(method, "get") and (method.get("tune") or method.get("search_space"))]
    if affected:
        from src.app.pipeline.exceptions import ConfigurationError

        raise ConfigurationError(
            "These methods declare search spaces without a tuning component: " + ", ".join(affected)
            + ". Add tuning configuration or remove their search spaces before running."
        )
