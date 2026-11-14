"""Resolve the common Theta worker and execution-resource interpreter."""
from src.app.core.execution import execution_backend, reject_selectors

DEFAULT_ENGINE_SCRIPT = "src/app/train.py"


def resolve_engine(method_cfg=None, cfg=None, *, paths=None):
    method = method_cfg or {}
    reject_selectors(method, "method")
    effective = dict(cfg.items()) if cfg is not None else {}
    effective.update(method)
    execution_backend(effective, paths=paths)
    return DEFAULT_ENGINE_SCRIPT, method.get("python") or effective.get("python")


def runner_supports_episode_outcomes(cfg, *, paths=None):
    from src.app.core.execution import algorithm_selection
    from src.app.core.execution_features import selected_features
    _, selected = algorithm_selection(cfg, paths=paths)
    return bool(selected and selected_features(selected, selected[1].metadata.get("backend")).episode_outcomes)
