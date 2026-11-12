"""Resolve the common Theta worker and execution-resource interpreter."""
from src.app.core.execution import execution_backend, reject_selectors

DEFAULT_ENGINE_SCRIPT = "src/app/train.py"


def resolve_engine(method_cfg=None, cfg=None):
    method = method_cfg or {}
    reject_selectors(method, "method")
    effective = dict(cfg.items()) if cfg is not None else {}
    effective.update(method)
    execution_backend(effective)
    return DEFAULT_ENGINE_SCRIPT, method.get("python") or effective.get("python")


def runner_supports_episode_outcomes(cfg):
    if cfg.get("paradigm") != "online_rl":
        return False
    from src.app.core.execution import algorithm_selection
    _, selected = algorithm_selection(cfg)
    return bool(selected and selected[1].metadata.get("episode_outcomes") is True)
