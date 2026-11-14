"""Mechanistic monitors for training engines outside Lightning.

    ``MonitorManager(cfg, run, model, algorithm, make_environment)``
                       the configured monitor components for one run. An engine calls
                       ``manager.collect(transitions, updates, final=...)`` at its update
                       boundary and ``manager.close()`` when training ends.
    ``preserve_rng()`` context manager that restores Python, NumPy and Torch RNG state, so
                       diagnostics do not change the training trajectory.

See docs/reference/MONITOR_CONTRACT.md.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "MonitorManager": "src.app.monitoring.runtime:MonitorManager",
    "preserve_rng": "src.app.monitoring.api:preserve_rng",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
