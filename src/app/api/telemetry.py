"""Run telemetry for training engines outside Lightning.

    ``init``                   open a run: ``run = init(group, experiment_id, agent, config=cfg)``.
    ``Run``                    the run handle; ``run.log(metrics, step=...)`` and ``run.finish()``.
    ``settings_from_config``   the configured sinks and system-metric interval.
    ``selection_from_config``  the user's metric selection (``keeps_section``, ``keeps``).
"""
from ._exports import lazy_exports

_EXPORTS = {
    "init": "src.app.telemetry.run:init",
    "Run": "src.app.telemetry.run:Run",
    "settings_from_config": "src.app.telemetry.run:settings_from_config",
    "selection_from_config": "src.app.telemetry.selection:selection_from_config",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
