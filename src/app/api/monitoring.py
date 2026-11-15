"""Mechanistic monitors for training engines outside Lightning.

    ``MonitorManager(cfg, run, model, algorithm, make_environment)``
                       the configured monitor components for one run. An engine calls
                       ``manager.collect(transitions, updates, final=...)`` at its update
                       boundary and ``manager.close()`` when training ends.
    ``preserve_rng()`` context manager that restores Python, NumPy and Torch RNG state, so
                       diagnostics do not change the training trajectory.

Monitor providers
    ``Result``         what a provider's ``collect(context)`` returns: scalars, arrays, details
                       and rich ``visuals``.
    ``Context``        what ``setup``/``collect`` receive: instance spec, model access, probes,
                       step counters and ``wants_output(suffix)``.
    ``ProbeSet``       fixed probe observations with their identity.
    ``ModelAccess``    scoring and module access to the monitored model; a model returns its own
                       from ``get_monitoring_adapter(algorithm)``.
    ``action_difference``, ``capture``, ``effective_rank``, ``layers_for``, ``linear_cka``,
    ``replace_units``, ``validate_params``
                       shared measurement helpers.

Engines that publish their own snapshots
    ``state_digest(module)``, ``write_snapshot(directory, context, result, ...)``,
    ``validate_visual_data(declaration, payload, arrays)``

See docs/reference/MONITOR_CONTRACT.md.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "MonitorManager": "src.app.monitoring.runtime:MonitorManager",
    **{name: f"src.app.monitoring.api:{name}" for name in (
        "preserve_rng", "Result", "Context", "ProbeSet", "ModelAccess", "state_digest", "write_snapshot")},
    **{name: f"src.app.monitoring.measurements:{name}" for name in (
        "action_difference", "capture", "effective_rank", "layers_for", "linear_cka", "replace_units",
        "validate_params")},
    "validate_visual_data": "src.app.telemetry.visuals:validate_visual_data",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
