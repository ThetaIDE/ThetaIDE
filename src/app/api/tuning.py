"""Portable tuner execution, objectives and trial artifacts.

See docs/reference/TUNER_EXECUTION_CONTRACT.md for the execution-plan contract.

Execution plans and artifacts
    ``TunerExecutionPlan``, ``TunerArtifact``, ``TunerLaunchBindings``,
    ``prepare_plan``, ``launch_bindings``, ``apply_overrides``, ``record_plan``,
    ``record_config``, ``record_overrides``, ``sqlite_snapshot``,
    ``check_artifact_admission``, ``request_artifact_snapshot``, ``commit_artifact``

Objectives and trials
    ``objective_metric``, ``objective_value``, ``get_sweep_direction``,
    ``trial_log_version``, ``trial_checkpoint_dir``, ``finish_trial_runs``
"""
from ._exports import lazy_exports

_PLAN = "src.app.pipeline.tuner_execution"
_EXPORTS = {
    **{name: f"{_PLAN}:{name}" for name in (
        "TunerExecutionPlan", "TunerArtifact", "TunerLaunchBindings", "prepare_plan", "launch_bindings",
        "apply_overrides", "record_plan", "record_config", "record_overrides", "sqlite_snapshot",
        "check_artifact_admission", "request_artifact_snapshot", "commit_artifact",
    )},
    "objective_metric": "src.app.pipeline.objective:objective_metric",
    "objective_value": "src.app.pipeline.objective:objective_value",
    "get_sweep_direction": "src.app.pipeline.commands:get_sweep_direction",
    "trial_log_version": "src.app.pipeline.trial_artifacts:trial_log_version",
    "trial_checkpoint_dir": "src.app.pipeline.trial_artifacts:trial_checkpoint_dir",
    "finish_trial_runs": "src.app.pipeline.variability:finish_trial_runs",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
