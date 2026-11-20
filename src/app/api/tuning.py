"""Portable tuner execution, objectives and trial artifacts.

See docs/reference/TUNER_EXECUTION_CONTRACT.md for the execution-plan contract.

Execution plans and artifacts
    ``TunerExecutionPlan``, ``TunerArtifact``, ``TunerLaunchBindings``,
    ``prepare_plan``, ``launch_bindings``, ``apply_overrides``, ``record_plan``,
    ``record_config``, ``record_overrides``, ``sqlite_snapshot``,
    ``check_artifact_admission``, ``request_artifact_snapshot``, ``commit_artifact``

Objectives and trials
    ``ObjectiveResult``, ``ObjectiveProgress``, ``RunResult``, ``TrialResults`` (objective component inputs, final and intermediate results),
    ``objective_metric``, ``objective_value``,
    ``trial_log_version``, ``trial_checkpoint_dir``, ``trial_checkpoint_dir_from_config``,
    ``finish_trial_runs``, ``trial_monitor`` (reports intermediate values and pruning for the active trial)
    ``tuning_enabled(cfg)`` whether the composed configuration runs a tuner
"""
from ._exports import lazy_exports

_PLAN = "src.app.pipeline.tuner_execution"
_EXPORTS = {
    **{name: f"{_PLAN}:{name}" for name in (
        "TunerExecutionPlan", "TunerArtifact", "TunerLaunchBindings", "prepare_plan", "launch_bindings",
        "apply_overrides", "record_plan", "record_config", "record_overrides", "sqlite_snapshot",
        "check_artifact_admission", "request_artifact_snapshot", "commit_artifact",
    )},
    **{name: f"src.app.objectives.api:{name}" for name in ("ObjectiveResult", "ObjectiveProgress", "RunResult", "TrialResults")},
    "objective_metric": "src.app.pipeline.objective:objective_metric",
    "objective_value": "src.app.pipeline.objective:objective_value",
    "trial_log_version": "src.app.pipeline.trial_artifacts:trial_log_version",
    "trial_checkpoint_dir": "src.app.pipeline.trial_artifacts:trial_checkpoint_dir",
    "trial_checkpoint_dir_from_config": "src.app.pipeline.trial_artifacts:trial_checkpoint_dir_from_config",
    "trial_monitor": "src.app.pipeline.tuners:trial_monitor",
    **{name: f"src.app.pipeline.local_budget_tuning:{name}" for name in (
        "LocalBudgetTuner", "LocalBudgetSweeper", "hydra_value", "progress",
    )},
    "trial_worker_results": "src.app.pipeline.execution_data:trial_worker_results",
    "runner_tuning_profile": "src.app.core.execution:runner_tuning_profile",
    "finish_trial_runs": "src.app.pipeline.variability:finish_trial_runs",
    "tuning_enabled": "src.app.pipeline.tuning_config:tuning_enabled",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
