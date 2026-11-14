"""Compatibility entry point for the allocation-based SLURM executor."""
from src.app.sites.execution import submit
from src.app.workspace import WorkspacePaths


def run_slurm_training(cfg, context):
    record = submit((context.get("workspace") or WorkspacePaths.for_process()).root, cfg.experiment_name,
                    context.get("sanitized_extra_args", []), cfg)
    return [record["job_id"]], [], True
