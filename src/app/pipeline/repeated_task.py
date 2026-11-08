"""Run a resolved configuration through the shared variance executor, then score it."""
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, open_dict

from src.app.pipeline.variability import make_plan, run_execution
from src.app.services.cancellation import TrainingCancelled


def execute_resolved(cfg, run, on_start, on_wait, *, script):
    """The same resource admission/cancellation path as ordinary training."""
    from src.app.pipeline.datasets import run_experiment
    directory = Path(run["logs"]) / run["agent"] / run["version"]
    directory.mkdir(parents=True, exist_ok=True)
    full = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(full):
        full.objective = None
        full.theta_tuning = None
        full.theta_execution = None
        full.theta_objective_runner = None
        full.variability = None
        full.hydra = OmegaConf.create(OmegaConf.to_container(HydraConfig.get(), resolve=False))
        full.hydra.mode = "RUN"
        full.hydra.sweep.dir = str(directory.resolve())
        full.hydra.sweep.subdir = "."
        full.hydra.job.chdir = False
    config_path, result_path = directory / "sample.yaml", directory / "sample_result.json"
    OmegaConf.save(full, config_path)
    targets = [f"replicate:{run['run_id']}", f"method:{run['method']}"]
    targets.extend(json.loads(os.environ.get("THETA_CANCEL_TARGETS", "[]")))
    runtime = cfg.get("theta_tuning") or {}
    if runtime.get("log_version"):
        targets.append(f"trial:{runtime['log_version']}")
    task_config = OmegaConf.to_container(cfg, resolve=True)
    task_config["entrypoint"] = script
    from src.app.core.provider_loader import resolve_resource
    run_experiment(["src.app.pipeline.task_worker", str(resolve_resource(script)),
                    str(config_path.resolve()), str(result_path.resolve())],
                   script_entrypoint="-m", python_executable=sys.executable,
                   task_config=task_config, output_log=run["output_log"],
                   cancel_targets=targets, on_start=on_start, on_wait=on_wait, resource_ticket=run.get("resource_ticket"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if "error" in result:
        raise RuntimeError(result["error"])


def run_objective(cfg, *, execute=None):
    """Compatibility API: execution is shared; the scoring component only sees results."""
    from src.app.objectives.api import validate_objective
    from src.app.objectives.evaluation import score_execution
    from src.app.pipeline.trial_artifacts import trial_log_version, trial_checkpoint_dir_from_config
    if cfg.get("objective") is not None:
        validate_objective(cfg)
    if os.environ.get("THETA_RESOURCE_POOL"):
        from src.app.services.resource_budget import ResourcePool
        ResourcePool(os.environ["THETA_RESOURCE_POOL"], os.environ.get("THETAIDE_SETTINGS_FILE")).release_coordinator_reservation(os.getpid())
    spec = cfg.get("theta_execution") or cfg.get("theta_objective_runner") or {}
    script = str(spec["script"])
    name = str(OmegaConf.select(cfg, "agent.name") or OmegaConf.select(cfg, "model.name") or "run").replace("/", "_")
    planning_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if planning_cfg.get("variability") is None:
        with open_dict(planning_cfg):
            planning_cfg.variability = {"training_runs": 1, "starting_seed": int(cfg.get("seed", 1))}
    plan = make_plan(planning_cfg, {"methods": {name: {"entrypoint": script}}, "is_interactive": True})
    version = trial_log_version(cfg) or "version_execution_" + uuid.uuid4().hex[:12]
    runtime = cfg.get("theta_tuning") or {}
    parent = {"kind": "trial" if runtime.get("study_name") else "method", "method": name,
              "study": runtime.get("study_name"), "trial_number": runtime.get("trial_number"), "log_version": version}
    logs = Path("results/logs") / str(cfg.group) / str(cfg.experiment_id) / name
    output = logs / version
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path("results/checkpoints") / str(cfg.group) / str(cfg.experiment_id) / name
    artifact_root = trial_checkpoint_dir_from_config(checkpoint_root, cfg) or checkpoint_root / version
    artifact_root.mkdir(parents=True, exist_ok=True)

    def dispatch(child, run, on_start, on_wait):
        if execute is None:
            execute_resolved(child, run, on_start, on_wait, script=script)
        else:
            # Test injection follows the same lifecycle without starting an OS process.
            with open_dict(child):
                child.objective = None
                child.theta_tuning = None
            directory = Path(run["logs"]) / run["agent"] / run["version"]
            directory.mkdir(parents=True, exist_ok=True)
            on_start(os.getpid())
            execute(child, directory)

    path = run_execution(cfg, {"methods": {name: {"entrypoint": script}}}, plan,
                         dispatch=dispatch, parent=parent, raise_on_failure=False)
    execution = json.loads(path.read_text(encoding="utf-8"))
    if any(run["status"] == "cancelled" for run in execution["runs"]):
        raise TrainingCancelled()
    from src.app.pipeline.objective import objective_metric
    default_metric = "eval/reward" if cfg.get("paradigm") == "online_rl" else "val/loss"
    metric = objective_metric(cfg, default_metric)
    scoring_cfg = cfg
    if cfg.get("objective") is None:
        # Repeated training is independent of custom objectives. The existing
        # reported-metric mode takes the mean final scalar across its seeds.
        scoring_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        with open_dict(scoring_cfg):
            scoring_cfg.objective = {"component": "final_metric", "params": {"metric": metric}}
    result, result_path = score_execution(scoring_cfg, path, output, artifact_root)
    from src.app import telemetry
    run = telemetry.init(str(cfg.group), str(cfg.experiment_id), name,
                         config=OmegaConf.to_container(cfg, resolve=True), version=version)
    run.log({metric: result.value}, step=0)
    run.artifact(result_path, kind="objective")
    run.finish()
    print(f"Score {metric}: {result.value:g} ({len(execution['runs'])} training runs)", flush=True)
    return result.value


@hydra.main(version_base=None, config_path="../../../in/config", config_name="config")
def main(cfg):
    try:
        return run_objective(cfg)
    except TrainingCancelled:
        return {"theta_trial_status": "pruned"}


if __name__ == "__main__":
    main()
