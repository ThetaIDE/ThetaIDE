"""Local training phase runner.

Exposes composable phase functions driven by the structured methods: dict.
Each method in cfg.methods is dispatched individually with proper overrides.
"""

import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.app.workspace import WorkspacePaths

from src.app.pipeline.commands import build_method_overrides
from src.app.pipeline.config import normalize_agent_name
from src.app.pipeline.datasets import fast_purge_dir, resolve_dataset_for_method, run_experiment
from src.app.pipeline.engine import resolve_engine
from src.app.pipeline.experiment_snapshot import save_experiment_snapshot, save_run_config
from src.app.pipeline.tuners import get_tuner, tuner_name
from src.app.pipeline.validation import paradigm_uses_static_dataset
from src.app.pipeline.plot_config import should_run_plotting
from src.app.pipeline.tuning_config import require_runnable_tuning, tuning_enabled

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_output_dirs(cfg, *, paths=None) -> None:
    """Purge and recreate checkpoint, log, and plot directories unless recovering."""
    paths = paths or WorkspacePaths.for_process()
    if cfg.get("variability_run_id"):
        # A replicate owns only its method/version subtree. Its parent prepares
        # the shared experiment roots once before launching any workers.
        agent_cfg = cfg.get("agent", {})
        agent_name = agent_cfg.get("name") if hasattr(agent_cfg, "get") else None
        agent_name = str(agent_name or cfg.get("variability_method", "agent")).replace("/", "_")
        version = str(cfg.get("variability_version", "version_0"))
        for base in ((paths.results / "checkpoints"), (paths.results / "logs")):
            run_dir = base / str(cfg.group) / str(cfg.experiment_id) / agent_name / version
            fast_purge_dir(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        return
    if cfg.get("recover", False) or tuning_enabled(cfg):
        return

    from src.app.pipeline.result_paths import nested_run_names
    saved = nested_run_names((paths.results / "logs") / cfg.group / cfg.experiment_id)

    def prepare(path):
        if saved and path.is_dir():
            for child in path.iterdir():
                if child.name in saved:
                    continue
                if child.is_dir() and not child.is_symlink():
                    fast_purge_dir(child)
                else:
                    child.unlink()
        else:
            fast_purge_dir(path)
        path.mkdir(parents=True, exist_ok=True)

    prepare((paths.results / "checkpoints") / cfg.group / cfg.experiment_id)
    prepare((paths.results / "logs") / cfg.group / cfg.experiment_id)
    if should_run_plotting(cfg):
        prepare((paths.results / "plots") / cfg.group / cfg.experiment_id)


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


def run_methods(cfg, context) -> None:
    """Run methods one at a time, or launch one training process per method concurrently."""
    context = {**context, "workspace": context.get("workspace") or WorkspacePaths.for_process()}
    methods = context["methods"]
    require_runnable_tuning(cfg, methods)
    parallel = bool(cfg.get("run_in_parallel", False) and len(methods) > 1)
    workers = len(methods) if parallel else min(1, len(methods))
    resource_pool = None
    tickets = {}
    if os.environ.get('THETA_RESOURCE_POOL'):
        from src.app.services.resource_budget import ResourcePool, orchestration_workers
        resource_pool = ResourcePool(os.environ['THETA_RESOURCE_POOL'], os.environ.get('THETAIDE_SETTINGS_FILE'))
        workers = min(workers, orchestration_workers(len(methods)))
        tickets = dict(zip(methods, resource_pool.enqueue(workers,
                          parent=os.environ.get('THETA_RESOURCE_QUEUE_PARENT'))))

    def execute(name, settings):
        if resource_pool and name not in tickets:
            tickets[name] = resource_pool.enqueue()[0]
        try:
            return _run_methods_sequential(cfg, {**context, "methods": {name: settings},
                                                "resource_ticket": tickets.get(name)})
        finally:
            if resource_pool:
                resource_pool.dequeue(tickets[name])

    try:
        if not parallel:
            for name, settings in methods.items():
                execute(name, settings)
            return
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="theta-method") as pool:
            futures = [pool.submit(execute, name, settings) for name, settings in methods.items()]
            for future in as_completed(futures):
                future.result()
    finally:
        if resource_pool:
            for ticket in tickets.values():
                resource_pool.dequeue(ticket)


def _run_methods_sequential(cfg, context) -> None:
    """Execute all methods declared in cfg.methods."""
    paths = context.get("workspace") or WorkspacePaths.for_process()
    methods = context["methods"]
    require_runnable_tuning(cfg, methods)
    sanitized_extra_args = context["sanitized_extra_args"]
    storage_url = context["storage_url"]
    paradigm = cfg.get("paradigm", "offline_rl")

    for method_name, method_cfg in methods.items():
        if hasattr(method_cfg, "items"):
            from omegaconf import DictConfig, OmegaConf

            if isinstance(method_cfg, DictConfig):
                method_cfg = OmegaConf.to_container(method_cfg, resolve=True)
            else:
                method_cfg = dict(method_cfg)

        agent_name = normalize_agent_name(method_name)
        method_is_sweep = tuning_enabled(cfg)

        study_name = None
        tuner = None
        if method_is_sweep:
            tuner = get_tuner(cfg, paths=paths)
            study_name = tuner.study_name(cfg, agent_name)

        # Resolve dataset for paradigms that train from a file rather than a simulator
        dataset_path = None
        if paradigm_uses_static_dataset(paradigm, paths=paths):
            try:
                dataset_path = resolve_dataset_for_method(method_name, method_cfg, cfg, paths=paths)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(1)
            print(f"Using dataset from: {dataset_path}")

        agent_str = f"agent={method_cfg.get('agent')}, " if method_cfg.get("agent") else ""
        mode_str = f" [{tuner_name(cfg)} sweep]" if method_is_sweep else ""
        print(f"\n=== Training: {method_name} ({agent_str}model={method_cfg.get('model')}){mode_str} ===")

        overrides = build_method_overrides(
            method_name=method_name,
            method_cfg=method_cfg,
            dataset_path=dataset_path,
            extra_args=sanitized_extra_args,
            cfg=cfg,
            study_name=study_name,
            is_sweep=method_is_sweep,
            workspace=paths,
        )

        if method_is_sweep:
            tuner.prepare(cfg, storage_url, study_name)

        engine_script, engine_python = resolve_engine(method_cfg, cfg, paths=paths)
        if tuner is not None:
            engine_script, engine_python = tuner.job_entrypoint(cfg, method_cfg, engine_script, engine_python)
        repeated = cfg.get("objective") is not None or (method_is_sweep and cfg.get("variability") is not None)
        if repeated:
            import json
            from omegaconf import OmegaConf
            from src.app.pipeline.commands import _format_hydra_val
            def objective_override(value):
                if isinstance(value, str):
                    return json.dumps(value)
                if isinstance(value, dict):
                    return "{" + ",".join(f"{key}:{objective_override(item)}" for key, item in value.items()) + "}"
                if isinstance(value, list):
                    return "[" + ",".join(objective_override(item) for item in value) + "]"
                return _format_hydra_val(value)
            # Carry resolved group defaults as well as experiment-local settings.
            for key in ("objective", "variability"):
                value = cfg.get(key)
                if value is not None:
                    value = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
                    overrides.append(f"++{key}={objective_override(value)}")
            engine_script = "src/app/pipeline/repeated_task.py"
        from src.app.services.cancellation import TrainingCancelled
        targets = [f"method:{agent_name}"]
        for arg in sanitized_extra_args:
            if arg.lstrip("+").startswith("variability_run_id="):
                targets.append("replicate:" + arg.split("=", 1)[1])
        try:
            run_experiment(
                overrides,
                paths=paths,
                site_cfg=getattr(cfg, "site", None),
                script_entrypoint=engine_script,
                coordinator=repeated,
                python_executable=engine_python,
                output_log=context.get("output_log"),
                cancel_targets=targets,
                **({"on_start": context["on_worker_start"]} if context.get("on_worker_start") else {}),
                **({"resource_ticket": context["resource_ticket"]} if context.get("resource_ticket") else {}),
                **({"on_wait": context["on_worker_wait"]} if context.get("on_worker_wait") else {}),
            )
        except TrainingCancelled:
            if context.get("propagate_cancellation"):
                raise
            if repeated:
                import os
                from src.app.pipeline.variability import finish_trial_runs
                finish_trial_runs(cfg.group, cfg.experiment_id, None, "cancelled",
                                  owner_pid=int(os.environ.get("THETA_PIPELINE_PID", os.getpid())), method=agent_name)
            cancel = getattr(tuner, "cancel", None)
            if callable(cancel):
                cancel(cfg, agent_name, storage_url, study_name)
            print(f"Cancelled {method_name}", flush=True)
            continue
        if method_is_sweep:
            tuner.finalize(cfg.group, cfg.experiment_id, agent_name, storage_url, study_name)



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def run_plotting_phase(cfg, context) -> None:
    """Run automated plotting after training completes."""
    # Replicate plotting is deferred to the variability coordinator, which runs
    # this phase once after all workers finish and the aggregate is complete.
    if cfg.get("variability_run_id") or not should_run_plotting(cfg):
        return
    from src.app.pipeline.datasets import run_plotting

    site_cfg = getattr(cfg, "site", None)
    run_plotting(
        cfg.experiment_id,
        style=cfg.get("plot_style", None),
        base_experiment=cfg.get("experiment_name", ""),
        site_cfg=site_cfg,
        paths=context.get("workspace"),
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_local_training(cfg, context) -> None:
    """Execute all phases sequentially: setup → methods → plot."""
    require_runnable_tuning(cfg, context.get("methods", {}))
    paths = context.get("workspace") or WorkspacePaths.for_process()
    # Every phase and worker uses the same selected workspace, never a later CWD guess.
    context = {**context, "workspace": paths}
    _setup_output_dirs(cfg, paths=paths)
    save_experiment_snapshot(cfg, context)
    save_run_config(cfg, paths=paths)
    run_methods(cfg, context)

    if should_run_plotting(cfg):
        run_plotting_phase(cfg, context)
