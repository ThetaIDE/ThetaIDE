"""Local training phase runner.

Exposes composable phase functions driven by the structured methods: dict.
Each method in cfg.methods is dispatched individually with proper overrides.
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.app.pipeline.commands import build_method_overrides
from src.app.pipeline.config import normalize_agent_name
from src.app.pipeline.datasets import fast_purge_dir, resolve_dataset_for_method, run_experiment
from src.app.pipeline.engine import resolve_engine
from src.app.pipeline.experiment_snapshot import save_experiment_snapshot, save_run_config
from src.app.pipeline.tuners import get_tuner, tuner_name
from src.app.pipeline.validation import paradigm_uses_static_dataset
from src.app.pipeline.plot_config import should_run_plotting
from src.app.pipeline.tuning_config import require_runnable_tuning, tuning_disabled

# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------


def _setup_output_dirs(cfg) -> None:
    """Purge and recreate checkpoint, log, and plot directories unless recovering."""
    if cfg.get("variability_run_id"):
        # A replicate owns only its method/version subtree. Its parent prepares
        # the shared experiment roots once before launching any workers.
        agent_cfg = cfg.get("agent", {})
        agent_name = agent_cfg.get("name") if hasattr(agent_cfg, "get") else None
        agent_name = str(agent_name or cfg.get("variability_method", "agent")).replace("/", "_")
        version = str(cfg.get("variability_version", "version_0"))
        for base in (Path("results/checkpoints"), Path("results/logs")):
            run_dir = base / str(cfg.group) / str(cfg.experiment_id) / agent_name / version
            fast_purge_dir(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
        return
    if cfg.get("recover", False) or cfg.get("tuning") is not None or cfg.get("sweep", False):
        return

    ckpt_dir = Path("results/checkpoints") / cfg.group / cfg.experiment_id
    fast_purge_dir(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp_log_dir = Path("results/logs") / cfg.group / cfg.experiment_id
    fast_purge_dir(exp_log_dir)
    exp_log_dir.mkdir(parents=True, exist_ok=True)

    if should_run_plotting(cfg):
        clean_exp = Path(cfg.experiment_id).stem
        exp_plot_dir = Path("results/plots") / cfg.group / clean_exp
        fast_purge_dir(exp_plot_dir)
        exp_plot_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Method dispatch
# ---------------------------------------------------------------------------


def run_methods(cfg, context) -> None:
    """Run methods one at a time, or launch one training process per method concurrently."""
    methods = context["methods"]
    require_runnable_tuning(cfg, methods)
    if not cfg.get("run_in_parallel", False) or len(methods) < 2:
        return _run_methods_sequential(cfg, context)

    with ThreadPoolExecutor(max_workers=len(methods), thread_name_prefix="theta-method") as pool:
        futures = [pool.submit(_run_methods_sequential, cfg,
                               {**context, "methods": {name: settings}})
                   for name, settings in methods.items()]
        for future in as_completed(futures):
            future.result()


def _run_methods_sequential(cfg, context) -> None:
    """Execute all methods declared in cfg.methods."""
    methods = context["methods"]
    require_runnable_tuning(cfg, methods)
    sanitized_extra_args = context["sanitized_extra_args"]
    storage_url = context["storage_url"]
    global_is_sweep = context.get("is_sweep", False)
    paradigm = cfg.get("paradigm", "offline_rl")

    for method_name, method_cfg in methods.items():
        if hasattr(method_cfg, "items"):
            from omegaconf import DictConfig, OmegaConf

            if isinstance(method_cfg, DictConfig):
                method_cfg = OmegaConf.to_container(method_cfg, resolve=True)
            else:
                method_cfg = dict(method_cfg)

        agent_name = normalize_agent_name(method_name)
        method_tune = method_cfg.get("tune") or method_cfg.get("search_space") or {}
        method_is_sweep = not tuning_disabled(cfg) and (bool(global_is_sweep) or bool(method_tune))

        study_name = None
        tuner = None
        if method_is_sweep:
            tuner = get_tuner(cfg)
            study_name = tuner.study_name(cfg, agent_name)

        # Resolve dataset for paradigms that train from a file rather than a simulator
        dataset_path = None
        if paradigm_uses_static_dataset(paradigm):
            try:
                dataset_path = resolve_dataset_for_method(method_name, method_cfg, cfg)
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
        )

        if method_is_sweep:
            tuner.prepare(cfg, storage_url, study_name)

        engine_script, engine_python = resolve_engine(method_cfg, cfg)
        if tuner is not None:
            engine_script, engine_python = tuner.job_entrypoint(cfg, method_cfg, engine_script, engine_python)
        run_experiment(
            overrides,
            site_cfg=getattr(cfg, "site", None),
            script_entrypoint=engine_script,
            python_executable=engine_python,
            output_log=context.get("output_log"),
        )
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
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_local_training(cfg, context) -> None:
    """Execute all phases sequentially: setup → methods → plot."""
    require_runnable_tuning(cfg, context.get("methods", {}))
    _setup_output_dirs(cfg)
    save_experiment_snapshot(cfg, context)
    save_run_config(cfg)
    run_methods(cfg, context)

    if should_run_plotting(cfg):
        run_plotting_phase(cfg, context)
