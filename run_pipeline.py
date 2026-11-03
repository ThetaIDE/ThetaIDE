"""Unified experiment pipeline orchestrator for Theta-IDE.

Coordinates experiment methods, parameter sweeps, plotting, and cluster submissions.
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
for p in [
    PROJECT_ROOT,
    SRC_DIR,
    os.path.join(SRC_DIR, "app"),
    os.path.join(SRC_DIR, "usr"),
    os.path.join(SRC_DIR, "usr", "models"),
    os.path.join(SRC_DIR, "usr", "environments"),
    os.path.join(SRC_DIR, "usr", "eval"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

from omegaconf import OmegaConf

from src.app.pipeline.compose import CONFIG_DIR, compose_experiment, validate_composed
from src.app.pipeline.config import normalize_agent_name, parse_methods_dict, resolve_experiment_config_name
from src.app.pipeline.datasets import run_plotting
from src.app.pipeline.exceptions import ConfigurationError
from src.app.pipeline.slurm import generate_sbatch_header, submit_sbatch


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: python run_pipeline.py <group>/<experiment_id> [Hydra Overrides]")
        print("Standard Orchestration Overrides (use Hydra syntax, e.g. key=value):")
        print(
            "  site=local            Interactive CLI execution (default, for local machine or cluster interactive node)"
        )
        print("  site=ncshare          Slurm cluster execution on NCShare (GPU consolidated, default)")
        print("  site=ncshare_gpu      Slurm cluster on NCShare GPU partition (consolidated)")
        print("  site=ncshare_common   Slurm cluster on NCShare CPU common partition (one job per method)")
        print("  site=arc              Slurm cluster execution on ARC")
        print("  plot_only=true        Run plotting phase only (no training, no Slurm submission)")
        print("  no_plot=true          Skip automatic plotting")
        print("  no_online=true        Skip online training phase")
        print("  no_offline=true       Skip offline training phase")
        print("  dry_run=true          Validate config and exit")
        print("  sweep=true            Run Optuna hyperparameter sweep")
        print("  dash=true             Launch Optuna dashboard during run")
        print("  dash_only=true        Launch Optuna dashboard and exit")
        print("  remake=true           Force recalculation of summaries")
        print("  consolidate=true      Consolidate Slurm jobs into 1 single GPU job")
        print("  experiment_id=...     Override the experiment directory name")
        sys.exit(0)

    raw_experiment = sys.argv[1]
    extra_args = sys.argv[2:]

    # Direct CLI experiment component export
    if raw_experiment == "export":
        if not extra_args:
            print("Usage: python run_pipeline.py export <group>/<experiment_id> [--include-weights] [--out bundle.zip]")
            sys.exit(1)
        target_exp = extra_args[0]
        include_weights = "--include-weights" in extra_args or "include_weights=true" in extra_args
        out_path = None
        for i, a in enumerate(extra_args):
            if a in ("--out", "-o") and i + 1 < len(extra_args):
                out_path = Path(extra_args[i + 1])
            elif a.startswith("out="):
                out_path = Path(a.split("=", 1)[1])

        from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions

        exporter = ExperimentExporter(workspace_root=Path(PROJECT_ROOT))
        opts = ExportOptions(include_checkpoints=include_weights, output_path=out_path)
        res = exporter.export(target_exp, options=opts)
        if res.success:
            print(f"Successfully exported experiment '{target_exp}' to {res.archive_path}")
            print(f"Total bundled assets: {len(res.files_included)}")
            sys.exit(0)
        else:
            print(f"Export failed: {res.error}")
            sys.exit(1)

    # Direct CLI component scaffolding
    if raw_experiment in ("scaffold", "new-plugin"):
        if len(extra_args) < 2:
            print("Usage: python run_pipeline.py scaffold <kind> <name> [--author '...'] [--desc '...']")
            print("Available kinds: model, agent, env, datamodule, evaluator")
            sys.exit(1)
        kind = extra_args[0].lower()
        name = extra_args[1]
        author = "Theta Researcher"
        desc = ""
        for i, a in enumerate(extra_args):
            if a == "--author" and i + 1 < len(extra_args):
                author = extra_args[i + 1]
            elif a == "--desc" and i + 1 < len(extra_args):
                desc = extra_args[i + 1]

        from src.app.services.plugin_scaffolder import PluginScaffolder, ScaffoldOptions

        scaffolder = PluginScaffolder(workspace_root=Path(PROJECT_ROOT))
        opts = ScaffoldOptions(name=name, kind=kind, author=author, description=desc)
        res = scaffolder.scaffold(opts)
        if res.success:
            print(f"Successfully scaffolded {kind} plugin '{name}' at {res.target_dir}")
            print("Generated files:")
            for f in res.created_files:
                print(f"  - {f}")
            sys.exit(0)
        else:
            print(f"Scaffolding failed: {res.error}")
            sys.exit(1)

    # Resolve experiment config path
    experiment_arg = resolve_experiment_config_name(raw_experiment)

    if any(a.lower() in ("export=true", "export=1", "--export") for a in extra_args):
        include_weights = any(a.lower() in ("include_weights=true", "include_checkpoints=true") for a in extra_args)
        from src.app.services.experiment_exporter import ExperimentExporter, ExportOptions

        exporter = ExperimentExporter(workspace_root=Path(PROJECT_ROOT))
        opts = ExportOptions(include_checkpoints=include_weights)
        res = exporter.export(experiment_arg, options=opts)
        if res.success:
            print(f"Successfully exported experiment '{experiment_arg}' to {res.archive_path}")
            print(f"Total bundled assets: {len(res.files_included)}")
            sys.exit(0)
        else:
            print(f"Export failed: {res.error}")
            sys.exit(1)

    # Load configuration (shared with the HTTP API so GUI and CLI configs compose identically)
    try:
        print(f"Loading Hydra configuration for '{experiment_arg}'...", flush=True)
        composed = compose_experiment(experiment_arg, extra_args)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    cfg = composed.cfg
    sanitized_extra_args = composed.launch_args
    is_sweep = composed.is_sweep

    # Pre-flight validation
    try:
        notices = validate_composed(composed)
        for n in notices:
            print(f"[Config Notice] {n}")
    except ConfigurationError as e:
        print(f"\n{e}\n")
        sys.exit(1)

    if cfg.get("dry_run", False):
        print(f"\n[Validation Success] Experiment config '{experiment_arg}' is valid and ready to run.")
        sys.exit(0)

    if cfg.get("plot_only", False):
        from src.app.pipeline.plot_config import should_run_plotting
        if not should_run_plotting(cfg):
            print("No figures or tables configured; skipping plotting.")
            sys.exit(0)
        print(f"\n=== Running Plotting Phase Only for '{experiment_arg}' ===")
        run_plotting(
            experiment=experiment_arg,
            style=cfg.get("plot_style", None),
            wipe=cfg.get("wipe", False),
            use_cache=cfg.get("use_cache", False),
        )
        sys.exit(0)

    # Load paradigm definition for component assembly
    paradigm_def = None
    try:
        from src.app.core.paradigm_loader import load_paradigm_definition

        paradigm_name = cfg.get("paradigm", None)
        if paradigm_name:
            paradigm_def = load_paradigm_definition(paradigm_name)
    except Exception as e:
        # Paradigm loading is best-effort during transition; log but don't abort
        print(f"[Notice] Could not load paradigm definition: {e}")

    # Execution mode is determined solely by the site profile: site=local -> interactive CLI, any other site -> Slurm cluster
    site_name = None
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "runtime") and hasattr(cfg.hydra.runtime, "choices"):
        site_name = cfg.hydra.runtime.choices.get("site")
    if not site_name and hasattr(cfg, "site"):
        site_name = getattr(cfg.site, "name", None)
    if not site_name:
        site_name = "local"

    if hasattr(cfg, "site") and OmegaConf.is_config(cfg.site) and "name" not in cfg.site:
        import omegaconf

        with omegaconf.open_dict(cfg.site):
            cfg.site.name = site_name

    is_interactive = site_name == "local"
    print(f"Execution Mode: {'Interactive (Local CLI)' if is_interactive else f'Slurm Cluster ({site_name})'}")

    methods_dict = parse_methods_dict(cfg)
    from src.app.pipeline.tuners import get_tuner, tuner_name

    uses_tuning = is_sweep or any(m.get("tune") or m.get("search_space") for m in methods_dict.values())
    tuner = get_tuner(cfg) if uses_tuning or cfg.get("dash") or cfg.get("dash_only") else None
    storage_url = tuner.storage(cfg) if tuner else None
    if is_interactive and storage_url and (cfg.get("dash") or cfg.get("dash_only")):
        dashboard = getattr(tuner, "launch_dashboard", None)
        if not callable(dashboard):
            raise ConfigurationError(f"Tuning component {tuner_name(cfg)!r} does not provide a dashboard.")
        dashboard(storage_url)
        if cfg.get("dash_only"):
            print("Dashboard running in persistent mode. Press Ctrl+C to exit.")
            import time

            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                sys.exit(0)

    if cfg.get("dash_only"):
        print("Error: Could not find tuning storage URL in configuration.")
        sys.exit(1)

    # Parse structured methods: dict with shared params support
    if not methods_dict:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_arg}' has no runnable methods declared under 'methods:'. "
            f"Declare methods as a dict with agent + model per entry."
        )

    # Reject legacy keys
    for legacy_key in ("online_methods", "offline_methods", "offline_datasets"):
        if cfg.get(legacy_key, None):
            raise ConfigurationError(
                f"[ConfigurationError] Legacy key '{legacy_key}' found in config. Use the 'methods:' dict instead."
            )

    print("Declared Methods:")
    from src.app.pipeline.tuning_config import tuning_disabled

    for name, mcfg in methods_dict.items():
        agent_str = f"agent={mcfg.get('agent')}, " if mcfg.get("agent") else ""
        tune_str = f" [{tuner_name(cfg)} sweep]" if (is_sweep or ((mcfg.get("tune") or mcfg.get("search_space")) and not tuning_disabled(cfg))) else ""
        print(f"  {name}: {agent_str}model={mcfg.get('model')}{tune_str}")

    # Build context for tasks
    context = {
        "experiment_yaml": (CONFIG_DIR / "experiment" / f"{composed.experiment}.yaml").read_bytes(),
        "is_interactive": is_interactive,
        "site_name": site_name,
        "sanitized_extra_args": sanitized_extra_args,
        "storage_url": storage_url,
        "is_sweep": is_sweep,
        "methods": methods_dict,
        "paradigm_def": paradigm_def,
    }

    from src.app.pipeline.variability import make_plan, run_execution

    variability_plan = make_plan(cfg, context)

    # Warn if stale task: key is present in config
    stale_task = cfg.get("task", None)
    if stale_task:
        print(
            f"[Deprecation Warning] 'task: {stale_task}' is no longer used. "
            f"Routing is now handled by 'paradigm:' and 'workflow:'. "
            f"Remove 'task:' from your experiment config."
        )

    # Dispatch: workflow engine or direct paradigm runner
    workflow_id = cfg.get("workflow", None)
    if variability_plan is not None:
        run_execution(cfg, context, variability_plan)
    elif workflow_id:
        from src.app.pipeline.workflow.executor import run_workflow

        from src.app.pipeline.experiment_snapshot import save_run_config

        save_run_config(cfg)
        run_workflow(workflow_id, cfg, context)
    elif is_interactive:
        from src.app.pipeline.local_runner import run_local_training

        run_local_training(cfg, context)
    else:
        from src.app.pipeline.slurm_runner import run_slurm_training

        run_slurm_training(cfg, context)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help"):
        print(f"Initializing Theta pipeline for: {sys.argv[1]} ...", flush=True)
    main()
