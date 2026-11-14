import os
import sys
from pathlib import Path

import argparse
import importlib
import pkgutil
import shutil

from plot.base import BasePlotter, deep_update, create_plotter
from src.app.core.plotter_registry import (
    PLOTTER_REGISTRY,
    auto_discover_plotters,
    get_plotter_class,
    list_registered_plotters,
)

from src.app.pipeline.plot_config import GENERAL_PLOT_OPTIONS, requested_plots


def discover_plotters(*, paths=None) -> dict:
    """Auto-discover all BasePlotter subclasses via the central Plotter Registry."""
    return {name: None for name in list_registered_plotters(paths=paths)}


def get_requested_plots(exp_cfg: dict, exp_id: str) -> dict:
    """Return the plotters explicitly selected by the composed experiment."""
    return requested_plots(exp_cfg)


def run_experiment_plots(
    exp_id: str,
    exp_config_name: str = None,
    style: str = None,
    wipe: bool = False,
    use_cache: bool = False,
    *, paths=None,
):
    print("\n==================================================")
    print(f"=== Auto-Generating Plots for Experiment: {exp_id} ===")
    print("==================================================")

    dummy_plotter = BasePlotter("manager", paths=paths)
    paths = dummy_plotter.workspace
    exp_cfg = dummy_plotter.get_experiment_config(exp_id, exp_config_name=exp_config_name)
    requested_modules = get_requested_plots(exp_cfg, exp_id)
    if not requested_modules:
        print("No figures or tables configured; skipping plotting.")
        return
    group = dummy_plotter.get_group(exp_id, exp_cfg)
    clean_exp = Path(exp_id).stem
    plots_sec = exp_cfg.get("plots", {})
    if isinstance(plots_sec, dict) and "output_dir" in plots_sec and plots_sec["output_dir"]:
        custom_dir = str(plots_sec["output_dir"]).format(group=group, exp_id=exp_id, clean_exp=clean_exp)
        output_dir = Path(custom_dir)
    elif "output_dir" in exp_cfg and exp_cfg["output_dir"]:
        custom_dir = str(exp_cfg["output_dir"]).format(group=group, exp_id=exp_id, clean_exp=clean_exp)
        output_dir = Path(custom_dir)
    else:
        result_id = exp_id.removeprefix(f"{group}/")
        output_dir = (paths.results / "plots") / group / result_id

    if wipe and output_dir.exists():
        print(f"Wiping existing plot directory: {output_dir}")
        shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = discover_plotters(paths=paths)
    effective_use_cache = use_cache or bool(exp_cfg.get("use_cache", False))

    for module_name, overrides in requested_modules.items():
        try:
            plotter_cls = get_plotter_class(module_name, paths=paths)
        except ValueError as exc:
            print(f"Cannot resolve plotter '{module_name}': {exc}")
            continue
        if plotter_cls is not None:
            plotter = create_plotter(plotter_cls, paths)
            try:
                plot_overrides = dict(overrides) if isinstance(overrides, dict) else {}
                configured = exp_cfg.get("plot_overrides", {})
                if isinstance(configured, dict) and isinstance(configured.get(module_name), dict):
                    plot_overrides = deep_update(plot_overrides, configured[module_name])
                if style:
                    plot_overrides["style"] = style
                if effective_use_cache:
                    plot_overrides["use_cache"] = True
                plotter.run(exp_id, cli_overrides=plot_overrides if plot_overrides else None)
            except Exception as e:
                print(f"Error running plotter '{module_name}' for '{exp_id}': {e}")
        else:
            print(f"Warning: Unknown plotter module '{module_name}' requested. Available: {list(registry.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orchestrate Experiment Plot Generation")
    parser.add_argument("experiment_id", type=str, help="Experiment ID to generate plots for")
    parser.add_argument(
        "--experiment",
        "-c",
        "--config",
        dest="experiment",
        type=str,
        default=None,
        help="Base experiment config name if different from experiment_id",
    )
    parser.add_argument("--style", type=str, default=None, help="Plot style config")
    parser.add_argument(
        "--wipe", "-w", action="store_true", help="Wipe existing plot directory before generating plots"
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        default=False,
        help="Plot from cached evaluation data instead of recomputing from checkpoints",
    )
    args = parser.parse_args()

    run_experiment_plots(
        args.experiment_id,
        exp_config_name=args.experiment,
        style=args.style,
        wipe=args.wipe,
        use_cache=args.use_cache,
    )
