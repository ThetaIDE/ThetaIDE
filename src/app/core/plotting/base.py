import os
import sys

from src.app.workspace import WorkspacePaths
from src.app.workspace import PROJECT_ROOT as APPLICATION_ROOT
PROJECT_ROOT = str(APPLICATION_ROOT)
_DEFAULT_PROJECT_ROOT = PROJECT_ROOT

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Qualitative palette guaranteeing distinct colors for methods on the same plot
QUALITATIVE_PALETTE = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def clean_label(name: str, style_override: dict | None = None) -> str:
    """Return human-readable display label for a method name, honoring style overrides."""
    if style_override and isinstance(style_override, dict) and style_override.get("label"):
        return str(style_override["label"])
    return str(name)


def get_canonical_method_name(name: str) -> str:
    """Map method name to normalized form."""
    return str(name).replace("/", "_")


def get_method_aliases(name: str) -> set[str]:
    """Return all known aliases (raw and normalized) for a method identifier."""
    raw = str(name)
    return {raw, raw.replace("/", "_")}


def get_style(name: str, style_override: dict | None = None) -> dict:
    """Return style dictionary for a method name, incorporating overrides."""
    base = {"label": str(name), "color": None, "marker": "o", "linestyle": "-"}
    if style_override and isinstance(style_override, dict):
        for k in ("label", "color", "marker", "linestyle"):
            if style_override.get(k) is not None:
                base[k] = style_override[k]
    return base


def get_style_info(name: str, style_override: dict | None = None) -> tuple[str | None, str, str]:
    """Return (color, linestyle, marker) tuple for plotting."""
    s = get_style(name, style_override=style_override)
    return s.get("color"), s.get("linestyle", "-"), s.get("marker", "o")


def moving_average(a: np.ndarray, n: int = 5) -> np.ndarray:
    if len(a) == 0:
        return np.array([])
    n = min(len(a), max(1, n))
    a_padded = np.pad(a, (n - 1, 0), mode="edge")
    ret = np.cumsum(a_padded, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def deep_update(base: dict, update: dict) -> dict:
    """Recursively updates a nested dictionary."""
    result = dict(base)
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


@dataclass(frozen=True)
class PanelSpec:
    """One drawable panel a plotter can contribute to a composed figure.

    `draw` takes an Axes the caller owns and returns whether anything was
    drawn. Plotters that cannot draw into a supplied Axes report no panels,
    and the Plots pane falls back to capturing the figures they make
    themselves (see plot/capture.py).
    """

    key: str
    label: str
    draw: Callable[[Any], bool]


# Bounded API-1 adapter: older plotters have a no-argument constructor calling
# BasePlotter.__init__. Scope the host's explicit workspace without changing CWD/env.
from contextvars import ContextVar
_PLOTTER_WORKSPACE = ContextVar("theta_plotter_workspace", default=None)


def create_plotter(cls, paths):
    import inspect
    parameters = inspect.signature(cls).parameters
    if "paths" in parameters:
        return cls(paths=paths)
    token = _PLOTTER_WORKSPACE.set(paths)
    try:
        return cls()
    finally:
        _PLOTTER_WORKSPACE.reset(token)


class BasePlotter:
    name: str = "base"

    # Set by plotters whose output is time-series metrics drawn through
    # draw_metric(), which is what makes them composable into a shared figure.
    draws_metric_series: bool = False

    def __init__(self, name: str, *, paths=None):
        self.name = name
        from src.app.core.provider_catalog import get_catalog
        paths = paths or _PLOTTER_WORKSPACE.get() or (WorkspacePaths.for_process() if PROJECT_ROOT == _DEFAULT_PROJECT_ROOT else WorkspacePaths(PROJECT_ROOT))
        self.workspace = paths
        selected = get_catalog(paths).resolve("plotter", name)
        self.plot_dir = selected[0].root if selected else paths.user("plotter")
        project_root = paths.root
        plot_cfg_dir = paths.config / "plot"
        stem = Path(name).stem
        matching_plot_configs = []
        if plot_cfg_dir.exists():
            matching_plot_configs = list(plot_cfg_dir.glob(f"**/{stem}.yaml"))
        candidate_configs = (
            [paths.resolve_config(f"plot/{name}.yaml")]
            + matching_plot_configs
            + [
                self.plot_dir / name / f"_{stem}_config.yaml",
                self.plot_dir / name / f"_{name}_config.yaml",
                self.plot_dir / name / "config.yaml",
                self.plot_dir / f"_{stem}_config.yaml",
                self.plot_dir / f"_{name}_config.yaml",
            ]
        )
        self.default_config_path = next((p for p in candidate_configs if p.exists()), candidate_configs[-1])
        self.default_cfg = self._load_yaml(self.default_config_path)

    def _load_yaml(self, path: Path) -> dict:
        if path.exists():
            with open(path) as f:
                return yaml.safe_load(f) or {}
        return {}

    def _resolve_config_defaults(self, raw_cfg: dict, visited: set | None = None) -> dict:
        """Recursively resolves Hydra-style defaults entries."""
        if visited is None:
            visited = set()

        resolved = dict(raw_cfg)
        defaults = resolved.pop("defaults", None)
        if not defaults or not isinstance(defaults, list):
            return resolved

        base_acc = {}
        for item in defaults:
            if isinstance(item, str):
                item_str = item.strip()
                if item_str == "_self_":
                    continue
                # Handle override patterns
                if item_str.startswith("override /env:"):
                    env_name = item_str.split(":", 1)[1].strip()
                    env_path = self.workspace.resolve_config(f"env/{env_name}.yaml")
                    if env_path.exists():
                        base_acc["env"] = self._load_yaml(env_path)
                    continue
                if item_str.startswith("override /agent:"):
                    agent_name = item_str.split(":", 1)[1].strip()
                    agent_path = self.workspace.resolve_config(f"agent/{agent_name}.yaml")
                    if agent_path.exists():
                        base_acc["agent"] = self._load_yaml(agent_path)
                    continue
                # Experiment base config (e.g. mimic/_base or cartpole/_base)
                base_cands = [
                    self.workspace.resolve_config(f"experiment/{item_str}.yaml"),
                    self.workspace.resolve_config(f"experiment/{Path(item_str).stem}.yaml"),
                ]
                for bcand in base_cands:
                    if bcand.exists() and str(bcand) not in visited:
                        visited.add(str(bcand))
                        parent_raw = self._load_yaml(bcand)
                        parent_resolved = self._resolve_config_defaults(parent_raw, visited)
                        base_acc = deep_update(base_acc, parent_resolved)
                        break
            elif isinstance(item, dict):
                for k, v in item.items():
                    if k.startswith("override /env"):
                        env_path = self.workspace.resolve_config(f"env/{v}.yaml")
                        if env_path.exists():
                            base_acc["env"] = self._load_yaml(env_path)
                    elif k.startswith("override /agent"):
                        agent_path = self.workspace.resolve_config(f"agent/{v}.yaml")
                        if agent_path.exists():
                            base_acc["agent"] = self._load_yaml(agent_path)
                    elif isinstance(v, str):
                        base_cand = self.workspace.resolve_config(f"experiment/{v}.yaml")
                        if base_cand.exists() and str(base_cand) not in visited:
                            visited.add(str(base_cand))
                            parent_raw = self._load_yaml(base_cand)
                            parent_resolved = self._resolve_config_defaults(parent_raw, visited)
                            base_acc = deep_update(base_acc, parent_resolved)

        return deep_update(base_acc, resolved)

    def get_experiment_config(self, exp_id: str, exp_config_name: str | None = None) -> dict:
        """Finds, resolves, and loads the experiment configuration YAML or saved run config."""
        clean_exp = Path(exp_id).stem

        group_hint = None
        if "/" in exp_id:
            group_hint = exp_id.split("/")[0]
        elif exp_config_name and "/" in exp_config_name:
            group_hint = exp_config_name.split("/")[0]

        # 1. Check saved config.yaml from the experiment's log or checkpoint directory first
        # to ensure we capture the true group and execution hyperparameters
        saved_cfg = {}
        for base_dir in [(self.workspace.results / "logs"), (self.workspace.results / "checkpoints")]:
            if base_dir.exists():
                if group_hint:
                    target_direct = base_dir / group_hint / exp_id.removeprefix(f"{group_hint}/") / "config.yaml"
                    if target_direct.exists():
                        saved_cfg = self._load_yaml(target_direct)
                        break
                    target_nested = list(base_dir.glob(f"{group_hint}/{clean_exp}/*/config.yaml"))
                    if target_nested:
                        saved_cfg = self._load_yaml(target_nested[0])
                        break
                matches = list(base_dir.glob(f"*/{clean_exp}/config.yaml"))
                if matches:
                    saved_cfg = self._load_yaml(matches[0])
                    break
                matches_nested = list(base_dir.glob(f"*/{clean_exp}/*/config.yaml"))
                if matches_nested:
                    saved_cfg = self._load_yaml(matches_nested[0])
                    break

        if not group_hint:
            if saved_cfg and saved_cfg.get("group"):
                group_hint = str(saved_cfg.get("group"))
            else:
                for base_dir in [(self.workspace.results / "logs"), (self.workspace.results / "checkpoints"), (self.workspace.results / "plots")]:
                    if base_dir.exists():
                        for g_dir in base_dir.iterdir():
                            if g_dir.is_dir() and (g_dir / clean_exp).exists():
                                group_hint = g_dir.name
                                break
                        if group_hint:
                            break

        # 2. Find live experiment YAML candidates
        candidates = []
        if exp_config_name:
            clean_base = Path(exp_config_name).stem
            candidates.extend(
                [
                    self.workspace.resolve_config(f"experiment/{exp_config_name}.yaml"),
                    self.workspace.resolve_config(f"experiment/{clean_base}.yaml"),
                ]
            )
            if group_hint:
                candidates.append(self.workspace.resolve_config(f"experiment/{group_hint}/{clean_base}.yaml"))
            candidates.extend([self.workspace.resolve_config(selector) for selector in self.workspace.config_resolver().selectors("experiment") if Path(selector).name == f"{clean_base}.yaml"])

        candidates.append(self.workspace.resolve_config(f"experiment") / f"{exp_id}.yaml")
        if group_hint:
            candidates.append(self.workspace.resolve_config(f"experiment/{group_hint}/{clean_exp}.yaml"))
        candidates.extend(
            [
                self.workspace.resolve_config(f"experiment/{exp_id}.yaml"),
                self.workspace.resolve_config(f"experiment/{clean_exp}.yaml"),
            ]
        )
        if not group_hint:
            candidates.extend([self.workspace.resolve_config(selector) for selector in self.workspace.config_resolver().selectors("experiment") if Path(selector).name == f"{clean_exp}.yaml"])

        live_cfg = {}
        for cand in candidates:
            if cand.exists():
                raw = self._load_yaml(cand)
                live_cfg = self._resolve_config_defaults(raw)
                break

        if live_cfg and saved_cfg:
            # Saved config represents the ground-truth hyperparameters and methods executed.
            # Live config allows overriding visualization-only keys (plots, default_plots, etc.).
            merged = dict(saved_cfg)
            for k, v in live_cfg.items():
                if k in ("plots", "plot_overrides", "default_plots", "style", "plot_style"):
                    merged[k] = v
                elif k not in merged:
                    merged[k] = v
            return merged
        elif live_cfg:
            return live_cfg
        elif saved_cfg:
            return saved_cfg

        return {}

    def get_group(self, exp_id: str, exp_config: dict) -> str:
        """Resolves group for the given experiment ID."""
        # If exp_id has a group prefix like mimic/mimic_test, prioritize it
        if "/" in exp_id:
            parts = exp_id.split("/")
            return parts[0]

        if "group" in exp_config and exp_config["group"]:
            return exp_config["group"]

        clean_exp = Path(exp_id).stem
        # Scan results/logs/*/clean_exp and results/checkpoints/*/clean_exp
        for base_dir in [(self.workspace.results / "logs"), (self.workspace.results / "checkpoints"), (self.workspace.results / "plots")]:
            if base_dir.exists():
                for g_dir in base_dir.iterdir():
                    if g_dir.is_dir() and (g_dir / clean_exp).exists():
                        return g_dir.name
        return "ungrouped"

    def get_effective_config(
        self, exp_id: str, cli_overrides: dict | None = None, exp_config_name: str | None = None
    ) -> tuple[dict, str, Path]:
        """
        Merges default module config < default_cfg
               < experiment config plots.<module_name> < plot_overrides.<module_name>
               < CLI overrides
        Returns (merged_config, group, output_dir).
        """
        exp_cfg = self.get_experiment_config(exp_id, exp_config_name=exp_config_name)
        group = self.get_group(exp_id, exp_cfg)
        clean_exp = Path(exp_id).stem

        # Extract general plot options and per-plotter options from experiment YAML
        general_plot_opts = {}
        exp_plot_opts = {}
        plots_sec = exp_cfg.get("plots", {})
        if isinstance(plots_sec, dict):
            # General plot parameters (e.g. title, xlabel, ylabel, smoothing_window, figsize, dpi)
            for k, v in plots_sec.items():
                if (
                    v is not None
                    and k != self.name
                    and k != "output_dir"
                    and not (isinstance(v, dict) and k not in ("style", "figsize"))
                ):
                    general_plot_opts[k] = v
            if self.name in plots_sec and isinstance(plots_sec[self.name], dict):
                exp_plot_opts = plots_sec[self.name]

        merged = deep_update(self.default_cfg, general_plot_opts)
        merged = deep_update(merged, exp_plot_opts)
        overrides = exp_cfg.get("plot_overrides", {})
        if isinstance(overrides, dict) and isinstance(overrides.get(self.name), dict):
            merged = deep_update(merged, overrides[self.name])
        if cli_overrides:
            merged = deep_update(merged, cli_overrides)

        # Retain method definitions and group for method-level plot styling
        if "methods" in exp_cfg:
            merged["methods"] = exp_cfg["methods"]
        merged["group"] = group
        merged["_exp_config"] = exp_cfg

        # Allow output_dir to be explicitly overridden in plotter config, experiment config, or CLI
        if "output_dir" in merged and merged["output_dir"]:
            custom_dir = str(merged["output_dir"]).format(group=group, exp_id=exp_id, clean_exp=clean_exp)
            output_dir = Path(custom_dir)
        elif "output_dir" in exp_cfg and exp_cfg["output_dir"]:
            custom_dir = str(exp_cfg["output_dir"]).format(group=group, exp_id=exp_id, clean_exp=clean_exp)
            output_dir = Path(custom_dir)
        else:
            result_id = exp_id.removeprefix(f"{group}/")
            output_dir = (self.workspace.results / "plots") / group / result_id
        output_dir.mkdir(parents=True, exist_ok=True)
        return merged, group, output_dir

    @staticmethod
    def get_active_aliases(exp_cfg: dict) -> tuple[set[str], bool]:
        """Extracts active method aliases from experiment config.

        Checks structured `methods` dictionary, as well as legacy `online_methods`
        and `offline_methods`. Returns (active_aliases, has_active_filter).
        """
        active_aliases = set()
        has_active_filter = False
        methods_dict = exp_cfg.get("methods", {})
        if methods_dict and isinstance(methods_dict, dict):
            has_active_filter = True
            for m in methods_dict.keys():
                active_aliases.update(get_method_aliases(m))
        for key in ["online_methods", "offline_methods"]:
            val = exp_cfg.get(key, [])
            if val:
                has_active_filter = True
                if isinstance(val, (list, tuple)):
                    methods = list(val)
                else:
                    methods = [item.strip() for item in str(val).split(",") if item.strip()]
                for m in methods:
                    active_aliases.update(get_method_aliases(m))
        return active_aliases, has_active_filter

    @staticmethod
    def is_method_active(method_name: str, active_aliases: set[str], has_active_filter: bool) -> bool:
        """Determines if a method directory name matches active aliases (including trial suffixes)."""
        if not has_active_filter:
            return True
        if method_name in active_aliases:
            return True
        parts = method_name.rsplit("_", 1)
        return len(parts) == 2 and parts[0] in active_aliases and parts[1].isdigit()

    def load_metrics(self, group: str, exp_id: str) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Loads metrics.csv files for all runs matching results/logs/[group]/[exp_id]/[method]/*.
        Filters by active methods from experiment config if defined.
        Returns dict: { method_name: { version_str: df } }
        """
        result_id = exp_id.removeprefix(f"{group}/")
        exp_dir = (self.workspace.results / "logs") / group / result_id
        if not exp_dir.exists():
            print(f"Warning: Log directory {exp_dir} not found.")
            return {}

        exp_cfg = self.get_experiment_config(exp_id)
        active_aliases, has_active_filter = self.get_active_aliases(exp_cfg)

        results = {}
        for method_dir in sorted(exp_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            raw_method_name = method_dir.name
            if not self.is_method_active(raw_method_name, active_aliases, has_active_filter):
                continue

            canon_name = raw_method_name
            if canon_name not in results:
                results[canon_name] = {}

            # Check version_X subdirectories
            version_dirs = sorted([d for d in method_dir.glob("version_*") if d.is_dir()])
            if not version_dirs:
                version_dirs = [method_dir]

            for v_dir in version_dirs:
                csv_path = v_dir / "metrics.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            v_key = (
                                f"{raw_method_name}_{v_dir.name}"
                                if canon_name in results and v_dir.name in results[canon_name]
                                else v_dir.name
                            )
                            results[canon_name][v_key] = df
                    except Exception as e:
                        print(f"Error reading {csv_path}: {e}")

        # Filter out empty method entries
        return {k: v for k, v in results.items() if v}

    # ── Metric series: data, drawing, and the file-writing wrapper ────────
    #
    # Split three ways so the same series can go to a file or into an Axes the
    # caller owns. The Plots pane composes several of these into one figure,
    # and matplotlib refuses to move an Axes between figures, so a plotter has
    # to be able to draw into one it is given rather than make its own.

    def resolve_method_styles(self, active_methods: list, cfg: dict, group: str) -> dict:
        """Per-method label, colour, linestyle and marker for one plot.

        Colours are assigned per plot rather than globally so every method on
        a given figure is visually distinct, even when two plots show
        different subsets of methods.
        """
        method_styles: dict[str, dict] = {}
        reserved_colors = set()
        methods_cfg = cfg.get("methods", {}) if isinstance(cfg.get("methods"), dict) else {}

        for m_name in active_methods:
            style_dict = {}
            m_entry = methods_cfg.get(m_name, {})
            if isinstance(m_entry, dict):
                if "style" in m_entry and isinstance(m_entry["style"], dict):
                    style_dict.update(m_entry["style"])
                for k in ("label", "color", "linestyle", "marker", "display_name"):
                    if k in m_entry and m_entry[k] is not None:
                        target_k = "label" if k == "display_name" else k
                        style_dict[target_k] = m_entry[k]

            if not style_dict:
                from src.app.pipeline.config import find_group_method_config

                gm_cfg = find_group_method_config(m_name, group=group)
                if gm_cfg and isinstance(gm_cfg, dict):
                    if "style" in gm_cfg and isinstance(gm_cfg["style"], dict):
                        style_dict.update(gm_cfg["style"])
                    for k in ("label", "color", "linestyle", "marker", "display_name"):
                        if k in gm_cfg and gm_cfg[k] is not None:
                            target_k = "label" if k == "display_name" else k
                            style_dict[target_k] = gm_cfg[k]

            if not style_dict.get("label"):
                style_dict["label"] = m_name
            if style_dict.get("color"):
                reserved_colors.add(style_dict["color"])
            method_styles[m_name] = style_dict

        palette_idx = 0
        for m_name in active_methods:
            if not method_styles[m_name].get("color"):
                while palette_idx < len(QUALITATIVE_PALETTE) and QUALITATIVE_PALETTE[palette_idx] in reserved_colors:
                    palette_idx += 1
                if palette_idx < len(QUALITATIVE_PALETTE):
                    chosen_color = QUALITATIVE_PALETTE[palette_idx]
                    palette_idx += 1
                else:
                    chosen_color = QUALITATIVE_PALETTE[palette_idx % len(QUALITATIVE_PALETTE)]
                    palette_idx += 1
                method_styles[m_name]["color"] = chosen_color
                reserved_colors.add(chosen_color)

            if not method_styles[m_name].get("linestyle"):
                method_styles[m_name]["linestyle"] = "-"
            if not method_styles[m_name].get("marker"):
                method_styles[m_name]["marker"] = "o"
        return method_styles

    @staticmethod
    def active_methods_for(runs_data: dict, metric: str) -> list:
        """Methods with at least one non-empty version of *metric*."""
        active = []
        for method_name, versions in sorted(runs_data.items()):
            for _v_name, df in versions.items():
                if metric in df.columns and not df.dropna(subset=[metric]).empty:
                    active.append(method_name)
                    break
        return active

    def collect_metric_series(self, runs_data: dict, metric: str, cfg: dict) -> tuple[dict, str | None]:
        """x/y arrays per method for *metric*, plus the x label they imply.

        Returns {method: (x, y, sem_or_None)}. SEM is present only when a
        method has several versions, which is what the shaded band shows.
        """
        x_axis_col = cfg.get("x_axis", "transitions")
        window = cfg.get("smoothing_window", 10)
        series: dict[str, tuple] = {}
        used_xlabel = None

        for method_name, versions in sorted(runs_data.items()):
            all_x, all_y = [], []
            for _v_name, df in versions.items():
                if metric not in df.columns:
                    continue
                valid_df = df.dropna(subset=[metric])
                if valid_df.empty:
                    continue

                x_vals = None
                if x_axis_col in df.columns and df[x_axis_col].notna().any():
                    full_x = df[x_axis_col].interpolate(method="linear").ffill().bfill()
                    s_x = full_x.loc[valid_df.index]
                    if not s_x.empty and s_x.nunique() > 1 and not s_x.isna().any():
                        x_vals = s_x.values
                        if used_xlabel is None:
                            used_xlabel = cfg.get("xlabel", x_axis_col.replace("_", " ").title())
                if x_vals is None:
                    if "step" in valid_df.columns and valid_df["step"].nunique() > 1:
                        x_vals = valid_df["step"].values
                        if used_xlabel is None:
                            used_xlabel = cfg.get("xlabel", "Training Steps")
                    elif "epoch" in valid_df.columns and valid_df["epoch"].nunique() > 1:
                        x_vals = valid_df["epoch"].values
                        if used_xlabel is None:
                            used_xlabel = cfg.get("xlabel", "Epoch")
                    else:
                        x_vals = valid_df.index.values
                        if used_xlabel is None:
                            used_xlabel = cfg.get("xlabel", "Index")
                all_x.append(x_vals)
                all_y.append(valid_df[metric].values)

            if not all_y:
                continue

            if len(all_y) > 1:
                min_len = min(len(y) for y in all_y)
                trimmed = np.array([y[:min_len] for y in all_y])
                y_mean = np.mean(trimmed, axis=0)
                y_sem = np.std(trimmed, axis=0) / np.sqrt(len(all_y))
                y_smoothed = moving_average(y_mean, window)
                sem_smoothed = moving_average(y_sem, window)
                series[method_name] = (all_x[0][: len(y_smoothed)], y_smoothed, sem_smoothed)
            else:
                y_smoothed = moving_average(all_y[0], window)
                series[method_name] = (all_x[0][: len(y_smoothed)], y_smoothed, None)
        return series, used_xlabel

    @staticmethod
    def resolve_labels(metric: str, exp_id: str, cfg: dict, used_xlabel: str | None) -> tuple[str, str, str]:
        """The xlabel, ylabel and title this metric should carry."""
        xlabel = cfg.get("xlabel") or used_xlabel or "Training Steps"

        if isinstance(cfg.get("ylabel"), dict):
            ylabel = cfg["ylabel"].get(metric, metric.replace("_", " ").title())
        elif cfg.get("ylabel"):
            ylabel = str(cfg["ylabel"])
        else:
            ylabel = metric.replace("_", " ").title()

        clean_metric = metric.split("/")[-1].replace("_", " ").title()
        if isinstance(cfg.get("title"), dict):
            title = cfg["title"].get(metric, f"{exp_id.upper()}: {metric}")
        elif cfg.get("title"):
            try:
                title = str(cfg["title"]).format(exp_id=exp_id.upper(), metric=metric, clean_metric=clean_metric)
            except Exception:
                title = str(cfg["title"])
        else:
            title = f"{exp_id.upper()}: {metric}"
        return xlabel, ylabel, title

    def draw_metric(self, ax, exp_id: str, group: str, metric: str, cfg: dict, runs_data: dict | None = None) -> bool:
        """Draw one metric into *ax*. False when there is nothing to draw.

        The Axes belongs to the caller, so this sets labels and a legend but
        never creates, saves or closes a figure.
        """
        if runs_data is None:
            runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            return False

        active_methods = self.active_methods_for(runs_data, metric)
        if not active_methods:
            return False

        method_styles = self.resolve_method_styles(active_methods, cfg, group)
        series, used_xlabel = self.collect_metric_series(runs_data, metric, cfg)

        style_cfg = cfg.get("style", {}) if isinstance(cfg.get("style"), dict) else {}
        line_width = style_cfg.get("line_width", 2.0)

        has_data = False
        for method_name, (x_plot, y_plot, sem) in series.items():
            if method_name not in method_styles:
                continue
            has_data = True
            style = method_styles[method_name]
            ax.plot(
                x_plot,
                y_plot,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=line_width,
            )
            if sem is not None:
                ax.fill_between(x_plot, y_plot - sem, y_plot + sem, color=style["color"], alpha=0.15)

        if not has_data:
            return False

        xlabel, ylabel, title = self.resolve_labels(metric, exp_id, cfg, used_xlabel)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(style_cfg.get("grid", True), alpha=style_cfg.get("grid_alpha", 0.3))
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return True

    def plot_metric_series(self, exp_id: str, group: str, output_dir: Path, metrics: list, cfg: dict):
        """Standard multi-method metric plotting with multi-version mean±SEM.

        Shared implementation used by ConvergencePlotter, LossesPlotter, and
        any future plotter that plots time-series metrics from metrics.csv.
        Writes one PNG per metric; draw_metric() is the same drawing against
        an Axes the caller supplies.
        """
        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            print(f"No log data found for experiment '{exp_id}' in group '{group}'.")
            return

        dpi = cfg.get("dpi", 300)
        figsize = tuple(cfg.get("figsize", [8, 5]))
        subdir = cfg.get("output_subdir", self.name)
        filename_prefix = cfg.get("filename_prefix", "")

        out_dir = (output_dir / subdir) if subdir else output_dir
        any_saved = False

        print(f"=== Generating {self.name.title()} Plots for '{exp_id}' ===")

        for metric in metrics:
            fig = plt.figure(figsize=figsize)
            if self.draw_metric(fig.gca(), exp_id, group, metric, cfg, runs_data=runs_data):
                out_dir.mkdir(parents=True, exist_ok=True)
                plt.tight_layout()
                safe_metric_name = metric.replace("/", "_")
                out_path = out_dir / f"{filename_prefix}{safe_metric_name}.png"
                plt.savefig(out_path, dpi=dpi)
                plt.close()
                print(f"  Saved: {out_path}")
                any_saved = True
            else:
                plt.close()

        if out_dir.exists() and not any_saved and not any(out_dir.iterdir()):
            try:
                out_dir.rmdir()
            except Exception:
                pass

    def panel_label(self, metric: str) -> str:
        """A short, unambiguous name for one metric's panel.

        The whole path rather than its last segment, because eval/reward and
        train/reward are different plots that would otherwise both be offered
        as "Reward". A leading segment that only repeats the plotter's own
        name is dropped, so losses/total_loss reads "Total Loss" and not
        "Losses Total Loss".
        """
        parts = [p for p in metric.split("/") if p]
        if len(parts) > 1 and parts[0].lower() == self.name.lower():
            parts = parts[1:]
        return " ".join(parts).replace("_", " ").title()

    def panels(self, exp_id: str, cli_overrides: dict | None = None) -> list:
        """The panels this plotter can draw into Axes supplied by a caller.

        Empty for plotters that build their own figures; the Plots pane
        captures those instead of composing them.
        """
        if not self.draws_metric_series:
            return []

        cfg, group, _output_dir = self.get_effective_config(exp_id, cli_overrides)
        metrics = cfg.get("metrics") or []
        if not metrics:
            return []

        runs_data = self.load_metrics(group, exp_id)
        if not runs_data:
            return []

        specs = []
        for metric in metrics:
            if not self.active_methods_for(runs_data, metric):
                continue  # nothing logged this metric: no empty panel to offer

            def draw(ax, _metric=metric):
                return self.draw_metric(ax, exp_id, group, _metric, cfg, runs_data=runs_data)

            specs.append(PanelSpec(key=f"{self.name}/{metric}", label=self.panel_label(metric), draw=draw))
        return specs

    def run(self, exp_id: str, cli_overrides: dict | None = None):
        raise NotImplementedError("Subclasses must implement run()")
