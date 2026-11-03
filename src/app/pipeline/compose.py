"""Shared experiment composition used by both the CLI (run_pipeline.py) and the HTTP API.

Keeping a single composition path is what guarantees that a config built in the GUI
resolves exactly like the same experiment launched from the command line.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.app.pipeline.config import resolve_experiment_config_name
from src.app.pipeline.validation import validate_experiment_config
from src.app.pipeline.tuning_config import tuning_disabled, tuning_enabled

CONFIG_DIR = Path(__file__).resolve().parents[3] / "in" / "config"

# The compose API cannot parse Optuna sweep operators or hydra internal configs
_SWEEP_MARKERS = ("interval(", "choice(", "range(", "hydra.", "hydra/")

# Hydra keeps global state, so concurrent API requests must not compose at the same time
_HYDRA_LOCK = threading.Lock()


@dataclass
class ComposedExperiment:
    experiment: str
    """Resolved experiment config path relative to in/config/experiment/."""
    cfg: DictConfig
    extra_args: list[str]
    """User-supplied overrides exactly as they would be typed on the command line."""
    launch_args: list[str]
    """Overrides forwarded to training subprocesses, including internal experiment tracking."""
    is_sweep: bool

    @property
    def argv(self) -> list[str]:
        """The equivalent command line invocation."""
        return ["python", "run_pipeline.py", self.experiment, *self.extra_args]


def compose_experiment(raw_experiment: str, extra_args: list[str] | tuple[str, ...] = ()) -> ComposedExperiment:
    """Compose the Hydra config for an experiment plus CLI-style overrides.

    Raises ValueError for an unknown experiment and Hydra exceptions for invalid overrides.
    """
    experiment = resolve_experiment_config_name(raw_experiment)
    extra_args = list(extra_args)

    overrides = [f"+experiment={experiment}", f"++experiment_name={experiment}"]
    for arg in extra_args:
        if any(marker in arg for marker in _SWEEP_MARKERS):
            continue
        if "=" in arg or arg.startswith("~"):
            overrides.append(arg)

    # Subprocesses need everything PLUS the internal experiment tracking
    launch_args = [*extra_args, f"++experiment_name={experiment}"]

    with _HYDRA_LOCK:
        GlobalHydra.instance().clear()
        initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR))
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)

    exp_stem = Path(experiment).stem
    if not cfg.get("experiment_id") or cfg.experiment_id == "default_exp":
        cfg.experiment_id = exp_stem

    exp_group = Path(experiment).parent.name if "/" in experiment else "ungrouped"
    if not cfg.get("group") or cfg.group == "ungrouped":
        cfg.group = exp_group

    # Ensure it's in extra args so it passes to children
    if not any("experiment_id=" in arg for arg in launch_args):
        launch_args.append(f"++experiment_id={cfg.experiment_id}")
    if not any("group=" in arg for arg in launch_args):
        launch_args.append(f"++group={cfg.group}")

    is_sweep = not tuning_disabled(cfg) and (
        tuning_enabled(cfg) or bool(cfg.get("sweep", False)) or "--multirun" in launch_args or "-m" in launch_args
    )
    launch_args = [a for a in launch_args if a not in ("--multirun", "-m")]
    if tuning_disabled(cfg):
        launch_args = [
            arg for arg in launch_args
            if not arg.lstrip("+").startswith(("hydra/sweeper=", "hydra.sweeper."))
        ]

    return ComposedExperiment(experiment, cfg, extra_args, launch_args, is_sweep)


def validate_composed(composed: ComposedExperiment) -> list[str]:
    """Run pre-flight paradigm validation. Returns notices; raises ConfigurationError."""
    notices = validate_experiment_config(composed.cfg, composed.experiment, is_sweep=composed.is_sweep)
    if composed.cfg.get("variability") is not None:
        from src.app.pipeline.config import parse_methods_dict
        from src.app.pipeline.variability import make_plan

        choices = composed.cfg.get("hydra", {}).get("runtime", {}).get("choices", {})
        site = choices.get("site") or (composed.cfg.get("site") or {}).get("name") or "local"
        make_plan(composed.cfg, {"methods": parse_methods_dict(composed.cfg),
                                 "is_sweep": composed.is_sweep, "is_interactive": site == "local"})
    return notices


def method_plans(composed: ComposedExperiment) -> dict[str, dict[str, Any]]:
    """Per-method settings after tier merging, plus the exact overrides each training subprocess receives.

    Method-level values (e.g. methods.<method>.lr) only reach agent.* here, so this, not the top-level
    agent block, is what a method actually trains with. Offline dataset paths are not resolved.
    """
    from src.app.pipeline.commands import build_method_overrides
    from src.app.pipeline.config import parse_methods_dict

    plans = {}
    for name, settings in parse_methods_dict(composed.cfg).items():
        method_sweeps = not tuning_disabled(composed.cfg) and (
            composed.is_sweep or bool(settings.get("tune") or settings.get("search_space"))
        )
        plans[name] = {
            "settings": settings,
            "train_overrides": build_method_overrides(
                method_name=name, method_cfg=settings, extra_args=composed.launch_args, cfg=composed.cfg,
                is_sweep=method_sweeps,
            ),
        }
    return plans


def effective_config(cfg: DictConfig) -> dict[str, Any]:
    """The resolved experiment config as plain data, without Hydra's runtime bookkeeping."""
    try:
        data = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        data = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(data, dict):
        return {}
    data.pop("hydra", None)
    return cast(dict[str, Any], data)


def comparable_config(cfg: DictConfig) -> dict[str, Any]:
    """Effective config minus fields that name the recipe file rather than describe the experiment."""
    data = effective_config(cfg)
    data.pop("experiment_name", None)
    return data
