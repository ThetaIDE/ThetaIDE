"""Shared experiment composition used by the CLI and in-process desktop services.

Keeping a single composition path is what guarantees that a config built in the GUI
resolves exactly like the same experiment launched from the command line.
"""

from __future__ import annotations

import functools
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from src.app.core.provider_discovery import provider_discovery_snapshot
from src.app.pipeline.config import resolve_experiment_config_name
from src.app.pipeline.result_paths import result_identity
from src.app.pipeline.tuning_config import tuning_disabled, tuning_enabled
from src.app.pipeline.validation import validate_experiment_config
from src.app.workspace import WorkspacePaths

def composition_workspace(config_dir=None, workspace=None):
    # Runtime workspace selection occurs at the entry point, never while importing this service.
    if workspace is not None:
        return workspace
    if config_dir is not None:
        return WorkspacePaths.from_config_root(config_dir)
    return WorkspacePaths.for_process()

def one_discovery(function):
    """Run `function` as one discovery read: workspace manifests, the provider catalog and config
    resolvers are built once and shared by every lookup inside it (and by an enclosing caller's).
    Without this, one validation rescans the workspace about ninety times."""
    @functools.wraps(function)
    def scoped(*args, **kwargs):
        with provider_discovery_snapshot():
            return function(*args, **kwargs)
    return scoped


# The compose API cannot parse Optuna sweep operators or hydra internal configs
_SWEEP_MARKERS = ("interval(", "choice(", "range(", "hydra.", "hydra/")

# Hydra keeps global state, so concurrent callers must not compose at the same time
_HYDRA_LOCK = threading.Lock()


@dataclass
class ComposedExperiment:
    experiment: str
    """Resolved experiment selector relative to the experiment config group."""
    cfg: DictConfig
    extra_args: list[str]
    """User-supplied overrides exactly as they would be typed on the command line."""
    launch_args: list[str]
    """Overrides forwarded to training subprocesses, including internal experiment tracking."""
    is_sweep: bool
    config_dir: Path | None = None
    workspace: WorkspacePaths | None = None

    @property
    def argv(self) -> list[str]:
        """The equivalent command line invocation."""
        return ["python", "run_pipeline.py", self.experiment, *self.extra_args]


@one_discovery
def compose_experiment(raw_experiment: str, extra_args: list[str] | tuple[str, ...] = (), *, config_dir: Path | None = None, workspace: WorkspacePaths | None = None) -> ComposedExperiment:
    """Compose the Hydra config for an experiment plus CLI-style overrides.

    Raises ValueError for an unknown experiment and Hydra exceptions for invalid overrides.
    """
    if config_dir is None and workspace is not None:
        config_dir = workspace.config
    workspace = composition_workspace(config_dir, workspace)
    experiment = resolve_experiment_config_name(raw_experiment, paths=workspace, config_dir=config_dir)
    extra_args = list(extra_args)

    overrides = [f"+experiment={experiment}", f"++experiment_name={experiment}"]
    for arg in extra_args:
        if any(marker in arg for marker in _SWEEP_MARKERS):
            continue
        if "=" in arg or arg.startswith("~"):
            overrides.append(arg)

    # Subprocesses need everything PLUS the internal experiment tracking
    launch_args = [*extra_args, f"++experiment_name={experiment}"]

    from src.app.core.data_loaders import loader_override_args
    overrides = loader_override_args(overrides)

    from src.app.config_resolution import configuration_context
    with _HYDRA_LOCK, configuration_context(workspace, config_dir):
        GlobalHydra.instance().clear()
        primary = Path(config_dir or workspace.config)
        search = workspace.config_searchpath()
        if not primary.is_dir():
            primary = workspace.framework_config
        if search and not any(arg.startswith("hydra.searchpath=") for arg in overrides):
            overrides.append("hydra.searchpath=" + str(search).replace("'", ""))
        initialize_config_dir(version_base=None, config_dir=str(primary))
        cfg = compose(config_name="config", overrides=overrides, return_hydra_config=True)

    cfg.group, cfg.experiment_id = result_identity(experiment, cfg.get("group"), cfg.get("experiment_id"))

    # Forward the resolved identity, including subdirectories, to every engine.
    launch_args = [arg for arg in launch_args if not arg.lstrip("+").startswith(("experiment_id=", "group="))]
    launch_args.extend(f"++{key}={json.dumps(str(cfg[key]))}" for key in ("experiment_id", "group"))

    is_sweep = tuning_enabled(cfg)
    launch_args = [a for a in launch_args if a not in ("--multirun", "-m")]
    if tuning_disabled(cfg):
        launch_args = [
            arg for arg in launch_args
            if not arg.lstrip("+").startswith(("hydra/sweeper=", "hydra.sweeper."))
        ]

    return ComposedExperiment(experiment, cfg, extra_args, launch_args, is_sweep, config_dir, workspace)


@one_discovery
def validate_composed(composed: ComposedExperiment, *, resuming: bool = False) -> list[str]:
    """Run pre-flight paradigm validation. Returns notices; raises ConfigurationError."""
    notices = validate_experiment_config(
        composed.cfg, composed.experiment, is_sweep=composed.is_sweep,
        config_dir=composed.config_dir, workspace=composed.workspace,
    )
    _validate_method_composition(composed)
    from src.app.sites import scheduler
    if scheduler(composed.cfg) != "local":
        from src.app.sites.execution import validate_allocation
        validate_allocation(composed.cfg, paths=composed.workspace)
    if composed.cfg.get("variability") is not None:
        from src.app.pipeline.config import parse_methods_dict
        from src.app.pipeline.seeds import validate_seed_ranges, with_launch_seeds
        from src.app.pipeline.variability import make_plan

        methods = parse_methods_dict(composed.cfg)
        seeded = with_launch_seeds(composed.cfg, methods, resuming=resuming)
        plan = make_plan(seeded, {"methods": methods, "is_sweep": composed.is_sweep, "is_interactive": True})
        if seeded.get("paradigm") == "online_rl":
            from src.app.pipeline.exceptions import ConfigurationError

            try:
                validate_seed_ranges(seeded, methods, plan)
            except ConfigurationError as exc:
                if not resuming:
                    raise
                notices.append(f"Resuming with the original study seeds. {exc}")
    return notices



def _validate_method_composition(composed: ComposedExperiment) -> None:
    """Preflight the same per-method Hydra overrides used by a tuner's sweeper.

    A representative value exercises path existence and Hydra composition before
    starting a study; dimension bounds and types are checked by the tuner.
    """
    from src.app.pipeline.component_validation import (
        schemas_for_method,
        validate_method_components,
        validate_sweep_dimensions,
    )
    from src.app.pipeline.config import parse_methods_dict
    from src.app.pipeline.exceptions import ConfigurationError
    from src.app.pipeline.tuning_config import tuning_disabled
    paths = composition_workspace(composed.config_dir, composed.workspace)
    methods = parse_methods_dict(composed.cfg, paths=paths)
    workspace = composition_workspace(composed.config_dir, composed.workspace).root
    schemas = {name: schemas_for_method(workspace, settings) for name, settings in methods.items()}
    sweep = not tuning_disabled(composed.cfg) and composed.is_sweep
    import json

    from hydra.core.override_parser.overrides_parser import OverridesParser
    from src.app.pipeline.tuners import get_tuner
    tuner = get_tuner(composed.cfg, paths=paths) if sweep else None
    inspect_override = getattr(tuner, "preview_override", None)
    validate_runner = getattr(tuner, "validate_runner", None)

    parser = OverridesParser.create()
    for name, plan in method_plans(composed).items():
        has_sweep = sweep and bool(plan["settings"].get("tune"))
        fixed, sampled = {}, {}
        validate_sweep_dimensions(plan["settings"], schemas.get(name, {}), name)
        try:
            for arg in plan["train_overrides"]:
                if arg in ("--multirun", "-m") or arg.startswith(("+experiment=", "hydra/", "++hydra.", "hydra.")):
                    continue
                override = parser.parse_override(arg)
                value = inspect_override(override) if has_sweep and callable(inspect_override) else None
                key = override.get_key_element()
                if has_sweep and value is not None:
                    sampled[key] = f"{key}={json.dumps(value)}"
                else:
                    fixed[key] = arg
            for key in fixed:
                sampled.pop(key, None)
            trial = compose_experiment(composed.experiment, [*fixed.values(), *sampled.values()], config_dir=composed.config_dir, workspace=composed.workspace)
            from src.app.core.execution import execution_backend
            execution_backend(trial.cfg, paths=paths)
            if has_sweep and callable(validate_runner):
                validate_runner(trial.cfg, paths=paths)
            from src.app.core.paradigm_loader import load_paradigm_definition
            runner = load_paradigm_definition(trial.cfg.get("paradigm"), paths=paths).runner_cls
            validator = getattr(runner, "validate_config", None)
            if callable(validator):
                validator(trial.cfg, paths=composition_workspace(composed.config_dir, composed.workspace))
            validate_method_components(trial.cfg, schemas.get(name, {}), name)
        except Exception as exc:
            raise ConfigurationError(f"Method '{name}' trial configuration cannot be composed: {exc}") from exc

@one_discovery
def method_plans(composed: ComposedExperiment) -> dict[str, dict[str, Any]]:
    """Per-method settings after tier merging, plus the exact overrides each training subprocess receives.

    Method-level values (e.g. methods.<method>.lr) only reach agent.* here, so this, not the top-level
    agent block, is what a method actually trains with. Offline dataset paths are not resolved.
    """
    from src.app.pipeline.commands import build_method_overrides
    from src.app.pipeline.config import parse_methods_dict

    plans = {}
    for name, settings in parse_methods_dict(composed.cfg, paths=composition_workspace(composed.config_dir, composed.workspace)).items():
        plans[name] = {
            "settings": settings,
            "train_overrides": build_method_overrides(
                method_name=name, method_cfg=settings, extra_args=composed.launch_args, cfg=composed.cfg,
                config_dir=composed.config_dir or composition_workspace(workspace=composed.workspace).config,
                workspace=composed.workspace,
            ),
        }
    return plans


def effective_config(cfg: DictConfig) -> dict[str, Any]:
    """The resolved experiment config as plain data, without Hydra's runtime bookkeeping."""
    config = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    config.pop("hydra", None)
    try:
        data = OmegaConf.to_container(config, resolve=True)
    except Exception:
        data = OmegaConf.to_container(config, resolve=False)
    if not isinstance(data, dict):
        return {}
    return cast(dict[str, Any], data)


def comparable_config(cfg: DictConfig) -> dict[str, Any]:
    """Effective config minus fields that name the recipe file rather than describe the experiment."""
    data = effective_config(cfg)
    data.pop("experiment_name", None)
    return data
