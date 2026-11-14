"""Single source of truth for training command construction.

Builds Hydra override lists from structured method config dicts.
"""

import json

from src.app.pipeline.config import normalize_agent_name
from src.app.pipeline.validation import paradigm_uses_agents, paradigm_uses_static_dataset


def _format_hydra_val(x):
    """Format Python values into Hydra override syntax."""
    if x is None:
        return "null"
    if isinstance(x, (list, tuple)):
        return "[" + ",".join(_format_hydra_val(item) for item in x) + "]"
    elif isinstance(x, dict):
        return "{" + ",".join(f"{k}:{_format_hydra_val(val)}" for k, val in x.items()) + "}"
    elif isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def build_method_overrides(
    method_name: str,
    method_cfg: dict,
    dataset_path=None,
    extra_args: list | None = None,
    cfg=None,
    study_name: str | None = None,
    is_sweep: bool = False,
    config_dir=None,
    workspace=None,
) -> list[str]:
    """Build Hydra override list from a structured methods: dict entry.

    Args:
        method_name:  Key in the experiment's methods mapping.
        method_cfg:   The dict value for that entry. Must contain 'agent' and 'model'.
        dataset_path: Resolved filesystem path to the offline dataset (offline only).
        extra_args:   Additional Hydra overrides forwarded from the CLI.
        cfg:          Full Hydra config (used for experiment_name, paradigm).
        study_name:   Selected tuner's study name (optional, for sweep runs).
        is_sweep:     Compatibility argument; composed cfg.tuning determines sweep mode.

    Returns:
        list[str]: Hydra override arguments ready to pass to train.py.
    """
    if hasattr(method_cfg, "items"):
        from omegaconf import DictConfig, OmegaConf

        if isinstance(method_cfg, DictConfig):
            method_cfg = OmegaConf.to_container(method_cfg, resolve=True)
        else:
            method_cfg = dict(method_cfg)

    from src.app.core.execution import reject_selectors
    reject_selectors(method_cfg, f"method {method_name}")
    from src.app.pipeline.tuning_config import require_runnable_tuning, tuning_enabled

    require_runnable_tuning(cfg, {method_name: method_cfg})
    is_sweep = tuning_enabled(cfg)

    paradigm = cfg.get("paradigm", "offline_rl") if cfg is not None else "offline_rl"
    from src.app.workspace import WorkspacePaths
    paths = workspace or (WorkspacePaths.from_config_root(config_dir) if config_dir is not None else WorkspacePaths.for_process())
    # Asked once: a paradigm with no allowed agents drives its methods by model.
    uses_agents = paradigm_uses_agents(paradigm, paths=paths)
    agent_val = method_cfg.get("agent")
    if isinstance(agent_val, dict):
        agent_algo = (
            agent_val.get("name") or agent_val.get("algorithm") or agent_val.get("type") or agent_val.get("algo")
        )
        agent_subparams = {k: v for k, v in agent_val.items() if k not in ("name", "algorithm", "type", "algo")}
    else:
        agent_algo = agent_val
        agent_subparams = {}

    model_val = method_cfg.get("model")
    if isinstance(model_val, dict):
        model_arch = (
            model_val.get("name") or model_val.get("architecture") or model_val.get("type") or model_val.get("base")
        )
        model_subparams = {k: v for k, v in model_val.items() if k not in ("name", "architecture", "type", "base")}
    else:
        model_arch = model_val
        model_subparams = {}

    agent_name = method_cfg.get("name", normalize_agent_name(method_name))
    experiment_name = cfg.get("experiment_name", "") if cfg is not None else ""

    if not model_arch:
        raise ValueError(f"Method '{method_name}' is missing required key 'model' (or 'model.name').")

    # The selected profile path and its registered implementation are distinct.
    # Keep the profile in Hydra's group selector; use its declared runtime ID.
    from pathlib import Path

    from src.app.pipeline.config import load_model_base_config
    from src.app.workspace import WorkspacePaths
    paths = workspace or (WorkspacePaths.from_config_root(config_dir) if config_dir is not None else WorkspacePaths.for_process())
    model_config_dir = (Path(config_dir) if config_dir is not None else paths.config) / "model"
    profile_defaults = load_model_base_config(str(model_arch), model_config_dir, paths=paths)
    runtime_arch = (profile_defaults.get("architecture") or profile_defaults.get("type")
                    or profile_defaults.get("name") or model_arch)

    overrides = [
        f"+experiment={experiment_name}",
        f"paradigm={paradigm}",
        f"model={model_arch}",
    ]
    if method_cfg.get("backend"):
        overrides.append(f"++backend={method_cfg['backend']}")
    # The orchestrator may derive these from the experiment path, so the
    # experiment YAML itself need not contain them. Jobs and artifact promotion
    # must nevertheless use the same resolved identifiers.
    for key in ("group", "experiment_id"):
        value = cfg.get(key) if cfg is not None else None
        if value is not None:
            overrides.append(f"++{key}={json.dumps(str(value))}")

    def _flatten_overrides(prefix, val):
        if isinstance(val, dict):
            for k, v in val.items():
                _flatten_overrides(f"{prefix}.{k}", v)
        else:
            formatted_v = _format_hydra_val(val)
            overrides.append(f"++{prefix}={formatted_v}")

    # Combine explicit subparams with inherited agent_params/model_params
    merged_agent_params = dict(method_cfg.get("agent_params", {}))
    merged_agent_params.update(agent_subparams)

    merged_model_params = dict(method_cfg.get("model_params", {}))
    merged_model_params.update(model_subparams)

    _flatten_overrides("model", merged_model_params)
    _flatten_overrides("agent", merged_agent_params)

    # Inputs belong to the run, never agent.* or model.*. Clear inherited maps
    # before installing a different provider so Hydra cannot retain its params.
    from src.app.core.data_loaders import LOADER_SLOTS, loader_selection
    for slot in LOADER_SLOTS:
        if slot in method_cfg or (cfg is not None and cfg.get(slot) is not None):
            selection, _ = loader_selection(cfg or {}, slot, method_cfg, paths=paths)
            overrides.append(f"++{slot}=null")
            if selection is not None:
                overrides.append(f"++{slot}={_format_hydra_val(selection)}")

    # Monitor instances are run-level settings, even when configured per method.
    # Hydra already composed the shared instances; apply this method's overrides
    # at the same root used by validation and MonitorManager, before CLI overrides.
    if "monitors" in method_cfg:
        if method_cfg["monitors"] is None:
            overrides.append("++monitors=null")
        else:
            _flatten_overrides("monitors", method_cfg["monitors"])

    if not uses_agents:
        overrides.append(f"++model.name={agent_name}")
        overrides.append(f"++model.architecture={runtime_arch}")
    else:
        if not agent_algo:
            raise ValueError(f"Method '{method_name}' is missing required key 'agent' for paradigm '{paradigm}'.")
        overrides.extend(
            [
                f"agent={agent_algo}",
                f"++agent.algorithm={agent_algo}",
                f"++agent.name={agent_name}",
                f"++model.architecture={runtime_arch}",
            ]
        )

    if paradigm_uses_static_dataset(paradigm, paths=paths) and dataset_path is not None:
        safe_ds_path = str(dataset_path)
        if any(c in safe_ds_path for c in "(), "):
            safe_ds_path = f'"{safe_ds_path}"'
        overrides.append(f"++dataset_path={safe_ds_path}")

    # Process remaining method-level or global hyperparameter overrides
    _INTERNAL_KEYS = {
        "agent",
        "model",
        "name",
        "agent_params",
        "model_params",
        "style",
        "label",
        "color",
        "marker",
        "linestyle",
        "display_name",
        "base",
        "template",
        "from_method",
        "tune",
        "search_space",
        "from_study",
        "entrypoint",
        "runner",
        "script",
        "engine",
        "backend",
        "python",
        "venv_python",
        "venv",
        "data_module",
        "symbolic_state_loader",
        "monitors",
    }
    _MODEL_KEYS = {
        "architecture",
        "hidden_sizes",
        "activation",
        "dropout",
        "d_model",
        "nhead",
        "dim_feedforward",
        "num_layers",
        "hidden_dim",
    }
    # Dynamically expand model keys if base model config is discovered
    try:
        from src.app.pipeline.config import load_model_base_config

        base_m_cfg = load_model_base_config(model_arch, model_config_dir, paths=paths)
        if base_m_cfg and isinstance(base_m_cfg, dict):
            _MODEL_KEYS.update(base_m_cfg.keys())
    except Exception:
        pass
    # UI-exposed model fields can be absent from an older workspace profile.
    # Route a fixed value to the same model path used by a sweep target.
    from pathlib import Path

    from src.app.pipeline.component_validation import schemas_for_method
    schema = schemas_for_method(paths, {"model": model_arch}, validation_only=False).get("model", {})
    _MODEL_KEYS.update(schema.get("properties", {}))

    for k, v in method_cfg.items():
        if k in _INTERNAL_KEYS:
            continue
        if k in merged_agent_params or k in merged_model_params:
            continue

        if not uses_agents:
            _flatten_overrides(f"model.{k}", v)
            if k in ("lr", "batch_size", "epochs", "eval_every_n_epochs", "weight_decay"):
                overrides.append(f"++{k}={_format_hydra_val(v)}")
        else:
            if k in _MODEL_KEYS:
                _flatten_overrides(f"model.{k}", v)
            else:
                _flatten_overrides(f"agent.{k}", v)
                if k in (
                    "epochs",
                    "eval_every_n_epochs",
                    "gamma",
                    "reward_scale",
                    "pos_action_weight",
                    "bellman_loss",
                    "weight_decay",
                ):
                    overrides.append(f"++{k}={_format_hydra_val(v)}")

    # The selected tuning component compiles its own job overrides.
    if is_sweep:
        from src.app.pipeline.tuners import get_tuner

        overrides = get_tuner(cfg, paths=paths).compile(
            overrides, cfg=cfg, method_cfg=method_cfg, extra_args=extra_args,
            paradigm=paradigm, study_name=study_name, uses_agents=uses_agents, model_keys=_MODEL_KEYS,
        )

    if extra_args:
        if not is_sweep:
            # Filter out multirun flags if this method is running as a single normal model
            clean_extra = [a for a in extra_args if a not in ("--multirun", "-m")
                           and not a.lstrip("+").startswith(("hydra/sweeper=", "hydra.sweeper."))]
            overrides.extend(clean_extra)
        else:
            overrides.extend(extra_args)

    from src.app.core.data_loaders import loader_override_args
    return loader_override_args(overrides)


def get_sweep_direction(cfg, paradigm: str) -> str:
    """Resolve the historical Hydra direction or the paradigm's default.

    Checks cfg.hydra.sweeper.direction first (explicit override).
    Defaults: online_rl → 'maximize' (reward), offline_rl → 'minimize' (loss).
    """
    if hasattr(cfg, "hydra") and hasattr(cfg.hydra, "sweeper"):
        explicit = cfg.hydra.sweeper.get("direction", None)
        if explicit:
            return str(explicit)
    return "maximize" if paradigm == "online_rl" else "minimize"
