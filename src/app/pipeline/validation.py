"""Pre-flight configuration validation module.

Validates experiment configs against their declared paradigm before any
training begins. Raises ConfigurationError on fatal mismatches so the
pipeline aborts cleanly at startup.

Paradigm definitions live in in/config/paradigms/. Adding a new paradigm
requires only a new YAML file — no changes to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.app.core.checkpoints import checkpoint_errors
from src.app.pipeline.config import (
    method_is_enabled,
    normalize_agent_name,
    parse_methods_dict,
    resolve_experiment_config_name,
)
from src.app.pipeline.datasets import resolve_dataset_path
from src.app.pipeline.exceptions import ConfigurationError
from src.app.telemetry.selection import selection_errors

# ---------------------------------------------------------------------------
# Paradigm registry loading
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PARADIGMS_DIR = _PROJECT_ROOT / "in" / "config" / "paradigms"


def _load_all_paradigms() -> dict[str, dict]:
    """Load all paradigm YAML files from in/config/paradigms/.

    Returns a dict mapping paradigm name → paradigm definition dict.
    """
    paradigms: dict[str, dict] = {}
    for yaml_path in sorted(_PARADIGMS_DIR.glob("**/*.yaml")):
        name = yaml_path.stem  # filename without extension = paradigm name
        with open(yaml_path) as f:
            defn = yaml.safe_load(f) or {}
        paradigms[name] = defn
    return paradigms


def list_paradigms() -> list[str]:
    """Return sorted list of registered paradigm names."""
    return sorted(_load_all_paradigms().keys())


def load_paradigm(paradigm_name: str) -> dict:
    """Load and return the definition dict for a named paradigm.

    Raises ConfigurationError if the paradigm is not found.
    """
    paradigms = _load_all_paradigms()
    if paradigm_name not in paradigms:
        raise ConfigurationError(
            f"[ConfigurationError] Unknown paradigm '{paradigm_name}'. "
            f"Available paradigms: {sorted(paradigms.keys())}. "
            f"To add a new paradigm create in/config/paradigms/<name>.yaml."
        )
    return paradigms[paradigm_name]


# ---------------------------------------------------------------------------
# Paradigm capability queries
# ---------------------------------------------------------------------------
#
# The pipeline used to branch on paradigm names - `paradigm == "supervised"`,
# `paradigm in ("offline_rl", "supervised")` - which meant every new paradigm
# had to be threaded through a dozen tuples before it could run. These two
# questions are what those branches were really asking, and both are already
# answered by the paradigm YAML, so a new file is enough to answer them.


def paradigm_uses_agents(paradigm_name: str | None) -> bool:
    """Whether methods in this paradigm name an RL agent harness.

    An `allowed_agents: []` declaration means no agent is permitted at all, so
    methods name a model only - supervised and unsupervised both work that way,
    having no policy that chooses actions. A paradigm that omits the key places
    no restriction, which is treated as the RL default.
    """
    try:
        defn = load_paradigm(paradigm_name or "")
    except ConfigurationError:
        return True  # Unknown paradigm: leave the RL assumption in place.
    requirement = _algorithm_requirement(defn)
    return requirement != "forbidden"


def paradigm_uses_static_dataset(paradigm_name: str | None) -> bool:
    """Whether this paradigm trains from a dataset file rather than a live simulator."""
    try:
        defn = load_paradigm(paradigm_name or "")
    except ConfigurationError:
        return False
    declared = defn.get("environment_source")
    if isinstance(declared, str) and declared:
        return declared == "static_dataset"
    data_module = defn.get("data_module") or defn.get("default_data_module")
    return bool(data_module and data_module != "SimulatorDataModule")


def paradigm_uses_live_environment(paradigm_name: str | None) -> bool:
    """Whether a paradigm requires a live environment rather than static data."""
    try:
        defn = load_paradigm(paradigm_name or "")
    except ConfigurationError:
        return True
    declared = defn.get("environment_source")
    if isinstance(declared, str) and declared:
        return declared == "live_environment"
    return not paradigm_uses_static_dataset(paradigm_name)


def _algorithm_requirement(paradigm_def: dict[str, Any]) -> str:
    """Return the declarative method contract with legacy-manifest fallback."""
    method = paradigm_def.get("method")
    declared = method.get("algorithm") if isinstance(method, dict) else None
    if isinstance(declared, str) and declared in ("required", "optional", "forbidden"):
        return declared
    allowed = paradigm_def.get("allowed_agents", None)
    if allowed is not None and not allowed:
        return "forbidden"
    if allowed or paradigm_def.get("constraints", {}).get("requires", {}).get("agent") == "required":
        return "required"
    return "optional"


# ---------------------------------------------------------------------------
# Constraint checking helpers
# ---------------------------------------------------------------------------


def _deep_get(cfg: Any, dotted_key: str, default=None):
    """Resolve a dotted key path against a Hydra cfg or plain dict.

    E.g. _deep_get(cfg, 'env.name') returns cfg.env.name.
    """
    parts = dotted_key.split(".")
    obj = cfg
    for part in parts:
        if obj is None:
            return default
        if isinstance(obj, dict):
            obj = obj.get(part)
        else:
            # OmegaConf / MagicMock — use getattr then .get
            try:
                obj = getattr(obj, part)
            except AttributeError:
                try:
                    obj = obj.get(part)
                except Exception:
                    return default
    return obj if obj is not None else default


_CONSTRAINT_CHECKERS = {
    "non_empty": lambda v: bool(v),
    "non_zero": lambda v: bool(v) and v != 0,
    "true": lambda v: v is True,
    "false": lambda v: v is False,
    True: lambda v: v is True,
    False: lambda v: v is False,
}


def _check_paradigm_constraints(cfg: Any, paradigm_name: str, constraints: dict, env_name: str) -> None:
    """Apply requires/forbids rules from the paradigm definition.

    Raises ConfigurationError on any violation.
    """
    for key, rule in (constraints.get("requires") or {}).items():
        if rule not in _CONSTRAINT_CHECKERS:
            continue  # unknown rule type, skip silently
        val = _deep_get(cfg, key)
        if not _CONSTRAINT_CHECKERS[rule](val):
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' requires '{key}' "
                f"to satisfy rule '{rule}', but got {val!r} "
                f"(env: '{env_name}')."
            )

    for key, rule in (constraints.get("forbids") or {}).items():
        if rule not in _CONSTRAINT_CHECKERS:
            continue
        val = _deep_get(cfg, key)
        if _CONSTRAINT_CHECKERS[rule](val):
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' forbids '{key}' "
                f"from satisfying rule '{rule}', but got {val!r} "
                f"(env: '{env_name}')."
            )


# ---------------------------------------------------------------------------
# Raw experiment YAML loader (for explicit override checks)
# ---------------------------------------------------------------------------


def _load_raw_experiment_yaml(experiment_name: str) -> dict:
    """Return the raw (pre-Hydra-composition) experiment YAML as a dict, or {} on failure."""
    rel_path = resolve_experiment_config_name(experiment_name)
    exp_path = Path("in/config/experiment") / f"{rel_path}.yaml"
    if not exp_path.exists():
        return {}
    with open(exp_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_experiment_config(cfg: Any, experiment_name: str, is_sweep: bool = False) -> list[str]:
    """Validate experiment configuration against its declared paradigm.

    Returns a list of non-fatal notice strings.
    Raises ConfigurationError on any fatal paradigm incompatibility.
    """
    notices: list[str] = []

    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    raw_exp = _load_raw_experiment_yaml(experiment_name)

    # --- Paradigm lookup ---
    paradigm_name = cfg.get("paradigm", None)
    if not paradigm_name:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' has no 'paradigm' declared. "
            f"Add 'paradigm: <name>' to its group _base.yaml. "
            f"Available paradigms: {list_paradigms()}."
        )

    paradigm_def = load_paradigm(paradigm_name)  # raises ConfigurationError if unknown
    constraints = paradigm_def.get("constraints", {})

    # --- eval_episodes: check raw YAML for explicit override ---
    allows_eval_episodes = paradigm_def.get("allows_eval_episodes", False)
    if not allows_eval_episodes:
        explicit_eval_ep = raw_exp.get("eval_episodes", None)
        resolved_eval_ep = cfg.get("eval_episodes", 0)
        eval_ep_to_check = explicit_eval_ep if explicit_eval_ep is not None else resolved_eval_ep
        if eval_ep_to_check and eval_ep_to_check > 0:
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' (env '{env_name}') disables "
                f"simulated gym rollouts — policy actions cannot alter historical trajectories. "
                f"Remove 'eval_episodes' from the experiment config."
            )

    # --- Apply paradigm constraints ---
    _check_paradigm_constraints(cfg, paradigm_name, constraints, env_name)

    # --- Periodic checkpoints: checkpoint_every_steps or num_checkpoints ---
    for problem in checkpoint_errors(cfg):
        raise ConfigurationError(f"[ConfigurationError] Experiment '{experiment_name}': {problem}")

    # --- Recorded metrics: telemetry.metrics ---
    env = cfg.get("env") if hasattr(cfg, "get") else None
    env_monitor = env.get("monitor_metric") if env is not None and hasattr(env, "get") else None
    monitor = paradigm_def.get("monitor_metric") or env_monitor
    for problem in selection_errors(cfg, monitor):
        raise ConfigurationError(f"[ConfigurationError] Experiment '{experiment_name}': {problem}")

    # --- Paradigm-Method/Agent Validation ---
    _validate_methods_for_paradigm(cfg, paradigm_name, paradigm_def)

    # --- Missing Components Validation ---
    missing_comps = find_missing_components(cfg)
    if missing_comps:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' references missing component(s): "
            f"{', '.join(missing_comps)}. Install them via the Component Hub."
        )

    # New-style sweep settings are validated before either launcher starts work.
    if "tuning" in cfg and cfg.get("tuning") is not None:
        _validate_tuning(cfg)

    # --- Optuna sweep direction notice ---
    if is_sweep:
        try:
            sweeper = cfg.get("hydra", {}).get("sweeper", {})
            direction = sweeper.get("direction", None)
            monitor = paradigm_def.get("monitor_metric", None)
            if direction == "maximize" and monitor and "loss" in monitor:
                notices.append(
                    f"Notice: Optuna direction is 'maximize' on paradigm '{paradigm_name}' "
                    f"which monitors '{monitor}'. Consider using 'direction: minimize'."
                )
        except Exception:
            pass

    # --- Offline dataset path checks (non-fatal) ---
    _validate_offline_dataset_paths(cfg, notices)

    return notices


def _validate_tuning(cfg: Any) -> None:
    from src.app.pipeline.commands import _sweep_expression

    tuning = cfg.get("tuning")
    if not hasattr(tuning, "get"):
        raise ConfigurationError("Sweep tuning must be a mapping.")
    if not isinstance(tuning.get("metric"), str) or not tuning.get("metric").strip():
        raise ConfigurationError("Sweep tuning requires an objective metric.")
    if tuning.get("direction") not in ("minimize", "maximize"):
        raise ConfigurationError("Sweep direction must be 'minimize' or 'maximize'.")
    for key, default in (("n_trials", 40), ("n_jobs", 1)):
        value = tuning.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ConfigurationError(f"Sweep {key} must be a positive integer.")
    missing = []
    for name, method in parse_methods_dict(cfg).items():
        dimensions = method.get("tune") or method.get("search_space") or {}
        if not dimensions:
            missing.append(name)
            continue
        for target, dimension in dimensions.items():
            try:
                if not isinstance(dimension, str) and not str(target).startswith(("agent.", "model.", "config.")):
                    raise ValueError(f"Typed sweep dimension {target!r} requires an explicit agent, model, or config path")
                _sweep_expression(target, dimension)
            except ValueError as exc:
                raise ConfigurationError(f"Method '{name}': {exc}") from exc
    if missing:
        raise ConfigurationError("Sweep methods require at least one dimension: " + ", ".join(missing))


def _validate_methods_for_paradigm(cfg: Any, paradigm_name: str, paradigm_def: dict) -> None:
    """Validate that methods declared in config match the paradigm's agent and model requirements.

    Raises:
        ConfigurationError: If any method is incompatible with the selected paradigm.
    """
    methods = parse_methods_dict(cfg)
    if not methods:
        return

    allowed_agents = paradigm_def.get("allowed_agents", None)
    algorithm_requirement = _algorithm_requirement(paradigm_def)
    forbidden_agents = paradigm_def.get("forbidden_agents", [])

    for method_name, method_cfg in methods.items():
        ag = method_cfg.get("agent")
        if isinstance(ag, dict):
            agent_algo = ag.get("name") or ag.get("algorithm") or ag.get("type") or ag.get("algo")
        elif isinstance(ag, str):
            agent_algo = ag
        else:
            agent_algo = None

        mo = method_cfg.get("model")
        if isinstance(mo, dict):
            model_arch = mo.get("name") or mo.get("architecture") or mo.get("type") or mo.get("base")
        elif isinstance(mo, str):
            model_arch = mo
        else:
            model_arch = None

        # Every method must have a model
        if not model_arch:
            raise ConfigurationError(
                f"[ConfigurationError] Method '{method_name}' is missing required key 'model'. "
                f"Every method must specify a model architecture (for example, 'model: package/model')."
            )

        # The method contract, not the paradigm's name, decides whether a
        # method may carry an algorithm harness.
        if algorithm_requirement == "forbidden":
            if agent_algo:
                raise ConfigurationError(
                    f"[ConfigurationError] Paradigm '{paradigm_name}' does not use RL agent harnesses "
                    f"(its method contract forbids an algorithm harness), "
                    f"but method '{method_name}' declared 'agent: {agent_algo}'."
                )

        # Forbidden agents check
        if forbidden_agents and agent_algo:
            clean_algo = agent_algo.lower().replace("-", "_")
            if any(
                clean_algo == f.lower()
                or clean_algo.startswith(f.lower() + "_")
                or clean_algo.startswith(f.lower() + "/")
                or clean_algo.endswith("/" + f.lower())
                for f in forbidden_agents
            ):
                raise ConfigurationError(
                    f"[ConfigurationError] Method '{method_name}' uses agent '{agent_algo}', "
                    f"which is forbidden in paradigm '{paradigm_name}'. Forbidden agents: {forbidden_agents}."
                )

        if algorithm_requirement == "required" and not agent_algo:
            allowed_str = f" (allowed: {allowed_agents})" if allowed_agents else ""
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' requires an agent harness{allowed_str}, "
                f"but method '{method_name}' has no 'agent' declared."
            )

        if agent_algo:
            # 1. Query agent-declared supported paradigms
            agent_paradigms = _get_agent_supported_paradigms(agent_algo)
            if agent_paradigms is not None:
                if paradigm_name.lower() not in agent_paradigms:
                    raise ConfigurationError(
                        f"[ConfigurationError] Method '{method_name}' uses agent '{agent_algo}', "
                        f"which is not allowed in paradigm '{paradigm_name}'."
                    )
            elif allowed_agents:
                # 2. Fallback to paradigm allowed_agents if defined
                clean_algo = agent_algo.lower().replace("-", "_")
                in_allowed = any(
                    clean_algo == a.lower()
                    or clean_algo.startswith(a.lower() + "_")
                    or clean_algo.startswith(a.lower() + "/")
                    or clean_algo.endswith("/" + a.lower())
                    for a in allowed_agents
                )
                if not in_allowed:
                    raise ConfigurationError(
                        f"[ConfigurationError] Method '{method_name}' uses agent '{agent_algo}', "
                        f"which is not allowed in paradigm '{paradigm_name}'. Allowed agents: {allowed_agents}."
                    )


def _get_agent_supported_paradigms(agent_algo: str) -> list[str] | None:
    """Query supported paradigms from agent class or config YAML."""
    clean_algo = agent_algo.lower().replace("-", "_")
    try:
        from src.usr.algorithms.agent_registry import AGENT_REGISTRY, auto_discover, get_agent_class

        if not AGENT_REGISTRY:
            auto_discover()
        agent_cls = None
        try:
            agent_cls = get_agent_class(clean_algo)
        except Exception:
            agent_cls = AGENT_REGISTRY.get(clean_algo) or AGENT_REGISTRY.get(agent_algo)

        if agent_cls is not None:
            if hasattr(agent_cls, "get_supported_paradigms"):
                return [str(p).lower() for p in agent_cls.get_supported_paradigms()]
            if hasattr(agent_cls, "supported_paradigms"):
                p = getattr(agent_cls, "supported_paradigms")
                if isinstance(p, (list, tuple, set)):
                    return [str(x).lower() for x in p]
                return [str(p).lower()]
            if hasattr(agent_cls, "compatible_paradigms"):
                p = getattr(agent_cls, "compatible_paradigms")
                if isinstance(p, (list, tuple, set)):
                    return [str(x).lower() for x in p]
                return [str(p).lower()]
    except Exception:
        pass

    # Search config files in in/config/agent/
    agent_dir = Path("in/config/agent")
    if agent_dir.exists():
        for p in [
            agent_dir / f"{agent_algo}.yaml",
            agent_dir / f"{agent_algo.replace('_', '/')}.yaml",
            agent_dir / f"{clean_algo}.yaml",
        ]:
            if p.is_file():
                try:
                    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                    paradigms = (
                        data.get("paradigms") or data.get("compatible_paradigms") or data.get("supported_paradigms")
                    )
                    if paradigms:
                        if isinstance(paradigms, list):
                            return [str(x).lower() for x in paradigms]
                        return [str(paradigms).lower()]
                except Exception:
                    pass

        # Glob match for nested package profiles like in/config/agent/<package>/<profile>.yaml
        for y in agent_dir.rglob("*.yaml"):
            if y.stem.lower() in (clean_algo, agent_algo.lower()):
                try:
                    data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
                    paradigms = (
                        data.get("paradigms") or data.get("compatible_paradigms") or data.get("supported_paradigms")
                    )
                    if paradigms:
                        if isinstance(paradigms, list):
                            return [str(x).lower() for x in paradigms]
                        return [str(paradigms).lower()]
                except Exception:
                    pass

    # Search installed component manifests in the algorithm component directory.
    base_dir = Path("src/usr/algorithms")
    if base_dir.exists():
        candidates = [
            base_dir / agent_algo,
            base_dir / clean_algo,
            base_dir / (agent_algo.split("/")[0] if "/" in agent_algo else agent_algo),
            base_dir / (clean_algo.split("_")[0] if "_" in clean_algo else clean_algo),
        ]
        for candidate_dir in candidates:
            if candidate_dir.is_dir():
                for meta_file in [candidate_dir / ".theta_component.json", candidate_dir / "plugin.json"]:
                    if meta_file.is_file():
                        try:
                            import json

                            meta = json.loads(meta_file.read_text(encoding="utf-8"))
                            paradigms = meta.get("paradigms") or meta.get("compatible_paradigms")
                            if paradigms:
                                if isinstance(paradigms, list):
                                    return [str(x).lower() for x in paradigms]
                                return [str(paradigms).lower()]
                        except Exception:
                            pass

    return None


def _agent_exists(agent_algo: str) -> bool:
    """Check whether an agent is registered or has an installed agent profile."""
    if not agent_algo:
        return False
    clean_algo = str(agent_algo).lower().replace("-", "_")
    try:
        from src.usr.algorithms.agent_registry import AGENT_REGISTRY, auto_discover

        if not AGENT_REGISTRY:
            auto_discover()
        if clean_algo in AGENT_REGISTRY or agent_algo in AGENT_REGISTRY:
            return True
    except Exception:
        pass

    prefix = str(agent_algo).split("/")[0].lower()
    project_root = _PROJECT_ROOT

    for root in dict.fromkeys((Path.cwd(), project_root)):
        algorithms_dir = root / "src/usr/algorithms"
        for d in (algorithms_dir / agent_algo, algorithms_dir / clean_algo, algorithms_dir / prefix):
            if d.is_dir():
                return True

        agent_dir = root / "in/config/agent"
        if agent_dir.exists():
            for p in (
                agent_dir / f"{agent_algo}.yaml",
                agent_dir / f"{agent_algo.replace('_', '/')}.yaml",
                agent_dir / f"{clean_algo}.yaml",
            ):
                if p.is_file():
                    return True
            for y in agent_dir.rglob("*.yaml"):
                if y.stem.lower() in (clean_algo, str(agent_algo).lower()):
                    return True
    return False


def _model_exists(name: str) -> bool:
    """Check whether a model architecture is registered, installed as a plugin, or configured in in/config/model/."""
    if not name:
        return True
    clean = str(name).lower().strip()
    stem = clean.split("/")[-1]

    # Model registry
    try:
        from src.app.core.model_registry import MODEL_REGISTRY, auto_discover_models

        if not MODEL_REGISTRY:
            auto_discover_models()
        if clean in MODEL_REGISTRY or stem in MODEL_REGISTRY:
            return True
        if any(
            clean == p or clean.startswith(p + "_") or stem == p or stem.startswith(p + "_") for p in MODEL_REGISTRY
        ):
            return True
    except Exception:
        pass

    # Configs in in/config/model
    cfg_root = Path("in/config/model")
    if cfg_root.is_dir():
        for p in [cfg_root / f"{clean}.yaml", cfg_root / f"{stem}.yaml", cfg_root / "basics" / f"{stem}.yaml"]:
            if p.is_file():
                return True
        for y in cfg_root.rglob("*.yaml"):
            if y.stem.lower() in (clean, stem):
                return True

    # Directories in src/usr/models
    models_dir = Path("src/usr/models")
    if models_dir.is_dir():
        for d in [models_dir / clean, models_dir / stem]:
            if d.is_dir():
                return True

    return False


def _env_exists(env_name: str) -> bool:
    """Check whether an environment exists as a plugin, config, registered env, or Gymnasium env."""
    if not env_name or env_name == "unknown":
        return True
    clean = str(env_name).strip().lower()
    stem = clean.split("/")[-1]

    cfg_root = Path("in/config/env")
    if cfg_root.is_dir():
        for p in [cfg_root / f"{clean}.yaml", cfg_root / f"{stem}.yaml"]:
            if p.is_file():
                return True
        for y in cfg_root.rglob("*.yaml"):
            if y.stem.lower() == stem or y.relative_to(cfg_root).with_suffix("").as_posix().lower() == clean:
                return True

    for envs_dir in [Path("in/envs"), Path("src/usr/environments")]:
        if envs_dir.is_dir():
            for d in [envs_dir / clean, envs_dir / stem]:
                if d.is_dir():
                    return True

    try:
        import gymnasium as gym

        for spec_id in gym.envs.registry.keys():
            spec_clean = spec_id.lower()
            if clean == spec_clean or stem == spec_clean:
                return True
            spec_stem = spec_clean.split("/")[-1].split("-v")[0]
            if stem == spec_stem or clean == spec_stem:
                return True
    except Exception:
        pass

    return False


def find_missing_components(cfg_or_dict: Any) -> list[str]:
    """Inspect an experiment configuration and return a list of all missing components.

    Checks environment, agents, models, and composite submodels against installed plugins,
    code registries, and in/config/ YAML files.
    """
    if not cfg_or_dict or "MagicMock" in type(cfg_or_dict).__name__:
        return []
    missing: list[str] = []

    # Extract methods
    if hasattr(cfg_or_dict, "methods"):
        methods = cfg_or_dict.methods
    elif isinstance(cfg_or_dict, dict):
        methods = cfg_or_dict.get("methods", {})
    else:
        methods = {}

    # Extract environment
    env_obj = None
    if hasattr(cfg_or_dict, "env"):
        env_obj = cfg_or_dict.env
    elif isinstance(cfg_or_dict, dict):
        env_obj = cfg_or_dict.get("env")

    env_val = None
    if isinstance(env_obj, str):
        env_val = env_obj
    elif isinstance(env_obj, dict):
        candidate_env = env_obj.get("name") or env_obj.get("env_id")
        env_val = candidate_env if isinstance(candidate_env, str) else None
    elif env_obj is not None:
        candidate_env = getattr(env_obj, "name", None) or getattr(env_obj, "env_id", None)
        env_val = candidate_env if isinstance(candidate_env, str) else None

    # If env was not resolved from object, check defaults overrides
    if not env_val:
        defaults = (
            getattr(cfg_or_dict, "defaults", None)
            if hasattr(cfg_or_dict, "defaults")
            else (cfg_or_dict.get("defaults") if isinstance(cfg_or_dict, dict) else None)
        )
        if defaults and isinstance(defaults, (list, tuple)):
            for d in defaults:
                if isinstance(d, dict) and "override /env" in d:
                    env_val = d["override /env"]
                elif isinstance(d, str) and d.startswith("override /env:"):
                    env_val = d.split(":", 1)[1].strip()

    paradigm = (
        getattr(cfg_or_dict, "paradigm", None)
        if hasattr(cfg_or_dict, "paradigm")
        else (cfg_or_dict.get("paradigm") if isinstance(cfg_or_dict, dict) else None)
    )

    if env_val and env_val != "unknown" and not _env_exists(str(env_val)):
        missing.append(f"environment: {env_val}")
    elif not env_val and paradigm_uses_live_environment(str(paradigm) if paradigm else None):
        missing.append("environment: (none configured)")

    # Check static datasets for offline paradigms
    if paradigm and paradigm_uses_static_dataset(str(paradigm)):
        is_wf = (
            getattr(cfg_or_dict, "workflow", None)
            if hasattr(cfg_or_dict, "workflow")
            else (cfg_or_dict.get("workflow") if isinstance(cfg_or_dict, dict) else None)
        )
        if not is_wf and hasattr(methods, "items"):
            try:
                from src.app.pipeline.datasets import resolve_dataset_for_method

                for m_name, m_spec in methods.items():
                    if not m_spec or not hasattr(m_spec, "get"):
                        continue
                    if not method_is_enabled(m_spec):
                        continue
                    try:
                        ds = resolve_dataset_for_method(m_name, m_spec, cfg_or_dict)
                        if not ds.exists():
                            missing.append(f"dataset: {ds}")
                    except Exception:
                        ds_name = None
                        if env_obj and hasattr(env_obj, "get"):
                            ds_name = env_obj.get("dataset_name") or env_obj.get("name")
                        missing.append(f"dataset: {ds_name or m_name}")
            except Exception:
                pass

    if hasattr(methods, "items"):
        for m_name, m_spec in methods.items():
            if not m_spec or not hasattr(m_spec, "get"):
                continue
            if not method_is_enabled(m_spec):
                continue

            # Check agent
            ag = m_spec.get("agent")
            if isinstance(ag, dict) or hasattr(ag, "get"):
                ag_str = ag.get("name") or ag.get("algorithm") or ag.get("type")
            elif ag:
                ag_str = str(ag)
            else:
                ag_str = None
            if ag_str and not _agent_exists(ag_str):
                missing.append(f"agent: {ag_str}")

            # Check model
            mo = m_spec.get("model")
            mo_str = None
            if isinstance(mo, str):
                mo_str = mo
            elif isinstance(mo, dict) or hasattr(mo, "get"):
                mo_str = mo.get("name") or mo.get("architecture") or mo.get("type") or mo.get("base")
                if not mo_str and len(mo) == 1:
                    first_k = next(iter(mo))
                    mo_str = first_k
                    val = mo[first_k]
            if mo_str and not _model_exists(mo_str):
                missing.append(f"model: {mo_str}")

    return list(dict.fromkeys(missing))


def _validate_offline_dataset_paths(cfg: Any, notices: list[str]) -> None:
    """Ensure static dataset paradigms have resolvable dataset files before launching."""
    paradigm_name = cfg.get("paradigm") if hasattr(cfg, "get") else getattr(cfg, "paradigm", None)
    if not paradigm_uses_static_dataset(paradigm_name):
        return
    is_wf = cfg.get("workflow") if hasattr(cfg, "get") else getattr(cfg, "workflow", None)
    if is_wf:
        return

    from src.app.pipeline.datasets import resolve_dataset_for_method

    methods = parse_methods_dict(cfg)
    for method_name, method_cfg in methods.items():
        try:
            ds_path = resolve_dataset_for_method(method_name, method_cfg, cfg)
            if not ds_path.exists():
                raise ConfigurationError(
                    f"[ConfigurationError] Method '{method_name}' in paradigm '{paradigm_name}' "
                    f"resolved to dataset '{ds_path}', but the path does not exist on disk."
                )
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(
                f"[ConfigurationError] Method '{method_name}' in paradigm '{paradigm_name}' "
                f"requires a valid dataset on disk: {e}"
            )
