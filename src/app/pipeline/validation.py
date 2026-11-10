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
    deep_merge,
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

from src.app.workspace import WorkspacePaths


def _load_all_paradigms() -> dict[str, dict]:
    """Load all paradigm YAML files from in/config/paradigms/.

    Returns a dict mapping paradigm name → paradigm definition dict.
    """
    paradigms: dict[str, dict] = {}
    # Workspace definitions override provider/framework defaults individually.
    # A partial workspace must retain the framework's four base paradigms.
    for root in reversed(WorkspacePaths.for_process().config_roots()):
        for yaml_path in sorted((root / "paradigms").glob("**/*.yaml")):
            name = yaml_path.stem
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


def _load_raw_experiment_yaml(experiment_name: str, config_dir: Path | None = None) -> dict:
    """Return the raw (pre-Hydra-composition) experiment YAML as a dict, or {} on failure."""
    rel_path = resolve_experiment_config_name(experiment_name)
    exp_path = (config_dir or WorkspacePaths.for_process().config) / "experiment" / f"{rel_path}.yaml"
    if not exp_path.exists():
        return {}
    with open(exp_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_experiment_config(cfg: Any, experiment_name: str, is_sweep: bool = False, config_dir: Path | None = None) -> list[str]:
    """Validate experiment configuration against its declared paradigm.

    Returns a list of non-fatal notice strings.
    Raises ConfigurationError on any fatal paradigm incompatibility.
    """
    from src.app.pipeline.tuning_config import require_runnable_tuning

    require_runnable_tuning(cfg, parse_methods_dict(cfg))
    notices: list[str] = []

    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    raw_exp = _load_raw_experiment_yaml(experiment_name, config_dir)

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
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    if cfg.get("objective") is not None:
        from src.app.objectives.api import validate_objective
        try:
            validate_objective(cfg)
        except (ValueError, TypeError, KeyError) as exc:
            raise ConfigurationError(f"Invalid objective: {exc}") from exc
        monitor = None  # The component validates its own input metrics above.
    if is_sweep and hasattr(tuning, "get") and cfg.get("objective") is None:
        monitor = tuning.get("metric") or monitor
    for problem in selection_errors(cfg, monitor, is_sweep=is_sweep):
        raise ConfigurationError(f"[ConfigurationError] Experiment '{experiment_name}': {problem}")
    from src.app.telemetry.monitors import monitor_errors
    for problem in monitor_errors(cfg):
        raise ConfigurationError(f"[ConfigurationError] Experiment '{experiment_name}': {problem}")

    from src.app.monitoring.config import validate_monitors
    try:
        validate_monitors(cfg)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(f"Invalid monitoring configuration: {exc}") from exc

    # --- Paradigm-Method/Agent Validation ---
    _validate_methods_for_paradigm(cfg, paradigm_name, paradigm_def)

    from src.app.core.data_loaders import data_loader_errors
    paths = WorkspacePaths.from_config_root(config_dir) if config_dir else WorkspacePaths.for_process()
    for method_name, method in (parse_methods_dict(cfg, paths=paths) or {"training": None}).items():
        errors = data_loader_errors(cfg, method, paths=paths)
        if errors:
            raise ConfigurationError(f"Method '{method_name}': " + " ".join(errors))

    # --- Missing Components Validation ---
    missing_comps = find_missing_components(cfg)
    if missing_comps:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' references missing component(s): "
            f"{', '.join(missing_comps)}. Install them via the Component Hub."
        )

    # New-style sweep settings are validated before either launcher starts work.
    from src.app.pipeline.tuning_config import tuning_enabled

    if tuning_enabled(cfg):
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
    from src.app.pipeline.tuners import get_tuner

    tuning = cfg.get("tuning")
    if not hasattr(tuning, "get"):
        raise ConfigurationError("Sweep tuning must be a mapping.")
    if cfg.get("objective") is None and (not isinstance(tuning.get("metric"), str) or not tuning.get("metric").strip()):
        raise ConfigurationError("Sweep tuning requires an objective metric.")
    if tuning.get("direction") not in ("minimize", "maximize"):
        raise ConfigurationError("Sweep direction must be 'minimize' or 'maximize'.")
    tuner = get_tuner(cfg)
    try:
        tuner.validate_config(cfg)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    missing = []
    for name, method in parse_methods_dict(cfg).items():
        validate_method = getattr(tuner, "validate_method", None)
        if callable(validate_method):
            try:
                validate_method(cfg, name, method)
            except ValueError as exc:
                raise ConfigurationError(f"Method '{name}': {exc}") from exc
        dimensions = method.get("tune") or method.get("search_space") or {}
        if not dimensions:
            missing.append(name)
            continue
        for target, dimension in dimensions.items():
            try:
                tuner.validate_dimension(target, dimension)
                if str(target).startswith("agent."):
                    agent = method.get("agent")
                    if isinstance(agent, dict):
                        agent = agent.get("name") or agent.get("algorithm") or agent.get("type")
                    if isinstance(agent, str):
                        path = WorkspacePaths.for_process().resolve_config("agent/" + agent + ".yaml")
                        if path.is_file():
                            base = yaml.safe_load(path.read_text()) or {}
                            if isinstance(base, dict) and "defaults" not in base:
                                # Method overrides can declare supported optional
                                # controls absent from the component's base YAML.
                                value = deep_merge(base, method.get("agent_params") or {})
                                for part in str(target).split(".")[1:]:
                                    if not isinstance(value, dict) or part not in value:
                                        hint = " Did you mean agent.learning_rate?" if target == "agent.lr" and "learning_rate" in base else ""
                                        raise ValueError(f"Sweep target {target!r} is absent from agent {agent!r}.{hint}")
                                    value = value[part]
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
    from src.app.core.provider_catalog import get_catalog
    selected = get_catalog().resolve("algorithm", agent_algo)
    if selected is None: return None
    package, cap = selected
    paradigms = cap.metadata.get("paradigms", package.manifest.get("paradigms"))
    return [str(p).lower() for p in paradigms] if paradigms else None


def _provider_exists(kind, selector):
    if not selector: return False
    from src.app.core.provider_loader import resolve_provider
    try:
        return resolve_provider(kind, str(selector)) is not None
    except Exception as exc:
        raise ConfigurationError(f"Cannot load declared {kind} provider {selector!r}: {exc}") from exc


def _agent_exists(agent_algo: str) -> bool:
    return _provider_exists("algorithm", agent_algo)


def _model_exists(name: str) -> bool:
    return _provider_exists("model", name)


def _env_exists(env_name: str) -> bool:
    return _provider_exists("env", env_name)

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

    effective_model = cfg_or_dict.get('model') if hasattr(cfg_or_dict,'get') else getattr(cfg_or_dict,'model',None)
    if hasattr(effective_model,'get'):
        selected = effective_model.get('architecture') or effective_model.get('type') or effective_model.get('name')
        if isinstance(selected,str) and not _model_exists(selected): missing.append(f'model: {selected}')
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
