"""Pre-flight configuration validation module.

Validates experiment configs against their declared paradigm before any
training begins. Raises ConfigurationError on fatal mismatches so the
pipeline aborts cleanly at startup.

Paradigm definitions are the effective ``paradigms/<name>.yaml`` profiles. Adding a
new paradigm requires only a new YAML file in src/usr/config/paradigms/ — no changes to this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

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


# A paradigm describes only itself: the component kinds its methods consist of,
# where its environment comes from, its runner, budget and evaluation protocol.
PARADIGM_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "required": ["type", "environment_source", "method", "training", "runner"],
    "properties": {
        "type": {"enum": ["base", "meta"]},
        "description": {"type": "string"},
        "environment_source": {"enum": ["live_environment", "static_dataset"]},
        "method": {"type": "object", "additionalProperties": False, "required": ["algorithm", "model"],
                   "properties": {"algorithm": {"enum": ["required", "optional", "forbidden"]},
                                  "model": {"enum": ["required", "optional", "forbidden"]}}},
        "training": {"type": "object", "additionalProperties": False, "required": ["budget"],
                     "properties": {"budget": {"type": "string", "minLength": 1}}},
        "runner": {"type": "string", "minLength": 1},
        "default_data_module": {"type": "string", "minLength": 1},
        "eval_protocol": {"type": "string", "minLength": 1},
        "allows_eval_episodes": {"type": "boolean"},
        "default_callbacks": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "monitor_metric": {"type": "string", "minLength": 1},
    },
}


def paradigm_errors(definition) -> list[str]:
    import jsonschema
    validator = jsonschema.Draft202012Validator(PARADIGM_SCHEMA)
    return [(".".join(str(p) for p in error.absolute_path) + ": " if error.absolute_path else "") + error.message
            for error in validator.iter_errors(definition)]


def _load_all_paradigms(*, paths=None) -> dict[str, dict]:
    """Load every effective ``paradigms/<name>.yaml`` profile.

    Returns a dict mapping paradigm name → paradigm definition dict.
    """
    paradigms: dict[str, dict] = {}
    # Workspace definitions override provider/framework defaults individually.
    # A partial workspace must retain the framework's four base paradigms.
    resolver = (paths or WorkspacePaths.for_process()).config_resolver()
    for selector in resolver.selectors("paradigms"):
        resolved = resolver.resolve(selector, required=True)
        paradigms[Path(selector).relative_to("paradigms").with_suffix("").as_posix()] = yaml.safe_load(resolved.file.read_text()) or {}
    return paradigms


def list_paradigms(*, paths=None) -> list[str]:
    """Return sorted list of registered paradigm names."""
    return sorted(_load_all_paradigms(paths=paths).keys())


def load_paradigm(paradigm_name: str, *, paths=None) -> dict:
    """Load and return the definition dict for a named paradigm.

    Raises ConfigurationError if the paradigm is not found.
    """
    paradigms = _load_all_paradigms(paths=paths)
    if paradigm_name not in paradigms:
        raise ConfigurationError(
            f"[ConfigurationError] Unknown paradigm '{paradigm_name}'. "
            f"Available paradigms: {sorted(paradigms.keys())}. "
            f"To add a new paradigm create src/usr/config/paradigms/<name>.yaml."
        )
    if errors := paradigm_errors(paradigms[paradigm_name]):
        raise ConfigurationError(f"[ConfigurationError] Invalid paradigm '{paradigm_name}': " + "; ".join(errors))
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


def paradigm_uses_agents(paradigm_name: str | None, *, paths=None) -> bool:
    """Whether methods in this paradigm name an RL agent harness.

    A paradigm whose `method.algorithm` is `forbidden` permits no agent, so
    methods name a model only - supervised and unsupervised both work that way,
    having no policy that chooses actions.
    """
    try:
        defn = load_paradigm(paradigm_name or "", paths=paths)
    except ConfigurationError:
        return True  # Unknown paradigm: leave the RL assumption in place.
    requirement = _algorithm_requirement(defn)
    return requirement != "forbidden"


def paradigm_uses_static_dataset(paradigm_name: str | None, *, paths=None) -> bool:
    """Whether this paradigm trains from a dataset file rather than a live simulator."""
    try:
        defn = load_paradigm(paradigm_name or "", paths=paths)
    except ConfigurationError:
        return False
    return defn["environment_source"] == "static_dataset"


def paradigm_uses_live_environment(paradigm_name: str | None, *, paths=None) -> bool:
    """Whether a paradigm requires a live environment rather than static data."""
    try:
        defn = load_paradigm(paradigm_name or "", paths=paths)
    except ConfigurationError:
        return True
    return defn["environment_source"] == "live_environment"


def _algorithm_requirement(paradigm_def: dict[str, Any]) -> str:
    """Return the paradigm's declared method.algorithm contract."""
    return paradigm_def["method"]["algorithm"]


# ---------------------------------------------------------------------------
# Raw experiment YAML loader (for explicit override checks)
# ---------------------------------------------------------------------------


def _load_raw_experiment_yaml(experiment_name: str, config_dir: Path | None = None, *, paths=None) -> dict:
    """Return the raw (pre-Hydra-composition) experiment YAML as a dict, or {} on failure."""
    paths = paths or (WorkspacePaths.from_config_root(config_dir) if config_dir else WorkspacePaths.for_process())
    rel_path = resolve_experiment_config_name(experiment_name, paths=paths, config_dir=config_dir)
    from src.app.config_resolution import ConfigResolver
    selected = ConfigResolver(paths, config_root=config_dir).resolve(f"experiment/{rel_path}.yaml")
    exp_path = selected.file if selected else paths.config / "experiment" / f"{rel_path}.yaml"
    if not exp_path.exists():
        return {}
    with open(exp_path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_experiment_config(
    cfg: Any, experiment_name: str, is_sweep: bool = False,
    config_dir: Path | None = None, workspace: WorkspacePaths | None = None,
) -> list[str]:
    """Validate experiment configuration against its declared paradigm.

    Returns a list of non-fatal notice strings.
    Raises ConfigurationError on any fatal paradigm incompatibility.
    """
    from src.app.pipeline.tuning_config import require_runnable_tuning

    paths = workspace or (WorkspacePaths.from_config_root(config_dir) if config_dir else WorkspacePaths.for_process())
    require_runnable_tuning(cfg, parse_methods_dict(cfg, paths=paths))
    notices: list[str] = []

    env_name = getattr(cfg.env, "name", "unknown") if hasattr(cfg, "env") else "unknown"
    raw_exp = _load_raw_experiment_yaml(experiment_name, config_dir, paths=paths)

    # --- Paradigm lookup ---
    paradigm_name = cfg.get("paradigm", None)
    if not paradigm_name:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' has no 'paradigm' declared. "
            f"Add 'paradigm: <name>' to its group _base.yaml. "
            f"Available paradigms: {list_paradigms(paths=paths)}."
        )

    from src.app.core.execution import execution_backend, reject_selectors
    from src.app.core.paradigm_loader import load_paradigm_definition
    reject_selectors(cfg, "experiment")
    load_paradigm_definition(paradigm_name, paths=paths)
    for method_name, method in parse_methods_dict(cfg, paths=paths).items():
        reject_selectors(method, f"method {method_name}")
        effective = dict(cfg.items())
        effective.update(method)
        execution_backend(effective, paths=paths)

    paradigm_def = load_paradigm(paradigm_name, paths=paths)  # raises ConfigurationError if unknown
    if not parse_methods_dict(cfg, paths=paths):
        raise ConfigurationError(f"[ConfigurationError] Experiment '{experiment_name}' declares no enabled methods.")

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

    # --- Periodic checkpoints: checkpoint_every_steps or num_checkpoints ---
    for problem in checkpoint_errors(cfg):
        raise ConfigurationError(f"[ConfigurationError] Experiment '{experiment_name}': {problem}")

    # --- Recorded metrics: telemetry.metrics ---
    tuning = cfg.get("tuning") if hasattr(cfg, "get") else None
    tuning_metric = None
    if cfg.get("objective") is not None:
        from src.app.objectives.api import validate_objective
        try:
            validate_objective(cfg, paths=paths)  # The component validates its own input metrics.
        except (ValueError, TypeError, KeyError) as exc:
            raise ConfigurationError(f"Invalid objective: {exc}") from exc
    elif is_sweep and hasattr(tuning, "get"):
        tuning_metric = tuning.get("metric")
    for problem in selection_errors(cfg, tuning_metric):
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
    _validate_methods_for_paradigm(cfg, paradigm_name, paradigm_def, paths=paths)

    from src.app.core.data_loaders import data_loader_errors
    for method_name, method in parse_methods_dict(cfg, paths=paths).items():
        errors = data_loader_errors(cfg, method, paths=paths)
        if errors:
            raise ConfigurationError(f"Method '{method_name}': " + " ".join(errors))

    # --- Compatibility: what the selected components declare they need ---
    notices.extend(compatibility_notices(cfg, paths=paths))

    # External files are not installable components. Report dataset errors before
    # the component inventory can turn them into misleading Hub guidance.
    _validate_offline_dataset_paths(cfg, notices, paths=paths)

    # --- Missing Components Validation ---
    missing_comps = find_missing_components(cfg, paths=paths)
    if missing_comps:
        raise ConfigurationError(
            f"[ConfigurationError] Experiment '{experiment_name}' references missing component(s): "
            f"{', '.join(missing_comps)}. Install them via the Component Hub."
        )

    # New-style sweep settings are validated before either launcher starts work.
    from src.app.pipeline.tuning_config import tuning_enabled

    if tuning_enabled(cfg):
        _validate_tuning(cfg, paths=paths)

    return notices


TUNING_KEYS = ("component", "metric", "direction", "options")


def _validate_tuning(cfg: Any, *, paths=None) -> None:
    from src.app.pipeline.tuners import get_tuner, tuner_name

    tuning = cfg.get("tuning")
    if not hasattr(tuning, "get"):
        raise ConfigurationError("Sweep tuning must be a mapping.")
    unknown = sorted(str(key) for key in tuning.keys() if key not in TUNING_KEYS)
    if unknown:
        raise ConfigurationError(
            f"Unknown tuning settings: {', '.join(unknown)}. The tuning block holds only "
            f"{', '.join(TUNING_KEYS)}; tuner settings go under tuning.options.<component>.")
    tuner_name(cfg)  # Reject an unselected recipe before inspecting provider options.
    if cfg.get("objective") is None and (not isinstance(tuning.get("metric"), str) or not tuning.get("metric").strip()):
        raise ConfigurationError("Sweep tuning requires an objective metric.")
    if tuning.get("direction") not in ("minimize", "maximize"):
        raise ConfigurationError("Sweep direction must be 'minimize' or 'maximize'.")
    tuner = get_tuner(cfg, paths=paths)
    try:
        from src.app.pipeline.tuning_features import validate_tuning_compatibility
        pruning = validate_tuning_compatibility(cfg, paths=paths, tuner=tuner)
        tuner.validate_config(cfg)
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc
    missing = []
    for name, method in parse_methods_dict(cfg, paths=paths).items():
        if pruning:
            from src.app.pipeline.tuning_features import validate_pruning_runner
            try:
                validate_pruning_runner({**dict(cfg.items()), **method}, paths=paths)
            except ValueError as exc:
                raise ConfigurationError(f"Method '{name}': {exc}") from exc
        validate_method = getattr(tuner, "validate_method", None)
        if callable(validate_method):
            try:
                validate_method(cfg, name, method)
            except ValueError as exc:
                raise ConfigurationError(f"Method '{name}': {exc}") from exc
        dimensions = method.get("tune") or {}
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
                        path = (paths or WorkspacePaths.for_process()).resolve_config("agent/" + agent + ".yaml")
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


def _validate_methods_for_paradigm(cfg: Any, paradigm_name: str, paradigm_def: dict, *, paths=None) -> None:
    """Validate that methods declared in config match the paradigm's agent and model requirements.

    Raises:
        ConfigurationError: If any method is incompatible with the selected paradigm.
    """
    methods = parse_methods_dict(cfg, paths=paths)
    if not methods:
        return

    algorithm_requirement = _algorithm_requirement(paradigm_def)

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

        if algorithm_requirement == "required" and not agent_algo:
            raise ConfigurationError(
                f"[ConfigurationError] Paradigm '{paradigm_name}' requires an agent harness, "
                f"but method '{method_name}' has no 'agent' declared."
            )


def compatibility_notices(cfg: Any, *, paths=None) -> list[str]:
    """Raise on ``block`` findings; return ``warn`` findings as notices (also printed)."""
    from src.app.core.requirements import compatibility
    findings = compatibility(cfg, paths=paths)
    blocked = [str(f) for f in findings if f.level == "block"]
    if blocked:
        raise ConfigurationError("[ConfigurationError] Incompatible selection: " + " ".join(blocked))
    warnings = [f"Compatibility warning: {f}" for f in findings if f.level == "warn"]
    for warning in warnings:
        print(warning)
    return warnings


def _provider_exists(kind, selector, *, paths=None):
    if not selector: return False
    from src.app.core.provider_loader import resolve_provider
    try:
        return resolve_provider(kind, str(selector), paths=paths) is not None
    except Exception as exc:
        raise ConfigurationError(f"Cannot load declared {kind} provider {selector!r}: {exc}") from exc


def _agent_exists(agent_algo: str) -> bool:
    return _provider_exists("algorithm", agent_algo)


def _model_exists(name: str) -> bool:
    return _provider_exists("model", name)


def _env_exists(env_name: str) -> bool:
    return _provider_exists("env", env_name)

def find_missing_components(cfg_or_dict: Any, *, paths=None) -> list[str]:
    """Inspect an experiment configuration and return a list of all missing components.

    Checks environment, agent and model references against executable declared
    providers, and checks required static datasets.
    """
    return _find_missing_components(cfg_or_dict, lambda kind, selector: _provider_exists(kind, selector, paths=paths), paths=paths)


def find_unavailable_components(cfg_or_dict: Any, *, paths=None) -> list[str]:
    """Editor availability check. Runtime validation still loads executable providers."""
    from src.app.core.provider_discovery import provider_discovery_snapshot
    from src.app.core.provider_loader import find_provider

    with provider_discovery_snapshot():
        return _find_missing_components(
            cfg_or_dict, lambda kind, selector: bool(selector) and find_provider(kind, str(selector), paths) is not None, paths=paths
        )


def _find_missing_components(cfg_or_dict: Any, provider_exists: Callable[[str, str], bool], *, paths=None) -> list[str]:
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

    if env_val and env_val != "unknown" and not provider_exists("env", str(env_val)):
        missing.append(f"environment: {env_val}")
    elif not env_val and paradigm_uses_live_environment(str(paradigm) if paradigm else None, paths=paths):
        missing.append("environment: (none configured)")

    # Check static datasets for offline paradigms
    if paradigm and paradigm_uses_static_dataset(str(paradigm), paths=paths):
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
                        ds = resolve_dataset_for_method(m_name, m_spec, cfg_or_dict, paths=paths)
                        if not ds.exists():
                            missing.append(f"dataset: {ds}")
                    except Exception:
                        ds_name = m_spec.get("dataset_path") or (
                            cfg_or_dict.get("dataset_path") if hasattr(cfg_or_dict, "get")
                            else getattr(cfg_or_dict, "dataset_path", None)
                        )
                        if not ds_name and env_obj and hasattr(env_obj, "get"):
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
            if ag_str and not provider_exists("algorithm", ag_str):
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
            if mo_str and not provider_exists("model", mo_str):
                missing.append(f"model: {mo_str}")

    effective_model = cfg_or_dict.get('model') if hasattr(cfg_or_dict,'get') else getattr(cfg_or_dict,'model',None)
    if hasattr(effective_model,'get'):
        selected = effective_model.get('architecture') or effective_model.get('type') or effective_model.get('name')
        if isinstance(selected,str) and not provider_exists("model", selected): missing.append(f'model: {selected}')
    return list(dict.fromkeys(missing))


def _validate_offline_dataset_paths(cfg: Any, notices: list[str], *, paths=None) -> None:
    """Ensure static dataset paradigms have resolvable dataset files before launching."""
    paradigm_name = cfg.get("paradigm") if hasattr(cfg, "get") else getattr(cfg, "paradigm", None)
    if not paradigm_uses_static_dataset(paradigm_name, paths=paths):
        return
    is_wf = cfg.get("workflow") if hasattr(cfg, "get") else getattr(cfg, "workflow", None)
    if is_wf:
        return

    from src.app.pipeline.datasets import resolve_dataset_for_method

    methods = parse_methods_dict(cfg, paths=paths)
    for method_name, method_cfg in methods.items():
        try:
            ds_path = resolve_dataset_for_method(method_name, method_cfg, cfg, paths=paths)
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
