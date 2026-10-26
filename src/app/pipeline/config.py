from pathlib import Path
from typing import Any, cast

import yaml


def normalize_agent_name(agent_config: str) -> str:
    """Convert hierarchical agent config paths and dataset IDs to filesystem-safe and Hydra-safe names.
    e.g. 'offline_custom/human_variant' -> 'offline_custom_human_variant'
         'ex132(w)' -> 'ex132_w'
    This must match agent.name as set in the Hydra overrides."""
    return agent_config.replace("/", "_").replace("(", "_").replace(")", "").replace("__", "_").rstrip("_")


def parse_method_list(val):
    """Parse a method list from Hydra config.
    Hydra/YAML returns a Python list for `[a, b]` syntax but a string for `"a, b"` syntax.
    This function handles both forms."""
    if not val:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if hasattr(val, "__iter__") and not isinstance(val, str):
        return list(val)
    return [item.strip() for item in str(val).split(",") if item.strip()]


def _config_to_dict(cfg: Any) -> dict[str, Any]:
    """Resolve a DictConfig into a plain dict (OmegaConf types its result as a broad union)."""
    from omegaconf import OmegaConf

    return cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))


_RESERVED_METHOD_KEYS = {
    "params",
    "_params_",
    "defaults",
    "_defaults_",
    "shared",
    "_shared_",
    "common",
    "_common_",
    "common_params",
}


_KNOWN_ALGORITHMS: set[str] = set()
_KNOWN_MODELS: set[str] = set()


def get_known_algorithms() -> set[str]:
    """Return all known algorithm names, including any registered dynamically."""
    algos = set(_KNOWN_ALGORITHMS)
    try:
        from src.usr.algorithms.agent_registry import AGENT_REGISTRY

        algos.update(AGENT_REGISTRY.keys())
    except Exception:
        pass
    agent_dir = Path("in/config/agent")
    if agent_dir.is_dir():
        for p in agent_dir.glob("**/*.yaml"):
            algos.add(p.stem)
            try:
                rel = p.relative_to(agent_dir).with_suffix("").as_posix()
                algos.add(rel)
            except Exception:
                pass
    return algos


def get_known_models() -> set[str]:
    """Return all known model architecture names, including any registered dynamically."""
    models = set(_KNOWN_MODELS)
    try:
        from src.app.core.model_registry import MODEL_REGISTRY

        models.update(MODEL_REGISTRY.keys())
    except Exception:
        pass
    model_dir = Path("in/config/model")
    if model_dir.is_dir():
        for p in model_dir.glob("**/*.yaml"):
            models.add(p.stem)
            try:
                rel = p.relative_to(model_dir).with_suffix("").as_posix()
                models.add(rel)
            except Exception:
                pass
    return models


_MODEL_KEYS = {
    "architecture",
    "hidden_sizes",
    "activation",
}


def resolve_study_best_params(from_study: str, method_name: str, group: str | None = None) -> dict:
    """Resolve and load best_params.yaml produced by an Optuna study.

    Searches:
      1. Explicit path: from_study
      2. results/checkpoints/<from_study>/<normalized_agent>/best_params.yaml
      3. results/checkpoints/<from_study>/<method_name>/best_params.yaml
      4. results/checkpoints/<from_study>/best_params.yaml
      5. results/checkpoints/<group>/<from_study>/<normalized_agent>/best_params.yaml
    """
    clean_agent = normalize_agent_name(method_name)
    candidates = [
        Path(from_study),
        Path("results/checkpoints") / from_study / clean_agent / "best_params.yaml",
        Path("results/checkpoints") / from_study / method_name / "best_params.yaml",
        Path("results/checkpoints") / from_study / "best_params.yaml",
    ]
    if group and "/" not in from_study:
        candidates.append(Path("results/checkpoints") / group / from_study / clean_agent / "best_params.yaml")
        candidates.append(Path("results/checkpoints") / group / from_study / method_name / "best_params.yaml")
        candidates.append(Path("results/checkpoints") / group / from_study / "best_params.yaml")

    for cand in candidates:
        if cand.is_file() and cand.suffix in (".yaml", ".yml"):
            try:
                with open(cand, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
        elif cand.is_dir() and (cand / "best_params.yaml").is_file():
            try:
                with open(cand / "best_params.yaml", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception:
                pass
    return {}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    if not isinstance(base, dict):
        base = {}
    if not isinstance(override, dict):
        return base
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def find_group_method_config(method_name: str, group: str | None = None) -> dict:
    """Find and load a group-level method config YAML file.

    Location:
        in/config/experiment/<group>/methods/<method_name>.yaml
    """
    if not group or group == "ungrouped":
        return {}

    clean_name = normalize_agent_name(method_name)
    candidates = [
        Path("in/config/experiment") / group / "methods" / f"{clean_name}.yaml",
    ]
    if method_name != clean_name:
        candidates.append(Path("in/config/experiment") / group / "methods" / f"{method_name}.yaml")

    for path in candidates:
        if path.is_file():
            try:
                with open(path, encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, dict):
                        return data
            except Exception as e:
                import logging

                logging.getLogger("theta.pipeline").warning("Failed to load method YAML %s: %s", path, e)
    return {}


def _extract_name_and_subparams(
    val,
    name_keys=("name", "algorithm", "type", "algo", "architecture", "base"),
    known_names=None,
):
    """Extract string identifier and subparameters from a string or dict value.

    Supports:
      1. String value: 'package/algorithm' -> ('package/algorithm', {})
      2. Dict with explicit name key: {'name': 'package/algorithm', 'lr': 3e-4} -> ('package/algorithm', {'lr': 3e-4})
      3. Dict with algorithm/model as key: {'my_model': {...}} -> ('my_model', {...})
      4. Nested module dict: {'custom_module': {'setting': 0.03}} -> ('custom_module', {...})
    """
    if isinstance(val, dict):
        for nk in name_keys:
            if nk in val and val[nk] is not None:
                ident = str(val[nk])
                subparams = {k: v for k, v in val.items() if k not in name_keys}
                return ident, subparams

        all_known = get_known_algorithms() | get_known_models()
        if known_names:
            all_known.update(known_names)

        matching_keys = [k for k in val if k in all_known]
        if len(matching_keys) == 1:
            k = matching_keys[0]
            sub = val[k]
            subparams = dict(sub) if isinstance(sub, dict) else ({} if sub is None else {"value": sub})
            for sib_k, sib_v in val.items():
                if sib_k != k:
                    subparams[sib_k] = sib_v
            return k, subparams

        if len(val) == 1:
            k = next(iter(val))
            sub = val[k]
            if isinstance(sub, dict):
                return k, dict(sub)
            elif sub is None:
                return k, {}

        return None, dict(val)

    elif isinstance(val, str):
        return val, {}

    return None, {}


def load_model_base_config(model_name: str, config_dir=None) -> dict:
    """Load the base YAML configuration for a model architecture from in/config/model/<model_name>.yaml
    or from installed component manifests in src/usr/models/<model_name>.
    """
    if not model_name:
        return {}
    import json
    from pathlib import Path

    import yaml

    if config_dir is None:
        base_dir = Path("in/config/model")
        if not base_dir.exists():
            base_dir = Path(__file__).resolve().parents[3] / "in" / "config" / "model"
        config_dir = base_dir

    target = Path(config_dir) / f"{model_name}.yaml"
    if not target.exists():
        found = list(Path(config_dir).rglob(f"{model_name}.yaml"))
        if found:
            target = found[0]

    if target.exists():
        try:
            with open(target) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return dict(data)
        except Exception:
            pass

    # Dynamic check: look in src/usr/models/ for component manifests or configs
    workspace = Path(__file__).resolve().parents[3]
    candidate_dirs = [
        Path("src/usr/models") / model_name,
        workspace / "src" / "usr" / "models" / model_name,
    ]
    for cdir in candidate_dirs:
        if cdir.is_dir():
            for meta_name in [".theta_component.json", "plugin.json"]:
                meta_file = cdir / meta_name
                if meta_file.is_file():
                    try:
                        meta = json.loads(meta_file.read_text(encoding="utf-8"))
                        cfg_rel = meta.get("config_path")
                        if cfg_rel:
                            cfg_path = (workspace / cfg_rel) if not Path(cfg_rel).is_absolute() else Path(cfg_rel)
                            if cfg_path.is_file():
                                with open(cfg_path) as f:
                                    data = yaml.safe_load(f)
                                if isinstance(data, dict):
                                    return dict(data)
                        if "parameters" in meta and isinstance(meta["parameters"], dict):
                            return dict(meta["parameters"])
                    except Exception:
                        pass

    return {}


def method_is_enabled(method_cfg: Any) -> bool:
    """Whether a configured method participates in execution and preflight checks."""
    try:
        return method_cfg.get("enabled", True) is not False
    except Exception:
        return True


def parse_methods_dict(cfg) -> dict[str, dict]:
    """Parse and normalize the structured methods: dict from Hydra config.

    Supports hierarchical parameter specification:
      1. Tier 2: Universal Experiment Method Parameters:
         - Top-level shared scalars/dicts in `methods.params` (e.g. epochs, gamma).
         - Hierarchically grouped agent-specific overrides (`params.agent.<component>`).
         - Hierarchically grouped model-specific overrides (`params.model.<component>`).
         - Direct algorithm/architecture blocks (`params.<component>`).
      2. Tier 3: Method-Level Declarations & Specific Overrides:
         - Declared inside `methods.<method_name>`.
         - Individual method configurations override shared parameters.
      3. Nested model parameters:
         - Preserved as declared for the model component to interpret or compose.

    Returns:
        dict[str, dict]: Normalized method_name -> method_config mapping.
    """
    from omegaconf import DictConfig, OmegaConf

    raw_methods = getattr(cfg, "methods", None) if not isinstance(cfg, dict) else cfg.get("methods")
    if raw_methods is None and hasattr(cfg, "get"):
        raw_methods = cfg.get("methods", None)

    if not raw_methods:
        return {}

    if isinstance(raw_methods, DictConfig):
        raw_methods_dict = _config_to_dict(raw_methods)
    elif hasattr(raw_methods, "items"):
        raw_methods_dict = dict(raw_methods)
    else:
        return {}

    # 1. Extract shared params from top-level config (e.g. cfg.params)
    top_params: dict[str, Any] = {}
    for top_key in ("params", "shared_params", "common_params"):
        val = getattr(cfg, top_key, None) if not isinstance(cfg, dict) else cfg.get(top_key)
        if val is None and hasattr(cfg, "get"):
            val = cfg.get(top_key, None)
        if val:
            if isinstance(val, DictConfig):
                val_dict = _config_to_dict(val)
            elif hasattr(val, "items"):
                val_dict = dict(val)
            else:
                val_dict = {}
            top_params = deep_merge(top_params, val_dict)

    # 2. Extract shared params from within methods dict (e.g. methods.params)
    method_level_params: dict[str, Any] = {}
    for res_key in _RESERVED_METHOD_KEYS:
        if res_key in raw_methods_dict:
            res_val = raw_methods_dict[res_key]
            if isinstance(res_val, DictConfig):
                res_dict = _config_to_dict(res_val)
            elif isinstance(res_val, dict):
                res_dict = dict(res_val)
            else:
                res_dict = {}
            method_level_params = deep_merge(method_level_params, res_dict)

    shared_params = deep_merge(top_params, method_level_params)

    # Parse shared agent and model blocks
    shared_agent = shared_params.get("agent", {})
    shared_model = shared_params.get("model", {})

    # Extract shared tuning search space
    shared_tune: dict[str, Any] = {}
    for tune_key in ("tune", "search_space"):
        if tune_key in shared_params and isinstance(shared_params[tune_key], dict):
            shared_tune = deep_merge(shared_tune, shared_params[tune_key])

    # Extract universal global parameters (excluding agent, model, tune, and standalone algo/arch blocks)
    universal_global = {}
    known_algos = get_known_algorithms()
    known_models = get_known_models()
    for k, v in shared_params.items():
        if k in ("agent", "model", "tune", "search_space"):
            continue
        if k in known_algos or k in known_models:
            continue
        universal_global[k] = v

    # Extract default agent and model from shared params if present
    default_agent_algo, default_agent_sub = None, {}
    if isinstance(shared_agent, str):
        default_agent_algo = shared_agent
    elif isinstance(shared_agent, dict):
        if any(nk in shared_agent for nk in ("name", "algorithm", "type", "algo")):
            default_agent_algo, default_agent_sub = _extract_name_and_subparams(shared_agent)
        else:
            known_in_agent = [k for k in shared_agent if k in known_algos]
            if len(known_in_agent) == 1:
                default_agent_algo = known_in_agent[0]

    default_model_arch, default_model_sub = None, {}
    if isinstance(shared_model, str):
        default_model_arch = shared_model
    elif isinstance(shared_model, dict):
        if any(nk in shared_model for nk in ("name", "architecture", "type", "base")):
            default_model_arch, default_model_sub = _extract_name_and_subparams(shared_model)

    # Infer group for group-level method loading
    group = getattr(cfg, "group", None) if not isinstance(cfg, dict) else cfg.get("group")
    if not group or group == "ungrouped":
        exp_name = getattr(cfg, "experiment_name", None) if not isinstance(cfg, dict) else cfg.get("experiment_name")
        if exp_name and "/" in str(exp_name):
            group = str(exp_name).split("/")[0]
        else:
            exp_id = getattr(cfg, "experiment_id", None) if not isinstance(cfg, dict) else cfg.get("experiment_id")
            if exp_id and "/" in str(exp_id):
                group = str(exp_id).split("/")[0]
            else:
                env_cfg = getattr(cfg, "env", None) if not isinstance(cfg, dict) else cfg.get("env")
                if isinstance(env_cfg, dict) and env_cfg.get("name"):
                    group = env_cfg.get("name")
                elif hasattr(env_cfg, "name"):
                    group = getattr(env_cfg, "name")

    # 3. Build resolved method configurations
    result = {}
    for method_name, method_cfg in raw_methods_dict.items():
        if method_name in _RESERVED_METHOD_KEYS:
            continue

        if isinstance(method_cfg, DictConfig):
            m_dict = _config_to_dict(method_cfg)
        elif isinstance(method_cfg, dict):
            m_dict = dict(method_cfg)
        else:
            m_dict = {}

        # Look up reusable group method definition if available (supporting explicit base/template or method_name)
        base_name = m_dict.get("base") or m_dict.get("template") or method_name
        base_group_method = find_group_method_config(base_name, group=group)
        if base_group_method:
            m_dict = deep_merge(base_group_method, m_dict)

        if not method_is_enabled(m_dict):
            continue
        # `enabled` is IDE metadata and must not be forwarded as a training parameter.
        m_dict.pop("enabled", None)

        # Resolve method's agent algo and subparams
        raw_m_agent = m_dict.get("agent", default_agent_algo)
        m_agent_algo, m_agent_sub = _extract_name_and_subparams(raw_m_agent)
        if not m_agent_algo and default_agent_algo:
            m_agent_algo = default_agent_algo

        # Resolve method's model arch and subparams
        raw_m_model = m_dict.get("model", default_model_arch)
        m_model_arch, m_model_sub = _extract_name_and_subparams(raw_m_model)
        if not m_model_arch and default_model_arch:
            m_model_arch = default_model_arch

        # Tier 2: Universal global parameters
        resolved_mcfg = dict(universal_global)

        # Tier 2: Algorithm-specific overrides from params.agent.<algo> or params.<algo>
        agent_params = dict(default_agent_sub)
        if m_agent_algo:
            if (
                isinstance(shared_agent, dict)
                and m_agent_algo in shared_agent
                and isinstance(shared_agent[m_agent_algo], dict)
            ):
                agent_params = deep_merge(agent_params, shared_agent[m_agent_algo])
            if m_agent_algo in shared_params and isinstance(shared_params[m_agent_algo], dict):
                agent_params = deep_merge(agent_params, shared_params[m_agent_algo])

        # Tier 2: Model-specific overrides from params.model.<arch> or params.<arch>
        model_params = dict(default_model_sub)
        if m_model_arch:
            if (
                isinstance(shared_model, dict)
                and m_model_arch in shared_model
                and isinstance(shared_model[m_model_arch], dict)
            ):
                model_params = deep_merge(model_params, shared_model[m_model_arch])
            if m_model_arch in shared_params and isinstance(shared_params[m_model_arch], dict):
                model_params = deep_merge(model_params, shared_params[m_model_arch])

        # Tier 3: Apply method-level overrides
        agent_params = deep_merge(agent_params, m_agent_sub)
        model_params = deep_merge(model_params, m_model_sub)

        # Support inheriting tuned parameters from a previous Optuna sweep via from_study
        study_ref = m_dict.get("from_study")
        if study_ref:
            resolved_mcfg["from_study"] = study_ref
            tuned_params = resolve_study_best_params(study_ref, method_name, group=group)
            if tuned_params:
                print(f"  [Loaded Tuned Params] {method_name} from study '{study_ref}': {tuned_params}")
                for pk, pv in tuned_params.items():
                    if pk.startswith("agent."):
                        agent_params[pk[len("agent.") :]] = pv
                    elif pk.startswith("model."):
                        model_params[pk[len("model.") :]] = pv
                    elif pk in _MODEL_KEYS:
                        model_params[pk] = pv
                    else:
                        agent_params[pk] = pv
            else:
                import logging

                logging.getLogger("theta.pipeline").warning(
                    "Method '%s' specified from_study='%s', but best_params.yaml was not found.",
                    method_name,
                    study_ref,
                )

        # Support method-level tuning search spaces (Tier 3) merged with universal (Tier 2)
        raw_tune = m_dict.get("tune") or m_dict.get("search_space") or {}
        combined_tune = (
            deep_merge(shared_tune, raw_tune)
            if isinstance(raw_tune, dict)
            else (dict(shared_tune) if shared_tune else {})
        )
        if combined_tune:
            resolved_mcfg["tune"] = combined_tune

        if not isinstance(m_dict, dict):
            m_dict = {}

        # Copy non-agent, non-model keys from m_dict (deep-merging if both are dicts)
        for k, v in m_dict.items():
            if k in ("agent", "model", "tune", "search_space", "from_study"):
                continue
            if k in agent_params:
                agent_params[k] = (
                    deep_merge(agent_params[k], v) if isinstance(agent_params[k], dict) and isinstance(v, dict) else v
                )
            if k in model_params:
                model_params[k] = (
                    deep_merge(model_params[k], v) if isinstance(model_params[k], dict) and isinstance(v, dict) else v
                )

            if k in resolved_mcfg and isinstance(resolved_mcfg[k], dict) and isinstance(v, dict):
                resolved_mcfg[k] = deep_merge(resolved_mcfg[k], v)
            else:
                resolved_mcfg[k] = v

        # Set resolved agent
        if isinstance(raw_m_agent, dict):
            resolved_agent_dict = dict(agent_params)
            resolved_agent_dict["name"] = m_agent_algo
            resolved_mcfg["agent"] = resolved_agent_dict
        else:
            resolved_mcfg["agent"] = m_agent_algo
            for ak, av in agent_params.items():
                if ak not in resolved_mcfg:
                    resolved_mcfg[ak] = av
        if agent_params:
            resolved_mcfg["agent_params"] = agent_params

        # Set resolved model
        if isinstance(raw_m_model, dict):
            resolved_model_dict = dict(model_params)
            resolved_model_dict["name"] = m_model_arch
            resolved_mcfg["model"] = resolved_model_dict
        else:
            resolved_mcfg["model"] = m_model_arch
            for mk, mv in model_params.items():
                if mk in resolved_mcfg and isinstance(resolved_mcfg[mk], dict) and isinstance(mv, dict):
                    resolved_mcfg[mk] = deep_merge(resolved_mcfg[mk], mv)
                else:
                    resolved_mcfg[mk] = mv
        if model_params:
            resolved_mcfg["model_params"] = model_params

        result[str(method_name)] = resolved_mcfg

    return result


def resolve_experiment_config_name(exp_input: str) -> str:
    """Resolve an experiment name (e.g. 'group_run' or 'group/run')
    to its relative Hydra config path inside in/config/experiment/."""
    from pathlib import Path

    exp_dir = Path("in/config/experiment")
    if not exp_dir.exists():
        return exp_input

    clean_input = exp_input[:-5] if exp_input.endswith(".yaml") else exp_input
    direct_path = exp_dir / f"{clean_input}.yaml"
    if direct_path.exists():
        return clean_input

    # Search recursively in group subdirectories
    matches = list(exp_dir.glob(f"**/{clean_input}.yaml"))
    if not matches:
        raise ValueError(f"Experiment config '{clean_input}.yaml' not found in {exp_dir}")
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous experiment name '{clean_input}'. Found multiple matches: {[str(m) for m in matches]}. Please specify the exact group/experiment path."
        )

    rel = matches[0].relative_to(exp_dir)
    return str(rel.with_suffix(""))
