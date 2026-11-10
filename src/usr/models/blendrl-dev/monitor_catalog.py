"""Standalone config-to-metric catalog; deliberately has no ML dependencies."""
from itertools import combinations
import re


def expert_names(module_types):
    return [f'{re.sub(r"[^a-z0-9_]+", "_", str(kind).lower()).strip("_") or "module"}_{index}'
            for index, kind in enumerate(module_types)]


def module_types(config):
    modules = config.get("modules")
    symbolic = config.get("symbolic") or config.get("symbolic_actor")
    neural = config.get("neural") or config.get("neural_actor")
    if modules is None and (symbolic is not None or neural is not None):
        modules = []
        if symbolic is not None:
            kind = (symbolic.get("type") or symbolic.get("name") or symbolic.get("reasoner")) if isinstance(symbolic, dict) else None
            modules.append({"type": kind or config.get("reasoner", "nsfr")})
        if neural is not None:
            modules.append({"module_type": "neural"})
    if not modules:
        rules = config.get("rules", "default")
        rules = rules.split(",") if isinstance(rules, str) else rules
        return ["neural"] + ["logic"] * len(rules)
    types = [item.get("module_type", item.get("type", "neural")) for item in modules]
    return ["logic" if kind in ("nsfr", "neumann") else kind for kind in types]


def routing_output_names(types, *, agreement=True):
    names = ["entropy", "entropy_normalized", "mixed_share"]
    experts = expert_names(types)
    for expert in experts:
        names += [f"{expert}/{key}" for key in ("weight_mean", "weight_std", "dominant_share", "exclusive_share")]
    names += [f"{kind}/weight_mean" for kind in ("neural", "logic") if kind in types]
    if agreement:
        for first, second in combinations(experts, 2):
            names += [f"agreement/{first}__{second}/{key}" for key in ("top_action", "js_divergence")]
    return names


def describe(config, context):
    types = module_types(config)
    names = routing_output_names(types)
    names += [f"logic_{index}/coverage" for index, kind in enumerate(types) if kind == "logic"]
    if config.get("blend_function") == "gumbel_softmax" and config.get("actor_mode", "hybrid") == "hybrid":
        names += ["soft_gate/" + name for name in routing_output_names(types, agreement=False)]
    return [{"name": f"blend/{phase}/{name}", "label": name.replace("/", " · ").replace("_", " "),
             "group": f"Routing · {phase.title()}", "requires": ["model_observations.v1"]}
            for phase in ("train", "eval") for name in names]
