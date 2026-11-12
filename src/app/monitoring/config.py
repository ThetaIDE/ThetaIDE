"""Headless monitor instances and compatibility; never imports ML providers."""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping


INSTANCE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
FIELDS = {"component", "target", "enabled", "interval_updates", "max_snapshots", "failure_policy", "probes", "params"}
PROBE_FIELDS = {"path", "count", "steps", "seed"}


def plain(value):
    from omegaconf import OmegaConf
    if OmegaConf.is_dict(value) and "hydra" in value:
        value = OmegaConf.masked_copy(value, [key for key in value if key != "hydra"])
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def instances(cfg, *, defaults=None):
    """Named instances; explicit entries override model-owned defaults by identity."""
    from src.app.pipeline.config import deep_merge
    explicit = plain({} if cfg.get("monitors") is None else cfg.get("monitors"))
    if not isinstance(explicit, dict):
        raise ValueError("monitors must be a mapping of instance names to configurations")
    defaults = plain({} if defaults is None else defaults)
    if not isinstance(defaults, dict):
        raise ValueError("Model monitor defaults must be a mapping")
    merged = deep_merge(defaults, explicit)
    result = {}
    for identity, value in merged.items():
        if not isinstance(identity, str) or not INSTANCE.fullmatch(identity):
            raise ValueError(f"Invalid monitor instance name: {identity!r}")
        if not isinstance(value, dict) or set(value) - FIELDS:
            raise ValueError(f"Monitor {identity}: expected only {sorted(FIELDS)}")
        spec = {"enabled": True, "target": "", "interval_updates": 10, "max_snapshots": 32,
                "failure_policy": "raise", "probes": {}, "params": {}, **value}
        if not isinstance(spec["enabled"], bool):
            raise ValueError(f"Monitor {identity}: enabled must be boolean")
        if not spec["enabled"]:
            continue
        if not isinstance(spec.get("component"), str) or not spec["component"]:
            raise ValueError(f"Monitor {identity}: component is required")
        if not isinstance(spec["target"], str):
            raise ValueError(f"Monitor {identity}: target must be a module path")
        for key in ("interval_updates", "max_snapshots"):
            if type(spec[key]) is not int or not 1 <= spec[key] <= 100000:
                raise ValueError(f"Monitor {identity}: {key} must be a positive integer <= 100000")
        if spec["failure_policy"] not in ("raise", "disable"):
            raise ValueError(f"Monitor {identity}: failure_policy must be raise or disable")
        if not isinstance(spec["params"], dict) or not isinstance(spec["probes"], dict):
            raise ValueError(f"Monitor {identity}: params and probes must be mappings")
        probes = {"count": 128, "steps": 512, "seed": 0, **spec["probes"]}
        if set(probes) - PROBE_FIELDS:
            raise ValueError(f"Monitor {identity}: unknown probe settings")
        for key, minimum, maximum in (("count", 4, 4096), ("steps", 4, 100000), ("seed", 0, 2**32-1)):
            if type(probes[key]) is not int or not minimum <= probes[key] <= maximum:
                raise ValueError(f"Monitor {identity}: probes.{key} must be an integer in [{minimum}, {maximum}]")
        if probes.get("path") is not None and not isinstance(probes["path"], str):
            raise ValueError(f"Monitor {identity}: probes.path must name an NPZ probe artifact")
        spec["probes"] = probes
        result[identity] = spec
    return result


def selected_model(cfg, catalog):
    value = cfg.get("model") or {}
    selector = value if isinstance(value, str) else next((value.get(k) for k in ("name", "architecture", "type") if value.get(k)), None)
    if not selector:
        return None
    selected = catalog.resolve("model", str(selector))
    if selected is None:
        import yaml
        path = catalog.paths.resolve_config(f"model/{selector}.yaml")
        profile = yaml.safe_load(path.read_text()) if path.is_file() else {}
        selected = catalog.resolve("model", str((profile or {}).get("name", selector)))
    return selected


def model_defaults(cfg, catalog):
    """Declarative defaults mirror optional model.get_monitor_specs() at runtime."""
    selected = selected_model(cfg, catalog)
    return plain(selected[1].metadata.get("monitor_defaults", {})) if selected else {}


def model_capabilities(cfg, catalog):
    selected = selected_model(cfg, catalog)
    return set(selected[1].metadata.get("monitoring_access", [])) if selected else set()


def runner_metadata(context, catalog, script=None):
    from src.app.core.execution import algorithm_selection
    _, selected = algorithm_selection(context, paths=catalog.paths)
    from src.app.telemetry.monitors import BUILTIN_MONITORING
    if selected:
        default = BUILTIN_MONITORING if selected[1].metadata.get("backend") == "lightning" else {}
        return selected[1].metadata.get("monitoring", default)
    return BUILTIN_MONITORING


def validate_context(context, *, catalog, script=None, configured=None):
    if configured is None:
        configured = instances(context, defaults=model_defaults(context, catalog))
    if not configured:
        return
    supported = set(runner_metadata(context, catalog, script).get("supports", [])) | model_capabilities(context, catalog)
    for identity, spec in configured.items():
        selected = catalog.resolve("monitor", spec["component"])
        if selected is None:
            raise ValueError(f"Monitor {identity}: component {spec['component']!r} is not installed")
        metadata = selected[1].metadata
        required = set(metadata.get("requires", ["probe_forward.v1"]))
        if missing := required - supported:
            raise ValueError(f"Monitor {identity}: runner/model lacks required hooks: {', '.join(sorted(missing))}")
        # Parameter schemas are also checked by the component constructor. Headless
        # validation uses JSON Schema without importing its implementation.
        schema = metadata.get("parameter_schema", selected[0].manifest.get("parameter_schema", {}))
        if schema:
            import jsonschema
            try:
                jsonschema.validate(spec["params"], schema)
            except jsonschema.ValidationError as exc:
                raise ValueError(f"Monitor {identity}: {exc.message}") from exc


def validate_monitors(cfg):
    from src.app.core.provider_catalog import get_catalog
    from src.app.pipeline.config import parse_methods_dict, deep_merge
    catalog = get_catalog()
    cfg = plain(cfg)
    methods = parse_methods_dict(cfg, paths=catalog.paths) if "methods" in cfg else {"run": {}}
    for identity, method in methods.items():
        try:
            validate_context(deep_merge(cfg, method), catalog=catalog)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Method {identity}: {exc}") from exc


def schema_for(path, cfg, catalog):
    """Use ordinary shared parameter controls for monitor component instances."""
    selectors = sorted({cap.id for package in catalog.packages for cap in package.capabilities if cap.kind == "monitor"})
    if len(path) == 1:
        return {"type": "object", "properties": {
            "activations": {"type": "object", "default": {"component": "activation-evolution", "target": "network", "enabled": True}},
            "network": {"type": "object", "default": {"component": "network-structure", "target": "network", "enabled": True}},
            "circuit": {"type": "object", "default": {"component": "causal-circuit", "target": "network", "enabled": True}}}}
    spec = (cfg.get("monitors") or {}).get(path[1]) or {}
    selected = catalog.resolve("monitor", spec.get("component", "")) if spec.get("component") else None
    parameters = selected[1].metadata.get("parameter_schema", selected[0].manifest.get("parameter_schema", {})) if selected else {}
    schema = {"type": "object", "required": ["component"], "properties": {
        "component": {"type": "string", "enum": selectors},
        "target": {"type": "string", "default": ""}, "enabled": {"type": "boolean", "default": True},
        "interval_updates": {"type": "integer", "minimum": 1, "default": 10},
        "max_snapshots": {"type": "integer", "minimum": 1, "default": 32},
        "failure_policy": {"type": "string", "enum": ["raise", "disable"], "default": "raise"},
        "probes": {"type": "object", "default": {}, "properties": {
            "path": {"type": "string"}, "count": {"type": "integer", "minimum": 4, "maximum": 4096, "default": 128},
            "steps": {"type": "integer", "minimum": 4, "default": 512},
            "seed": {"type": "integer", "minimum": 0, "default": 0}}},
        "params": {**parameters, "default": {}}}}
    for key in path[2:]:
        schema = schema.get("properties", {}).get(key, {})
    return schema
