"""Headless monitor discovery and live-stream selection.

Components advertise ``monitoring.provides`` in capability metadata. Each output
has a metric ``name`` (an exact name or a prefix ending in /), a label, and
``requires`` runner hooks. Runners advertise ``monitoring.supports``. Discovery
reads declarations and optional standalone catalogs, never model/runner modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable

from .selection import ALWAYS_KEPT


@dataclass(frozen=True)
class MonitorSpec:
    name: str
    label: str
    source: str
    description: str = ""
    group: str = ""
    kind: str = "scalar_series"
    artifact_schema: str = ""
    axes: tuple[str, ...] = ()
    value_semantics: str = ""
    instance: str = ""


def _matches(pattern: str, name: str) -> bool:
    return pattern == "*" or (name.startswith(pattern) if pattern.endswith("/") else name == pattern)


def monitor_errors(cfg: Any) -> list[str]:
    telemetry = cfg.get("telemetry") if hasattr(cfg, "get") else None
    value = telemetry.get("monitors") if hasattr(telemetry, "get") else None
    if value is None:
        return []
    if not hasattr(value, "get") or set(value) - {"disabled", "enabled", "removed"}:
        return ["'telemetry.monitors' must be a mapping with optional 'enabled', 'disabled' and 'removed' settings."]
    if not isinstance(value.get("enabled", True), bool):
        return ["'telemetry.monitors.enabled' must be a boolean."]
    for field in ("disabled", "removed"):
        patterns = value.get(field, [])
        if isinstance(patterns, str) or not isinstance(patterns, Sequence) or not all(
            isinstance(name, str) and name.strip() == name and name and ("*" not in name or name == "*")
            for name in patterns
        ):
            return [f"'telemetry.monitors.{field}' must be a list of metric names or prefixes ending in '/' ('*' disables all)."]
    return []


class MonitorSelection:
    """An exclusion list: absent/empty streams everything, '*' streams no metrics.

    Run lifecycle, errors, artifacts and configuration are always delivered.
    Unknown exclusions survive component changes and take effect if restored.
    """
    def __init__(self, disabled=()):
        self.disabled = tuple(disabled)

    def keeps(self, name: str) -> bool:
        return not self.disabled or not any(_matches(pattern, name) for pattern in self.disabled)

    def with_toggled(self, names, enabled: bool, *, known=()) -> MonitorSelection:
        """Change a metric/group atomically, preserving other declared choices.

        Restoring children covered by an older namespace or '*' exclusion
        expands that exclusion into individual choices for unaffected outputs.
        """
        names = tuple(dict.fromkeys(names))
        disabled = list(self.disabled)
        if not enabled:
            return MonitorSelection([*disabled, *(name for name in names if name not in disabled)])
        disabled = [pattern for pattern in disabled if not any(
            _matches(pattern, name) or (name.endswith("/") and pattern.startswith(name)) for name in names)]
        restored = MonitorSelection(disabled)
        for name in known:
            if not any(_matches(target, name) for target in names) and not self.keeps(name) and restored.keeps(name):
                disabled.append(name)
        return MonitorSelection(disabled)

    def filter(self, values: dict, *, namespace: str = "") -> dict:
        return {name: value for name, value in values.items()
                if name in ALWAYS_KEPT or self.keeps(namespace + str(name))}

    def filter_record(self, rec: dict) -> dict | None:
        """Filter live records, including summaries and bare system sample keys."""
        if not self.disabled:
            return rec
        kind = rec.get("type")
        if kind in ("metrics", "system", "finish"):
            field = "summary" if kind == "finish" else "data"
            values = self.filter(rec.get(field) or {}, namespace="system/" if kind == "system" else "")
            if kind != "finish" and not (set(values) - ALWAYS_KEPT):
                return None
            return {**rec, field: values}
        return rec


def monitors_from_config(cfg: Any) -> MonitorSelection:
    errors = monitor_errors(cfg)
    if errors:
        raise ValueError(errors[0])
    telemetry = cfg.get("telemetry") if hasattr(cfg, "get") else None
    value = telemetry.get("monitors") if hasattr(telemetry, "get") else None
    return MonitorSelection([*value.get("disabled", []), *value.get("removed", [])]
                            if value is not None and value.get("enabled", True) else ())


def validate_monitoring_metadata(value: Any) -> None:
    """Validate the public manifest contract at installation/discovery time."""
    if not isinstance(value, dict) or set(value) - {"supports", "provides", "catalog"}:
        raise ValueError("monitoring must contain only supports, provides and catalog")
    if "catalog" in value:
        from pathlib import PurePosixPath
        module, separator, function = str(value["catalog"]).partition(":")
        path = PurePosixPath(module)
        if not separator or not function.isidentifier() or path.is_absolute() or ".." in path.parts or path.suffix != ".py":
            raise ValueError("monitoring.catalog must be a package-relative Python file:function")
    supports = value.get("supports", [])
    if not isinstance(supports, list) or not all(isinstance(v, str) and v.strip() for v in supports):
        raise ValueError("monitoring.supports must be a list of runner hook names")
    provides = value.get("provides", [])
    if not isinstance(provides, list):
        raise ValueError("monitoring.provides must be a list of monitor declarations")
    names = set()
    for item in provides:
        if not isinstance(item, dict):
            raise ValueError("Each monitor declaration must be a mapping")
        name, requires = item.get("name"), item.get("requires", ["scalars.v1"])
        if not isinstance(name, str) or not name.strip() or name != name.strip() or "*" in name or name in ALWAYS_KEPT:
            raise ValueError("Monitor names must be metric names or prefixes ending in '/' (not counters or wildcards)")
        if name in names:
            raise ValueError(f"Duplicate monitor name: {name}")
        names.add(name)
        from .visuals import validate_visual
        validate_visual(item)
        if not isinstance(requires, list) or not requires or not all(isinstance(v, str) and v.strip() for v in requires):
            raise ValueError("Monitor requires must be a nonempty list of runner hook names")
        for field in ("label", "description", "group"):
            if field in item and not isinstance(item[field], str):
                raise ValueError(f"Monitor {field} must be a string")


BUILTIN_MONITORING = {
    "supports": ["scalars.v1", "system.v1", "lightning_callbacks.v1"],
    # Training scalars belong to the selected model/algorithm, not the generic
    # Lightning transport. Do not invent namespace entries for unknown outputs.
    "provides": [],
}


def system_monitors(cfg=None) -> list[MonitorSpec]:
    """Individual shared resource fields, without probing or importing torch."""
    import importlib.util
    import os
    import sys
    fields = [("cpu_percent", "Process CPU (%)")]
    if importlib.util.find_spec("psutil") is not None:
        fields += [("rss_mb", "Process memory (MB)"), ("threads", "Threads"), ("children", "Child processes"),
                   ("host_cpu_percent", "Host CPU (%)"), ("host_mem_percent", "Host memory (%)")]
    else:
        if sys.platform == "linux":
            fields.append(("rss_mb", "Process memory (MB)"))
        if hasattr(os, "getloadavg"):
            fields.append(("host_load_1m", "Host load (1 minute)"))
    trainer = (cfg or {}).get("trainer") or {}
    if trainer.get("accelerator") in ("gpu", "cuda"):
        devices = trainer.get("devices", 1)
        ids = devices if isinstance(devices, list) else range(devices if isinstance(devices, int) and devices > 0 else 1)
        for device in ids:
            fields += [(f"gpu{device}_mem_allocated_mb", f"GPU {device} allocated memory (MB)"),
                       (f"gpu{device}_mem_reserved_mb", f"GPU {device} reserved memory (MB)")]
    return [MonitorSpec("system/" + name, title, "System resources") for name, title in fields]


def validate_monitor_catalog_source(path) -> None:
    """Catalogs are standalone, stdlib-only config descriptions, never models."""
    import ast
    import sys
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [item.name for item in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Monitor catalogs must be standalone (no package-relative imports)")
            modules = [node.module or ""]
        else:
            continue
        if any(module.split(".")[0] not in sys.stdlib_module_names for module in modules):
            raise ValueError("Monitor catalogs may import only the Python standard library")


_CATALOG_MODULES = {}


def _catalog_outputs(package, entry, config, context):
    """Load only the declared lightweight file, bypassing package __init__."""
    import runpy
    import copy
    module, _, function = entry.partition(":")
    path = package.resource(module)
    stamp = (str(path), path.stat().st_mtime_ns, path.stat().st_size)
    if stamp not in _CATALOG_MODULES:
        validate_monitor_catalog_source(path)
        # run_path does not execute the component's package initializer.
        _CATALOG_MODULES[stamp] = runpy.run_path(str(path))
    outputs = _CATALOG_MODULES[stamp][function](copy.deepcopy(config), copy.deepcopy(context))
    validate_monitoring_metadata({"provides": outputs})
    return outputs


def available_monitors(cfg: dict, *, catalog=None, script: str | None = None,
                       read_config: Callable[[str, str], dict] | None = None,
                       include_shared: bool = True, errors: list[str] | None = None) -> list[MonitorSpec]:
    """Union of selected components' outputs, intersected with each method's runner hooks.

    Uses the pipeline's method inheritance and runner dispatch. A component may
    use a capability selector, alias, or config profile; profiles are resolved
    through the same workspace paths as execution. Duplicate providers fail
    visibly rather than guessing ownership.
    UI callers may collect standalone errors without losing component outputs;
    execution and validation remain strict when errors is omitted.
    """
    from src.app.core.provider_catalog import get_catalog
    from src.app.pipeline.config import deep_merge, parse_methods_dict
    from src.app.pipeline.engine import DEFAULT_ENGINE_SCRIPT, resolve_engine
    from omegaconf import OmegaConf
    import yaml

    catalog = catalog or get_catalog()
    paths = catalog.paths

    def load_config(kind, selector):
        if read_config is not None:
            return read_config(kind, selector)
        path = paths.resolve_config(f"{kind}/{selector}.yaml")
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}

    def component(kind, value):
        if isinstance(value, str):
            selector, params = value, {}
        elif isinstance(value, dict):
            selector = next((value[key] for key in ("name", "type", "architecture", "algorithm", "component", "class", "base", "algo")
                             if isinstance(value.get(key), str)), None)
            params = value
        else:
            return {}, None
        if not selector:
            return params, None
        profile = load_config({"algorithm": "agent", "datamodule": "data", "evaluator": "eval"}.get(kind, kind), selector)
        effective = deep_merge(profile, params)
        resolved = catalog.resolve(kind, selector)
        if resolved is None:
            for key in ("name", "type", "architecture", "algorithm"):
                identity = effective.get(key)
                if isinstance(identity, str):
                    resolved = catalog.resolve(kind, identity)
                    if resolved:
                        break
        return effective, resolved

    specs = {}

    def add(metadata, source, supported, *, package=None, config=None, context=None):
        outputs = list(metadata.get("provides", []))
        if metadata.get("catalog") and package is not None:
            outputs += _catalog_outputs(package, metadata["catalog"], config or {}, context or {})
        for item in outputs:
            if set(item.get("requires", ["scalars.v1"])) <= supported:
                name = item["name"]
                specs.setdefault(name, MonitorSpec(name, item.get("label", name), source,
                                                   item.get("description", ""), item.get("group", ""),
                                                   item.get("kind", "scalar_series"), item.get("artifact_schema", ""),
                                                   tuple(item.get("axes", [])), item.get("value_semantics", ""),
                                                   (config or {}).get("instance", "")))

    methods = parse_methods_dict(cfg, paths=paths) if cfg.get("methods") else {}
    # An explicitly empty experiment has no runner yet; a runtime cfg has no methods.
    contexts = list(methods.values()) if "methods" in cfg else [{}]
    for method in contexts:
        context = deep_merge(cfg, method)
        agent, selected_agent = component("algorithm", context.get("agent"))
        context["agent"] = agent
        model, _ = component("model", context.get("model"))
        context["model"] = deep_merge(model, context.get("model_params") or {})
        runner, _ = resolve_engine(method, OmegaConf.create(context))
        runner = script or runner
        metadata = BUILTIN_MONITORING if runner == DEFAULT_ENGINE_SCRIPT else {}
        runner_package = None
        if runner != DEFAULT_ENGINE_SCRIPT:
            from src.app.core.provider_loader import resolve_resource
            try:
                source = resolve_resource(runner, paths=paths).resolve()
            except (ValueError, FileNotFoundError):
                source = None
            candidates = [selected_agent] if selected_agent else []
            candidates += [(p, c) for p in catalog.packages if p.origin != "python_distribution"
                           for c in p.capabilities if c.kind == "algorithm" and (p, c) not in candidates]
            for package, cap in candidates:
                if source is not None and package.resource(cap.entry_point.split(":", 1)[0]).resolve() == source:
                    metadata = cap.metadata.get("monitoring", {})
                    runner_package = package
                    break
        supported = set(metadata.get("supports", []))
        add(metadata, "Runner", supported, package=runner_package, config=agent, context=context)
        if include_shared and runner == DEFAULT_ENGINE_SCRIPT and (context.get("trainer") or {}).get("accelerator") in ("gpu", "cuda"):
            for key, title in (("peak_alloc_gb", "Peak GPU allocated memory (GB)"),
                               ("peak_reserved_gb", "Peak GPU reserved memory (GB)")):
                specs.setdefault("gpu/" + key, MonitorSpec("gpu/" + key, title, "Runner"))
        if include_shared and "system.v1" in supported:
            for spec in system_monitors(context):
                specs.setdefault(spec.name, spec)

        visited = set()

        def visit(kind, value):
            effective, resolved = component(kind, value)
            if resolved:
                package, cap = resolved
                key = (package.id, cap.kind, cap.id)
                if key in visited:
                    return
                visited.add(key)
                add(cap.metadata.get("monitoring", {}), package.manifest.get("name", package.id), supported,
                    package=package, config=effective, context=context)
                # Nested model selectors use the component's declared schema domains.
                def children(values, schema):
                    if not isinstance(values, dict):
                        return
                    for key, field in schema.get("properties", {}).items():
                        if key not in values:
                            continue
                        if field.get("x-theta-domain") == "model":
                            visit("model", values[key])
                        elif field.get("type") == "object":
                            children(values[key], field)
                children(effective, package.manifest.get("parameter_schema", {}))
                visited.remove(key)

        for kind, key in (("algorithm", "agent"), ("model", "model"), ("env", "env"),
                          ("datamodule", "data_module"), ("evaluator", "evaluator")):
            value = context.get(key)
            for selected in value if isinstance(value, list) else [value]:
                visit(kind, selected)
        from src.app.monitoring.config import instances, model_defaults, model_capabilities, validate_context
        try:
            configured = instances(context, defaults=model_defaults(context, catalog))
        except (ValueError, TypeError, KeyError, OSError) as exc:
            if errors is None:
                raise
            errors.append(str(exc))
            configured = {}
        for identity, instance in configured.items():
            try:
                validate_context(context, catalog=catalog, script=script, configured={identity: instance})
                selected = catalog.resolve("monitor", instance["component"])
                if selected:
                    package, cap = selected
                    add(cap.metadata.get("monitoring", {}), f"Monitor · {identity}", supported | model_capabilities(context, catalog),
                        package=package, config={**instance["params"], "instance": identity, "target": instance["target"]}, context=context)
            except (ValueError, TypeError, KeyError, OSError) as exc:
                if errors is None:
                    raise
                errors.append(str(exc))
        from src.app.pipeline.tuning_config import tuning_enabled
        if tuning_enabled(context) or context.get("theta_tuning"):
            tuning = context.get("theta_tuning") or context.get("tuning") or {}
            visit("tuner", tuning.get("component", "optuna"))
    return sorted(specs.values(), key=lambda spec: (spec.source, spec.group, spec.name))
