"""Resolve experiment defaults and method-specific data input providers.

Training batches and inference-time state conversion are separate slots. Provider
requirements live in component manifests, so the pipeline never names a model.
"""
from __future__ import annotations

import copy
import yaml

from src.app.workspace import WorkspacePaths

LOADER_SLOTS = ("data_module", "symbolic_state_loader")
SLOT_ROLES = {"data_module": "training", "symbolic_state_loader": "symbolic_state"}


def _profile(group, name, paths):
    path = paths.resolve_config(f"{group}/{name}.yaml")
    if not path.is_file():
        from src.app.core.provider_catalog import get_catalog
        dest = f"in/config/{group}/{name}.yaml"
        for package in get_catalog(paths).packages:
            relative = package.manifest.get("payload", {}).get("configs", {}).get(dest)
            if relative:
                path = package.root / relative
                break
    return (yaml.safe_load(path.read_text()) or {}) if path.is_file() else {}


def plain(value):
    from omegaconf import OmegaConf
    return OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)


def loader_name(value):
    if hasattr(value, "get"):
        return value.get("name") or value.get("class") or value.get("type") or ""
    return value if isinstance(value, str) else ""


def merge_loader_selection(base, override):
    """A new provider replaces all old parameters; the same provider inherits."""
    from src.app.pipeline.config import deep_merge
    if override is None:
        return None
    old, new = loader_name(base), loader_name(override)
    if old and (not new or old == new):
        base = {"name": base} if isinstance(base, str) else plain(base)
        override = {"name": override} if isinstance(override, str) else plain(override)
        if isinstance(base, dict) and isinstance(override, dict):
            return deep_merge(base, override)
    return plain(override)


def loader_override_args(arguments):
    """Make an explicit CLI provider selection replace Hydra's inherited map.

    Parameter-only CLI edits continue to merge. Set the provider before its
    parameters, as with other ordered command-line overrides.
    """
    from hydra.core.override_parser.overrides_parser import OverridesParser
    parser = OverridesParser.create()
    result = []
    for argument in arguments:
        key = argument.split("=", 1)[0].lstrip("+")
        slot = key.split(".", 1)[0]
        if slot in LOADER_SLOTS and key in (slot, slot + ".name", slot + ".class", slot + ".type"):
            override = parser.parse_override(argument)
            if not override.is_sweep_override():
                value = override.value()
                if (key != slot and value) or (key == slot and loader_name(value)):
                    result.append(f"++{slot}=null")
        result.append(argument)
    return result


def loader_selection(cfg, slot, method=None, *, paths=None):
    """Return a normalized selection and its source without importing providers."""
    paths = paths or WorkspacePaths.for_process()
    value, source = cfg.get(slot), "experiment"
    if method is not None and slot in method:
        value = merge_loader_selection(value, method[slot])
        source = "method"
    if value is None and slot == "data_module":
        path = paths.resolve_config("paradigms/" + str(cfg.get("paradigm", "online_rl")) + ".yaml")
        paradigm = yaml.safe_load(path.read_text()) if path.is_file() else {}
        value = (paradigm or {}).get("default_data_module") or (paradigm or {}).get("data_module")
        source = "paradigm"
    name = loader_name(value)
    if not name:
        return None, source
    profile = _profile("data", name, paths)
    provider = profile.get("data_module") or profile.get("name") or name
    reserved = {"name", "class", "type", "data_module", "role", "paradigms", "compatible_paradigms", "params"}
    params = {k: v for k, v in profile.items() if k not in reserved}
    params.update(profile.get("params") or {})
    if hasattr(value, "items"):
        params.update({k: plain(v) for k, v in value.items() if k not in reserved})
        from src.app.pipeline.config import deep_merge
        params = deep_merge(params, plain(value.get("params") or {}))
    return {"name": provider, "params": params}, source


def _selected_component(cfg, method, kind, paths):
    from src.app.pipeline.config import deep_merge
    value = method.get(kind, cfg.get(kind, {})) if method is not None else cfg.get(kind, {})
    name = loader_name(value)
    if hasattr(value, "get"):
        name = value.get("architecture") or value.get("algorithm") or name
    defaults = _profile(kind, name, paths) if name else {}
    params = method.get(kind + "_params", {}) if method is not None else {}
    config = deep_merge(defaults, plain(params))
    if hasattr(value, "items"):
        config = deep_merge(config, plain(value))
    return name, config


def input_requirements(cfg, method=None, *, paths=None):
    """Collect conditional requirements from the selected algorithm and model."""
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    catalog = get_catalog(paths)
    result = []
    for kind, capability_kind in (("model", "model"), ("agent", "algorithm")):
        name, config = _selected_component(cfg, method, kind, paths)
        if not name:
            continue
        selected = catalog.resolve(capability_kind, str(config.get("architecture") or config.get("algorithm") or config.get("name") or name))
        if selected is None:
            # A profile selector may differ from the capability's public ID.
            owners = [(p, c) for p in catalog.packages for c in p.capabilities
                      if c.kind == capability_kind and f"in/config/{kind}/{name}.yaml" in p.manifest.get("payload", {}).get("configs", {})]
            selected = owners[0] if len(owners) == 1 else None
        if selected is None:
            continue
        package, capability = selected
        for requirement in capability.metadata.get("input_requirements", package.manifest.get("input_requirements", [])):
            matches = True
            for key, allowed in requirement.get("when", {}).items():
                value = config
                for part in key.split("."):
                    value = value.get(part) if hasattr(value, "get") else None
                if value not in allowed:
                    matches = False
            if matches:
                result.append({**requirement, "consumer": name})
    return result


def loader_metadata(selection, *, paths=None):
    from src.app.core.provider_catalog import get_catalog
    if not selection:
        return {}
    selected = get_catalog(paths or WorkspacePaths.for_process()).resolve("datamodule", selection["name"])
    if selected is None:
        return {}
    package, capability = selected
    return {"id": capability.id, "paradigms": package.manifest.get("paradigms"), **capability.metadata}


def state_loader_choices(*, paths=None):
    """State providers and distinct profiles, without importing model packages."""
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    choices = {}
    for package in get_catalog(paths).packages:
        for capability in package.capabilities:
            if capability.kind == "datamodule" and capability.metadata.get("role") == "symbolic_state":
                choices[capability.id] = {}
        for dest, relative in package.manifest.get("payload", {}).get("configs", {}).items():
            if not dest.startswith("in/config/data/"):
                continue
            path = paths.resolve_config(dest.removeprefix("in/config/"))
            if not path.is_file():
                path = package.root / relative
            profile = yaml.safe_load(path.read_text()) or {}
            if profile.get("role") == "symbolic_state":
                choices[dest.removeprefix("in/config/data/").removesuffix(".yaml")] = profile
    return choices


def data_loader_errors(cfg, method=None, *, paths=None):
    """Shared authoring/preflight contract check. Does not execute plugin code."""
    paths = paths or WorkspacePaths.for_process()
    requirements = input_requirements(cfg, method, paths=paths)
    errors = []
    env = cfg.get("env")
    if (hasattr(env, "get") and env.get("data_module") is not None and cfg.get("data_module") is None
            and (method is None or method.get("data_module") is None)):
        errors.append("Move env.data_module to the experiment's data_module default. "
                      "scripts/migrate_data_loaders.py can migrate authored recipes.")
    for slot in LOADER_SLOTS:
        selection, _ = loader_selection(cfg, slot, method, paths=paths)
        needs = [r for r in requirements if r["slot"] == slot]
        if selection is None:
            errors.extend(f"{r['consumer']} requires {slot} producing {r['contract']}. Select one in Data Loaders." for r in needs)
            continue
        metadata = loader_metadata(selection, paths=paths)
        if not metadata:
            # Built-in training modules predate manifest capabilities.
            if slot != "data_module" or selection["name"] not in {
                "RLDataModule", "RLReplayBufferDataModule", "SimulatorDataModule", "SupervisedDataModule",
                "CrossValidationDataModule", "TabularNpzDataModule", "SupervisedNpzDataModule", "UnsupervisedDataModule",
            }:
                errors.append(f"Unknown {slot} provider '{selection['name']}'.")
            elif needs:
                errors.extend(f"{r['consumer']} requires {r['contract']}; {selection['name']} has no declared output contract."
                              for r in needs)
            continue
        role = metadata.get("role", "training")
        if role != SLOT_ROLES[slot]:
            errors.append(f"{selection['name']} supplies {role}, but {slot} needs {SLOT_ROLES[slot]}.")
        supported = metadata.get("paradigms")
        if supported and cfg.get("paradigm") not in supported:
            errors.append(f"{selection['name']} does not support paradigm '{cfg.get('paradigm')}'.")
        for requirement in needs:
            if requirement["contract"] not in metadata.get("output_contracts", []):
                errors.append(f"{requirement['consumer']} requires {requirement['contract']}; {selection['name']} does not produce it.")
        schema = metadata.get("parameter_schema")
        if schema:
            from jsonschema import Draft202012Validator
            errors.extend(f"{slot} {selection['name']}: {e.message}" for e in Draft202012Validator(schema).iter_errors(selection["params"]))
    return list(dict.fromkeys(errors))


def create_state_loader(cfg, *, paths=None):
    """Instantiate an inference-time converter, never a model or training runner."""
    from src.app.core.provider_loader import load_capability
    from src.app.core.provider_catalog import get_catalog
    paths = paths or WorkspacePaths.for_process()
    selection, _ = loader_selection(cfg, "symbolic_state_loader", paths=paths)
    if selection is None:
        raise ValueError("A symbolic observation builder is required. Select symbolic_state_loader in Data Loaders.")
    selected = get_catalog(paths).resolve("datamodule", selection["name"])
    if selected is None or selected[1].metadata.get("role") != "symbolic_state":
        raise ValueError(f"'{selection['name']}' is not a symbolic state loader.")
    loader = load_capability(*selected, paths=paths)(**selection["params"])
    if not callable(loader):
        raise ValueError("A symbolic state loader must be callable.")
    return loader
