"""Resolve the experiment's and each method's data module selection.

A data module is one of the component kinds a paradigm's method consists of.
What other components need from it is declared in their compatibility rules
(src/app/core/requirements.py), never here.
"""
from __future__ import annotations

import copy
import yaml

from src.app.workspace import WorkspacePaths
from src.app.core.provider_discovery import discovery_cached
from src.app.core.provider_metadata import profile_values

SLOT = "data_module"


@discovery_cached
def _profile(group, name, paths):
    path = paths.resolve_config(f"{group}/{name}.yaml")
    if not path.is_file():
        from src.app.core.provider_catalog import get_catalog
        selector = f"{group}/{name}.yaml"
        for package in get_catalog(paths).packages:
            relative = package.manifest.get("payload", {}).get("configs", {}).get(selector)
            if relative:
                path = package.root / relative
                break
    return profile_values(path) if path.is_file() else {}


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
        if slot == SLOT and key in (slot, slot + ".name", slot + ".class", slot + ".type"):
            override = parser.parse_override(argument)
            if not override.is_sweep_override():
                value = override.value()
                if (key != slot and value) or (key == slot and loader_name(value)):
                    result.append(f"++{slot}=null")
        result.append(argument)
    return result


def loader_selection(cfg, slot=SLOT, method=None, *, paths=None):
    """Return a normalized data module selection and its source without importing providers."""
    if slot != SLOT:
        raise ValueError(f"Unknown data selection {slot!r}; methods select a {SLOT}")
    paths = paths or WorkspacePaths.for_process()
    value, source = cfg.get(slot), "experiment"
    if method is not None and slot in method:
        value = merge_loader_selection(value, method[slot])
        source = "method"
    if value is None and cfg.get("paradigm"):
        path = paths.resolve_config("paradigms/" + str(cfg.get("paradigm")) + ".yaml")
        paradigm = yaml.safe_load(path.read_text()) if path.is_file() else {}
        value = (paradigm or {}).get("default_data_module")
        source = "paradigm"
    name = loader_name(value)
    if not name:
        return None, source
    profile = _profile("data", name, paths)
    provider = profile.get("data_module") or profile.get("name") or name
    reserved = {"name", "class", "type", "data_module", "params"}
    params = {k: v for k, v in profile.items() if k not in reserved}
    params.update(profile.get("params") or {})
    if hasattr(value, "items"):
        params.update({k: plain(v) for k, v in value.items() if k not in reserved})
        from src.app.pipeline.config import deep_merge
        params = deep_merge(params, plain(value.get("params") or {}))
    return {"name": provider, "params": params}, source


def loader_metadata(selection, *, paths=None):
    from src.app.core.provider_metadata import capability_metadata
    if not selection:
        return {}
    return capability_metadata("datamodule", selection["name"], paths or WorkspacePaths.for_process())


def data_loader_errors(cfg, method=None, *, paths=None):
    """The selected data module is a known provider and its parameters match its schema."""
    from src.app.core.provider_metadata import BUILTIN_SELECTORS
    paths = paths or WorkspacePaths.for_process()
    selection, _ = loader_selection(cfg, SLOT, method, paths=paths)
    if selection is None:
        return []
    metadata = loader_metadata(selection, paths=paths)
    if not metadata:
        return [] if selection["name"] in BUILTIN_SELECTORS["datamodule"] else [f"Unknown {SLOT} provider '{selection['name']}'."]
    schema = metadata.get("parameter_schema")
    if not schema:
        return []
    from jsonschema import Draft202012Validator
    return list(dict.fromkeys(f"{SLOT} {selection['name']}: {e.message}"
                              for e in Draft202012Validator(schema).iter_errors(selection["params"])))
