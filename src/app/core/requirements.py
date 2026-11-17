"""Compatibility requirements: components declare, core evaluates.

A capability states what it needs from the rest of an experiment in ``requires``
(rules at an author-chosen ``block`` or ``warn`` level), the contracts it satisfies
in ``provides``, and optionally a ``preflight`` hook for checks that need code.
Declaring nothing means compatible with everything. This module is the only place
those declarations are read; validation, forms, the run record and the SDK call it.

Rules are evaluated per method against the method's selections: the component
kinds below, plus every component a selected capability references through a
component-reference parameter (``x-theta-domain``). See
docs/reference/COMPATIBILITY_REQUIREMENTS.md.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from src.app.workspace import WorkspacePaths

LEVELS = ("block", "warn")
COMPARATORS = ("in", "equals", "min", "max", "present")
# Config key of a method's selection -> capability kind.
SELECTION_KINDS = {"agent": "algorithm", "model": "model", "env": "env", "data_module": "datamodule",
                   "evaluator": "evaluator"}
_MISSING = object()

RULE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["level"],
    "properties": {
        "level": {"enum": list(LEVELS)},
        "config": {"type": "string", "minLength": 1},
        "kind": {"type": "string", "minLength": 1},
        "in": {"type": "array"},
        "equals": {},
        "min": {"type": "number"},
        "max": {"type": "number"},
        "present": {"type": "boolean"},
        "capability": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}},
        "contract": {"type": "string", "minLength": 1},
        "when": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "array"}},
        "message": {"type": "string", "minLength": 1},
    },
}


def rule_error(rule) -> str | None:
    """Describe what is wrong with one declared rule, or return None."""
    import jsonschema
    try:
        jsonschema.validate(rule, RULE_SCHEMA)
    except jsonschema.ValidationError as exc:
        return exc.message
    if ("config" in rule) == ("kind" in rule):
        return "a rule names exactly one of config or kind"
    if "config" in rule:
        used = [key for key in COMPARATORS if key in rule]
        if len(used) != 1 or {"capability", "contract"} & set(rule):
            return "a config rule has exactly one comparator: in, equals, min, max or present"
    elif ("capability" in rule) == ("contract" in rule) or set(COMPARATORS) & set(rule):
        return "a kind rule has exactly one of capability or contract"
    return None


@dataclass(frozen=True)
class Finding:
    """One rule (or preflight message) that a method's selections do not satisfy."""
    capability: str
    package: str
    method: str | None
    level: str
    message: str
    rule: dict | None = None

    def as_dict(self):
        return {"capability": self.capability, "package": self.package, "method": self.method,
                "level": self.level, "message": self.message, "rule": self.rule}

    def __str__(self):
        where = f"Method '{self.method}': " if self.method else ""
        return f"{where}{self.message}"


@dataclass
class Selection:
    """A component a method selects, resolved to its capability when installed."""
    key: str
    kind: str
    name: str
    config: dict
    package: Any = None
    capability: Any = None
    path: tuple = ()
    children: list = field(default_factory=list)
    contract: str | None = None
    parent: Any = None

    @property
    def identity(self):
        return (self.package.id, self.capability.id) if self.capability is not None else None


def _plain(value):
    from omegaconf import OmegaConf
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    if hasattr(value, "items") and not isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return copy.deepcopy(value)


def selection_name(value) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "get"):
        for key in ("name", "algorithm", "architecture", "type", "class"):
            if isinstance(value.get(key), str) and value.get(key):
                return value[key]
    return ""


def _profile(kind, name, paths):
    from src.app.config_resolution import GROUP_KINDS
    from src.app.core.data_loaders import _profile
    group = next((g for g, k in GROUP_KINDS.items() if k == kind), kind)
    try:
        return _profile(group, name, paths)
    except (ValueError, OSError):
        return {}


def resolve(kind, name, profile, paths):
    """The (package, capability) a selector names: its capability ID, else its profile's owner."""
    from src.app.core.provider_catalog import DuplicateProvider, get_catalog
    from src.app.core.provider_loader import find_provider
    from src.app.config_resolution import GROUP_KINDS
    if not name:
        return None
    declared = {"algorithm": profile.get("algorithm"), "model": profile.get("architecture"),
                "datamodule": profile.get("data_module")}.get(kind)
    for selector in dict.fromkeys(s for s in (declared, profile.get("name"), name) if isinstance(s, str) and s):
        try:
            found = find_provider(kind, selector, paths)
        except DuplicateProvider:
            return None
        if found is not None:
            return found
    group = next((g for g, k in GROUP_KINDS.items() if k == kind), kind)
    owners = [(p, c) for p in get_catalog(paths).packages for c in p.capabilities
              if c.kind == kind and f"{group}/{name}.yaml" in p.manifest.get("payload", {}).get("configs", {})]
    return owners[0] if len(owners) == 1 else None


def _reference(kind, value, paths, *, key, path, seen, contract=None, parent=None):
    name = selection_name(value)
    if not name:
        return None
    profile = _profile(kind, name, paths)
    from src.app.pipeline.config import deep_merge
    config = deep_merge(copy.deepcopy(profile), _plain(value) if hasattr(value, "items") else {})
    selection = Selection(key, kind, name, config, path=path, contract=contract, parent=parent)
    _attach(selection, profile, paths, seen)
    return selection


def _attach(selection, profile, paths, seen):
    found = resolve(selection.kind, selection.name, profile, paths)
    if found is None:
        return
    selection.package, selection.capability = found
    if selection.identity in seen:
        return
    seen = seen | {selection.identity}
    for name, (child_kind, contract) in reference_parameters(selection.package, selection.capability, selection.name, paths).items():
        value = selection.config.get(name)
        child = _reference(child_kind, value, paths, key=selection.key, path=(*selection.path, name), seen=seen,
                           contract=contract, parent=selection)
        if child is not None:
            selection.children.append(child)


def reference_parameters(package, capability, name=None, paths=None) -> dict[str, tuple[str, str | None]]:
    """Component-reference parameters: name -> (referenced capability kind, required contract or None)."""
    from src.app.core.provider_catalog import kind_name
    schema = capability.metadata.get("parameter_schema")
    if not isinstance(schema, dict):
        schema = package.manifest.get("parameter_schema") or {}
        if name and paths is not None:
            from src.app.config_resolution import GROUP_KINDS
            group = next((g for g, k in GROUP_KINDS.items() if k == capability.kind), capability.kind)
            schema = (schema.get("x-theta-config-schemas") or {}).get(f"{group}/{name}.yaml", schema)
    return {key: (kind_name(node["x-theta-domain"]), node.get("x-theta-contract"))
            for key, node in (schema.get("properties") or {}).items()
            if isinstance(node, dict) and isinstance(node.get("x-theta-domain"), str)}


def method_selections(cfg, method=None, *, paths=None) -> dict[str, Selection]:
    """The method's top-level selections by config key (agent, model, env, data_module, evaluator)."""
    from src.app.pipeline.config import deep_merge
    from src.app.core.data_loaders import loader_selection
    paths = paths or WorkspacePaths.for_process()
    method = method or {}
    result = {}
    for key, kind in SELECTION_KINDS.items():
        if key == "data_module":
            selected, _ = loader_selection(cfg, key, method or None, paths=paths)
            if selected is None:
                continue
            name, config, profile = selected["name"], {"name": selected["name"], **selected["params"]}, {}
        else:
            value = method[key] if key in method else (cfg.get(key) if hasattr(cfg, "get") else None)
            name = selection_name(value)
            if not name:
                continue
            profile = _profile(kind, name, paths) if key != "env" else {}
            config = deep_merge(copy.deepcopy(profile), _plain(method.get(key + "_params") or {}))
            if hasattr(value, "items"):
                config = deep_merge(config, _plain(value))
        selection = Selection(key, kind, name, config, path=(key,))
        _attach(selection, profile, paths, frozenset())
        result[key] = selection
    return result


def _walk(selections):
    for selection in selections:
        yield selection
        yield from _walk(selection.children)


def _lookup(path, selections, effective):
    head, _, rest = path.partition(".")
    if head in SELECTION_KINDS and rest:
        selection = selections.get(head)
        if selection is None:
            return _MISSING
        value = selection.config
    else:
        value, rest = effective, path
    for part in rest.split("."):
        if hasattr(value, "get") and part in value:
            value = value[part]
        else:
            return _MISSING
    return _plain(value)


def _compare(rule, value):
    if "present" in rule:
        return (value is not _MISSING and value is not None) == rule["present"]
    if "in" in rule:
        return (None if value is _MISSING else value) in rule["in"]
    if "equals" in rule:
        return (None if value is _MISSING else value) == rule["equals"]
    if value is _MISSING or value is None:
        return True  # min and max apply only when the value is set
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return value >= rule["min"] if "min" in rule else value <= rule["max"]


def _shown(value):
    return "nothing" if value is _MISSING or value is None else repr(value)


def _describe(owner, rule, value):
    if "kind" in rule:
        wanted = (f"a {rule['kind']} providing {rule['contract']!r}" if "contract" in rule
                  else f"{rule['kind']} {' or '.join(rule['capability'])}")
        return f"{owner} requires {wanted}; got {_shown(value)}."
    comparator = next(key for key in COMPARATORS if key in rule)
    wanted = {"present": "to be set" if rule.get("present") else "to be unset",
              "in": f"in {rule.get('in')}", "equals": f"= {rule.get('equals')!r}",
              "min": f">= {rule.get('min')}", "max": f"<= {rule.get('max')}"}[comparator]
    return f"{owner} requires {rule['config']} {wanted}; got {_shown(value)}."


def _allowed(rule, selection, paths):
    from src.app.core.provider_catalog import DuplicateProvider, get_catalog
    if "contract" in rule:
        return selection.capability is not None and rule["contract"] in (selection.capability.metadata.get("provides") or [])
    names = {selection.name.lower()}
    if selection.capability is not None:
        names |= {selection.capability.id.lower(), *(a.lower() for a in selection.capability.aliases)}
    for allowed in rule["capability"]:
        if allowed.lower() in names:
            return True
        try:
            found = get_catalog(paths).resolve(rule["kind"], allowed)
        except DuplicateProvider:
            found = None
        if found is not None and selection.identity == (found[0].id, found[1].id):
            return True
    return False


def _top(kind, selections):
    from src.app.core.provider_catalog import kind_name
    kind = kind_name(kind)
    return next((s for s in selections.values() if s.kind == kind), None)


def rule_findings(selection, selections, effective, method_name, *, paths):
    """Findings of one selected capability's declared rules."""
    findings = []
    rules = selection.capability.metadata.get("requires") or []
    owner = selection.capability.id
    for rule in rules:
        if not all(_compare({"in": allowed}, _lookup(path, selections, effective))
                   for path, allowed in (rule.get("when") or {}).items()):
            continue
        if "kind" in rule:
            target = _top(rule["kind"], selections)
            satisfied = target is not None and _allowed(rule, target, paths)
            value = target.name if target is not None else _MISSING
        else:
            value = _lookup(rule["config"], selections, effective)
            satisfied = _compare(rule, value)
        if not satisfied:
            findings.append(Finding(f"{selection.kind}:{owner}", selection.package.id, method_name, rule["level"],
                                    rule.get("message") or _describe(owner, rule, value), dict(rule)))
    return findings


def contract_findings(selection, method_name):
    """A reference parameter narrowed by ``x-theta-contract`` must select a capability providing it."""
    if not selection.contract or selection.parent is None or selection.parent.capability is None:
        return []
    if selection.capability is not None and selection.contract in (selection.capability.metadata.get("provides") or []):
        return []
    parent = selection.parent
    where = ".".join(selection.path)
    return [Finding(f"{parent.kind}:{parent.capability.id}", parent.package.id, method_name, "block",
                    f"{parent.capability.id} requires {where} to provide {selection.contract!r}; "
                    f"{selection.name!r} does not.", {"config": where, "contract": selection.contract})]


def preflight_findings(selection, cfg, method_name, *, paths):
    """Run a selected capability's declared preflight hook."""
    from dataclasses import replace
    from src.app.core.provider_loader import load_capability
    reference = selection.capability.metadata.get("preflight")
    if reference is None:
        return []
    package, capability = selection.package, selection.capability
    hook = load_capability(package, replace(capability, entry_point=reference), paths=paths)
    if not callable(hook):
        raise ValueError(f"{package.id}: preflight is not callable")
    result = hook(cfg, method_name, paths=paths) or {}
    if not hasattr(result, "get") or set(result) - set(LEVELS) or not all(
            isinstance(result.get(level, []), (list, tuple)) and all(isinstance(m, str) for m in result.get(level, []))
            for level in LEVELS):
        raise ValueError(f"{package.id}: preflight must return {{'block': [...], 'warn': [...]}} message lists")
    return [Finding(f"{capability.kind}:{capability.id}", package.id, method_name, level, message)
            for level in LEVELS for message in result.get(level, [])]


def _effective(cfg, method):
    """The experiment with the method's settings over it (Hydra's own runtime node excluded)."""
    effective = {key: _plain(cfg[key]) for key in cfg.keys() if key != "hydra"} if hasattr(cfg, "keys") else {}
    effective.update(_plain(method or {}))
    return effective


def method_findings(cfg, method_name=None, method=None, *, paths=None, preflight=True):
    paths = paths or WorkspacePaths.for_process()
    selections = method_selections(cfg, method, paths=paths)
    effective = _effective(cfg, method)
    findings = []
    for selection in _walk(selections.values()):
        findings.extend(contract_findings(selection, method_name))
        if selection.capability is None:
            continue
        findings.extend(rule_findings(selection, selections, effective, method_name, paths=paths))
        if preflight:
            findings.extend(preflight_findings(selection, cfg, method_name, paths=paths))
    return _unique(findings)


def _unique(findings):
    seen, result = set(), []
    for finding in findings:
        key = (finding.capability, finding.method, finding.level, finding.message)
        if key not in seen:
            seen.add(key)
            result.append(finding)
    return result


def compatibility(cfg, *, paths=None, preflight=True) -> list[Finding]:
    """Every finding of every enabled method's selected capabilities."""
    from src.app.pipeline.config import parse_methods_dict
    from src.app.core.provider_discovery import provider_discovery_snapshot
    paths = paths or WorkspacePaths.for_process()
    with provider_discovery_snapshot():
        methods = parse_methods_dict(cfg, paths=paths) or {None: None}
        findings = []
        for name, method in methods.items():
            findings.extend(method_findings(cfg, name, method, paths=paths, preflight=preflight))
    return findings


def candidate_findings(cfg, method, key, candidate, *, paths=None) -> list[Finding]:
    """Declared findings if ``candidate`` became the method's ``key`` selection.

    Forms call this for every candidate of a picker; it reads declarations only and
    never runs preflight hooks. Only findings that involve the candidate are returned:
    its own rules and the other selections' rules about its kind or configuration.
    """
    if key not in SELECTION_KINDS:
        raise ValueError(f"Unknown selection {key!r}; expected one of {sorted(SELECTION_KINDS)}")
    paths = paths or WorkspacePaths.for_process()
    method = dict(_plain(method or {}))
    method[key] = candidate
    method.pop(key + "_params", None)
    selections = method_selections(cfg, method, paths=paths)
    effective = _effective(cfg, method)
    findings = []
    for selection in _walk(selections.values()):
        if selection.capability is None:
            continue
        for finding in rule_findings(selection, selections, effective, None, paths=paths):
            rule = finding.rule or {}
            mine = selection.key == key
            about = (rule.get("kind") and SELECTION_KINDS.get(key) == _kind(rule["kind"])) or \
                str(rule.get("config", "")).split(".", 1)[0] == key
            if mine or about:
                findings.append(finding)
    return _unique(findings)


def _kind(kind):
    from src.app.core.provider_catalog import kind_name
    return kind_name(kind)


def reference_findings(cfg, method, key, path, kind, candidate, *, paths=None) -> list[Finding]:
    """Declared findings of a component-reference parameter's candidate (its own rules)."""
    paths = paths or WorkspacePaths.for_process()
    selections = method_selections(cfg, method, paths=paths)
    effective = _effective(cfg, method)
    child = _reference(_kind(kind), candidate, paths, key=key, path=path, seen=frozenset())
    if child is None or child.capability is None:
        return []
    return rule_findings(child, selections, effective, None, paths=paths)


def _experiment_value(path, effective):
    value = effective
    for part in path.split("."):
        if hasattr(value, "get") and part in value:
            value = value[part]
        else:
            return _MISSING
    return _plain(value)


def figure_findings(cfg, capability, *, paths=None) -> list[Finding]:
    """Declared findings of a figure plotter, a ``(package, capability)``, for a composed experiment.

    A figure belongs to the whole experiment rather than to one method: ``config`` rules
    (and ``when`` conditions) read the composed experiment, and a ``kind`` rule is satisfied
    when any enabled method's selection of that kind satisfies it. Preflight hooks never run.
    """
    from src.app.pipeline.config import parse_methods_dict
    paths = paths or WorkspacePaths.for_process()
    package, capability = capability
    effective = _effective(cfg, None)
    owner = capability.id
    selections = None
    findings = []
    for rule in capability.metadata.get("requires") or []:
        if not all(_compare({"in": allowed}, _experiment_value(path, effective))
                   for path, allowed in (rule.get("when") or {}).items()):
            continue
        if "kind" in rule:
            if selections is None:
                methods = parse_methods_dict(cfg, paths=paths) or {None: None}
                selections = [method_selections(cfg, method, paths=paths) for method in methods.values()]
            targets = [t for t in (_top(rule["kind"], s) for s in selections) if t is not None]
            satisfied = any(_allowed(rule, target, paths) for target in targets)
            value = ", ".join(sorted({t.name for t in targets})) if targets else _MISSING
        else:
            value = _experiment_value(rule["config"], effective)
            satisfied = _compare(rule, value)
        if not satisfied:
            findings.append(Finding(f"plotter:{owner}", package.id, None, rule["level"],
                                    rule.get("message") or _describe(owner, rule, value), dict(rule)))
    return _unique(findings)


def provides(kind, candidate, contract, *, paths=None) -> bool:
    """Whether a candidate selection's capability lists ``contract`` in ``provides``."""
    paths = paths or WorkspacePaths.for_process()
    name = selection_name(candidate)
    found = resolve(_kind(kind), name, _profile(_kind(kind), name, paths), paths) if name else None
    return found is not None and contract in (found[1].metadata.get("provides") or [])


def warnings_record(cfg, *, paths=None) -> list[dict]:
    """The warn findings a run records with its configuration."""
    return [f.as_dict() for f in compatibility(cfg, paths=paths) if f.level == "warn"]


IDENTITY_KEYS = frozenset({"name", "class", "type", "algorithm", "architecture", "data_module"})


def load_selected(kind, selection, *, paths=None):
    """Load the component a configuration selects and call it with its parameters.

    ``selection`` is a selector or a mapping with ``name`` and parameters, as written in
    a component-reference parameter. The selected profile's values are the defaults;
    the mapping's values override them. Returns what the capability's entry point returns.
    """
    import sys
    from src.app.core.provider_loader import _caller_paths, load_capability
    from src.app.pipeline.config import deep_merge
    if paths is None:
        paths = _caller_paths(sys._getframe(1).f_globals.get("__name__"))
    kind = _kind(kind)
    name = selection_name(selection)
    if not name:
        raise ValueError(f"Select a {kind}")
    profile = _profile(kind, name, paths)
    found = resolve(kind, name, profile, paths)
    if found is None:
        raise ValueError(f"No installed {kind} provides {name!r}")
    config = deep_merge(copy.deepcopy(profile), _plain(selection) if hasattr(selection, "items") else {})
    params = {k: v for k, v in config.items() if k not in IDENTITY_KEYS and k != "params"}
    params = deep_merge(params, config.get("params") or {})
    return load_capability(*found, paths=paths)(**params)
