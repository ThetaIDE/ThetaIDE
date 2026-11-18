"""Formal JSON Schema definition and validator for .theta_component.json manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.app.core.requirements import RULE_SCHEMA

COMPONENT_MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ThetaComponentManifest",
    "description": "Manifest metadata describing an installable Theta-IDE research component or plugin.",
    "type": "object",
    "required": ["id", "name", "kind", "version", "storage_version", "api_version", "capabilities", "payload"],
    "additionalProperties": False,
    "properties": {
        "license": {"type": "string"},
        "repository": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "icon": {"type": "string"},
        "entry_point": {"type": "string"},
        "default_enabled": {"type": "boolean"},
        "extra": {"type": "object"},
        "origin": {"type": "string"},
        "parent": {"type": "object"},
        "installed_from": {"type": ["string", "object"]},
        "sha256": {"type": "string"},
        # Installation records the lifecycle writes into an installed package's manifest.
        "authoring_hash": {"type": "string"},
        "installed_files": {"type": "array"},
        "source_hashes": {"type": "object"},
        "asset_hashes": {"type": "object"},
        "target_dir": {"type": "string"},
        "retained_edits": {"type": "array", "items": {"type": "string"}},
        "saved_local_changes": {},
        "storage_version": {"type": "integer", "const": 3},
        "api_version": {"type": "integer", "const": 1},
        "capabilities": {"type": "array", "items": {"type": "object", "required": ["id", "kind", "entry_point"],
            "additionalProperties": False,
            "if": {"properties": {"kind": {"const": "monitor"}}}, "then": {"required": ["hooks"]},
            "properties": {"id": {"type": "string", "minLength": 1}, "kind": {"type": "string"},
                           "entry_point": {"type": "string", "minLength": 1},
                           "aliases": {"type": "array", "items": {"type": "string"}},
                           "label": {"type": "string", "minLength": 1},
                           "description": {"type": "string", "minLength": 1},
                           "activation_plugin": {"type": "string", "minLength": 1},
                           "contribution_api": {"type": "integer", "minimum": 1},
                           "readiness": {"type": "object", "additionalProperties": False, "properties": {
                               "config": {"type": "array", "items": {"type": "string", "minLength": 1}},
                               "distributions": {"type": "array", "items": {"type": "string", "minLength": 1}}}},
                           "contributions": {"type": "array", "items": {"type": "object",
                               "required": ["id", "slot"], "additionalProperties": False, "properties": {
                                   "id": {"type": "string", "minLength": 1}, "slot": {"type": "string", "minLength": 1},
                                   "label": {"type": "string"}, "availability": {"enum": ["enabled", "ready"]},
                                   "entry_point": {"type": "string", "pattern": r"^.+\.py:.+$"}}}},
                           "scheduler_api": {"type": "integer", "enum": [2]},
                           "execution_contract": {"type": "integer", "minimum": 1},
                           "backend": {"type": "string"},
                           "tuning_features": {"type": "object", "additionalProperties": False, "properties": {
                               "objective_modes": {"type": "array", "items": {"enum": ["metric", "component"]}},
                               "pruning": {"type": "boolean"},
                               "pruning_objectives": {"type": "boolean"},
                               "pruning_repeated_runs": {"type": "boolean"}}},
                           "execution_features": {
                               "type": "object", "additionalProperties": False,
                               "properties": {
                                   "evaluation": {"enum": ["provider", "configured", "both"]},
                                   "resume_modes": {"type": "array", "items": {"enum": ["training", "inference"]}},
                                   "checkpoint_formats": {"type": "array", "items": {"type": "string", "minLength": 1}},
                                   "tuning": {"type": "boolean"}, "winner_promotion": {"type": "boolean"},
                                   "episode_outcomes": {"type": "boolean"},
                                   "intermediate_values": {"type": "boolean"},
                               }},
                           "requires": {"type": "array", "items": RULE_SCHEMA,
                                        "description": "Compatibility rules for each method that selects this capability."},
                           "provides": {"type": "array", "items": {"type": "string", "minLength": 1},
                                        "description": "Opaque contract IDs this capability satisfies."},
                           "preflight": {"type": "string", "pattern": r"^.+\.py:.+$"},
                           "resource_hint_hook": {"type": "string", "pattern": r"^.+\.py:.+$"},
                           "resource_hints": {"type": "object", "additionalProperties": False, "properties": {
                               "base_memory_gib": {"type": "number", "exclusiveMinimum": 0},
                               "cpu": {"type": "number", "exclusiveMinimum": 0},
                               "accelerator": {"type": "boolean"}}},
                           "resource_root": {"type": "string", "minLength": 1},
                           "results": {"type": "object", "additionalProperties": False, "required": ["category", "patterns"],
                               "properties": {"category": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
                                              "patterns": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}},
                               "description": "Run loggers: the results category and file patterns they write."},
                           "hooks": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1},
                                     "description": "Monitors: runtime hooks the runner or model must offer."},
                           "monitoring": {"type": "object"},
                           "monitoring_access": {"type": "array", "items": {"type": "string", "minLength": 1}},
                           "monitor_defaults": {"type": "object"},
                           "parameter_schema": {"type": "object"}}}},
        "workspace_resources": {"type": "object", "description": "Workspace resource directories the package reads: "
                                "name -> {path: package directory with its own entries (optional), depth: entry depth}.",
                                "propertyNames": {"pattern": "^[a-z][a-z0-9_]*$"},
                                "additionalProperties": {"type": "object", "additionalProperties": False, "required": ["depth"],
                                    "properties": {"path": {"type": "string", "minLength": 1},
                                                   "depth": {"type": "integer", "minimum": 1, "maximum": 4}}}},
        "exports": {"type": "object", "description": "Public cross-package objects: name -> 'file.py:attribute'.",
                    "propertyNames": {"pattern": r"^[A-Za-z_][A-Za-z0-9_.]*$"},
                    "additionalProperties": {"type": "string", "pattern": r"^[^:]+\.py:[A-Za-z_][A-Za-z0-9_.]*$"}},
        "payload": {"type": "object", "required":["implementations","resources","configs"], "properties": {
            "implementations": {"type": "array", "items": {"type": "string"}},
            "resources": {"type": "array", "items": {"type": "string"}},
            "configs": {"type": "object", "additionalProperties": {"type": "string"}}}},
        "id": {
            "type": "string",
            "pattern": "^[a-zA-Z0-9_-]+$",
            "description": "Unique identifier for the component (alphanumeric, dashes, underscores).",
        },
        "name": {
            "type": "string",
            "minLength": 1,
            "description": "Human-readable display title.",
        },
        "kind": {
            "type": "string",
            "enum": [
                "site",
                "monitor",
                "model",
                "algorithm",
                "objective",
                "tuner",
                "method",
                "agent",
                "env",
                "datamodule",
                "data",
                "evaluator",
                "eval",
                "plotter",
                "plot",
                "experiment",
                "plugin",
                "suite",
                "package",
            ],
            "description": "Component archetype.",
        },
        "version": {
            "type": "string",
            "pattern": r"^\d+\.\d+\.\d+.*$",
            "description": "Semantic version string (e.g. 1.0.0 or 0.1.0-alpha).",
        },
        "description": {
            "type": "string",
            "description": "Detailed explanation of the component's functionality.",
        },
        "author": {
            "type": ["object", "string"],
            "description": "Author or research lab information.",
        },
        "config_path": {
            "type": "string",
            "description": "Primary config selector, relative to the config root (e.g. model/mlp.yaml).",
        },
        "config_paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of configuration YAML files bundled with this component.",
        },
        "compatibility": {
            "type": "object",
            "description": "Software version constraints (e.g. theta, torch, python).",
        },
        "dependencies": {
            "type": "object",
            "properties": {
                "pip": {"type": "array", "items": {"type": "string"}},
                "components": {"type": "array", "items": {"oneOf": [{"type": "string"}, {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}, "version": {"type": "string"}}}]}},
            },
        },
        "suite": {
            "type": "string",
            "minLength": 1,
            "description": "ID of the installable package that supplies this internal component.",
        },
        "hidden": {
            "type": "boolean",
            "description": "Whether to omit this component from normal Hub discovery.",
        },
        "aliases": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": "Older or alternate IDs that resolve to this package.",
        },
        "namespace": {
            "type": "string",
            "minLength": 1,
            "description": "Optional configuration subdirectory for a component family.",
        },
        "test_suite": {
            "type": "object",
            "description": "Conformance cases packaged by a downloadable test suite.",
            "required": ["format", "cases"],
            "properties": {
                "format": {"const": "theta.test-suite/v1"},
                "cases": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "required": ["id", "title", "description", "experiment", "expectation", "execution"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "title": {"type": "string", "minLength": 1},
                            "description": {"type": "string", "minLength": 1},
                            "experiment": {"type": "string", "minLength": 1},
                            "expectation": {"enum": ["runnable", "blocked"]},
                            "execution": {"enum": ["preflight", "run"]},
                            "overrides": {"type": "array", "items": {"type": "string", "minLength": 1}},
                            "blocked_reason": {"type": "string", "minLength": 1},
                            "expected_error": {"type": "string", "minLength": 1},
                        },
                        "allOf": [
                            {
                                "if": {"properties": {"expectation": {"const": "blocked"}}},
                                "then": {
                                    "required": ["blocked_reason"],
                                    "properties": {"execution": {"const": "preflight"}},
                                },
                            },
                            {
                                "if": {"properties": {"expectation": {"const": "runnable"}}},
                                "then": {"properties": {"execution": {"const": "run"}}},
                            },
                        ],
                    },
                },
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "title", "description", "config_path", "required"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "title": {"type": "string", "minLength": 1},
                            "description": {"type": "string", "minLength": 1},
                            "config_path": {"type": "string", "minLength": 1},
                            "required": {"type": "object", "minProperties": 1},
                        },
                    },
                },
            },
        },
        "created_at": {
            "type": "string",
            "description": "ISO-8601 timestamp of package creation.",
        },
        "parameter_schema": {
            "type": "object",
            "required": ["type", "properties"],
            "properties": {
                "type": {"const": "object"},
                "properties": {
                    "type": "object",
                    "additionalProperties": {"type": "object"},
                },
                "required": {"type": "array", "items": {"type": "string"}},
                "additionalProperties": {"type": ["boolean", "object"]},
                "x-theta-validate": {"type": "boolean", "description": "Opt this parameter schema into pipeline validation."},
                "x-theta-config-schemas": {
                    "type": "object",
                    "additionalProperties": {"type": "object"},
                    "description": "Per-config-path JSON Schemas for components that bundle multiple profiles.",
                },
            },
            "description": "JSON Schema object describing component parameters. Use properties for nested keys, enum for dropdown choices, x-theta-domain: <kind> for a picker of installed components of that kind (narrowed by x-theta-contract), x-theta-resource: <name> for entries of a declared workspace resource, and x-theta-config-schemas for profile-specific schemas.",
        },
    },
}

SEMVER_REGEX = re.compile(r"^\d+\.\d+\.\d+.*$")
ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\./]+$")
VALID_KINDS = {
    "site",
    "algorithm", "objective", "tuner",
    "monitor",
    "model",
    "method",
    "agent",
    "env",
    "datamodule",
    "data",
    "evaluator",
    "eval",
    "plotter",
    "plot",
    "experiment",
    "plugin",
    "suite",
    "package",
}


def config_selector(value) -> str:
    """Return a package config selector such as ``model/mlp.yaml``.

    Selectors are relative to the config root: absolute paths, ``..`` and
    workspace-layout prefixes are rejected.
    """
    from pathlib import PurePosixPath
    text = str(value) if isinstance(value, str) else ""
    path = PurePosixPath(text)
    if (not text or "\\" in text or ":" in text or path.is_absolute() or ".." in path.parts
            or len(path.parts) < 2 or path.parts[0] in ("in", "src") or path.suffix not in (".yaml", ".yml")
            or path.as_posix() != text):
        raise ValueError(f"Invalid config selector {value!r}: use <group>/<name>.yaml relative to the config root")
    return text


def validate_config_selectors(data: Dict[str, Any]) -> None:
    """Every config destination, schema key and recorded profile is a logical selector."""
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    configs = payload.get("configs") if isinstance(payload.get("configs"), dict) else {}
    schemas = (data.get("parameter_schema") or {}).get("x-theta-config-schemas", {}) if isinstance(data.get("parameter_schema"), dict) else {}
    suite = data.get("test_suite") if isinstance(data.get("test_suite"), dict) else {}
    contracts = [c.get("config_path") for c in suite.get("components", []) if isinstance(c, dict) and c.get("config_path")]
    for value in [*configs, *(schemas if isinstance(schemas, dict) else {}), *data.get("config_paths", []),
                  *([data["config_path"]] if data.get("config_path") else []), *contracts]:
        config_selector(value)


def validate_manifest_data(data: Dict[str, Any]) -> Tuple[bool, str | None]:
    """Validate component manifest dictionary against schema.

    Uses `jsonschema` if available, falling back to a pure-Python schema validator.

    Returns:
        (True, None) if valid.
        (False, error_message) if invalid.
    """
    if not isinstance(data, dict):
        return False, "Manifest must be a JSON object dictionary."

    if data.get("storage_version") != 3 or data.get("api_version") != 1:
        return False, "Unsupported archive storage_version or provider api_version"
    for field in ('capabilities','payload'):
        if field not in data: return False, f"Missing native manifest field: {field}"
    if not isinstance(data['capabilities'],list) or not isinstance(data['payload'],dict):
        return False, "capabilities must be an array and payload must be an object"
    if not all(k in data['payload'] for k in ('implementations','resources','configs')):
        return False, "Declare all three native payload sections"
    for cap in data['capabilities']:
        if not isinstance(cap,dict) or not all(isinstance(cap.get(k),str) and cap[k] for k in ('id','kind','entry_point')):
            return False, "Every capability requires id, kind and entry_point"
        if "tuning_features" in cap:
            from src.app.pipeline.tuning_features import TuningFeatures
            try:
                TuningFeatures.from_metadata(cap)
            except ValueError as exc:
                return False, str(exc)
    try:
        validate_config_selectors(data)
    except ValueError as exc:
        return False, str(exc)
    exports = data.get("exports", {})
    if not isinstance(exports, dict):
        return False, "exports must map public names to 'file.py:attribute'"
    implementations = set(data["payload"].get("implementations") or [])
    for name, target in exports.items():
        file, _, attribute = target.partition(":") if isinstance(target, str) else ("", "", "")
        if not isinstance(name, str) or not name or not file.endswith(".py") or not attribute:
            return False, f"Export {name!r} must name 'file.py:attribute'"
        if file not in implementations:
            return False, f"Export {name!r} must come from a declared payload implementation, not {file!r}"
    # Try standard jsonschema library first
    try:
        import jsonschema

        try:
            jsonschema.validate(instance=data, schema=COMPONENT_MANIFEST_SCHEMA)
            return True, None
        except jsonschema.ValidationError as err:
            return False, f"Manifest schema validation error at '{'.'.join(str(p) for p in err.path)}': {err.message}"
    except ImportError:
        pass

    # Pure Python fallback validation
    for req in ("id", "name", "kind", "version"):
        if req not in data:
            return False, f"Missing required manifest field: '{req}'"
        if not data[req]:
            return False, f"Required field '{req}' cannot be empty."

    comp_id = str(data["id"])
    if not ID_REGEX.match(comp_id):
        return (
            False,
            f"Invalid 'id': '{comp_id}'. Must contain alphanumeric characters, underscores, dashes, or slashes.",
        )

    kind = str(data["kind"]).lower()
    if kind not in VALID_KINDS:
        return False, f"Invalid 'kind': '{kind}'. Allowed kinds: {sorted(VALID_KINDS)}"

    ver = str(data["version"])
    if not SEMVER_REGEX.match(ver):
        return False, f"Invalid 'version': '{ver}'. Must adhere to semantic versioning (e.g. 1.0.0)."

    test_suite = data.get("test_suite")
    if test_suite is not None:
        if kind not in {"suite", "package"}:
            return False, "A 'test_suite' manifest must have kind 'suite' or 'package'."
        if not isinstance(test_suite, dict) or test_suite.get("format") != "theta.test-suite/v1":
            return False, "'test_suite.format' must be 'theta.test-suite/v1'."
        cases = test_suite.get("cases")
        if not isinstance(cases, list) or not cases:
            return False, "'test_suite.cases' must contain at least one case."
        case_ids: set[str] = set()
        for index, case in enumerate(cases):
            if not isinstance(case, dict):
                return False, f"'test_suite.cases[{index}]' must be an object."
            missing = [
                key
                for key in ("id", "title", "description", "experiment", "expectation", "execution")
                if not case.get(key)
            ]
            if missing:
                return False, f"'test_suite.cases[{index}]' is missing required fields: {', '.join(missing)}."
            if case["id"] in case_ids:
                return False, f"Duplicate test-suite case id: '{case['id']}'."
            case_ids.add(case["id"])
            if case["expectation"] not in {"runnable", "blocked"}:
                return False, f"Invalid test-suite expectation for '{case['id']}'."
            if case["expectation"] == "blocked" and not case.get("blocked_reason"):
                return False, f"Blocked test-suite case '{case['id']}' requires 'blocked_reason'."
            execution = case.get("execution")
            if execution not in {"preflight", "run"}:
                return False, f"Invalid test-suite execution mode for '{case['id']}'."
            if case["expectation"] == "runnable" and execution != "run":
                return False, f"Runnable test-suite case '{case['id']}' requires execution 'run'."
            if case["expectation"] == "blocked" and execution != "preflight":
                return False, f"Blocked test-suite case '{case['id']}' requires execution 'preflight'."
        contracts = test_suite.get("components", [])
        if not isinstance(contracts, list):
            return False, "'test_suite.components' must be a list."
        for index, contract in enumerate(contracts):
            if not isinstance(contract, dict):
                return False, f"'test_suite.components[{index}]' must be an object."
            fields = ("id", "title", "description", "config_path", "required")
            missing = [key for key in fields if not contract.get(key)]
            if missing or not isinstance(contract.get("required"), dict):
                return False, f"'test_suite.components[{index}]' has invalid required fields."

    return True, None


def validate_manifest_file(manifest_path: str | Path) -> Tuple[bool, str | None]:
    """Validate a .theta_component.json file from disk."""
    path = Path(manifest_path)
    if not path.is_file():
        return False, f"Manifest file not found: {path}"

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"Failed to parse JSON manifest '{path}': {exc}"

    return validate_manifest_data(data)
