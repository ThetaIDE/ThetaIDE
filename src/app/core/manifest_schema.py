"""Formal JSON Schema definition and validator for .theta_component.json manifests."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

COMPONENT_MANIFEST_SCHEMA: Dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ThetaComponentManifest",
    "description": "Manifest metadata describing an installable Theta-IDE research component or plugin.",
    "type": "object",
    "required": ["id", "name", "kind", "version", "storage_version", "api_version", "capabilities", "payload"],
    "properties": {
        "storage_version": {"type": "integer", "enum": [2]},
        "api_version": {"type": "integer", "const": 1},
        "capabilities": {"type": "array", "items": {"type": "object", "required": ["id", "kind", "entry_point"],
            "properties": {"id": {"type": "string", "minLength": 1}, "kind": {"type": "string"},
                           "entry_point": {"type": "string", "minLength": 1},
                           "aliases": {"type": "array", "items": {"type": "string"}},
                           "role": {"enum": ["training", "symbolic_state"]},
                           "output_contracts": {"type": "array", "items": {"type": "string", "minLength": 1}},
                           "parameter_schema": {"type": "object"},
                           "input_requirements": {"type": "array", "items": {
                               "type": "object", "required": ["slot", "contract"], "properties": {
                                   "slot": {"enum": ["data_module", "symbolic_state_loader"]},
                                   "contract": {"type": "string", "minLength": 1},
                                   "when": {"type": "object", "additionalProperties": {"type": "array"}}
                               }}}}}},
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
                "rules",
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
        "target_path": {
            "type": "string",
            "description": "Relative workspace destination directory.",
        },
        "config_path": {
            "type": "string",
            "description": "Primary YAML configuration path in in/config/.",
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
        "provides": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": "Additional component IDs installed by this package.",
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
        "paradigms": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Learning paradigms compatible with this component (e.g. ['online_rl', 'offline_rl', 'supervised']).",
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
            "description": "JSON Schema object describing component parameters. Use properties for nested keys, enum for dropdown choices, x-theta-domain: model for installed model component choices, and x-theta-config-schemas for profile-specific schemas.",
        },
    },
}

SEMVER_REGEX = re.compile(r"^\d+\.\d+\.\d+.*$")
ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\./]+$")
VALID_KINDS = {
    "site",
    "algorithm", "objective", "tuner", "rules",
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


def validate_manifest_data(data: Dict[str, Any]) -> Tuple[bool, str | None]:
    """Validate component manifest dictionary against schema.

    Uses `jsonschema` if available, falling back to a pure-Python schema validator.

    Returns:
        (True, None) if valid.
        (False, error_message) if invalid.
    """
    if not isinstance(data, dict):
        return False, "Manifest must be a JSON object dictionary."

    if data.get("storage_version") != 2 or data.get("api_version") != 1:
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
