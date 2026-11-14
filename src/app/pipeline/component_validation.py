"""Optional component parameter validation against their declared JSON Schema.

Legacy parameter schemas describe UI controls and may not model runtime values.
A schema opts into validation with x-theta-validate: true, at either the package
or profile schema level. Components without that declaration remain compatible.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from omegaconf import OmegaConf

from src.app.pipeline.exceptions import ConfigurationError


def schemas_for_method(workspace: Path, settings: dict, *, validation_only: bool = True) -> dict[str, dict]:
    wanted = {}
    for kind in ("agent", "model"):
        selected = settings.get(kind)
        if isinstance(selected, dict):
            selected = selected.get("name") or selected.get("algorithm") or selected.get("architecture") or selected.get("type")
        if isinstance(selected, str) and selected:
            wanted[f"in/config/{kind}/{selected}.yaml"] = kind
    result = {}
    from src.app.workspace import WorkspacePaths
    paths = workspace if isinstance(workspace, WorkspacePaths) else WorkspacePaths(workspace)
    resolver = paths.config_resolver()
    for config_path, kind in wanted.items():
        relative = config_path.removeprefix("in/config/")
        package, _ = resolver.owner(relative)
        if package is None:
            continue
        data = package.manifest
        schema = data.get("parameter_schema")
        if not isinstance(schema, dict):
            continue
        profiles = schema.get("x-theta-config-schemas") or {}
        declared = list(data.get("config_paths") or [])
        if data.get("config_path"):
            declared.append(data["config_path"])
        selected_schema = profiles.get(config_path, schema)
        if not isinstance(selected_schema, dict):
            continue
        if not validation_only or selected_schema.get("x-theta-validate", schema.get("x-theta-validate")) is True:
            result[kind] = selected_schema
    return result


def validate_method_components(cfg, schemas: dict[str, dict], method: str) -> None:
    for kind, schema in schemas.items():
        try:
            Draft202012Validator.check_schema(schema)
            instance = OmegaConf.to_container(cfg.get(kind), resolve=True)
            errors = sorted(Draft202012Validator(schema).iter_errors(instance), key=lambda e: str(list(e.absolute_path)))
        except SchemaError as exc:
            raise ConfigurationError(f"Method '{method}' has invalid {kind} validation schema: {exc.message}") from exc
        if errors:
            error = errors[0]
            suffix = "".join(f".{part}" for part in error.absolute_path)
            raise ConfigurationError(f"Method '{method}' {kind}{suffix}: {error.message}")


def validate_sweep_dimensions(settings: dict, schemas: dict[str, dict], method: str) -> None:
    """Check every choice and both bounds against the opted-in field schema."""
    dimensions = settings.get("tune") or settings.get("search_space") or {}
    for target, dimension in dimensions.items():
        if not isinstance(target, str) or not hasattr(dimension, "get"):
            continue
        parts = target.split(".")
        schema = schemas.get(parts[0])
        if not schema:
            continue
        for part in parts[1:]:
            if part.isdigit() and schema.get("type") == "array":
                schema = schema.get("items", {})
            else:
                schema = schema.get("properties", {}).get(part, {})
            if not isinstance(schema, dict) or not schema:
                break
        if not schema:
            continue  # additionalProperties and untyped component extensions are allowed
        choices = dimension.get("values") if dimension.get("type") == "choice" else [dimension.get("low"), dimension.get("high")]
        if not isinstance(choices, (list, tuple)):
            continue
        try:
            Draft202012Validator.check_schema(schema)
            validator = Draft202012Validator(schema)
            for value in choices:
                error = next(validator.iter_errors(value), None)
                if error:
                    raise ConfigurationError(f"Method '{method}' sweep {target}: {error.message}")
        except SchemaError as exc:
            raise ConfigurationError(f"Method '{method}' has invalid schema for {target}: {exc.message}") from exc
