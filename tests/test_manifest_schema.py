"""Unit tests for formal manifest schema validation."""
import json
from pathlib import Path
import pytest

from src.app.core.manifest_schema import (
    COMPONENT_MANIFEST_SCHEMA,
    validate_manifest_data,
    validate_manifest_file,
)


def native(**fields):
    """A package format v3 manifest without capabilities, plus the fields under test."""
    return {"storage_version": 3, "api_version": 1, "capabilities": [],
            "payload": {"implementations": [], "resources": [], "configs": {}}, **fields}


def test_valid_manifest():
    valid = native(
        id="my_custom_model",
        name="My Custom Policy",
        kind="model",
        version="1.0.0",
        description="Valid model component",
        author={"name": "Lab Researcher"},
        compatibility={"theta": ">=2.0.0"},
    )
    is_valid, err = validate_manifest_data(valid)
    assert is_valid is True
    assert err is None


def test_missing_required_fields():
    invalid = native(id="missing_kind", name="Missing Kind", version="1.0.0")
    is_valid, err = validate_manifest_data(invalid)
    assert is_valid is False
    assert "kind" in err.lower()


def test_invalid_kind_enum():
    invalid = native(id="bad_kind", name="Bad Kind", kind="not_a_real_kind", version="1.0.0")
    is_valid, err = validate_manifest_data(invalid)
    assert is_valid is False
    assert "kind" in err.lower()


def test_invalid_semver():
    invalid = native(id="bad_ver", name="Bad Version", kind="model", version="invalid-version-string")
    is_valid, err = validate_manifest_data(invalid)
    assert is_valid is False
    assert "version" in err.lower()


def test_validate_manifest_file(tmp_path):
    manifest_path = tmp_path / ".theta_component.json"
    manifest_path.write_text(
        json.dumps(native(id="file_test", name="File Test", kind="env", version="0.2.1")),
        encoding="utf-8",
    )
    is_valid, err = validate_manifest_file(manifest_path)
    assert is_valid is True
    assert err is None


def test_validate_nonexistent_manifest_file(tmp_path):
    is_valid, err = validate_manifest_file(tmp_path / "missing.json")
    assert is_valid is False
    assert "not found" in err.lower()


def test_test_suite_manifest_requires_expectations_and_block_reasons():
    valid = native(**{
        "id": "conformance_suite",
        "name": "Conformance Suite",
        "kind": "suite",
        "version": "1.0.0",
        "test_suite": {
            "format": "theta.test-suite/v1",
            "cases": [
                {
                    "id": "runnable", "title": "Runnable config", "description": "Must validate.",
                    "experiment": "suite/runnable", "expectation": "runnable", "execution": "run",
                },
                {
                    "id": "blocked", "title": "Missing method", "description": "Must be blocked.",
                    "experiment": "suite/blocked", "expectation": "blocked", "execution": "preflight", "blocked_reason": "No method is configured.",
                },
            ],
        },
    })
    assert validate_manifest_data(valid) == (True, None)

    valid["test_suite"]["cases"][1].pop("blocked_reason")
    is_valid, error = validate_manifest_data(valid)
    assert is_valid is False
    assert "blocked_reason" in error


def test_test_suite_manifest_requires_real_execution_for_runnable_case():
    suite = native(**{
        "id": "conformance_suite", "name": "Conformance Suite", "kind": "suite", "version": "1.0.0",
        "test_suite": {"format": "theta.test-suite/v1", "cases": [{
            "id": "runnable", "title": "Runnable", "description": "Must execute.",
            "experiment": "suite/runnable", "expectation": "runnable",
        }]},
    })
    valid, error = validate_manifest_data(suite)
    assert valid is False
    assert "execution" in error
