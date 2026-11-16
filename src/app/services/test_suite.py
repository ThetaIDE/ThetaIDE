"""Manifest-driven conformance suites for Theta Hub packages.

Suites deliberately define expectations instead of treating every invalid recipe
as a failed test.  A blocked case proves that pre-flight validation prevents an
unsafe or underspecified launch before the runner starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

TEST_SUITE_FORMAT = "theta.test-suite/v1"


class TestSuiteManifestError(ValueError):
    """Raised when a package's test_suite declaration is malformed."""

    __test__ = False


@dataclass(frozen=True)
class TestCase:
    id: str
    title: str
    description: str
    experiment: str
    expectation: str
    execution: str
    blocked_reason: str | None = None
    expected_error: str | None = None
    overrides: tuple[str, ...] = ()


@dataclass(frozen=True)
class ComponentContract:
    """A configuration capability that an installed component must provide."""

    id: str
    title: str
    description: str
    config_path: str
    required: Mapping[str, Any]


@dataclass(frozen=True)
class TestSuite:
    id: str
    name: str
    cases: tuple[TestCase, ...]
    components: tuple[ComponentContract, ...] = ()


@dataclass(frozen=True)
class TestCaseResult:
    case: TestCase
    passed: bool
    actual: str
    message: str


@dataclass(frozen=True)
class ComponentContractResult:
    contract: ComponentContract
    passed: bool
    message: str


def parse_test_suite_manifest(manifest: Mapping[str, Any]) -> TestSuite:
    """Parse a validated test-suite manifest into an immutable suite object."""
    declaration = manifest.get("test_suite")
    if not isinstance(declaration, Mapping) or declaration.get("format") != TEST_SUITE_FORMAT:
        raise TestSuiteManifestError(f"Expected test_suite.format '{TEST_SUITE_FORMAT}'.")

    raw_cases = declaration.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise TestSuiteManifestError("A test suite needs at least one case.")

    cases: list[TestCase] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, Mapping):
            raise TestSuiteManifestError("Each test-suite case must be an object.")
        required = ("id", "title", "description", "experiment", "expectation")
        missing = [key for key in required if not isinstance(raw.get(key), str) or not raw[key].strip()]
        if missing:
            raise TestSuiteManifestError(f"Test-suite case is missing: {', '.join(missing)}.")
        case_id = raw["id"]
        if case_id in seen:
            raise TestSuiteManifestError(f"Duplicate test-suite case id: '{case_id}'.")
        seen.add(case_id)
        expectation = raw["expectation"]
        if expectation not in {"runnable", "blocked"}:
            raise TestSuiteManifestError(f"Invalid expectation '{expectation}' for '{case_id}'.")
        blocked_reason = raw.get("blocked_reason")
        if expectation == "blocked" and (not isinstance(blocked_reason, str) or not blocked_reason.strip()):
            raise TestSuiteManifestError(f"Blocked case '{case_id}' needs a blocked_reason.")
        execution = raw.get("execution", "preflight")
        if execution not in {"preflight", "run"}:
            raise TestSuiteManifestError(f"Invalid execution mode '{execution}' for '{case_id}'.")
        if expectation == "runnable" and execution != "run":
            raise TestSuiteManifestError(
                f"Runnable case '{case_id}' must use execution: 'run'; pre-flight alone does not prove it runs."
            )
        if expectation == "blocked" and execution != "preflight":
            raise TestSuiteManifestError(f"Blocked case '{case_id}' must use execution: 'preflight'.")
        overrides = raw.get("overrides", [])
        if not isinstance(overrides, list) or not all(isinstance(item, str) and item for item in overrides):
            raise TestSuiteManifestError(f"Case '{case_id}' has invalid overrides.")
        cases.append(
            TestCase(
                id=case_id,
                title=raw["title"],
                description=raw["description"],
                experiment=raw["experiment"],
                expectation=expectation,
                execution=execution,
                blocked_reason=blocked_reason if isinstance(blocked_reason, str) else None,
                expected_error=raw.get("expected_error") if isinstance(raw.get("expected_error"), str) else None,
                overrides=tuple(overrides),
            )
        )

    contracts: list[ComponentContract] = []
    raw_contracts = declaration.get("components", [])
    if not isinstance(raw_contracts, list):
        raise TestSuiteManifestError("test_suite.components must be a list.")
    component_ids: set[str] = set()
    for raw in raw_contracts:
        if not isinstance(raw, Mapping):
            raise TestSuiteManifestError("Each component contract must be an object.")
        required_fields = ("id", "title", "description", "config_path", "required")
        missing = [key for key in required_fields if not raw.get(key)]
        if missing:
            raise TestSuiteManifestError(f"Component contract is missing: {', '.join(missing)}.")
        if not isinstance(raw["required"], Mapping):
            raise TestSuiteManifestError(f"Component contract '{raw['id']}' has invalid required fields.")
        if raw["id"] in component_ids:
            raise TestSuiteManifestError(f"Duplicate component contract id: '{raw['id']}'.")
        component_ids.add(raw["id"])
        contracts.append(
            ComponentContract(
                id=raw["id"],
                title=raw["title"],
                description=raw["description"],
                config_path=raw["config_path"],
                required=dict(raw["required"]),
            )
        )

    suite_id = manifest.get("id")
    if not isinstance(suite_id, str) or not suite_id:
        raise TestSuiteManifestError("Test-suite manifest needs an id.")
    return TestSuite(
        id=suite_id, name=str(manifest.get("name", suite_id)), cases=tuple(cases), components=tuple(contracts)
    )


def load_test_suite_manifest(path: str | Path) -> TestSuite:
    """Load a suite from the .theta_component.json installed with its package."""
    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TestSuiteManifestError(f"Could not read test-suite manifest '{manifest_path}': {exc}") from exc
    return parse_test_suite_manifest(manifest)


def evaluate_test_suite(
    suite: TestSuite,
    preflight: Callable[[str], Mapping[str, Any] | bool],
    execute: Callable[[TestCase], Mapping[str, Any] | bool] | None = None,
) -> tuple[TestCaseResult, ...]:
    """Run each declared contract, including a real runner for runnable cases.

    ``preflight`` can be :func:`src.app.services.config_service.compose_report` or
    another callable returning a mapping with ``valid`` and optional ``errors``.
    Runnable cases require ``execute``. A suite that claims a recipe can run but
    only supplies pre-flight validation is rejected at manifest-parse time.
    """
    results: list[TestCaseResult] = []
    for case in suite.cases:
        report = preflight(case.experiment)
        valid = bool(report.get("valid")) if isinstance(report, Mapping) else bool(report)
        actual = "runnable" if valid else "blocked"
        errors = report.get("errors", []) if isinstance(report, Mapping) else []
        error_text = "\n".join(
            str(error.get("message", error)) if isinstance(error, Mapping) else str(error) for error in errors
        )
        matches_error = not case.expected_error or case.expected_error.lower() in error_text.lower()
        passed = actual == case.expectation and matches_error
        if passed and case.execution == "run":
            if execute is None:
                passed = False
                message = "Runnable case needs an execution callback."
            else:
                execution_result = execute(case)
                ran = (
                    bool(execution_result.get("success"))
                    if isinstance(execution_result, Mapping)
                    else bool(execution_result)
                )
                if not ran:
                    passed = False
                    detail = (
                        execution_result.get("message", "runner returned failure")
                        if isinstance(execution_result, Mapping)
                        else "runner returned failure"
                    )
                    message = f"Pre-flight passed but execution failed: {detail}"
                else:
                    message = "Pre-flight passed and the recipe executed successfully."
        elif passed:
            message = "Pre-flight result matches the declared expectation."
        elif actual != case.expectation:
            message = f"Expected {case.expectation}, but pre-flight reported {actual}."
        else:
            message = f"Expected validation message containing: {case.expected_error}"
        results.append(TestCaseResult(case=case, passed=passed, actual=actual, message=message))
    return tuple(results)


def evaluate_component_contracts(suite: TestSuite, workspace_dir: str | Path) -> tuple[ComponentContractResult, ...]:
    """Verify that installed components retain the configuration capabilities a suite needs."""
    from src.app.workspace import WorkspacePaths
    paths = workspace_dir if isinstance(workspace_dir, WorkspacePaths) else WorkspacePaths(Path(workspace_dir))
    results: list[ComponentContractResult] = []
    for contract in suite.components:
        resolved = paths.config_resolver().resolve(contract.config_path)
        config_path = resolved.file if resolved is not None else None
        if config_path is None or not config_path.is_file():
            results.append(ComponentContractResult(contract, False, f"Missing config: {contract.config_path}"))
            continue
        try:
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            results.append(ComponentContractResult(contract, False, f"Invalid YAML: {exc}"))
            continue
        missing = _missing_required_values(config, contract.required)
        if missing:
            results.append(
                ComponentContractResult(contract, False, f"Missing or incompatible values: {', '.join(missing)}")
            )
        else:
            results.append(ComponentContractResult(contract, True, "Configuration capability contract is satisfied."))
    return tuple(results)


def run_pipeline_case(
    case: TestCase, workspace_dir: str | Path, python: str | None = None, timeout: int = 120
) -> dict[str, Any]:
    """Run one suite recipe using the same CLI path as a user launch."""
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(workspace_dir)
    command = [python or sys.executable, str(paths.application / "run_pipeline.py"), case.experiment, *case.overrides]
    completed = subprocess.run(command, cwd=paths.root, env={**os.environ, **paths.process_environment()},
                               capture_output=True, text=True, timeout=timeout)
    return {
        "success": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "message": (completed.stderr or completed.stdout).strip(),
    }


def _missing_required_values(config: Mapping[str, Any], required: Mapping[str, Any], prefix: str = "") -> list[str]:
    """Return dotted paths whose expected scalar, nested object, or list values do not match."""
    missing: list[str] = []
    for key, expected in required.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        actual = config.get(key) if isinstance(config, Mapping) else None
        if isinstance(expected, Mapping):
            if not isinstance(actual, Mapping):
                missing.append(path)
            else:
                missing.extend(_missing_required_values(actual, expected, path))
        elif isinstance(expected, list):
            if not isinstance(actual, list) or any(value not in actual for value in expected):
                missing.append(path)
        elif actual != expected:
            missing.append(path)
    return missing
