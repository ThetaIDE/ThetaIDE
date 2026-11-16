"""Tests for manifest-driven Theta Hub conformance suites."""

import json

import pytest

from src.app.services.test_suite import (
    TestSuiteManifestError,
    evaluate_component_contracts,
    evaluate_test_suite,
    load_test_suite_manifest,
    parse_test_suite_manifest,
)


def _manifest():
    return {
        "id": "suite-fixture",
        "name": "Install Test Fixture",
        "kind": "suite",
        "version": "1.0.0",
        "test_suite": {
            "format": "theta.test-suite/v1",
            "cases": [
                {
                    "id": "online-smoke", "title": "Online smoke recipe", "description": "A complete online RL recipe must be runnable.",
                    "experiment": "suite_fixture/online_smoke", "expectation": "runnable", "execution": "run",
                },
                {
                    "id": "missing-method", "title": "Missing method is blocked", "description": "A training recipe without a method must not launch.",
                    "experiment": "suite_fixture/missing_method", "expectation": "blocked",
                    "execution": "preflight", "blocked_reason": "The online-RL contract requires a method.",
                    "expected_error": "methods",
                },
            ],
        },
    }


def test_suite_records_human_readable_case_contracts(tmp_path):
    manifest_path = tmp_path / ".theta_component.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    suite = load_test_suite_manifest(manifest_path)

    assert [case.title for case in suite.cases] == ["Online smoke recipe", "Missing method is blocked"]
    assert suite.cases[1].blocked_reason == "The online-RL contract requires a method."


def test_suite_evaluator_accepts_expected_block_and_rejects_wrong_outcome():
    suite = parse_test_suite_manifest(_manifest())

    reports = {
        "suite_fixture/online_smoke": {"valid": True},
        "suite_fixture/missing_method": {"valid": False, "errors": [{"message": "methods are required"}]},
    }
    results = evaluate_test_suite(suite, reports.__getitem__, lambda _: {"success": True})
    assert all(result.passed for result in results)

    wrong = evaluate_test_suite(suite, lambda _: {"valid": True})
    assert wrong[1].passed is False
    assert "Expected blocked" in wrong[1].message

    no_runner = evaluate_test_suite(suite, reports.__getitem__)
    assert no_runner[0].passed is False
    assert "execution callback" in no_runner[0].message


def test_component_contracts_verify_nested_and_list_capabilities(tmp_path):
    manifest = _manifest()
    manifest["test_suite"]["components"] = [{
        "id": "ppo", "title": "PPO", "description": "Online algorithm.",
        "config_path": "agent/sb3/ppo.yaml",
        "required": {"entrypoint": "src/usr/algorithms/example/runner.py", "learning_rate": 0.0003},
    }]
    from src.app.workspace import WorkspacePaths
    target = WorkspacePaths(tmp_path).config / "agent/sb3"
    target.mkdir(parents=True)
    (target / "ppo.yaml").write_text("entrypoint: src/usr/algorithms/example/runner.py\nlearning_rate: 0.0003\n", encoding="utf-8")

    results = evaluate_component_contracts(parse_test_suite_manifest(manifest), tmp_path)
    assert results[0].passed is True

    (target / "ppo.yaml").write_text("entrypoint: src/usr/algorithms/example/runner.py\n", encoding="utf-8")
    results = evaluate_component_contracts(parse_test_suite_manifest(manifest), tmp_path)
    assert results[0].passed is False
    assert "learning_rate" in results[0].message


def test_blocked_case_requires_reason():
    manifest = _manifest()
    manifest["test_suite"]["cases"][1].pop("blocked_reason")
    with pytest.raises(TestSuiteManifestError, match="blocked_reason"):
        parse_test_suite_manifest(manifest)


def test_runnable_case_requires_real_execution_contract():
    manifest = _manifest()
    manifest["test_suite"]["cases"][0].pop("execution")
    with pytest.raises(TestSuiteManifestError, match="execution"):
        parse_test_suite_manifest(manifest)


