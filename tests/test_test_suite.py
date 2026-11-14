"""Tests for manifest-driven Theta Hub conformance suites."""

import json
import hashlib
from zipfile import ZipFile

import pytest

from src.app.services.test_suite import (
    TestSuiteManifestError,
    evaluate_component_contracts,
    evaluate_test_suite,
    load_test_suite_manifest,
    parse_test_suite_manifest,
)
from frontend.hub.installer import HubInstaller
from frontend.hub.models import HubComponent


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
        "required": {"entrypoint": "src/usr/algorithms/example/runner.py", "compatible_paradigms": ["online_rl"]},
    }]
    from src.app.workspace import WorkspacePaths
    target = WorkspacePaths(tmp_path).config / "agent/sb3"
    target.mkdir(parents=True)
    (target / "ppo.yaml").write_text("entrypoint: src/usr/algorithms/example/runner.py\ncompatible_paradigms: [online_rl]\n", encoding="utf-8")

    results = evaluate_component_contracts(parse_test_suite_manifest(manifest), tmp_path)
    assert results[0].passed is True

    (target / "ppo.yaml").write_text("entrypoint: src/usr/algorithms/example/runner.py\n", encoding="utf-8")
    results = evaluate_component_contracts(parse_test_suite_manifest(manifest), tmp_path)
    assert results[0].passed is False
    assert "compatible_paradigms" in results[0].message


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


def test_downloadable_suite_installs_components_and_experiment_group(tmp_path):
    source = tmp_path / "suite-source"
    source.mkdir()
    manifest = _manifest()
    manifest["dependencies"] = {"components": ["basic_suite"]}
    manifest["config_paths"] = [
        "src/usr/config/experiment/suite_fixture/_base.yaml",
        "src/usr/config/experiment/suite_fixture/online_smoke.yaml",
        "src/usr/config/experiment/suite_fixture/missing_method.yaml",
    ]
    (source / ".theta_component.json").write_text(json.dumps(manifest), encoding="utf-8")
    experiment_dir = source / "src/usr/config/experiment/suite_fixture"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "_base.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")
    (experiment_dir / "online_smoke.yaml").write_text("methods: {}\n", encoding="utf-8")
    (experiment_dir / "missing_method.yaml").write_text("methods: {}\n", encoding="utf-8")

    archive = tmp_path / "suite-fixture.zip"
    with ZipFile(archive, "w") as zf:
        for path in source.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(source).as_posix())

    component = HubComponent.from_dict({
        **manifest,
        "releases": {
            "1.0.0": {
                "tag": "v1.0.0",
                "url": archive.as_uri(),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        },
    })
    workspace = tmp_path / "workspace"
    installer = HubInstaller(workspace, tmp_path / "state")

    assert installer.install(component) is True
    installed_manifest = workspace / ".thetaide" / "suites" / component.id / ".theta_component.json"
    installed_suite = load_test_suite_manifest(installed_manifest)
    assert [case.experiment for case in installed_suite.cases] == [
        "suite_fixture/online_smoke", "suite_fixture/missing_method",
    ]
    assert installed_suite.cases[0].execution == "run"
    assert json.loads(installed_manifest.read_text(encoding="utf-8"))["dependencies"]["components"] == ["basic_suite"]
    assert (workspace / "src/usr/config/experiment/suite_fixture/online_smoke.yaml").is_file()
    assert (workspace / "src/usr/config/experiment/suite_fixture/missing_method.yaml").is_file()
