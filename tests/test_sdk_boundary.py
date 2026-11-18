"""Packages reach the host only through the Theta SDK, other packages only through declared exports,
and libraries only through declared requirements; core and the desktop name no package.

Covered: optional components, bundled component packages and app plugins, the tracked authored
packages under src/usr and the fixture packages. Installed packages are checked when installed
(the installer refuses private imports) and when built (the package builder runs these checks).
"""
import ast
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.app.components.sdk_boundary import (package_errors, package_references, private_references,
                                             undeclared_imports, violations, window_access)

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "src/plugins/components/bundled", ROOT / "src/plugins/app/bundled", ROOT / "tests/fixtures"]
HOST_TREES = [ROOT / "src/app", ROOT / "frontend", ROOT / "hydra_plugins", ROOT / "run_pipeline.py"]


def guarded_files():
    return sorted(p for root in GUARDED for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def package_roots():
    """Every package this repository tracks: bundled, authored (src/usr) and fixtures."""
    listed = subprocess.run(["git", "ls-files", "*.theta_component.json"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.split()
    return sorted((ROOT / name).parent for name in listed)


def application_requirements():
    from src.app.distribution import application_project
    return application_project(extras=("gui",))[1]


def host_files():
    return sorted(p for root in HOST_TREES for p in ([root] if root.is_file() else root.rglob("*.py"))
                  if "__pycache__" not in p.parts and ".venv" not in p.parts)


def test_guarded_trees_exist():
    assert len(guarded_files()) > 20


@pytest.mark.parametrize("path", guarded_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_only_the_supported_sdk(path):
    assert private_references(path) == [], f"{path.relative_to(ROOT)} imports private host modules"


def test_tracked_packages_cover_every_origin():
    roots = {root.relative_to(ROOT).parts[0] if root.parts[-3:-1] != ("src", "usr") else "" for root in package_roots()}
    assert {"src", "tests"} <= roots
    assert any(root.is_relative_to(ROOT / "src/usr") for root in package_roots())


@pytest.mark.parametrize("root", package_roots(), ids=lambda p: str(p.relative_to(ROOT)))
def test_package_code_keeps_the_boundary_and_declares_its_imports(root):
    manifest = json.loads((root / ".theta_component.json").read_text())
    assert package_errors(manifest, root) == []
    assert undeclared_imports(manifest, root, application_requirements()) == []


def test_core_and_desktop_import_no_package_by_path():
    # The provider loader alone maps loaded packages to their private module names.
    loader = ROOT / "src/app/core/provider_loader.py"
    found = [f"{p.relative_to(ROOT)}:{line}: {name}" for p in host_files() if p != loader
             for line, name in package_references(p)]
    assert found == []


def test_core_and_desktop_name_no_optional_or_authored_package():
    """Static half of 'core never branches on a package': no non-bundled package ID appears as a string."""
    bundled = {json.loads(p.read_text())["id"] for p in [*(ROOT / "src/plugins/components/bundled").glob("*/.theta_component.json"),
                                                          *(ROOT / "src/plugins/app/bundled").glob("*/plugin.json")]}
    # Package IDs; capability IDs are ordinary words ("tuning", "losses") that name config keys too.
    names = {json.loads((root / ".theta_component.json").read_text())["id"] for root in package_roots()}
    # Registered exception: the local site.
    allowed = set()
    names -= bundled | {"local", "default"}
    found = []
    for path in host_files():
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in names
                    and (relative, node.value) not in allowed):
                found.append(f"{relative}:{node.lineno}: {node.value!r}")
    assert found == []


CORE_TREES = [ROOT / "src/app", ROOT / "hydra_plugins", ROOT / "run_pipeline.py", ROOT / "frontend/results",
              ROOT / "frontend/config_panel", ROOT / "frontend/config_model.py", ROOT / "frontend/config_viewer.py",
              ROOT / "frontend/parameter_actions.py"]


@pytest.mark.parametrize("token", ["symbolic_state", "ruleset", "rules_root", "compatible_paradigms",
                                   "allowed_agents", "tensorboard", "tfevents", "slurm_submit_dir", "optuna", "ncshare",
                                   "sb3", "d3rlpy"])
def test_core_has_no_component_special_cases(token):
    """Compatibility, resources, result categories, sweepers and scheduler shells belong to their packages."""
    found = []
    files = sorted(p for root in CORE_TREES for p in ([root] if root.is_file() else root.rglob("*.py"))
                   if "__pycache__" not in p.parts and ".venv" not in p.parts)
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and token in node.value.lower():
                found.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
            elif isinstance(node, ast.Name) and token in node.id.lower():
                found.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}")
    assert found == []


def test_the_installer_refuses_package_code_with_private_imports(tmp_path):
    from types import SimpleNamespace
    from src.app.components.archive import adapt
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.py").write_text("from src.app.core.model_registry import register_model\n")
    (source / "test_model.py").write_text("from src.app.workspace import WorkspacePaths\n")
    manifest = {"id": "leaky", "name": "Leaky", "kind": "model", "version": "1.0.0", "storage_version": 3, "api_version": 1,
                "capabilities": [{"id": "leaky", "kind": "model", "entry_point": "model.py:Model"}],
                "payload": {"implementations": ["model.py", "test_model.py"], "resources": [], "configs": {}}}
    (source / ".theta_component.json").write_text(json.dumps(manifest))
    component = SimpleNamespace(id="leaky", kind="model", version="1.0.0")
    with pytest.raises(ValueError, match="model.py:1: imports private host module src.app.core.model_registry"):
        adapt(source, component, tmp_path / "staged")
    (source / "model.py").write_text("from src.app.api.models import register_model\n")
    adapt(source, component, tmp_path / "staged")  # its own tests may use the host


def test_undeclared_imports_follow_requirements_and_skip_optional_and_local_modules(tmp_path):
    (tmp_path / "helpers.py").write_text("")
    (tmp_path / "pkg.py").write_text(
        "import os, numpy, helpers\nimport undeclared_lib\nimport declared_lib.sub\n"
        "try:\n    import optional_lib\nexcept ImportError:\n    optional_lib = None\n"
        "def f(atari):\n    if atari:\n        import ale_py\n")
    manifest = {"dependencies": {"pip": ["declared-lib>=1"]}, "payload": {"implementations": ["pkg.py", "helpers.py"]}}
    assert undeclared_imports(manifest, tmp_path, ["numpy>=1.24"]) == [
        "pkg.py:2: imports undeclared_lib, which neither the application nor dependencies.pip provides"]


def test_bundled_plugins_do_not_reach_into_the_host_window():
    offenders = [f"{p.relative_to(ROOT)}:{n}" for p in guarded_files() for n in window_access(p)]
    assert offenders == []


def test_violations_report_private_imports_and_window_access(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("from src.app.api.models import register_model\n"
                                            "from src.app.core.types import ActionResult\n")
    (tmp_path / "pkg" / "b.py").write_text("def f(context):\n    return context.window\n")
    (tmp_path / "pkg" / "c.py").write_text("from src.usr.models.other.model import Net\n")
    assert violations(tmp_path) == ["pkg/a.py:2: imports private host module src.app.core.types",
                                    "pkg/b.py:2: reaches into the host window",
                                    "pkg/c.py:1: imports package module src.usr.models.other.model by path; use a declared export"]


def test_the_checker_rejects_private_imports(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import src.app.api.sites\nfrom src.app.api import desktop\n"
        "def f():\n    from src.app.sites.storage import safe_path\n"
        "    __import__('frontend.theme')\n    return ['-m', 'src.app.pipeline.task_worker']\n")
    assert [name for _, name in private_references(sample)] == [
        "src.app.sites.storage", "frontend.theme", "src.app.pipeline.task_worker"]


def test_sdk_exports_are_canonical_host_objects():
    from src.app.api import execution, models, sites, tuning, workspace
    from src.app.core import model_registry
    from src.app.objectives import api as objectives
    from src.app.pipeline import tuner_execution
    from src.app.sites import storage
    from src.app.workspace import WorkspacePaths

    assert workspace.WorkspacePaths is WorkspacePaths
    assert sites.safe_path is storage.safe_path
    assert tuning.TunerExecutionPlan is tuner_execution.TunerExecutionPlan
    assert tuning.ObjectiveResult is objectives.ObjectiveResult
    assert models.register_model is model_registry.register_model
    from src.app.core.execution import RunResult
    from src.app.monitoring.runtime import MonitorManager
    from src.app.api import monitoring
    assert execution.RunResult is RunResult and execution.RunResult is not tuning.RunResult
    assert monitoring.MonitorManager is MonitorManager
    from src.app.monitoring import api as monitor_api, measurements
    from src.app.core import data_loaders, execution as core_execution
    from src.app.api import data
    assert monitoring.Result is monitor_api.Result and monitoring.ModelAccess is monitor_api.ModelAccess
    assert monitoring.validate_params is measurements.validate_params
    assert execution.execute_backend is core_execution.execute_backend
    assert data.loader_selection is data_loaders.loader_selection
    from src.app.api import config as sdk_config, packages
    from src.app.core import requirements, provider_loader
    from src.app.api import SDK_VERSION
    assert SDK_VERSION == "3.4"
    assert sdk_config.compatibility is requirements.compatibility
    assert packages.load_selected is requirements.load_selected
    assert packages.resource_entries is provider_loader.resource_entries
    for removed in ("create_state_loader", "state_loader_choices", "input_requirements"):
        assert not hasattr(data, removed)
    with pytest.raises(ImportError):
        importlib.import_module("src.app.api.rulesets")
    from src.app.api import results
    from src.app.services import catalog
    assert results.list_runs is catalog.list_runs and results.read_run_file is catalog.read_run_file
    assert execution.task_worker_command("py", "a", 1) == ["py", "-u", "-m", "src.app.pipeline.task_worker", "a", "1"]
    for module in ("workspace", "errors", "config", "execution", "tuning", "sites", "models", "agents", "plotting",
                   "packages", "environments", "data", "evaluation", "telemetry", "monitoring", "results",
                   "desktop"):
        sdk = importlib.import_module(f"src.app.api.{module}")
        for name in sdk.__all__:
            assert getattr(sdk, name) is not None, f"src.app.api.{module}.{name}"
    with pytest.raises(AttributeError):
        sites.not_an_export


def test_headless_sdk_imports_without_qt_or_optional_libraries():
    import subprocess
    code = ("import sys\nfrom src.app.api import workspace, errors, config, execution, tuning, sites, packages,"
            " evaluation, telemetry, results\n"
            "for m in (workspace, errors, config, execution, tuning, sites, packages, evaluation, telemetry,"
            " results):\n"
            "    [getattr(m, n) for n in m.__all__]\n"
            "bad = [k for k in sys.modules if k.split('.')[0] in ('PyQt6', 'optuna', 'sqlalchemy')]\n"
            "assert not bad, bad\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("spec,ok", [(">=3.0,<4", True), ("~=3.0", True), (">=2", True), (">=2.0,<3", False), (None, False),
                                     (">=1.0,<2", False), ("~=1.0", False), ("not a range", False), ("", False), (3, False)])
def test_sdk_requirement_check(spec, ok):
    from src.app.api import sdk_requirement_error
    assert (sdk_requirement_error(spec) is None) is ok
