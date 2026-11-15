"""Optional components, bundled component packages and bundled app plugins use only the supported Theta SDK."""
import importlib
import sys
from pathlib import Path

import pytest

from src.app.components.sdk_boundary import private_references, violations, window_access

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "optional_components", ROOT / "src/plugins/components/bundled", ROOT / "src/plugins/app/bundled"]


def guarded_files():
    return sorted(p for root in GUARDED for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_guarded_trees_exist():
    assert len(guarded_files()) > 20


@pytest.mark.parametrize("path", guarded_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_only_the_supported_sdk(path):
    assert private_references(path) == [], f"{path.relative_to(ROOT)} imports private host modules"


def test_bundled_plugins_do_not_reach_into_the_host_window():
    offenders = [f"{p.relative_to(ROOT)}:{n}" for p in guarded_files() for n in window_access(p)]
    assert offenders == []


def test_violations_report_private_imports_and_window_access(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("from src.app.api.models import register_model\n"
                                            "from src.app.core.types import ActionResult\n")
    (tmp_path / "pkg" / "b.py").write_text("def f(context):\n    return context.window\n")
    assert violations(tmp_path) == ["pkg/a.py:2: imports private host module src.app.core.types",
                                    "pkg/b.py:2: reaches into the host window"]


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
    assert data.create_state_loader is data_loaders.create_state_loader
    from src.app.api import results
    from src.app.services import catalog
    assert results.list_runs is catalog.list_runs and results.read_run_file is catalog.read_run_file
    assert execution.task_worker_command("py", "a", 1) == ["py", "-u", "-m", "src.app.pipeline.task_worker", "a", "1"]
    for module in ("workspace", "errors", "config", "execution", "tuning", "sites", "models", "agents", "plotting",
                   "packages", "environments", "data", "evaluation", "rulesets", "telemetry", "monitoring", "results",
                   "desktop"):
        sdk = importlib.import_module(f"src.app.api.{module}")
        for name in sdk.__all__:
            assert getattr(sdk, name) is not None, f"src.app.api.{module}.{name}"
    with pytest.raises(AttributeError):
        sites.not_an_export


def test_headless_sdk_imports_without_qt_or_optional_libraries():
    import subprocess
    code = ("import sys\nfrom src.app.api import workspace, errors, config, execution, tuning, sites, packages,"
            " evaluation, rulesets, telemetry, results\n"
            "for m in (workspace, errors, config, execution, tuning, sites, packages, evaluation, rulesets, telemetry,"
            " results):\n"
            "    [getattr(m, n) for n in m.__all__]\n"
            "bad = [k for k in sys.modules if k.split('.')[0] in ('PyQt6', 'optuna', 'sqlalchemy')]\n"
            "assert not bad, bad\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("spec,ok", [(">=2.0,<3", True), ("~=2.0", True), (">=2", True), (None, False),
                                     (">=1.0,<2", False), ("~=1.0", False), ("not a range", False), ("", False), (3, False)])
def test_sdk_requirement_check(spec, ok):
    from src.app.api import sdk_requirement_error
    assert (sdk_requirement_error(spec) is None) is ok
