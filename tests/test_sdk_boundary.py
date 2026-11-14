"""Optional components and bundled app plugins use only the supported Theta SDK."""
import ast
import importlib
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARDED = [ROOT / "optional_components", ROOT / "src/plugins/app/bundled"]
HOST = re.compile(r"^(src\.app|frontend)(\.|$)")
SDK = re.compile(r"^src\.app\.api(\.|$)")
# A string naming a private host module, e.g. a `python -m src.app.x` argument.
MODULE_STRING = re.compile(r"^(src\.app|frontend)(\.[A-Za-z_][A-Za-z0-9_]*)+$")


def guarded_files():
    return sorted(p for root in GUARDED for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def private_references(path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names = [node.module]
        elif isinstance(node, ast.Call):
            func = node.func
            called = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""
            if called in ("__import__", "import_module") and node.args and isinstance(node.args[0], ast.Constant):
                names = [str(node.args[0].value)]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and MODULE_STRING.match(node.value):
            names = [node.value]
        found += [(node.lineno, name) for name in names if HOST.match(name) and not SDK.match(name)]
    return sorted(set(found))


def test_guarded_trees_exist():
    assert len(guarded_files()) > 20


@pytest.mark.parametrize("path", guarded_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_imports_only_the_supported_sdk(path):
    assert private_references(path) == [], f"{path.relative_to(ROOT)} imports private host modules"


def test_bundled_plugins_do_not_reach_into_the_host_window():
    pattern = re.compile(r"\bcontext\.window\b|\bcontext\._window\b|\.context\.window\b")
    offenders = [f"{p.relative_to(ROOT)}:{n}" for p in guarded_files()
                 for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if pattern.search(line)]
    assert offenders == []


def test_the_checker_rejects_private_imports(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import src.app.api.sites\nfrom src.app.api import desktop\n"
        "def f():\n    from src.app.sites.storage import safe_path\n"
        "    __import__('frontend.theme')\n    return ['-m', 'src.app.pipeline.task_worker']\n")
    assert [name for _, name in private_references(sample)] == [
        "src.app.sites.storage", "frontend.theme", "src.app.pipeline.task_worker"]


def test_sdk_exports_are_canonical_host_objects():
    from src.app.api import execution, sites, tuning, workspace
    from src.app.pipeline import tuner_execution
    from src.app.sites import storage
    from src.app.workspace import WorkspacePaths

    assert workspace.WorkspacePaths is WorkspacePaths
    assert sites.safe_path is storage.safe_path
    assert tuning.TunerExecutionPlan is tuner_execution.TunerExecutionPlan
    assert execution.task_worker_command("py", "a", 1) == ["py", "-u", "-m", "src.app.pipeline.task_worker", "a", "1"]
    for module in ("workspace", "errors", "config", "execution", "tuning", "sites", "desktop"):
        sdk = importlib.import_module(f"src.app.api.{module}")
        for name in sdk.__all__:
            assert getattr(sdk, name) is not None, f"src.app.api.{module}.{name}"
    with pytest.raises(AttributeError):
        sites.not_an_export


def test_headless_sdk_imports_without_qt_or_optional_libraries():
    import subprocess
    code = ("import sys\nfrom src.app.api import workspace, errors, config, execution, tuning, sites\n"
            "for m in (workspace, errors, config, execution, tuning, sites):\n"
            "    [getattr(m, n) for n in m.__all__]\n"
            "bad = [k for k in sys.modules if k.split('.')[0] in ('PyQt6', 'optuna', 'sqlalchemy')]\n"
            "assert not bad, bad\n")
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("spec,ok", [(None, True), (">=1.0,<2", True), ("~=1.0", True),
                                     (">=2", False), ("not a range", False), ("", False), (3, False)])
def test_sdk_requirement_check(spec, ok):
    from src.app.api import sdk_requirement_error
    assert (sdk_requirement_error(spec) is None) is ok
