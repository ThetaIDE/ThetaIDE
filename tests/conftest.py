import functools
import os
import shutil
import sys
from pathlib import Path

import pytest

try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft < 4096:
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(4096, hard), hard))
except Exception:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TEST_WORKSPACE_CONFIG = Path(__file__).parent / "fixtures" / "workspace_config"
TEST_WORKSPACE_USR = Path(__file__).parent / "fixtures" / "workspace_usr"

for p in [
    str(PROJECT_ROOT),
    str(SRC_DIR),
    str(SRC_DIR / "app"),
    str(SRC_DIR / "usr"),
    str(SRC_DIR / "usr" / "models"),
    str(SRC_DIR / "usr" / "environments"),
    str(SRC_DIR / "usr" / "eval"),
]:
    if p not in sys.path:
        sys.path.insert(0, p)



_RESULT_KINDS = ("logs", "checkpoints", "plots", "executions", "analysis")


def _checkout_result_groups():
    from src.app.workspace import WorkspacePaths
    results = WorkspacePaths(PROJECT_ROOT).results
    return {f"{kind}/{group.name}" for kind in _RESULT_KINDS
            for group in (results / kind).glob("*") if group.is_dir()}


def pytest_sessionstart(session):
    session.config._theta_result_groups = _checkout_result_groups()


def pytest_sessionfinish(session, exitstatus):
    """Fail a run whose tests wrote results into the checkout instead of their own workspace."""
    leaked = sorted(_checkout_result_groups() - session.config._theta_result_groups)
    if leaked:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter is not None:
            reporter.write_line("Tests wrote results into the checkout: " + ", ".join(
                f"src/usr/results/{name}" for name in leaked), red=True)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@functools.cache
def checkout_workspace():
    """A workspace holding exactly this checkout's tracked ``src/usr`` files.

    Fresh and populated checkouts look the same here: packages the developer installed,
    ignored recipes, rules and datasets, and ``.thetaide`` state are absent. Built once per
    test session; tests that change workspace content use their own ``tmp_path`` workspace.
    """
    import atexit
    import subprocess
    import tempfile
    root = Path(tempfile.mkdtemp(prefix="theta-checkout-workspace-"))
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    listed = subprocess.run(["git", "ls-files", "-z", "--", "src/usr"], cwd=PROJECT_ROOT,
                            capture_output=True, check=True).stdout.split(b"\0")
    for name in filter(None, (item.decode() for item in listed)):
        source = PROJECT_ROOT / name
        if source.is_file() and not source.is_symlink():
            (root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, root / name)
    return root


@pytest.fixture(scope="session", autouse=True)
def _session_workspace():
    """Select the checkout workspace before module- and class-scoped fixtures run."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("THETA_WORKSPACE", str(checkout_workspace()))
        patch.delenv("THETA_DATA_DIR", raising=False)
        yield


@pytest.fixture(autouse=True)
def tracked_workspace(_session_workspace, monkeypatch):
    """Select the checkout workspace as every test's process workspace.

    ``WorkspacePaths.for_process()``, ``get_catalog()``, Hydra composition and worker
    subprocesses then never resolve the developer's own ``src/usr``, installed packages or
    ``.thetaide`` state. Tests that need other packages build their own workspace and select
    it with ``THETA_WORKSPACE``.
    """
    root = checkout_workspace()
    monkeypatch.setenv("THETA_WORKSPACE", str(root))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    return root


def copy_research_config(target):
    """Copy every non-bundled profile the checkout workspace resolves into ``target``.

    Bundled profiles still come from the application. Tracked user files and authored
    package profiles become editable files of an isolated workspace; the developer's
    installed packages and ignored recipes are never read.
    """
    from src.app.core.provider_catalog import DuplicateProvider
    from src.app.workspace import WorkspacePaths
    target = Path(target)
    resolver = WorkspacePaths(checkout_workspace()).config_resolver()
    for selector in resolver.selectors():
        try:
            resolved = resolver.resolve(selector)
        except DuplicateProvider:
            continue
        if resolved is None or resolved.origin in ("bundled", "framework"):
            continue
        destination = target / selector
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(resolved.file, destination)
    return target


def build_fixture_workspace(root):
    """Copy the checkout workspace plus test profiles and fixture packages into ``root``.

    The tracked packages (``sb3-dev``, ``d3rlpy-dev``, ``gym_classic_control_dev``,
    ``blendrl-dev``) provide the algorithms, environment and composite model the fixture
    recipes select; ``test-models`` provides ``dnn``.
    """
    from src.app.workspace import WorkspacePaths
    root = Path(root)
    shutil.copytree(checkout_workspace(), root, dirs_exist_ok=True)
    shutil.copytree(TEST_WORKSPACE_USR, root / "src/usr", dirs_exist_ok=True)
    shutil.copytree(TEST_WORKSPACE_CONFIG, WorkspacePaths(root).config, dirs_exist_ok=True)
    return root


@pytest.fixture
def fixture_workspace(tmp_path, monkeypatch):
    """An isolated workspace: see ``build_fixture_workspace``."""
    build_fixture_workspace(tmp_path)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    return tmp_path


@pytest.fixture(scope="session")
def _optuna_workspace_root(tmp_path_factory):
    from component_fixtures import install_optional
    base = tmp_path_factory.mktemp("optuna-workspace")
    root = build_fixture_workspace(base / "workspace")
    install_optional(root, "optuna", base / "optional-archives")
    return root


@pytest.fixture
def optuna_workspace(_optuna_workspace_root, monkeypatch):
    """The fixture workspace with the optional Optuna tuner installed, selected for the process.

    Built once per session: tests using it must not change its files.
    """
    root = _optuna_workspace_root
    monkeypatch.setenv("THETA_WORKSPACE", str(root))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    try:
        from frontend.config_panel.catalogs import invalidate_config_tree
    except ImportError:  # no GUI stack installed
        yield root
        return
    invalidate_config_tree()
    yield root
    invalidate_config_tree()


@pytest.fixture
def tuner_monitor_workspace(tmp_path):
    """A ready independent monitor contribution, with provider imports forbidden."""
    from src.app.workspace import WorkspacePaths
    root = tmp_path / 'src/usr/packages/lattice'
    shutil.copytree(PROJECT_ROOT / 'tests/fixtures/contribution_tuner', root)
    (root / 'provider.py').write_text('raise AssertionError("monitor imported tuner")')
    return (WorkspacePaths(tmp_path, application=tmp_path / 'empty-app'),
            {'tuning': {'component': 'lattice', 'options': {'lattice': {'token': 'configured'}}}})


@pytest.fixture
def optional_tuner_metadata(monkeypatch):
    """Exercise bundled field metadata without requiring optional training libraries."""
    from src.app.core import contributions
    available = contributions._distribution_available
    monkeypatch.setattr(contributions, '_distribution_available',
                        lambda name: name in ('optuna', 'hydra-optuna-sweeper', 'sqlalchemy') or available(name))


# Qt requires QtWebEngineWidgets to be imported before any QApplication exists.
# Test modules build one at import time, so whichever module pytest collects
# first would otherwise decide whether the web engine is usable for all of them
# — and the modules that need it fail to import and skip silently rather than
# failing. Importing it here, before any test module is collected, removes the
# ordering dependency.
try:
    from PyQt6 import QtWebEngineWidgets  # noqa: F401
except ImportError:
    pass


@pytest.fixture(autouse=True)
def _delete_leftover_widgets():
    """Free the windows a test leaves behind.

    close() only hides a widget, and deleteLater() is not honoured without a
    running event loop, so every Window a test builds (~800 widgets) used to
    live until the session ended. app.setStyleSheet() restyles every live
    widget, so each new Window got slower until GUI tests hit the timeout.
    """
    yield
    if "PyQt6.QtWidgets" not in sys.modules:
        return
    from PyQt6.QtCore import QEvent
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return
    for widget in app.topLevelWidgets():
        widget.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


@pytest.fixture(autouse=True)
def _no_hub_network(monkeypatch):
    """Keep the Component Hub offline in tests.

    Opening a HubView starts fetch_index_async: a thread that downloads the registry from
    GitHub and signals the view when it is done. Tests close the view straight away, so the
    thread outlived it (and wrote a .thetaide cache into the workspace); on CI that aborted
    the whole run. Tests that need components set client.components themselves.
    """
    if "frontend.hub.client" not in sys.modules:
        try:
            import frontend.hub.client  # noqa: F401
        except ImportError:  # no GUI stack installed
            return
    monkeypatch.setattr(
        sys.modules["frontend.hub.client"].HubClient, "fetch_index_async", lambda self, force=False: None
    )


@pytest.fixture
def own_workspace(tmp_path, monkeypatch):
    """Select a copy of the checkout workspace in ``tmp_path``, so a run's results land in ``tmp_path``."""
    shutil.copytree(checkout_workspace(), tmp_path, dirs_exist_ok=True)
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    return tmp_path


@pytest.fixture
def workspace_results(tmp_path, monkeypatch):
    """Select ``tmp_path`` as the process workspace and return its results folder."""
    from src.app.workspace import WorkspacePaths
    monkeypatch.setenv("THETA_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("THETA_DATA_DIR", raising=False)
    return WorkspacePaths.for_process().results


@pytest.fixture
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def tmp_results(tmp_path):
    """Create a temporary results directory structure."""
    for sub in ["logs", "plots", "checkpoints", "datasets"]:
        (tmp_path / sub).mkdir()
    return tmp_path
