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

collect_ignore_glob = ["in/envs/*"]


@pytest.fixture
def fixture_workspace(tmp_path):
    """Build test configs outside the user's workspace, even if tests are interrupted."""
    target = tmp_path / "in" / "config"
    shutil.copytree(PROJECT_ROOT / "in" / "config", target)
    # Hub source profiles can be read-only. These are isolated test copies and
    # must be editable before overlaying fixtures or exercising editor saves.
    for path in target.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o200)
    shutil.copytree(TEST_WORKSPACE_CONFIG, target, dirs_exist_ok=True)
    return tmp_path


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
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config_dir():
    """Return the Hydra config directory."""
    return PROJECT_ROOT / "in" / "config"


@pytest.fixture
def tmp_results(tmp_path):
    """Create a temporary results directory structure."""
    for sub in ["logs", "plots", "checkpoints", "datasets"]:
        (tmp_path / sub).mkdir()
    return tmp_path
