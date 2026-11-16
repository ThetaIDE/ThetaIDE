"""Tracked source describes only the current workspace layout."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Both the path form (in/envs) and the module form (in.envs) of the retired roots.
RETIRED_PATHS = re.compile(r"(?<![\w.])in[/.](config|datasets|envs|inputs)\b")
# Tests may build rejected layouts, and documentation records history (its current
# guides are swept by hand). Results are run outputs, not source.
EXEMPT = ("tests/", "docs/", "src/usr/results/")


def tracked_text_files():
    try:
        listed = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("the layout guard needs a Git checkout")
    for name in filter(None, listed.decode().split("\0")):
        path = ROOT / name
        if name.startswith(EXEMPT) or not path.is_file() or path.is_symlink():
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary payloads carry no paths worth guarding


def test_tracked_source_never_mentions_retired_workspace_paths():
    offenders = [f"{name}:{number}: {line.strip()[:120]}"
                 for name, text in tracked_text_files()
                 for number, line in enumerate(text.splitlines(), start=1)
                 if RETIRED_PATHS.search(line)]
    assert not offenders, "use src/usr/config or src/usr/datasets:\n" + "\n".join(offenders)


# A results path spelled as a literal: "results/...", Path("results"), root / "results", or a
# SQLite URL under it. `WorkspacePaths.results` is the one definition.
RESULTS_LITERAL = re.compile(r"""["']results/|/\s*["']results["']|Path\(\s*["']results|sqlite:///results""")
RESULTS_OWNER = "src/app/workspace.py"
# Verbatim third-party research source, mirrored for provenance.
VERBATIM = ("src/usr/packages/themes-dev/reference/",)


def test_results_paths_come_only_from_the_workspace():
    offenders = [f"{name}:{number}: {line.strip()[:120]}"
                 for name, text in tracked_text_files()
                 if name.endswith((".py", ".yaml", ".yml")) and name != RESULTS_OWNER and not name.startswith(VERBATIM)
                 for number, line in enumerate(text.splitlines(), start=1)
                 if RESULTS_LITERAL.search(line)]
    assert not offenders, "derive results paths from WorkspacePaths.results:\n" + "\n".join(offenders)


# The repository root joined with src/usr (ROOT / "src/usr/...", parents[1] / "src/usr", or a
# working-directory-relative Path("src/usr...")): the developer's own workspace.
CHECKOUT_USR = re.compile(r"""((\b(ROOT|REPO|PROJECT_ROOT|project)|parents\[1\]|parent\.parent)\s*/\s*["']|Path\(\s*["'])src/usr(?!/results)""")
# Lint over tracked repository files, not a workspace read.
REPOSITORY_LINT = ("tests/test_sdk_boundary.py", "tests/test_workspace_defaults.py", "tests/test_layout_guard.py")


def test_tests_never_read_the_checkout_workspace():
    """Tests run in workspaces built from tests/fixtures; the developer's src/usr is theirs."""
    tests = ROOT / "tests"
    offenders = [f"{path.relative_to(ROOT)}:{number}: {line.strip()[:120]}"
                 for path in sorted(tests.rglob("*.py")) if "fixtures" not in path.relative_to(tests).parts
                 and str(path.relative_to(ROOT)) not in REPOSITORY_LINT
                 for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
                 if CHECKOUT_USR.search(line)]
    assert not offenders, "build the workspace from tests/fixtures instead:\n" + "\n".join(offenders)
