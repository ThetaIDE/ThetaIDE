"""Tracked source describes only the current workspace layout."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RETIRED_PATHS = re.compile(r"\bin/(config|datasets|envs|inputs)\b")
# Tests may build rejected layouts, and documentation records history (its current
# guides are swept by hand). Results are run outputs, not source.
EXEMPT = ("tests/", "docs/", "results/")


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
