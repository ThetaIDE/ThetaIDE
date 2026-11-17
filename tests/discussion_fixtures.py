"""The Discussion tests' workspace: the base workspace plus ``tests/fixtures/discussion``.

``tests/fixtures/discussion`` is laid out like ``src/usr``: an experiment tree (``figs`` with a
``sweeps`` subfolder, and ``other``), the ``fixture_figures`` plot component (``fixture_curve``
provides ``figure/vega-lite-v1``; ``fixture_table`` does not), run metrics for
``figs/online`` and one saved figure with its discussion text.
"""

import shutil
from pathlib import Path

DISCUSSION_FIXTURE = Path(__file__).parent / "fixtures" / "discussion"


def build_discussion_workspace(root):
    from conftest import base_workspace
    root = Path(root)
    shutil.copytree(base_workspace(), root, dirs_exist_ok=True)
    shutil.copytree(DISCUSSION_FIXTURE, root / "src/usr", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return root
