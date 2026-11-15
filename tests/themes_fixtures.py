"""The tracked themes-dev package in a test workspace, with its local inputs generated.

`resources/synthetic.npz` and `resources/synthetic_time.csv` are demonstration inputs
kept out of Git (see the package's `resources/.gitignore`). Tests copy the tracked
package from the checkout workspace and generate both with the package's own generators.
"""
import shutil
import sys
from pathlib import Path

from conftest import checkout_workspace

PACKAGE = checkout_workspace() / 'src/usr/packages/themes-dev'
FIXTURES = Path(__file__).parent / 'fixtures'

if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))


def themes_package(workspace_root):
    """Copy themes-dev into `workspace_root` and generate its resources; return the package directory."""
    from themes_al.synthetic import write_resources
    target = Path(workspace_root) / 'src/usr/packages/themes-dev'
    shutil.copytree(PACKAGE, target, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build'))
    write_resources(target / 'resources')
    return target


def parity_fixture():
    """Golden outputs of the original THEMES source, kept local; skip when this checkout lacks them."""
    import pytest
    path = FIXTURES / 'themes_parity.npz'
    if not path.is_file():
        pytest.skip('tests/fixtures/themes_parity.npz holds local golden outputs of the original THEMES source')
    return path
