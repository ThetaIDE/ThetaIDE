"""Backend component hot-reloading service for Theta-IDE.

Allows researchers to modify model layers, loss functions, environment step logic,
or data modules on disk and reload them live into the running process without
restarting Theta-IDE.
"""

from __future__ import annotations

import logging
from typing import List

log = logging.getLogger(__name__)


def reload_component_modules(target_prefix: str | None = None) -> List[str]:
    """Invalidate the loaded modules of the workspace's authored packages.

    Args:
        target_prefix: Optional provider module prefix; only the authored package
            whose prefix matches is reloaded. If None, every authored package is.

    Returns:
        The module prefixes of the invalidated packages; each loads fresh on next use.
    """
    from src.app.components.transactions import assert_idle, workspace_lock
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import invalidate, module_prefix
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths.for_process()
    with workspace_lock(paths):
        assert_idle(paths)
        reloaded = []
        for package in get_catalog(paths).packages:
            if package.origin != "user":
                continue
            prefix = module_prefix(package, paths)
            if target_prefix and not (prefix.startswith(target_prefix) or target_prefix.startswith(prefix)):
                continue
            invalidate(paths, package.root, package.id)
            reloaded.append(prefix)
        return reloaded
