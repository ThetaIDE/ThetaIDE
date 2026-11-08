"""Backend component hot-reloading service for Theta-IDE.

Allows researchers to modify model layers, loss functions, environment step logic,
or data modules on disk and reload them live into the running process without
restarting Theta-IDE.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import List, Optional

log = logging.getLogger(__name__)


def reload_component_modules(target_prefix: str | None = None) -> List[str]:
    """Reload active research modules in sys.modules and refresh component registries.

    Args:
        target_prefix: Optional module name prefix to reload (e.g. 'src.usr.models.custom_model').
            If None, reloads all active user research modules ('src.usr.' and 'in.envs.').

    Returns:
        List of successfully reloaded module names.
    """
    from src.app.components.transactions import assert_idle, workspace_lock
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import invalidate, module_prefix
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths.for_process()
    with workspace_lock(paths):
        assert_idle(paths)
        if target_prefix:
            # Explicit legacy decorator modules can lack catalog metadata. Keep this
            # bounded compatibility path without scanning/importing arbitrary providers.
            legacy = [(name, module) for name, module in list(sys.modules.items())
                      if module is not None and name.startswith(target_prefix) and name.startswith(('src.usr.', 'usr.', 'in.envs.'))]
            if legacy:
                reloaded = []
                for name, module in legacy:
                    source = getattr(module, '__file__', None)
                    if source:
                        from pathlib import Path
                        owner = next((p for p in get_catalog(paths).packages if p.origin != 'python_distribution' and Path(source).is_relative_to(p.root)), None)
                        if owner and not owner.writable:
                            raise ValueError(f'Managed provider {owner.id} cannot be hot-edited; create an editable copy')
                    if getattr(module, '__name__', '').startswith('src.app.'):
                        continue  # compatibility contracts keep their shared registry/class identity
                    if source and owner:
                        invalidate(paths, owner.root, owner.id)
                        from src.app.core.provider_loader import load_capability
                        for cap in owner.capabilities:
                            if cap.kind == 'objective': continue
                            try: load_capability(owner, cap, paths=paths)
                            except ImportError as exc: log.debug('Optional provider unavailable after reload: %s',exc)
                    else:
                        importlib.reload(module)
                    reloaded.append(name)
                return reloaded
        packages = [p for p in get_catalog(paths).packages if p.origin == "user"]
        reloaded = []
        for package in packages:
            prefix = module_prefix(package, paths)
            aliases = [prefix]
            if target_prefix and not any(alias.startswith(target_prefix) or target_prefix.startswith(alias) for alias in aliases):
                continue
            invalidate(paths, package.root, package.id)
            reloaded.append(prefix)
        return reloaded
