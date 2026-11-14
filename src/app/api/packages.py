"""Cross-package exports.

    ``provider_export(package_id, name)``  load an object another package declares in its
                                           manifest ``exports``. The calling package must list
                                           ``package_id`` in ``dependencies.components``, and a
                                           declared version range must match the installed version.

See docs/reference/PLUGIN_STORAGE.md#cross-package-exports.
"""
from ._exports import lazy_exports

_EXPORTS = {"provider_export": "src.app.core.provider_loader:provider_export"}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
