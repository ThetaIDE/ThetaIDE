"""Cross-package exports.

    ``provider_export(package_id, name)``  load an object another package declares in its
                                           manifest ``exports``. The calling package must list
                                           ``package_id`` in ``dependencies.components``, and a
                                           declared version range must match the installed version.
    ``configured_export(reference)``       load the export a configuration value names as
                                           ``export://<package>/<name>`` (for hooks that users
                                           select in profiles); no dependency declaration is needed.
    ``environment_resource(environment, relative)``
                                           a file under the ``resource_root`` an environment
                                           capability declares, for packages that pair with it.
    ``selected_capability(kind, selector)``
                                           the provider capability a configured selector names
                                           (``.id``, ``.kind``, ``.metadata``, ``.load()``, and its
                                           package's ``.package_id``, ``.package_version`` and
                                           ``.package_sha256``), or None, for runners that accept
                                           an independently selected component.

See docs/reference/PLUGIN_STORAGE.md#cross-package-exports.
"""
from ._exports import lazy_exports

_EXPORTS = {name: f"src.app.core.provider_loader:{name}" for name in ("provider_export", "configured_export", "environment_resource", "selected_capability")}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
