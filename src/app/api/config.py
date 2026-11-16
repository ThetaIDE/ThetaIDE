"""Recipe and configuration helpers.

``parse_methods_dict``  the enabled methods of a recipe with inherited settings.
``ConfigurationFiles``  read/write access to editable profiles, creating user
                        overrides rather than editing provider defaults.
``compatibility(cfg, *, paths)``
                        the compatibility findings of a composed configuration: every
                        selected capability's ``requires`` rules and ``preflight`` hook,
                        per method (``.capability``, ``.method``, ``.level``, ``.message``).
"""
from ._exports import lazy_exports

_EXPORTS = {
    "parse_methods_dict": "src.app.pipeline.config:parse_methods_dict",
    "ConfigurationFiles": "src.app.components.configuration:ConfigurationFiles",
    "compatibility": "src.app.core.requirements:compatibility",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
