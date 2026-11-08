"""Theta-IDE root package.

Separation:
- app: Core framework modules (core, pipeline, data, api, train, dataset_utils)
- usr: Installed or locally authored components (methods, models, environments, evaluators)
"""

# Historical saved Python targets resolve to canonical bundled objects.
from src.app.core.import_aliases import install as _install_aliases
_install_aliases()
from src.app.core.provider_loader import install_namespace_finder as _install_provider_aliases
_install_provider_aliases()
