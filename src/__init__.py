"""Theta-IDE root package.

Separation:
- app: Core framework modules (core, pipeline, data, api, train, dataset_utils)
- usr: Installed or locally authored components (methods, models, environments, evaluators)
"""

# Authored and installed package modules import under src.plugins._workspace.
from src.app.core.provider_loader import install_namespace_finder as _install_namespace_finder
_install_namespace_finder()
