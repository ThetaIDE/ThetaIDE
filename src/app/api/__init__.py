"""Supported Theta extension SDK.

Packages (components, app plugins, authored user packages) import host
functionality from the modules of this package only:

``src.app.api.workspace``  workspace/state/application paths and guarded edits
``src.app.api.errors``     shared exception types
``src.app.api.config``     method parsing and editable configuration files
``src.app.api.execution``  engines, worker processes, interpreters and resource admission
``src.app.api.tuning``     portable tuner execution, objectives and trial artifacts
``src.app.api.models``     model provider registration and construction (loads PyTorch)
``src.app.api.agents``     Lightning agent base classes and registration (loads Lightning)
``src.app.api.environments`` vectorized environment base class
``src.app.api.data``       data modules, registration and transition datasets
``src.app.api.evaluation`` evaluator registration
``src.app.api.rulesets``   ruleset discovery and resolution
``src.app.api.telemetry``  run telemetry for engines outside Lightning
``src.app.api.monitoring`` mechanistic monitors for engines outside Lightning
``src.app.api.plotting``   plotter base class and shared figure styling
``src.app.api.packages``   declared cross-package exports and selected capabilities
``src.app.api.sites``      remote sites: profiles, transport, receipts, policy and publication
``src.app.api.desktop``    app-plugin base classes, host context and shared Qt widgets

Every export is the canonical host object (identity is preserved), loaded on
first attribute access so headless imports never pull in Qt or optional
libraries. Anything else under ``src.app`` or ``frontend`` is private host
implementation and may change without notice.

The SDK is versioned independently of package storage formats. Minor versions
only add exports or context services; a major version may remove them. App
plugins declare the range they were written for as ``"sdk": ">=2.0,<3"`` in
their capability metadata; the host refuses activation when it cannot satisfy
it.
"""
from __future__ import annotations

SDK_VERSION = "2.3"


def sdk_requirement_error(spec) -> str | None:
    """Explain why an SDK requirement is unsatisfied, or return None.

    Every app plugin must declare the range it was written for.
    """
    if spec is None:
        return "declares no sdk requirement; declare the range it was written for, such as '>=2.0,<3'"
    if not isinstance(spec, str) or not spec.strip():
        return "declares an invalid sdk requirement; use a PEP 440 range such as '>=2.0,<3'"
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import Version
    try:
        allowed = SpecifierSet(spec)
    except InvalidSpecifier:
        return f"declares an invalid sdk requirement {spec!r}; use a PEP 440 range such as '>=2.0,<3'"
    if Version(SDK_VERSION) not in allowed:
        return f"requires Theta SDK {spec}, but this host provides SDK {SDK_VERSION}; install a compatible release"
    return None
