"""Supported Theta extension SDK.

Packages (components, app plugins, authored user packages) import host
functionality from the modules of this package only:

``src.app.api.workspace``  workspace/state/application paths and guarded edits
``src.app.api.errors``     shared exception types
``src.app.api.config``     method parsing and editable configuration files
``src.app.api.execution``  engines, worker processes, interpreters and resource admission
``src.app.api.tuning``     portable tuner execution, objectives and trial artifacts
``src.app.api.sites``      remote sites: profiles, transport, receipts, policy and publication
``src.app.api.desktop``    app-plugin base classes, host context and shared Qt widgets

Every export is the canonical host object (identity is preserved), loaded on
first attribute access so headless imports never pull in Qt or optional
libraries. Anything else under ``src.app`` or ``frontend`` is private host
implementation and may change without notice.

The SDK is versioned independently of package storage formats. Minor versions
only add exports or context services; a major version may remove them. App
plugins declare the range they were written for as ``"sdk": ">=1.0,<2"`` in
their capability metadata; the host refuses activation when it cannot satisfy
it. This namespace becomes ``src.ide.api`` at consolidation, with forwarding.
"""
from __future__ import annotations

SDK_VERSION = "1.0"


def sdk_requirement_error(spec) -> str | None:
    """Explain why an SDK requirement is unsatisfied, or return None.

    An absent requirement is accepted: packages written before the SDK was
    versioned target its first release.
    """
    if spec is None:
        return None
    if not isinstance(spec, str) or not spec.strip():
        return "declares an invalid sdk requirement; use a PEP 440 range such as '>=1.0,<2'"
    from packaging.specifiers import InvalidSpecifier, SpecifierSet
    from packaging.version import Version
    try:
        allowed = SpecifierSet(spec)
    except InvalidSpecifier:
        return f"declares an invalid sdk requirement {spec!r}; use a PEP 440 range such as '>=1.0,<2'"
    if Version(SDK_VERSION) not in allowed:
        return f"requires Theta SDK {spec}, but this host provides SDK {SDK_VERSION}; install a compatible release"
    return None
