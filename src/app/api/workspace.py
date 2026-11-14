"""Workspace, state and application roots, and guarded edits of managed files.

``WorkspacePaths``       the selected workspace; never infer it from the CWD.
``implementation_edit``  context manager guarding an edit of a package or authored
                         implementation file (leases, journal and rollback).
"""
from ._exports import lazy_exports

_EXPORTS = {
    "WorkspacePaths": "src.app.workspace:WorkspacePaths",
    "implementation_edit": "src.app.components.transactions:implementation_edit",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
