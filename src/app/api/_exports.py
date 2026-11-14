"""Lazy, identity-preserving re-exports for the SDK modules."""
from __future__ import annotations

import importlib


def lazy_exports(module_name: str, table: dict[str, str]):
    """Return ``(__getattr__, __dir__)`` resolving ``name -> "module:attr"`` on access.

    Nothing is cached, so a host object replaced on its defining module (for
    example by a test monkeypatch) is what the next lookup returns.
    """
    def __getattr__(name):
        try:
            target = table[name]
        except KeyError:
            raise AttributeError(f"module {module_name!r} has no attribute {name!r}") from None
        module, _, attr = target.partition(":")
        loaded = importlib.import_module(module)
        return getattr(loaded, attr) if attr else loaded

    def __dir__():
        return sorted(table)

    return __getattr__, __dir__
