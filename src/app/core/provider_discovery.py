"""Operation-scoped metadata reuse, without a persistent filesystem cache."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps


@dataclass
class DiscoverySnapshot:
    manifests: dict = field(default_factory=dict)
    catalogs: dict = field(default_factory=dict)
    values: dict = field(default_factory=dict)


_SNAPSHOT = ContextVar("provider_discovery_snapshot", default=None)


def current_snapshot():
    return _SNAPSHOT.get()


def discovery_cached(function):
    """Reuse pure metadata calculations with hashable arguments during a build.

    Copies keep caller edits out of cached profiles and metadata. Calls outside
    a discovery scope remain fresh; errors are never retained.
    """
    @wraps(function)
    def cached(*args, **kwargs):
        snapshot = current_snapshot()
        if snapshot is None:
            return function(*args, **kwargs)
        key = (function, args, tuple(sorted(kwargs.items())))
        if key not in snapshot.values:
            snapshot.values[key] = deepcopy(function(*args, **kwargs))
        return deepcopy(snapshot.values[key])
    return cached


@contextmanager
def provider_discovery_snapshot():
    """Reuse discovery during one synchronous read operation.

    Nested operations share their caller's snapshot. On exit (including errors),
    the outer operation discards it, so the next form build discovers external
    additions, removals and edits. Do not hold this scope across event-loop turns.
    """
    if current_snapshot() is not None:
        yield
        return
    token = _SNAPSHOT.set(DiscoverySnapshot())
    try:
        yield
    finally:
        _SNAPSHOT.reset(token)


def invalidate_discovery_snapshot():
    """Lifecycle mutations must also invalidate an enclosing read operation."""
    snapshot = current_snapshot()
    if snapshot is not None:
        snapshot.manifests.clear()
        snapshot.catalogs.clear()
        snapshot.values.clear()
