"""Headless extension contributions, readiness and lifecycle reconciliation v1.

Discovery reads metadata only. Hosts mount contributions in slots they support;
providers do not gain access to private desktop objects through this contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
from typing import Callable

from src.app.workspace import WorkspacePaths
from .provider_discovery import discovery_cached

API_VERSION = 1
STATE_SCHEMA = 'theta.extension-state/v1'


@discovery_cached
def _distribution_available(name):
    try:
        version(name)
        return True
    except PackageNotFoundError:
        return False


def extension_key(package, capability):
    return f'{package.origin}:{package.id}:{capability.kind}:{capability.id}'


def _state(paths):
    path = paths.data_dir / 'extensions.json'
    value = json.loads(path.read_text()) if path.is_file() else {'schema': STATE_SCHEMA, 'disabled': []}
    if not isinstance(value, dict) or value.get('schema') != STATE_SCHEMA or not isinstance(value.get('disabled'), list) or any(not isinstance(x, str) for x in value['disabled']):
        raise ValueError('Invalid extension state; expected theta.extension-state/v1')
    return value


@dataclass(frozen=True)
class ExtensionStatus:
    state: str
    reason: str = ''

    @property
    def enabled(self):
        return self.state in ('setup', 'ready')


@dataclass(frozen=True)
class Contribution:
    package: object
    capability: object
    id: str
    slot: str
    label: str
    availability: str = 'ready'
    entry_point: str | None = None
    status: ExtensionStatus = ExtensionStatus('ready')

    @property
    def key(self):
        return extension_key(self.package, self.capability) + ':' + self.id

    @property
    def signature(self):
        return (str(self.package.root), self.package.version, self.capability.entry_point,
                self.id, self.slot, self.label, self.availability, self.entry_point, self.status)


def capability_status(package, capability, *, paths=None, config=None):
    paths = paths or WorkspacePaths.for_process()
    metadata = capability.metadata
    api = metadata.get('contribution_api')
    if api is not None and (type(api) is not int or api != API_VERSION):
        return ExtensionStatus('incompatible', f'{package.id}@{package.version} requires contribution API {api!r}; host supports {API_VERSION}')
    if extension_key(package, capability) in _state(paths)['disabled']:
        return ExtensionStatus('disabled', f'{capability.kind} {capability.id!r} is disabled in this workspace; enable it to use this recipe')
    requirements = metadata.get('readiness') or {}
    missing = []
    for path in requirements.get('config', []):
        value = config or {}
        for part in path.split('.'):
            value = value.get(part) if hasattr(value, 'get') else None
        if value is None or value == '' or value == {} or value == []:
            missing.append('configure ' + path)
    for distribution in requirements.get('distributions', []):
        if not _distribution_available(distribution):
            missing.append('install dependency ' + distribution)
    return ExtensionStatus('setup', '; '.join(missing)) if missing else ExtensionStatus('ready')


def ensure_enabled(package, capability, *, paths=None):
    """Runtime loading checks enablement/API without blocking setup editors."""
    status = capability_status(package, capability, paths=paths)
    if status.state in ('disabled', 'incompatible'):
        from src.app.pipeline.exceptions import ConfigurationError
        raise ConfigurationError(status.reason)


class ContributionRegistry:
    def __init__(self, paths=None):
        self.paths = paths or WorkspacePaths.for_process()

    def status(self, kind, selector, *, config=None):
        from .provider_catalog import get_catalog
        selected = get_catalog(self.paths).resolve(kind, selector)
        if selected is None:
            return ExtensionStatus('absent', f'{kind} {selector!r} is not installed; install a compatible provider to use this recipe')
        return capability_status(*selected, paths=self.paths, config=config)

    def contributions(self, slot=None, *, config=None):
        from .provider_catalog import get_catalog
        catalog = get_catalog(self.paths)
        found = []
        for package in catalog.packages:
            for capability in package.capabilities:
                if capability.metadata.get('contribution_api') != API_VERSION:
                    continue
                # Capability ownership overrides govern contributions as well as execution.
                if catalog.resolve(capability.kind, capability.id) != (package, capability):
                    continue
                status = capability_status(package, capability, paths=self.paths, config=config)
                if not status.enabled:
                    continue
                for item in capability.metadata.get('contributions', []):
                    availability = item.get('availability', 'ready')
                    if slot is not None and item['slot'] != slot:
                        continue
                    if availability == 'ready' and status.state != 'ready':
                        continue
                    if availability not in ('enabled', 'ready'):
                        raise ValueError('Contribution availability must be enabled or ready')
                    found.append(Contribution(package, capability, item['id'], item['slot'],
                        item.get('label', item['id']), availability, item.get('entry_point'), status))
        return sorted(found, key=lambda item: item.key)

    def set_enabled(self, kind, selector, enabled):
        if type(enabled) is not bool:
            raise ValueError('Extension enabled state must be a boolean')
        from .provider_catalog import get_catalog
        from src.app.components.transactions import workspace_lock, assert_idle, write_json
        with workspace_lock(self.paths):
            selected = get_catalog(self.paths).resolve(kind, selector)
            if selected is None:
                raise ValueError(f'Cannot change missing extension: {kind}:{selector}')
            package, capability = selected
            assert_idle(self.paths, package.id)
            state = _state(self.paths)
            key = extension_key(package, capability)
            disabled = set(state['disabled'])
            disabled.discard(key) if enabled else disabled.add(key)
            write_json(self.paths.data_dir / 'extensions.json', {**state, 'disabled': sorted(disabled)})
            from .provider_discovery import invalidate_discovery_snapshot
            invalidate_discovery_snapshot()


def load_contribution(contribution, context, *, paths=None):
    """Explicit activation, separate from headless discovery."""
    if not contribution.entry_point:
        raise ValueError('This contribution is rendered by the host and has no factory')
    paths = paths or WorkspacePaths.for_process()
    ensure_enabled(contribution.package, contribution.capability, paths=paths)
    if contribution.package.entrypoint is not None:
        raise ValueError('Package-relative contribution factories require a native package')
    from .provider_loader import load_module
    entry, _, attribute = contribution.entry_point.partition(':')
    factory = load_module(contribution.package, entry, paths=paths)
    for name in attribute.split('.'):
        factory = getattr(factory, name)
    if not callable(factory):
        raise ValueError('Contribution factory is not callable')
    return factory(context)


class ContributionScope:
    """Mount a host slot and release removed/disabled/replaced contributions."""
    def __init__(self, mount: Callable):
        self.mount = mount
        self.active = {}

    def reconcile(self, contributions):
        contributions = list(contributions)
        desired = {item.key: item for item in contributions}
        if len(desired) != len(contributions):
            raise ValueError('Duplicate contribution identity')
        errors = []
        for key in reversed(list(self.active)):
            signature, cleanup = self.active[key]
            if key not in desired or desired[key].signature != signature:
                del self.active[key]
                try:
                    if cleanup: cleanup()
                except Exception as exc: errors.append(exc)
        if errors:
            raise RuntimeError('Contribution cleanup failed: ' + '; '.join(str(exc) for exc in errors)) from errors[0]
        added = []
        try:
            for key, item in desired.items():
                if key in self.active:
                    continue
                cleanup = self.mount(item)
                if cleanup is not None and not callable(cleanup):
                    raise TypeError('Contribution mount must return a cleanup callback or None')
                self.active[key] = (item.signature, cleanup)
                added.append(key)
        except BaseException as original:
            cleanup_errors = []
            for key in reversed(added):
                _, cleanup = self.active.pop(key)
                try:
                    if cleanup: cleanup()
                except Exception as exc: cleanup_errors.append(exc)
            if cleanup_errors:
                raise RuntimeError('Contribution activation/rollback failed: ' + '; '.join(str(exc) for exc in cleanup_errors)) from original
            raise

    def close(self):
        self.reconcile([])
