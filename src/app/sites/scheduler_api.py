"""Selected optional scheduler API v1. Core retains receipt and artifact policy."""
from src.app.workspace import WorkspacePaths


def scheduler_selector(*, cfg=None, record=None):
    if record is not None:
        if 'scheduler_component' in record:
            selector = record['scheduler_component']
        elif record.get('schema_version') == 1:
            # Bounded reader for already-created Slurm receipts, never new recipes.
            selector = 'slurm'
        else:
            raise ValueError('Receipt needs an explicit compatible scheduler_component')
    else:
        selector = ((cfg or {}).get('site') or {}).get('scheduler')
    if not isinstance(selector, str) or not selector.strip() or selector == 'local':
        raise ValueError('Select an installed scheduler using site.scheduler')
    return selector


def get_scheduler(*, cfg=None, record=None, paths=None):
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_capability
    from src.app.core.contributions import capability_status
    paths = paths or WorkspacePaths.for_process()
    selector = scheduler_selector(cfg=cfg, record=record)
    selected = get_catalog(paths).resolve('scheduler', selector)
    if selected is None:
        raise ValueError(f'Scheduler {selector!r} is not installed in {paths.root}; install its optional component before running this recipe')
    package, capability = selected
    if capability.metadata.get('scheduler_api') != 1:
        raise ValueError(f'Scheduler {selector!r} requires scheduler_api=1')
    status = capability_status(package, capability, paths=paths, config=cfg or (record or {}).get('config') or {})
    if not status.enabled:
        raise ValueError(f'Scheduler {selector!r}: {status.reason}')
    provider = load_capability(package, capability, paths=paths)
    provider = provider() if isinstance(provider, type) else provider
    if getattr(provider, 'api_version', None) != 1:
        raise ValueError(f'Scheduler {selector!r} requires runtime api_version=1')
    return provider


def scheduler_object(selector, entry, *, paths=None):
    """Fixed API-1 compatibility imports, resolved through installed ownership."""
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import load_module
    from src.app.core.contributions import ensure_enabled
    paths = paths or WorkspacePaths.for_process()
    selected = get_catalog(paths).resolve('scheduler', selector)
    if selected is None:
        raise ImportError(f'Optional scheduler {selector!r} is unavailable; install its component')
    package, capability = selected
    ensure_enabled(package, capability, paths=paths)
    file, attribute = entry.split(':', 1)
    module = load_module(package, file, paths=paths)
    return getattr(module, attribute)
