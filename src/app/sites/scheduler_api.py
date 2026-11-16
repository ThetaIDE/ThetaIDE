"""Selected optional scheduler API v2. Core retains receipt, bundle and artifact policy."""
from src.app.workspace import WorkspacePaths


def scheduler_selector(*, cfg=None, record=None):
    if record is not None:
        if 'scheduler_component' not in record:
            raise ValueError('Unsupported site receipt: it records no scheduler_component')
        selector = record['scheduler_component']
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
    if capability.metadata.get('scheduler_api') != 2:
        raise ValueError(f'Scheduler {selector!r} requires scheduler_api=2')
    status = capability_status(package, capability, paths=paths, config=cfg or (record or {}).get('config') or {})
    if not status.enabled:
        raise ValueError(f'Scheduler {selector!r}: {status.reason}')
    provider = load_capability(package, capability, paths=paths)
    provider = provider() if isinstance(provider, type) else provider
    if getattr(provider, 'api_version', None) != 2:
        raise ValueError(f'Scheduler {selector!r} requires runtime api_version=2')
    return provider
