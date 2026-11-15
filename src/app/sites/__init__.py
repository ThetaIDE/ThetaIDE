"""Site profiles, cluster submissions and explicit project/result transfers."""
from __future__ import annotations
from pathlib import Path


def site_name(cfg):
    return str(cfg.get('hydra', {}).get('runtime', {}).get('choices', {}).get('site')
               or (cfg.get('site') or {}).get('name') or 'local')


def scheduler(cfg):
    site = cfg.get('site') or {}
    if 'scheduler' not in site:
        name = site_name(cfg)
        if name == 'local' or name.startswith('local/'):
            return 'local'
        # Never run a remote profile locally by accident, or pick an installed scheduler implicitly.
        raise ValueError(f'Site profile {name!r} does not declare site.scheduler; set it to an installed '
                         'scheduler capability or to local.')
    value = site['scheduler']
    if not isinstance(value, str) or not value.strip():
        raise ValueError('site.scheduler must name a scheduler capability or local')
    return value


def profiles(paths, *, available=None):
    """Selectable profiles, grouped by their owning site; ``_``-prefixed files are shared defaults."""
    from .catalog import all_profiles
    available = all_profiles(paths) if available is None else available
    return {key: path for key, path in available.items() if not Path(key).name.startswith('_')}
