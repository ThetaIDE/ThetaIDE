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
        # Before scheduler components, an undeclared scheduler meant Slurm. Never run such
        # a profile locally by accident, and never pick an installed scheduler implicitly.
        raise ValueError(f'Site profile {name!r} does not declare site.scheduler. Profiles written before '
                         "scheduler components implied 'slurm'; set scheduler to an installed scheduler "
                         'capability or to local.')
    value = site['scheduler']
    if not isinstance(value, str) or not value.strip():
        raise ValueError('site.scheduler must name a scheduler capability or local')
    return value


def profiles(paths, *, available=None):
    """Selectable profiles, grouped by their owning site; compatibility aliases stay hidden."""
    from .catalog import all_profiles, normalize
    available = all_profiles(paths) if available is None else available
    return {key: path for key, path in available.items()
            if not Path(key).name.startswith('_') and normalize(key, available) == key}
