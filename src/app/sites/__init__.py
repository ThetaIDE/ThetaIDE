"""Site profiles, cluster submissions and explicit project/result transfers."""
from __future__ import annotations
from pathlib import Path


def site_name(cfg):
    return str(cfg.get('hydra', {}).get('runtime', {}).get('choices', {}).get('site')
               or (cfg.get('site') or {}).get('name') or 'local')


def scheduler(cfg):
    name = site_name(cfg)
    value = (cfg.get('site') or {}).get('scheduler', 'local' if name == 'local' else 'slurm')
    if value not in ('local', 'slurm'):
        raise ValueError(f'Unsupported site scheduler: {value!r}')
    return value


def profiles(paths):
    """Selectable profiles, grouped by their owning site; compatibility aliases stay hidden."""
    from .catalog import all_profiles, normalize
    available = all_profiles(paths)
    return {key: path for key, path in available.items()
            if not Path(key).name.startswith('_') and normalize(key, available) == key}
