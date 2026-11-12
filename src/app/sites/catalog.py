"""Component-owned site defaults and named Hydra profiles."""
from pathlib import Path
import json
import yaml
from omegaconf import OmegaConf
from .storage import safe_path

ALIASES = {'ncshare_common': 'ncshare/common', 'ncshare_gpu': 'ncshare/gpu'}


def all_profiles(paths):
    result = {}
    for root in paths.config_roots():
        directory = root / 'site'
        if directory.is_dir():
            for path in sorted(directory.rglob('*.yaml')):
                result.setdefault(path.relative_to(directory).with_suffix('').as_posix(), path)
    return result


def normalize(selector, available):
    selector = ALIASES.get(selector, selector)
    if '/' not in selector and selector + '/default' in available:
        selector += '/default'
    return selector


def load(root, selector):
    from src.app.workspace import WorkspacePaths
    available = all_profiles(WorkspacePaths(root))
    selector = normalize(selector, available)
    def read(name, parents=()):
        if name in parents: raise ValueError('Cyclic site defaults')
        if name not in available: raise ValueError(f'Unknown site profile: {name}')
        data = OmegaConf.load(available[name])
        merged = OmegaConf.create({})
        for entry in data.get('defaults', []):
            if entry == '_self_':
                merged = OmegaConf.merge(merged, OmegaConf.masked_copy(data, [k for k in data if k != 'defaults']))
            elif isinstance(entry, str):
                dependency = entry.removeprefix('/site/') if entry.startswith('/site/') else entry
                merged = OmegaConf.merge(merged, read(dependency, (*parents, name)))
        if '_self_' not in data.get('defaults', []):
            merged = OmegaConf.merge(merged, OmegaConf.masked_copy(data, [k for k in data if k != 'defaults']))
        return merged
    return OmegaConf.to_container(read(selector), resolve=True)


def defaults_path(root, site):
    component = ALIASES.get(site, site).split('/')[0]
    return safe_path(root, f'src/usr/sites/{component}/configs/site/{component}/_defaults.yaml')


def create(root, name, data=None, variants=None):
    component = safe_path(root, 'src/usr/sites/' + name)
    directory = component / 'configs/site' / name
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(data or {'scheduler': 'slurm', 'venv_dir': 'venv', 'resources': {'nodes': 1, 'cores': 4, 'gpus': 0, 'memory': '8G', 'time': '01:00:00'}})
    data.setdefault('connection', {'host': None, 'remote_root': None, 'python': 'venv/bin/python3',
        'ssh_config': f'src/usr/sites/{name}/ssh_config', 'ssh_connect_timeout_seconds': 60,
        'rsync_executable': 'rsync', 'remote_rsync': 'rsync'})
    data.setdefault('shortcuts', {'directory': f'src/usr/sites/{name}/bin'})
    (directory / '_defaults.yaml').write_text('# @package site\n' + yaml.safe_dump(data, sort_keys=False))
    for profile, overrides in {'default': {}, **(variants or {})}.items():
        value = {'defaults': [f'/site/{name}/_defaults', '_self_'], **overrides}
        (directory / (profile + '.yaml')).write_text('# @package site\n' + yaml.safe_dump(value, sort_keys=False))
    (component / 'site.py').write_text('"""Site metadata; execution uses the shared scheduler adapter."""\nSITE = ' + repr({'scheduler': data.get('scheduler', 'slurm')}) + '\n')
    (component / 'ssh_config').write_text('# Site-owned OpenSSH aliases and overrides. Private keys stay outside the project.\nInclude ~/.ssh/config\n')
    configs = {f'in/config/site/{name}/{p.name}': f'configs/site/{name}/{p.name}' for p in directory.glob('*.yaml')}
    manifest = {'id': name + '-site', 'name': name, 'kind': 'site', 'version': '1.0.0', 'storage_version': 2, 'api_version': 1,
                'capabilities': [{'id': name, 'kind': 'site', 'entry_point': 'site.py:SITE'}],
                'config_paths': list(configs), 'payload': {'implementations': ['site.py'], 'resources': ['ssh_config'], 'configs': configs}}
    (component / '.theta_component.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return component


def configure(root, selector, values):
    path = defaults_path(root, selector)
    if not path.exists(): create(root, ALIASES.get(selector, selector).split('/')[0])
    data = yaml.safe_load(path.read_text()) or {}
    data['connection'] = {**data.get('connection', {}), **values}
    path.write_text('# @package site\n' + yaml.safe_dump(data, sort_keys=False))
    return data['connection']
