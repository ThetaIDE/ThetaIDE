"""Component-owned site defaults and named Hydra profiles."""
from pathlib import Path
import json
import yaml
from omegaconf import OmegaConf
from .storage import safe_path
from src.app.core.provider_discovery import discovery_cached

@discovery_cached
def all_profiles(paths):
    # One resolver for the whole listing: each resolver scans every installed manifest.
    resolver = paths.config_resolver()
    def resolve(selector):
        resolved = resolver.resolve(selector)
        return resolved.file if resolved is not None else paths.config / selector
    return {Path(selector).relative_to("site").with_suffix("").as_posix(): resolve(selector)
            for selector in resolver.selectors("site")}


def _forwarding_target(path):
    """Target of a historical top-level selector file that only forwards to one site profile."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return None
    if not isinstance(data, dict) or set(data) != {'defaults'} or not isinstance(data['defaults'], list):
        return None
    targets = [entry for entry in data['defaults'] if entry != '_self_']
    if len(targets) != 1 or not isinstance(targets[0], str) or not targets[0].startswith('/site/'):
        return None
    return targets[0].removeprefix('/site/')


def normalize(selector, available):
    """Canonical profile for a selector; provider-owned forwarding files define compatibility aliases."""
    if '/' not in selector and selector in available:
        target = _forwarding_target(available[selector])
        # Only nested, selectable profiles are forwarding targets; authored variants stay distinct.
        if target and '/' in target and not Path(target).name.startswith('_') and target in available:
            selector = target
    if '/' not in selector and selector + '/default' in available:
        selector += '/default'
    return selector


def _paths(root):
    from src.app.workspace import WorkspacePaths
    return root if isinstance(root, WorkspacePaths) else WorkspacePaths(root)


def load(root, selector, *, available=None):
    """Merged site profile; pass ``available`` (from all_profiles) when loading several."""
    available = all_profiles(_paths(root)) if available is None else available
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
    paths = _paths(root)
    component = normalize(site, all_profiles(paths)).split('/')[0]
    return paths.resolve_config(f'site/{component}/_defaults.yaml')


def create(root, name, data=None, variants=None, *, scheduler=None):
    """Author a site component. `scheduler` names an installed scheduler capability;
    omitted, the profile records none and launch reports the missing selection."""
    component = safe_path(root, 'src/usr/sites/' + name)
    directory = component / 'configs/site' / name
    directory.mkdir(parents=True, exist_ok=True)
    data = dict(data or {'scheduler': scheduler, 'venv_dir': 'venv', 'resources': {'nodes': 1, 'cores': 4, 'gpus': 0, 'memory': '8G', 'time': '01:00:00'}})
    data.setdefault('connection', {'host': None, 'remote_root': None, 'python': 'venv/bin/python3',
        'ssh_config': f'src/usr/sites/{name}/ssh_config', 'ssh_connect_timeout_seconds': 60,
        'rsync_executable': 'rsync', 'remote_rsync': 'rsync'})
    data.setdefault('shortcuts', {'directory': f'src/usr/sites/{name}/bin'})
    note = '' if data.get('scheduler') else '# Set scheduler to the capability ID of an installed scheduler component.\n'
    (directory / '_defaults.yaml').write_text('# @package site\n' + note + yaml.safe_dump(data, sort_keys=False))
    for profile, overrides in {'default': {}, **(variants or {})}.items():
        value = {'defaults': [f'/site/{name}/_defaults', '_self_'], **overrides}
        (directory / (profile + '.yaml')).write_text('# @package site\n' + yaml.safe_dump(value, sort_keys=False))
    (component / 'site.py').write_text('"""Site metadata; execution uses the shared scheduler adapter."""\nSITE = ' + repr({'scheduler': data.get('scheduler')}) + '\n')
    (component / 'ssh_config').write_text('# Site-owned OpenSSH aliases and overrides. Private keys stay outside the project.\nInclude ~/.ssh/config\n')
    configs = {f'in/config/site/{name}/{p.name}': f'configs/site/{name}/{p.name}' for p in directory.glob('*.yaml')}
    manifest = {'id': name + '-site', 'name': name, 'kind': 'site', 'version': '1.0.0', 'storage_version': 2, 'api_version': 1,
                'capabilities': [{'id': name, 'kind': 'site', 'entry_point': 'site.py:SITE'}],
                'config_paths': list(configs), 'payload': {'implementations': ['site.py'], 'resources': ['ssh_config'], 'configs': configs}}
    (component / '.theta_component.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return component


def configure(root, selector, values, *, scheduler=None):
    from src.app.components.configuration import ConfigurationFiles
    paths = _paths(root)
    component = normalize(selector, all_profiles(paths)).split('/')[0]
    key = f'site/{component}/_defaults.yaml'
    resolved = paths.config_resolver().resolve(key)
    if resolved is None:
        create(paths.root, component, scheduler=scheduler)
        resolved = paths.config_resolver().resolve(key, required=True)
    # Authored site defaults are already user-owned; managed defaults need an explicit copy.
    path = (resolved.file if resolved.package and resolved.package.origin == 'user'
            else ConfigurationFiles(paths).create_override(key))
    data = yaml.safe_load(path.read_text()) or {}
    data['connection'] = {**data.get('connection', {}), **values}
    ConfigurationFiles(paths).write(path, '# @package site\n' + yaml.safe_dump(data, sort_keys=False), allow_external=True)
    return data['connection']
