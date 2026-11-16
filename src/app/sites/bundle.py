"""Explicit remote bundles (``theta.bundle/v1``).

A bundle is built from one composed experiment: the application build, the
providers the experiment selects with their declared component dependencies,
the workspace configuration and authored rules it uses, and the workspace
identity files. Results, datasets, unselected providers and other experiments
are never included. One manifest describes it, and extraction, activation,
execution and continuation accept exactly this schema.
"""
from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .storage import digest, read_json, safe_path, workspace_id, write_json

SCHEMA = 'theta.bundle/v1'
MANIFEST = '.thetaide/bundle.json'
REQUIREMENTS = '.thetaide/requirements.txt'
# The application a worker needs; the desktop and app plugins are not part of it.
APPLICATION_FILES = ('run_pipeline.py', 'pyproject.toml', 'README.md', 'LICENSE.md', 'MANIFEST.in',
                     'src/__init__.py', 'src/plugins/__init__.py', 'src/plugins/components/__init__.py',
                     'src/plugins/inventory.json')
APPLICATION_TREES = ('src/app', 'hydra_plugins', 'src/plugins/components/bundled')
WORKSPACE_STATE = ('provider_overrides.json', 'workspace.json', 'extensions.json')
EXCLUDED = {'__pycache__', '.git', 'node_modules', '.venv', 'venv', '.pytest_cache',
            '.DS_Store', '.env', '.aws', '.ssh'}


@dataclass
class Selection:
    """What one experiment uses, found by composing it in its workspace."""
    paths: object
    experiment: str
    overrides: list
    cfg: object
    packages: list = field(default_factory=list)
    configs: list = field(default_factory=list)


def _strings(value, found):
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str): found.add(key)
            _strings(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value: _strings(item, found)
    elif isinstance(value, str):
        found.add(value)
    return found


def _selected_packages(cfg, paths, configs):
    """Packages the composed configuration names, owns profiles it read, or depends on."""
    from omegaconf import OmegaConf
    from src.app.core.provider_catalog import DuplicateProvider, get_catalog
    from src.app.core.provider_loader import _declared_dependencies, find_provider
    catalog = get_catalog(paths)
    data = OmegaConf.to_container(cfg, resolve=False)
    choices = ((data.pop('hydra', None) or {}).get('runtime') or {}).get('choices') or {}
    strings = {value.strip() for value in _strings([data, choices], set()) if value.strip() and len(value) < 256}
    kinds = sorted({capability.kind for package in catalog.packages for capability in package.capabilities})
    selected = {}

    def add(package):
        if package is not None: selected.setdefault(package.id, package)

    for value in strings:
        if value.startswith(('export://', 'package://')):
            add(catalog.package(value.split('://', 1)[1].split('/', 1)[0]))
            continue
        for kind in kinds:
            try:
                resolved = find_provider(kind, value, paths) if kind == 'algorithm' else catalog.resolve(kind, value)
            except DuplicateProvider:
                continue
            if resolved: add(resolved[0])
    for resolved in configs:
        add(resolved.package)
    pending = list(selected.values())
    while pending:
        package = pending.pop()
        for dependency, _version in _declared_dependencies(package.manifest):
            owner = catalog.package(dependency)
            if owner is None:
                raise ValueError(f'Package {package.id} depends on component {dependency!r}, which is not installed')
            if owner.id not in selected:
                selected[owner.id] = owner
                pending.append(owner)
    app_roots = [root.resolve() for root in paths.app_roots]
    return sorted((package for package in selected.values()
                   if not any(package.root.resolve().is_relative_to(root) for root in app_roots)),
                  key=lambda package: package.id)


def select(root, experiment, overrides=(), *, paths=None):
    """Compose the experiment in its workspace and record what it uses."""
    from src.app.config_resolution import recording
    from src.app.pipeline.compose import compose_experiment
    from src.app.pipeline.config import parse_methods_dict
    from src.app.workspace import WorkspacePaths
    paths = paths or WorkspacePaths(root)
    with recording() as configs:
        composed = compose_experiment(experiment, list(overrides), workspace=paths)
        parse_methods_dict(composed.cfg, paths=paths)
        paradigm = composed.cfg.get('paradigm')
        if paradigm: paths.config_resolver().resolve(f'paradigms/{paradigm}.yaml')
    unique = list({resolved.file: resolved for resolved in configs}.values())
    return Selection(paths, composed.experiment, list(overrides), composed.cfg,
                     _selected_packages(composed.cfg, paths, unique), unique)


def _walk(source, prefix, files):
    if not source.is_dir(): return
    for current, children, names in os.walk(source):
        kept = []
        for entry in [*children, *names]:
            path = Path(current) / entry
            if entry in EXCLUDED: continue
            relative = prefix + '/' + path.relative_to(source).as_posix()
            if path.is_symlink():
                raise ValueError(f'Bundle symlink needs an explicit cluster copy: {relative}')
            if entry in children: kept.append(entry)
            elif path.is_file() and path.suffix not in ('.pyc', '.pyo'): _add(files, relative, path)
        children[:] = kept


def _add(files, relative, path):
    if files.get(relative, path) != path:
        raise ValueError(f'Bundle path {relative} has two sources: {files[relative]} and {path}')
    files[relative] = path


def _within(path, root, what):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f'{what} {path} is outside {root}; it cannot be bundled')
    return path.relative_to(Path(root).resolve()).as_posix()


def _application_project(application):
    """Name, version and requirements of the application: its source tree's ``pyproject.toml``,
    or the installed distribution's metadata when the application is installed."""
    pyproject = Path(application) / 'pyproject.toml'
    if pyproject.is_file():
        import tomllib
        project = tomllib.loads(pyproject.read_text()).get('project', {})
        return {'name': project.get('name'), 'version': project.get('version')}, list(project.get('dependencies', []))
    from importlib import metadata
    names = metadata.packages_distributions().get('run_pipeline') or []
    if len(names) != 1:
        raise ValueError(f'The application at {application} is neither a source tree nor one installed distribution')
    distribution = metadata.distribution(names[0])
    requirements = [value for value in distribution.requires or [] if 'extra ==' not in value]
    return {'name': distribution.metadata['Name'], 'version': distribution.version}, requirements


def _set_digest(hashes):
    return hashlib.sha256(''.join(f'{key}\0{value}\n' for key, value in sorted(hashes.items())).encode()).hexdigest()


def _contracts(capability):
    return {key: value for key, value in capability.metadata.items()
            if key == 'execution_contract' or key.endswith('_api')}


def build(selection, destination):
    """Copy the selection into ``destination`` and write its manifest."""
    from src.app.core.provider_catalog import provider_hash
    paths = selection.paths
    destination = Path(destination)
    files, application = {}, {}
    for name in APPLICATION_FILES:
        if (paths.application / name).is_file(): _add(application, name, paths.application / name)
    for tree in APPLICATION_TREES:
        _walk(paths.application / tree, tree, application)
    for name, path in application.items(): _add(files, name, path)
    providers, requirements = [], set()
    for package in selection.packages:
        entry = {'id': package.id, 'version': package.version, 'origin': package.origin,
                 'storage_version': package.manifest.get('storage_version'),
                 'api_version': package.manifest.get('api_version'),
                 'capabilities': [{'kind': c.kind, 'id': c.id, **_contracts(c)} for c in package.capabilities]}
        if package.origin == 'python_distribution':
            distribution = package.manifest.get('distribution')
            requirements.add(f'{distribution}=={package.version}')
            providers.append({**entry, 'digest': None, 'path': None})
            continue
        if package.origin == 'bundled':
            root = _within(package.root, paths.application, f'Package {package.id}')
        else:
            root = _within(package.root, paths.root, f'Package {package.id}')
            _walk(package.root, root, files)
        requirements.update((package.manifest.get('dependencies') or {}).get('pip') or [])
        providers.append({**entry, 'digest': provider_hash(package), 'path': root})
    configs = []
    experiments = paths.config / 'experiment'
    if paths.config.is_dir():
        for current, children, names in os.walk(paths.config):
            children[:] = [c for c in children if c not in EXCLUDED and Path(current, c) != experiments]
            for name in names:
                path = Path(current) / name
                if name in EXCLUDED: continue
                if path.is_symlink(): raise ValueError(f'Bundle symlink needs an explicit cluster copy: {path}')
                _add(files, _within(path, paths.root, 'Configuration'), path)
    for resolved in sorted(selection.configs, key=lambda item: item.selector):
        if resolved.origin == 'framework':
            relative = _within(resolved.file, paths.application, 'Configuration')
        else:
            relative = _within(resolved.file, paths.root, 'Configuration')
            if resolved.origin == 'workspace': _add(files, relative, resolved.file)
        configs.append({'selector': resolved.selector, 'owner': resolved.package.id if resolved.package else None,
                        'origin': resolved.origin, 'sha256': digest(resolved.file), 'path': relative})
    rules = paths.rules
    if rules.is_dir(): _walk(rules, _within(rules, paths.root, 'Rules'), files)
    for name in WORKSPACE_STATE:
        if (paths.data_dir / name).is_file(): _add(files, '.thetaide/' + name, paths.data_dir / name)
    hashes = {}
    for relative, source in sorted(files.items()):
        target = safe_path(destination, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative] = digest(target)
    listed = sorted(requirements)
    identity, application_requirements = _application_project(paths.application)
    # One file sets up a remote environment: the application's requirements, then the providers'.
    requirement_file = safe_path(destination, REQUIREMENTS)
    requirement_file.parent.mkdir(parents=True, exist_ok=True)
    requirement_file.write_text(''.join(line + '\n' for line in [*application_requirements, *listed]))
    hashes[REQUIREMENTS] = digest(requirement_file)
    from . import site_name
    site = selection.cfg.get('site') or {}
    execution = site.get('execution') or {}
    from omegaconf import OmegaConf
    inputs = OmegaConf.to_container(execution.get('inputs'), resolve=True) if OmegaConf.is_config(execution.get('inputs')) else list(execution.get('inputs') or [])
    manifest = {
        'schema': SCHEMA,
        'bundle_sha256': _set_digest(hashes),
        'project_id': workspace_id(paths.root),
        'created_at': datetime.now(timezone.utc).isoformat(),
        'experiment': {'selector': selection.experiment, 'overrides': list(selection.overrides),
                       'site': site_name(selection.cfg)},
        'application': {**identity,
                        'sha256': _set_digest({name: hashes[name] for name in application})},
        'providers': providers,
        'configs': configs,
        'inputs': [{'source': str(item['source']), 'target': str(item['target'])} for item in inputs],
        'requirements': {'application': application_requirements, 'providers': listed},
        'files': hashes,
    }
    write_json(destination / MANIFEST, manifest)
    return manifest


def prepare(root, experiment, overrides, destination):
    """Select, build and check the bundle of one experiment."""
    manifest = build(select(root, experiment, overrides), destination)
    check(destination, manifest)
    return manifest


def check(directory, manifest):
    """The experiment composes inside the bundle and every provider it names is there."""
    from src.app.pipeline.validation import find_unavailable_components
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths(directory, application=directory)
    experiment = manifest['experiment']
    try:
        selection = select(directory, experiment['selector'], experiment['overrides'], paths=paths)
    except Exception as exc:
        raise ValueError(f'The bundle of {experiment["selector"]} does not compose on its own: {exc}') from exc
    missing = find_unavailable_components(selection.cfg, paths=paths)
    absent = {package.id for package in selection.packages} - {item['id'] for item in manifest['providers']}
    if missing or absent:
        raise ValueError(f'The bundle of {experiment["selector"]} lacks ' + ', '.join(sorted({*missing, *absent})))


def _allowed(relative):
    from src.app.workspace import WorkspacePaths
    if relative.startswith('.thetaide/'):
        return relative in (REQUIREMENTS, *('.thetaide/' + name for name in WORKSPACE_STATE))
    excluded = (WorkspacePaths(Path('/')).results_relative, 'src/usr/datasets')
    return not any(relative == prefix or relative.startswith(prefix + '/') for prefix in excluded)


def verify(directory):
    """The manifest of the bundle in ``directory``; every listed file must match it."""
    manifest = read_json(Path(directory) / MANIFEST)
    if manifest.get('schema') != SCHEMA:
        raise ValueError(f'Unsupported bundle schema {manifest.get("schema")!r}; expected {SCHEMA}')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files: raise ValueError('Bundle manifest lists no files')
    if _set_digest(files) != manifest.get('bundle_sha256'): raise ValueError('Bundle digest mismatch')
    for relative, expected in files.items():
        if not _allowed(relative): raise ValueError(f'Bundle may not contain {relative}')
        path = safe_path(directory, relative)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Bundle file failed verification: {relative}')
    return manifest


def current(root):
    """The manifest of the bundle activated in a remote workspace, or an empty dict."""
    return read_json(Path(root) / MANIFEST)


def archive(directory, destination):
    """Zip a verified bundle; archives contain regular files only."""
    import zipfile
    manifest = verify(directory)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in [*sorted(manifest['files']), MANIFEST]:
            bundle.write(Path(directory) / name, name)
    return manifest


def extract(archive_path, destination, *, max_bytes=256 * 1024**2):
    """Validate the entire archive before writing. Never extract links or traversal."""
    import stat
    import zipfile
    with zipfile.ZipFile(archive_path) as bundle:
        seen, total = set(), 0
        for item in bundle.infolist():
            safe_path(destination, item.filename)
            mode = item.external_attr >> 16
            if item.is_dir() or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError('Bundle archives accept regular files only')
            if item.filename in seen: raise ValueError('Duplicate archive member')
            seen.add(item.filename)
            total += item.file_size
            if total > max_bytes: raise ValueError('Bundle archive exceeds size limit')
        for item in bundle.infolist():
            target = safe_path(destination, item.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output, 1024 * 1024)
    manifest = verify(destination)
    if seen != set(manifest['files']) | {MANIFEST}:
        raise ValueError('Bundle archive members differ from its manifest')
    return manifest
