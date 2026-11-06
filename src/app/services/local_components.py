"""Workspace component collection. Visibility never participates in runtime lookup."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import yaml

from .component_exporter import ComponentExporter, ComponentExportOptions, find_component_assets

_LOCK = threading.RLock()


def _write_operation(path, operation):
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.tmp', delete=False) as file:
        json.dump(operation, file)
        temporary = Path(file.name)
    temporary.replace(path)


def _write_profile(path, data):
    # Hydra package directives live in the opening comment block and affect
    # composition. Keep that block while updating structured selectors/assets.
    leading = []
    for line in path.read_text().splitlines(keepends=True):
        if line.strip() and not line.lstrip().startswith('#'):
            break
        leading.append(line)
    path.write_text(''.join(leading) + yaml.safe_dump(data, sort_keys=False))


def normalize_kind(kind):
    return {"agent": "algorithm", "method": "algorithm", "data": "datamodule",
            "eval": "evaluator", "plot": "plotter", "models": "model"}.get(kind, kind)


class LocalComponents:
    def __init__(self, workspace: Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / '.thetaide' / 'components'
        self.state_path = self.root / 'collection.json'
        self._recover_operations()

    def _recover_operations(self):
        # An active process owns its operation; only recover abandoned installs.
        for journal in (self.root / 'operations').glob('*.json'):
            operation = json.loads(journal.read_text())
            try:
                os.kill(operation['pid'], 0)
                continue
            except ProcessLookupError:
                pass
            except PermissionError:
                continue
            for value in reversed(operation['paths']):
                path = Path(value)
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            for destination, backup in operation.get('restore', {}).items():
                destination, backup = Path(destination), Path(backup)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if backup.is_dir():
                    shutil.copytree(backup, destination, symlinks=True)
                else:
                    shutil.copy2(backup, destination)
            if operation.get('backup_root'):
                shutil.rmtree(operation['backup_root'], ignore_errors=True)
            if operation.get('remove_record', True):
                with _LOCK:
                    state = self.state()
                    state['records'].pop(operation['id'], None)
                    self._write(state)
            journal.unlink()
        for record in self.state()['records'].values():
            destination = Path(record['package'])
            backup = destination.with_name(destination.name + '.previous')
            if backup.exists() and not destination.exists():
                backup.rename(destination)
        for backup in (self.root / 'packages').glob('*.previous'):
            destination = backup.with_name(backup.name[:-len('.previous')])
            if not destination.exists():
                backup.rename(destination)

    def state(self):
        if not self.state_path.exists():
            return {'records': {}, 'hidden': []}
        return json.loads(self.state_path.read_text())

    def _write(self, state):
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(self.state_path)

    def set_hidden(self, component_id, hidden=True):
        with _LOCK:
            state = self.state()
            values = set(state['hidden'])
            values.add(component_id) if hidden else values.discard(component_id)
            state['hidden'] = sorted(values)
            self._write(state)

    def hidden(self, component_id):
        return component_id in self.state()['hidden']

    def hidden_config_paths(self):
        hidden = set(self.state()['hidden'])
        paths = set()
        roots = ('src/usr/models', 'src/usr/algorithms', 'src/usr/methods', 'src/usr/data', 'src/usr/eval', 'in/envs', 'plot')
        for manifest in (path for root in roots for path in (self.workspace / root).rglob('.theta_component.json')):
            try:
                meta = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            if meta.get('id') in hidden:
                paths.update(meta.get('config_paths', []))
                if meta.get('config_path'):
                    paths.add(meta['config_path'])
        return {(self.workspace / path).resolve() for path in paths}

    def installed_config(self, component, installer):
        target = installer.resolve_target_dir(component)
        manifest = target / '.theta_component.json'
        if not manifest.is_file():
            raise ValueError(f'Component is not installed: {component.id}')
        meta = json.loads(manifest.read_text())
        for relative in [meta.get('config_path'), *meta.get('config_paths', [])]:
            if relative and (self.workspace / relative).is_file():
                return self.workspace / relative
        return manifest

    def _shared_paths(self, component_id):
        paths = set()
        roots = ('src/usr/models', 'src/usr/algorithms', 'src/usr/methods', 'src/usr/data', 'src/usr/eval', 'in/envs', 'plot')
        for root in roots:
            for manifest in (self.workspace / root).rglob('.theta_component.json'):
                data = json.loads(manifest.read_text())
                if data.get('id') == component_id:
                    continue
                for relative in data.get('installed_files', []) + data.get('config_paths', []):
                    paths.add((self.workspace / relative).resolve())
        return paths

    def owned_paths(self, assets):
        paths = set(Path(path).resolve() for path in assets['files_to_delete'])
        meta = assets.get('meta') or {}
        for relative in meta.get('installed_files', []) + meta.get('asset_paths', []):
            path = (self.workspace / relative).resolve()
            if not path.is_relative_to(self.workspace):
                raise ValueError(f'Owned path escapes workspace: {relative}')
            if path.exists():
                paths.add(path)
        for directory in list(paths):
            if directory.is_dir():
                paths.update(path.resolve() for path in directory.rglob('*') if path.is_file() and '__pycache__' not in path.parts)
        return sorted(paths)

    def _export(self, config, destination):
        assets = find_component_assets(Path(config), workspace_dir=self.workspace)
        result = ComponentExporter(self.workspace).export(Path(config).resolve(), ComponentExportOptions(
            export_format='folder', output_path=destination,
            version=(assets.get('meta') or {}).get('version', '0.1.0'),
            include_configs=Path(config).suffix.lower() in ('.yaml', '.yml'), generate_git_scaffold=False))
        if not result.success:
            raise ValueError(result.error)
        package = Path(result.export_path)
        manifest = package / '.theta_component.json'
        meta = json.loads(manifest.read_text())
        if normalize_kind(meta['kind']) in ('model', 'algorithm', 'env', 'datamodule', 'evaluator', 'plotter') and assets.get('code_dir'):
            from frontend.hub.installer import HubInstaller
            installed = assets.get('meta') or {}
            meta['saved_local_changes'] = HubInstaller(self.workspace, self.workspace / '.thetaide').has_local_changes(Path(assets['code_dir']), installed)
        meta['kind'] = normalize_kind(meta['kind'])
        shared = self._shared_paths(meta['id'])
        source_root = Path(assets['code_dir']).resolve() if assets.get('code_dir') else None
        for path in shared:
            if source_root and path.is_relative_to(source_root):
                continue
            packaged = package / path.relative_to(self.workspace)
            if packaged.is_file():
                packaged.unlink()
        for key in ('asset_hashes', 'authoring_hash', 'source_hashes', 'config_hashes', 'config_backups', 'installed_files', 'installed_from', 'sha256'):
            meta.pop(key, None)
        manifest.write_text(json.dumps(meta, indent=2) + '\n')
        # Keep packaged metadata independent of installation bookkeeping too.
        for nested in package.rglob('.theta_component.json'):
            if nested != manifest:
                nested_meta = json.loads(nested.read_text())
                for key in ('asset_hashes', 'authoring_hash', 'installed_files', 'config_backups', 'source_hashes', 'config_hashes', 'installed_from', 'sha256'):
                    nested_meta.pop(key, None)
                nested.write_text(json.dumps(nested_meta, indent=2) + '\n')
        return package

    def snapshot(self, config, source='hub'):
        """Retain the current saved installation, never the original download."""
        assets = find_component_assets(Path(config), workspace_dir=self.workspace)
        cid = assets['comp_id']
        record = self.state()['records'].get(cid, {})
        destination = Path(record.get('package', self.root / 'packages' / cid))
        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            package = self._export(config, Path(temporary) / 'package')
            backup = destination.with_name(cid + '.previous')
            if backup.exists():
                shutil.rmtree(backup)
            if destination.exists():
                destination.rename(backup)
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(package), destination)
                with _LOCK:
                    state = self.state()
                    old = state['records'].get(cid, {})
                    state['records'][cid] = {'package': str(destination), 'source': old.get('source', source)}
                    self._write(state)
            except Exception:
                if destination.exists():
                    shutil.rmtree(destination)
                if backup.exists():
                    backup.rename(destination)
                raise
            if backup.exists():
                shutil.rmtree(backup)
        return destination

    def entries(self):
        from frontend.hub.models import HubComponent
        entries = []
        for cid, record in self.state()['records'].items():
            manifest = Path(record['package']) / '.theta_component.json'
            if manifest.is_file():
                data = json.loads(manifest.read_text())
                data['local_source'] = record['source']
                data['retained_source'] = record['source'] == 'hub'
                entries.append(HubComponent.from_dict(data))
        return entries

    def install(self, component, installer):
        """Install a retained package with rollback for every owned destination."""
        record = self.state()['records'][component.id]
        package = Path(record['package'])
        from frontend.hub.models import HubComponent, ReleaseInfo
        local = HubComponent.from_dict(json.loads((package / '.theta_component.json').read_text()))
        target = installer.resolve_target_dir(local).resolve()
        paths = {target}
        for file in package.rglob('*'):
            relative = file.relative_to(package)
            if file.is_file() and relative.parts[0] in ('src', 'in', 'plot', 'results'):
                destination = (self.workspace / relative).resolve()
                if not destination.is_relative_to(self.workspace):
                    raise ValueError(f'Installation path escapes workspace: {destination}')
                if not destination.is_relative_to(target):
                    paths.add(destination)
        paths = sorted(paths)
        self.root.mkdir(parents=True, exist_ok=True)
        journal = self.root / 'operations' / f'deploy-{component.id}.json'
        journal.parent.mkdir(parents=True, exist_ok=True)
        # Reserve before capturing any files so simultaneous installers cannot
        # restore another operation's source or remove its destinations.
        try:
            with journal.open('x') as reserved:
                json.dump({'id': component.id, 'pid': os.getpid(), 'paths': [], 'remove_record': False}, reserved)
        except FileExistsError as exc:
            raise ValueError(f'Installation already in progress: {component.id}') from exc
        backup_root = Path(tempfile.mkdtemp(prefix='rollback-', dir=self.root))
        restore = {}
        prepared = False
        finalized = False
        try:
            for index, path in enumerate(paths):
                if path.exists():
                    backup = backup_root / str(index)
                    if path.is_dir():
                        shutil.copytree(path, backup, symlinks=True)
                    else:
                        shutil.copy2(path, backup)
                    restore[str(path)] = str(backup)
            operation = {'id': component.id, 'pid': os.getpid(), 'paths': list(map(str, paths)),
                         'restore': restore, 'backup_root': str(backup_root), 'remove_record': False}
            _write_operation(journal, operation)
            prepared = True
            with tempfile.TemporaryDirectory() as temporary:
                archive = Path(shutil.make_archive(str(Path(temporary) / 'component'), 'zip', package))
                local.releases = {local.version: ReleaseInfo(local.version, archive.as_uri(), hashlib.sha256(archive.read_bytes()).hexdigest())}
                result = installer.install(local, install_dependencies=False)
                finalized = True
                return result
        except BaseException:
            if prepared:
                for path in reversed(paths):
                    if path.is_dir():
                        shutil.rmtree(path)
                    elif path.exists():
                        path.unlink()
                for destination, backup in restore.items():
                    destination = Path(destination)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if Path(backup).is_dir():
                        shutil.copytree(backup, destination, symlinks=True)
                    else:
                        shutil.copy2(backup, destination)
            finalized = True
            raise
        finally:
            if finalized or not prepared:
                journal.unlink(missing_ok=True)
                shutil.rmtree(backup_root, ignore_errors=True)

    def uninstall(self, component, installer):
        target = installer.resolve_target_dir(component)
        meta = json.loads((target / '.theta_component.json').read_text())
        shared = self._shared_paths(component.id)
        if any(path.is_relative_to(target.resolve()) for path in shared):
            raise ValueError(f'Another installed component owns files inside {target}; resolve ownership before uninstalling.')
        config = next((relative for relative in [meta.get('config_path'), *meta.get('config_paths', [])]
                       if relative and (self.workspace / relative).is_file()), None)
        if not config:
            config = (target / '.theta_component.json').relative_to(self.workspace).as_posix()
        self.snapshot(self.workspace / config, source=component.extra.get('local_source', 'hub'))
        # Removal must never cascade to shared child components.
        meta['dependencies'] = {}
        meta['installed_components'] = []
        meta['target_dir'] = str(target)
        return installer.uninstall_by_metadata(meta, cascade=False)

    def deletion_details(self, component, installer):
        record = self.state()['records'].get(component.id)
        if not record or record['source'] != 'local':
            raise ValueError('Only private local components can be deleted.')
        paths = [Path(record['package'])]
        retained = installer._retained_config_dir(component.id)
        if retained.exists():
            paths.append(retained)
        target = installer.resolve_target_dir(component)
        manifest = target / '.theta_component.json'
        if manifest.exists():
            meta = json.loads(manifest.read_text())
            paths.append(target)
            for relative in meta.get('installed_files', []) + meta.get('config_paths', []):
                path = (self.workspace / relative).resolve()
                if not path.is_relative_to(self.workspace):
                    raise ValueError(f'Owned path escapes workspace: {path}')
                if path.exists() and not path.is_relative_to(target):
                    paths.append(path)
        shared = self._shared_paths(component.id)
        if any(path.is_relative_to(target) for path in shared):
            raise ValueError(f'Another installed component owns files inside {target}; resolve ownership before deleting.')
        paths = [path for path in paths if path.resolve() not in shared]
        references = []
        for experiment in (self.workspace / 'in/config/experiment').rglob('*.yaml'):
            if component.id in experiment.read_text():
                references.append(experiment)
        return list(dict.fromkeys(paths)), references

    def delete(self, component, installer):
        """Stage owned files together, then use the operating system Trash."""
        from PyQt6.QtCore import QFile
        paths, _ = self.deletion_details(component, installer)
        staging = Path(tempfile.mkdtemp(prefix='trash-', dir=self.root))
        moved = []
        try:
            for index, path in enumerate(paths):
                if path.exists():
                    dest = staging / str(index) / path.name
                    dest.parent.mkdir()
                    shutil.move(str(path), dest)
                    moved.append((path, dest))
            (staging / 'locations.json').write_text(json.dumps({str(dest.relative_to(staging)): str(path) for path, dest in moved}, indent=2))
            success, _ = QFile.moveToTrash(str(staging))
            if not success:
                raise OSError('The operating system could not move the component to Trash.')
        except Exception:
            for path, dest in reversed(moved):
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(dest), path)
            shutil.rmtree(staging)
            raise
        with _LOCK:
            state = self.state()
            state['records'].pop(component.id)
            state['hidden'] = [cid for cid in state['hidden'] if cid != component.id]
            self._write(state)

    def deep_copy(self, config, new_id, name=None, destination=None):
        if name is not None and not name.strip():
            raise ValueError('Display name cannot be empty.')
        if not re.fullmatch(r'[a-z][a-z0-9_]*', new_id):
            raise ValueError('Model ID must start with a lowercase letter and contain lowercase letters, digits or underscores.')
        for source in (self.workspace / 'src/usr/models').rglob('*.py'):
            if '__pycache__' in source.parts:
                continue
            try:
                parsed = ast.parse(source.read_text())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(parsed):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'register_model':
                    if any(isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.lower() == new_id for arg in node.args):
                        raise ValueError(f'Runtime identity already exists at {source}:{node.lineno}: {new_id}')
        cached_index = self.workspace / '.thetaide/cache/hub/index.json'
        if cached_index.is_file():
            index = json.loads(cached_index.read_text())
            for entry in index.get('components', []) if isinstance(index, dict) else index:
                if new_id in [entry['id'], *entry.get('aliases', []), *entry.get('provides', [])]:
                    raise ValueError(f'Component identity already exists in the Hub catalog: {new_id}')
        if (self.workspace / 'in/config/model' / new_id).exists():
            raise ValueError(f'Config namespace already exists: {new_id}')
        assets = find_component_assets(Path(config), workspace_dir=self.workspace)
        if normalize_kind(assets['kind']) != 'model' or not assets.get('code_dir'):
            raise ValueError('Independent Deep Copy currently supports directory-based models only.')
        assets['kind'] = normalize_kind(assets['kind'])
        old_root = Path(assets['code_dir']).relative_to(self.workspace).as_posix()
        new_root = f'src/usr/models/{new_id}'
        destination = Path(destination).resolve() if destination else self.root / 'packages' / new_id
        active_roots = [self.workspace / root for root in ('src', 'in', 'plot', 'results')]
        if any(destination.is_relative_to(root) or root.is_relative_to(destination) for root in active_roots):
            raise ValueError('Retained package destination must be separate from active source/config/asset directories.')
        for record in self.state()['records'].values():
            retained = Path(record['package']).resolve()
            if destination.is_relative_to(retained) or retained.is_relative_to(destination):
                raise ValueError(f'Package destination overlaps an existing retained component: {retained}')
        if destination.exists() or (self.workspace / new_root).exists() or new_id in self.state()['records']:
            raise ValueError(f'Component ID or destination already exists: {new_id}')
        self.root.mkdir(parents=True, exist_ok=True)
        installed = []
        journal = self.root / 'operations' / f'{new_id}.json'
        journal.parent.mkdir(parents=True, exist_ok=True)
        try:
            with journal.open('x') as file:
                json.dump({'id': new_id, 'pid': os.getpid(), 'paths': []}, file)
        except FileExistsError as exc:
            raise ValueError(f'A Deep Copy operation already owns this ID: {new_id}') from exc
        try:
            with tempfile.TemporaryDirectory(dir=self.root) as temporary:
                _write_operation(journal, {'id': new_id, 'pid': os.getpid(), 'paths': [temporary]})
                package = self._export(config, Path(temporary) / 'package')
                meta = json.loads((package / '.theta_component.json').read_text())
                profiles = list(dict.fromkeys(meta.get('config_paths') or ([meta['config_path']] if meta.get('config_path') else [])))
                remap = {old_root: new_root}
                model_profiles = [profile for profile in profiles if Path(profile).parts[:3] == ('in', 'config', 'model')]
                common = Path(os.path.commonpath([str(Path(p).parent) for p in model_profiles])) if model_profiles else Path('in/config/model')
                owner_names = (assets['comp_id'], Path(old_root).name)
                for profile in profiles:
                    path = Path(profile)
                    if profile in model_profiles:
                        relative = path.relative_to(common)
                        if len(model_profiles) == 1 and common == Path('in/config/model'):
                            remap[profile] = f'in/config/model/{new_id}{path.suffix}'
                        else:
                            remap[profile] = f'in/config/model/{new_id}/{relative.as_posix()}'
                    else:
                        parts = list(path.parts)
                        if parts[:2] != ['in', 'config'] or len(parts) < 4:
                            raise ValueError(f'Unsupported owned config path: {profile}')
                        if len(parts) > 4 and parts[3] in owner_names:
                            parts[3] = new_id
                        else:
                            parts.insert(3, new_id)
                        remap[profile] = Path(*parts).as_posix()
                if len(set(remap.values())) != len(remap):
                    raise ValueError('Owned paths collide after assigning the new identity.')
                # A model can own auxiliary plotting modules. Move their whole
                # package and rewrite only imports inside those known namespaces.
                auxiliary_roots = set()
                for owned in package.rglob('*.py'):
                    relative = owned.relative_to(package).as_posix()
                    if relative.startswith(old_root + '/'):
                        continue
                    parts = Path(relative).parts
                    if len(parts) >= 3 and parts[0] == 'plot' and parts[1] in owner_names and (package / 'plot' / parts[1] / '__init__.py').is_file():
                        auxiliary_roots.add(f'plot/{parts[1]}')
                    else:
                        raise ValueError(f'Owned Python module outside a supported component namespace: {relative}')
                for auxiliary in sorted(auxiliary_roots):
                    remap[auxiliary] = f'plot/{new_id}'
                code_roots = [old_root, *sorted(auxiliary_roots)]
                module_remap = {root.replace('/', '.'): remap[root].replace('/', '.') for root in code_roots}
                for owned in package.rglob('*'):
                    if not owned.is_file():
                        continue
                    relative = owned.relative_to(package).as_posix()
                    if any(relative.startswith(root + '/') for root in code_roots) or relative in profiles or relative.split('/')[0] not in ('src', 'in', 'plot', 'results'):
                        continue
                    parts = list(Path(relative).parts)
                    owner_index = next((i for i, part in enumerate(parts[1:], 1) if part in owner_names), None)
                    if owner_index is not None:
                        parts[owner_index] = new_id
                    else:
                        parts.insert(len(parts) - 1, new_id)
                    remap[relative] = Path(*parts).as_posix()
                asset_remap = {old: new for old, new in remap.items() if old not in profiles and old != old_root}
                for directory in meta.get('asset_paths', []):
                    if not (package / directory).is_dir():
                        continue
                    new_directories = set()
                    for old, new in asset_remap.items():
                        if old.startswith(directory.rstrip('/') + '/'):
                            suffix = Path(old).relative_to(directory)
                            new_directory = Path(new)
                            for _ in suffix.parts:
                                new_directory = new_directory.parent
                            new_directories.add(new_directory.as_posix())
                    if len(new_directories) == 1:
                        asset_remap[directory] = new_directories.pop()
                config_selectors = {
                    Path(old).relative_to('in/config').with_suffix('').as_posix():
                    Path(new).relative_to('in/config').with_suffix('').as_posix()
                    for old, new in remap.items() if old in profiles
                }
                def remap_defaults(data):
                    if not isinstance(data, dict) or not isinstance(data.get('defaults'), list):
                        return data
                    defaults = []
                    for entry in data['defaults']:
                        if isinstance(entry, str):
                            replacement = config_selectors.get(entry.lstrip('/'))
                            defaults.append(('/' if entry.startswith('/') else '') + replacement if replacement else entry)
                        elif isinstance(entry, dict):
                            item = {}
                            for group, selection in entry.items():
                                if not isinstance(group, str) or not isinstance(selection, str):
                                    item[group] = selection
                                    continue
                                directive = 'override ' if group.startswith('override ') else ('optional ' if group.startswith('optional ') else '')
                                raw_group, _, package_suffix = group[len(directive):].partition('@')
                                key = raw_group.lstrip('/') + '/' + selection
                                replacement = config_selectors.get(key)
                                if replacement:
                                    depth = len(selection.split('/'))
                                    parts = replacement.split('/')
                                    new_group = '/'.join(parts[:-depth])
                                    selection = '/'.join(parts[-depth:])
                                    group = directive + ('/' if raw_group.startswith('/') else '') + new_group + ('@' + package_suffix if package_suffix else '')
                                item[group] = selection
                            defaults.append(item)
                        else:
                            defaults.append(entry)
                    data['defaults'] = defaults
                    return data
                source_path_remap = {**remap, **asset_remap}
                source = package / old_root
                registrations = []
                python_files = sorted({path for root in code_roots for path in (package / root).rglob('*.py')})
                for path in python_files:
                    for node in ast.walk(ast.parse(path.read_text())):
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'register_model':
                            if not node.args or not all(isinstance(arg, ast.Constant) and isinstance(arg.value, str) for arg in node.args):
                                raise ValueError(f'Unsupported registration at {path.relative_to(package)}:{node.lineno}')
                            registrations.append((path, node))
                if len(registrations) != 1:
                    locations = ', '.join(f'{p.relative_to(package)}:{n.lineno}' for p, n in registrations)
                    raise ValueError(f'Expected one literal @register_model call; found {len(registrations)}. Identity locations: {locations}')
                aliases = {arg.value for arg in registrations[0][1].args}
                for path in python_files:
                    text = path.read_text()
                    lines = text.splitlines(keepends=True)
                    edits = []
                    for node in ast.walk(ast.parse(text)):
                        replacement = None
                        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in source_path_remap:
                            replacement = repr(source_path_remap[node.value])
                        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'register_model':
                            replacement = f'register_model({new_id!r})'
                        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                            for before, after in module_remap.items():
                                if node.value == before or node.value.startswith((before + '.', before + ':')):
                                    replacement = repr(after + node.value[len(before):])
                                    break
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            for before, after in module_remap.items():
                                if node.module == before or node.module.startswith(before + '.'):
                                    node.module = after + node.module[len(before):]
                                    replacement = ast.unparse(node)
                                    break
                                if node.module == before.rsplit('.', 1)[0]:
                                    for alias in node.names:
                                        if alias.name == before.rsplit('.', 1)[1]:
                                            alias.asname = alias.asname or alias.name
                                            alias.name = after.rsplit('.', 1)[1]
                                            replacement = ast.unparse(node)
                        elif isinstance(node, ast.Import):
                            for alias in node.names:
                                for before, after in module_remap.items():
                                    if alias.name == before or alias.name.startswith(before + '.'):
                                        if alias.asname is None:
                                            raise ValueError(f'Use an aliased import or from-import before copying: {path.relative_to(package)}:{node.lineno}: {ast.unparse(node)}')
                                        alias.name = after + alias.name[len(before):]
                                        replacement = ast.unparse(node)
                                        break
                        if replacement:
                            start = sum(len(line) for line in lines[:node.lineno-1]) + len(lines[node.lineno-1].encode()[:node.col_offset].decode())
                            end = sum(len(line) for line in lines[:node.end_lineno-1]) + len(lines[node.end_lineno-1].encode()[:node.end_col_offset].decode())
                            edits.append((start, end, replacement))
                    for start, end, replacement in sorted(edits, reverse=True):
                        text = text[:start] + replacement + text[end:]
                    path.write_text(text)
                for profile in model_profiles:
                    path = package / profile
                    data = yaml.safe_load(path.read_text())
                    if not isinstance(data, dict):
                        raise ValueError(f'Model profile must be a mapping: {profile}')
                    selector = data if any(key in data for key in ('name', 'architecture', 'type')) else data.get('model', data)
                    def owns_selector(value):
                        return isinstance(value, str) and (value in aliases or value.split('/')[-1] in aliases)
                    if isinstance(selector, str) and owns_selector(selector):
                        data['model'] = new_id
                    elif isinstance(selector, dict):
                        keys = [key for key in ('name', 'architecture', 'type') if key in selector]
                        if not keys:
                            selector['architecture'] = new_id
                        for key in keys:
                            if not owns_selector(selector[key]):
                                raise ValueError(f'Cannot remap selector {profile}: {key}={selector[key]!r}; expected one of {sorted(aliases)}')
                            selector[key] = new_id
                    else:
                        raise ValueError(f'Cannot remap model selector at {profile}: {selector!r}')
                    _write_profile(path, data)
                def mapped(value):
                    if isinstance(value, str):
                        for old, new in {**remap, **asset_remap}.items():
                            if value == old or value.startswith(old + '/'):
                                return new + value[len(old):]
                            if old.startswith('in/config/') and value == old[len('in/config/'):]:
                                return new[len('in/config/'):]
                        for before, after in module_remap.items():
                            if value == before or value.startswith((before + '.', before + ':')):
                                return after + value[len(before):]
                        return value
                    if isinstance(value, list):
                        return [mapped(v) for v in value]
                    if isinstance(value, dict):
                        return {mapped(k): mapped(v) for k, v in value.items()}
                    return value
                for profile in profiles:
                    path = package / profile
                    _write_profile(path, remap_defaults(mapped(yaml.safe_load(path.read_text()))))
                if not model_profiles:
                    profile = f'in/config/model/{new_id}/default.yaml'
                    path = package / profile
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(yaml.safe_dump({'architecture': new_id}, sort_keys=False))
                    meta['config_path'] = profile
                    meta['config_paths'] = [profile, *meta.get('config_paths', [])]
                meta = mapped(meta)
                # Only the owning profile's selector schema changes identity;
                # nested child-component schemas keep their shared identities.
                schema = meta.get('parameter_schema') or {}
                profile_schemas = schema.get('x-theta-config-schemas') or {}
                selector_schemas = [schema, *(profile_schemas[remap[p]] for p in model_profiles if remap[p] in profile_schemas)]
                for selector_schema in selector_schemas:
                    properties = selector_schema.get('properties', {})
                    for selector_key in ('name', 'architecture', 'type'):
                        domain = properties.get(selector_key)
                        if not isinstance(domain, dict):
                            continue
                        for key in ('const', 'default'):
                            if isinstance(domain.get(key), str) and domain[key] in aliases:
                                domain[key] = new_id
                        if isinstance(domain.get('enum'), list):
                            domain['enum'] = [new_id if isinstance(value, str) and value in aliases else value for value in domain['enum']]
                if model_profiles:
                    primary = meta.get('config_path')
                    if primary not in [remap[p] for p in model_profiles]:
                        meta['config_path'] = remap[model_profiles[0]]
                meta.update(id=new_id, name=name or new_id, version='0.1.0', origin='local', parent={'id': assets['comp_id'], 'version': (assets.get('meta') or {}).get('version')})
                for key in ('saved_local_changes', 'aliases', 'provides', 'installed_from', 'sha256', 'installed_files', 'config_hashes', 'config_backups', 'user_config_paths', 'installed_components', 'suite', 'hidden'):
                    meta.pop(key, None)
                for old, new in remap.items():
                    target = package / new
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(package / old), target)
                (package / new_root / '.theta_component.json').unlink(missing_ok=True)
                (package / '.theta_component.json').write_text(json.dumps(meta, indent=2) + '\n')
                from src.app.core.manifest_schema import validate_manifest_file
                valid, error = validate_manifest_file(package / '.theta_component.json')
                if not valid:
                    raise ValueError(error)
                for path in package.rglob('*'):
                    relative = path.relative_to(package)
                    if path.is_file() and relative.parts[0] in ('src', 'in', 'plot', 'results'):
                        target = self.workspace / relative
                        if target.exists():
                            raise ValueError(f'Installation path already exists: {target}')
                        installed.append(target)
                installed.append(self.workspace / new_root)
                journal.parent.mkdir(parents=True, exist_ok=True)
                _write_operation(journal, {'id': new_id, 'pid': os.getpid(), 'paths': [str(p) for p in [*installed, destination]]})
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(package), destination)
                with _LOCK:
                    state = self.state()
                    state['records'][new_id] = {'source': 'local', 'package': str(destination)}
                    self._write(state)
                from frontend.hub.installer import HubInstaller
                from frontend.hub.models import HubComponent
                self.install(HubComponent.from_dict(meta), HubInstaller(self.workspace, self.workspace / '.thetaide'))
            journal.unlink(missing_ok=True)
            return self.workspace / meta['config_path']
        except Exception:
            for path in reversed(installed):
                if path.is_dir():
                    shutil.rmtree(path)
                elif path.exists():
                    path.unlink()
            if destination.exists():
                shutil.rmtree(destination)
            with _LOCK:
                state = self.state()
                state['records'].pop(new_id, None)
                self._write(state)
            journal.unlink(missing_ok=True)
            raise

    def context(self, config):
        assets = find_component_assets(Path(config), workspace_dir=self.workspace)
        root = Path(assets['code_dir'])
        meta = assets.get('meta') or {}
        paths = [p for p in self.owned_paths(assets) if p.is_file() and not p.is_relative_to(root)]
        kind = normalize_kind(assets['kind'])
        registration = []
        for path in root.rglob('*.py'):
            try:
                parsed = ast.parse(path.read_text())
            except SyntaxError as exc:
                registration.append(f'{path}:{exc.lineno}: syntax error; repair before running')
                continue
            for node in ast.walk(parsed):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith('register_'):
                    registration.append(f'{path}:{node.lineno}: {ast.unparse(node)}')
        signatures = []
        for source in root.rglob('*.py'):
            try:
                parsed = ast.parse(source.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(parsed):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ('__init__', 'forward'):
                    signatures.append(f'- {source}:{node.lineno}: `{node.name}({ast.unparse(node.args)})`')
                    doc = ast.get_docstring(node)
                    if doc:
                        signatures.append('  ' + doc[:600].replace('\n', ' '))
        try:
            revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=self.workspace, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            revision = 'unknown revision'

        profile_lines = []
        for profile in paths:
            if profile.suffix not in ('.yaml', '.yml'):
                continue
            try:
                defaults = yaml.safe_load(profile.read_text())
            except yaml.YAMLError:
                profile_lines.append(f'- {profile}: invalid YAML; repair before running')
                continue
            profile_lines.append(f'- {profile}: `{str(defaults)[:500]}`')
        if kind == 'model':
            contracts = f"""Factory: [{self.workspace / 'src/app/core/model_registry.py'}]({self.workspace / 'src/app/core/model_registry.py'}).
`build_model` accepts a name or mapping (name/architecture/type and constructor parameters).
It infers obs_dim/n_actions from env, supplies device/env and dimension aliases, and passes
accepted constructor arguments. The selected profile's architecture is the runtime ID;
the profile path remains the experiment selector. Shapes/outputs are component-specific:
see the forward signature/docstrings above and the consuming algorithm before changing them.
Optional hooks: [{self.workspace / 'src/app/core/protocols.py'}]({self.workspace / 'src/app/core/protocols.py'}).
Working example: {self.workspace / 'src/usr/models/sequential_spec/sequential_spec.py'}.
Child model selectors remain shared dependencies."""
        else:
            registries = {'algorithm': 'src/usr/algorithms/agent_registry.py', 'datamodule': 'src/app/core/datamodule_registry.py',
                          'evaluator': 'src/app/core/eval_registry.py', 'plotter': 'src/app/core/plotter_registry.py'}
            contract_path = self.workspace / registries.get(kind, 'src/app/core/protocols.py')
            contracts = f"Registry/contract: [{contract_path}]({contract_path}). Inspect this component's entry points, constructor and consumer for its required inputs/outputs; model factory conventions do not apply to this kind."

        text = f'''# {assets['comp_name']} developer context (generated)

ID: {assets['comp_id']} · kind: {assets['kind']} · version: {meta.get('version', 'unknown')}
Parent: {meta.get('parent', 'none')}
Project: {self.workspace} · Theta revision: {revision}
Installed source: {root}

Edit these installed files directly. Save affects the next fresh experiment process.
The retained collection package is a snapshot, not a synchronized working copy.
Keep handwritten notes in a separate file; this document is refreshed on open.

## Owned files and entry points
{chr(10).join('- ' + str(p) for p in paths)}
Source modules/assets: all files beneath {root}
{chr(10).join('- ' + entry for entry in registration)}
Manifest/schema: {root / '.theta_component.json'} (including x-theta-config-schemas).
Package paths are workspace-relative; prefix them with {self.workspace} to edit active files.

## Runtime contracts
{chr(10).join(signatures)}
{contracts}

## Profiles and defaults (bounded preview; installed files are authoritative)
{chr(10).join(profile_lines[:8])}
Profile schemas: {list((meta.get('parameter_schema') or {}).get('x-theta-config-schemas', {}))}

On a fork, change registration aliases, package imports, profile selectors/paths and schema keys.

## Existing commands
From {self.workspace}, using the Python environment that runs Theta:
- python run_pipeline.py <group>/<experiment> dry_run=true (inspect/validate an existing experiment)
- python run_pipeline.py <group>/<experiment> (run it in a fresh process)
- python -m pytest tests/test_local_components.py
Use your own CLI/editor/AI tools and component tests. No publication occurs on save.
'''
        path = root / 'COMPONENT_CONTEXT.md'
        path.write_text(text)
        return assets, path
