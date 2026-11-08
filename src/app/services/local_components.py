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

from src.app.workspace import WorkspacePaths
from src.app.services.rulesets import reject_legacy_rules_paths, require_canonical_rules_layout

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
    def __init__(self, workspace: Path | WorkspacePaths):
        self.paths = workspace if isinstance(workspace, WorkspacePaths) else WorkspacePaths(workspace)
        self.workspace = self.paths.root
        self.root = self.paths.collection
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
        for manifest in self.paths.manifests():
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
            from src.app.core.provider_catalog import get_catalog
            owner = get_catalog(self.paths).package(component.id)
            if owner is None: raise ValueError(f'Component is not installed: {component.id}')
            manifest = owner.root / '.theta_component.json'
        meta = json.loads(manifest.read_text())
        for relative in [meta.get('config_path'), *meta.get('config_paths', [])]:
            if relative and (self.workspace / relative).is_file():
                return self.workspace / relative
        return manifest

    def _shared_paths(self, component_id):
        paths = set()
        from src.app.core.provider_catalog import get_catalog
        for package in get_catalog(self.paths).packages:
            if package.id == component_id: continue
            paths.update((self.workspace / relative).resolve() for relative in package.manifest.get('config_paths',[]))
        return paths

    def owned_paths(self, assets):
        require_canonical_rules_layout(self.workspace)
        paths = set(Path(path).resolve() for path in assets['files_to_delete'])
        meta = assets.get('meta') or {}
        reject_legacy_rules_paths(meta)
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
        # Export preserves source modes; retained packages and copies must be editable.
        for path in package.rglob('*'):
            if path.is_file() and not path.is_symlink():
                path.chmod(path.stat().st_mode | 0o200)
        manifest = package / '.theta_component.json'
        meta = json.loads(manifest.read_text())
        if normalize_kind(meta['kind']) in ('model', 'algorithm', 'env', 'datamodule', 'evaluator', 'plotter') and assets.get('code_dir'):
            from src.app.components.installer import HubInstaller
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
        from src.app.components.models import HubComponent
        entries = []
        for cid, record in self.state()['records'].items():
            if record['source'] != 'local':
                continue  # Old Hub snapshots are never an installation source.
            manifest = Path(record['package']) / '.theta_component.json'
            if manifest.is_file():
                data = json.loads(manifest.read_text())
                data['local_source'] = record['source']
                entries.append(HubComponent.from_dict(data))
        return entries

    def discard_hub_snapshot(self, component_id):
        """Remove obsolete cached Hub source, without touching private copies."""
        with _LOCK:
            state = self.state()
            record = state['records'].get(component_id)
            if not record or record['source'] == 'local':
                return
            package = Path(record['package']).resolve()
            packages = (self.root / 'packages').resolve()
            if package.is_relative_to(packages) and package != packages and package.is_dir():
                for path in package.rglob('*'):
                    if path.is_file() and not path.is_symlink():
                        path.chmod(path.stat().st_mode | 0o200)
                shutil.rmtree(package)
            state['records'].pop(component_id)
            self._write(state)

    def install(self, component, installer):
        """Import a retained archive through the managed installer; never deploy authoring paths."""
        from src.app.core.provider_catalog import get_catalog
        owner = get_catalog(self.paths).package(component.id)
        if owner and owner.writable: raise ValueError("Existing user source cannot be installed over; export it or archive the editable copy explicitly")
        record = self.state()['records'][component.id]
        package = Path(record['package'])
        from src.app.components.models import HubComponent, ReleaseInfo
        metadata = json.loads((package / '.theta_component.json').read_text())
        local = HubComponent.from_dict(metadata)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(shutil.make_archive(str(Path(temporary) / 'component'), 'zip', package))
            local.releases = {local.version: ReleaseInfo(local.version, archive.as_uri(), hashlib.sha256(archive.read_bytes()).hexdigest())}
            return installer.install(local, install_dependencies=False)

    def uninstall(self, component, installer):
        """Ordinary uninstall only acts on managed package roots."""
        from src.app.core.provider_catalog import get_catalog
        owner = get_catalog(self.paths).package(component.id)
        if owner and owner.writable:
            raise ValueError("User source cannot be uninstalled. Hide it, explicitly archive the editable copy, or delete it through the authoring workflow.")
        return installer.uninstall(component)

    def archive_authored(self, component, installer=None):
        """Explicit development operation: retain a user implementation, then retire owned files."""
        from src.app.core.provider_catalog import get_catalog
        from src.app.components.transactions import workspace_lock, assert_idle, FileTransaction, remove
        owner = get_catalog(self.paths).package(component.id)
        if owner is None or not owner.writable:
            raise ValueError("Only confirmed user-owned implementations can be archived")
        profiles = [*owner.manifest.get('config_paths', []), *owner.manifest.get('user_config_paths', [])]
        primary = next((self.workspace / p for p in profiles if (self.workspace / p).is_file()), owner.root / '.theta_component.json')
        self.snapshot(primary, source='local')
        shared = self._shared_paths(component.id)
        owned = [owner.root, *(self.paths.config_file(p) for p in profiles)]
        owned = [p for p in owned if p.resolve() not in shared]
        for path in owned:
            if not (path.resolve().is_relative_to(self.paths.research) or path.resolve().is_relative_to(self.paths.config)):
                raise ValueError(f'Authored archive requires normalization of external owned asset: {path}')
        with workspace_lock(self.paths):
            assert_idle(self.paths)
            with FileTransaction(self.paths, 'archive-authored', component.id) as transaction:
                transaction.protect(owned)
                for path in owned: remove(path)
        from src.app.core.provider_loader import invalidate
        invalidate(self.paths, owner.root)
        if installer and installer.on_change_callback: installer.on_change_callback(component.id, 'uninstall')

    def restore_authored(self, component):
        """Explicit development restore into user/config roots, without invoking installation."""
        from src.app.components.transactions import workspace_lock, assert_idle, FileTransaction
        package = Path(self.state()['records'][component.id]['package'])
        from src.app.components.archive import validate_native_package
        meta = json.loads((package / '.theta_component.json').read_text())
        validate_native_package(meta,package)
        kind = normalize_kind(meta['kind'])
        target = self.paths.user(kind if kind in ('model','algorithm','env','datamodule','evaluator','plotter','objective','tuner','plugin','rules') else 'package') / component.id
        if target.exists(): raise ValueError(f'Authoring destination already exists: {target}')
        profiles = meta['payload']['configs']
        with workspace_lock(self.paths):
            assert_idle(self.paths)
            with FileTransaction(self.paths, 'restore-authored', component.id) as transaction:
                transaction.protect([target, *[self.paths.config_file(dest) for dest in profiles]])
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(package,target)
                for dest,relative in profiles.items():
                    active=self.paths.config_file(dest);active.parent.mkdir(parents=True,exist_ok=True)
                    if not active.exists(): shutil.copy2(package/relative,active)
                self._authoring_failure_point('restore', component.id)
        from src.app.core.provider_loader import invalidate
        invalidate(self.paths,target)
        return True

    def deletion_details(self, component, installer):
        record = self.state()['records'].get(component.id)
        if not record or record['source'] != 'local':
            raise ValueError('Only private local components can be deleted.')
        paths = [Path(record['package'])]
        retained = installer._retained_config_dir(component.id)
        if retained.exists():
            paths.append(retained)
        from src.app.core.provider_catalog import get_catalog
        owner = get_catalog(self.paths).package(component.id)
        target = owner.root if owner and owner.writable else installer.resolve_target_dir(component)
        manifest = target / '.theta_component.json'
        if manifest.exists():
            meta = json.loads(manifest.read_text())
            paths.append(target)
            for relative in [*meta.get('config_paths', []),*meta.get('user_config_paths', [])]:
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

    def delete(self, component, installer, *, trash=None):
        """Stage owned files together, then use the operating system Trash."""
        from src.app.core.provider_catalog import get_catalog
        owner = get_catalog(self.paths).package(component.id)
        if owner and not owner.writable:
            raise ValueError(f"{component.id} belongs to {owner.origin}; use managed uninstall or create an editable copy")
        if trash is None:
            raise ValueError('A system Trash adapter is required to delete a local component.')
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
            success, _ = trash(str(staging))
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
        if name is not None and not name.strip(): raise ValueError('Display name cannot be empty')
        assets = find_component_assets(Path(config), workspace_dir=self.workspace)
        if not assets['is_component']: raise ValueError('Create editable copy requires a declared native package')
        from src.app.components.forking import fork_package
        return fork_package(self.paths, assets['comp_id'], new_id, name)

    def _authoring_failure_point(self, boundary, identity):
        """Injection hook for explicit authoring transaction verification."""

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
Working example: {self.paths.bundled_components / 'sequential_spec/sequential_spec.py'}.
Child model selectors remain shared dependencies."""
        else:
            registries = {'algorithm': 'src/app/core/agent_registry.py', 'datamodule': 'src/app/core/datamodule_registry.py',
                          'evaluator': 'src/app/core/eval_registry.py', 'plotter': 'src/app/core/plotter_registry.py'}
            contract_path = self.workspace / registries.get(kind, 'src/app/core/protocols.py')
            contracts = f"Registry/contract: [{contract_path}]({contract_path}). Inspect this component's entry points, constructor and consumer for its required inputs/outputs; model factory conventions do not apply to this kind."

        text = f'''# {assets['comp_name']} developer context (generated)

ID: {assets['comp_id']} · kind: {assets['kind']} · version: {meta.get('version', 'unknown')}
Parent: {meta.get('parent', 'none')}
Project: {self.workspace} · Theta revision: {revision}
Installed source: {root}

Hub originals are read-only: make an independent component copy before editing.
Local component edits affect the next fresh experiment process.
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
- python run_pipeline.py component inspect {assets['comp_id']} (identity, active profiles and source)
- python run_pipeline.py component copy {assets['comp_id']} new_model_id (independent copy of a supported model)
- python run_pipeline.py component config --help (profile inspection, copying, editing and validation)
- python run_pipeline.py component validate-experiment <group>/<experiment> (preflight and effective model sources)
- python run_pipeline.py <group>/<experiment> dry_run=true (inspect/validate an existing experiment)
- python run_pipeline.py <group>/<experiment> (run it in a fresh process)
- python -m pytest tests/test_local_components.py
Use your own CLI/editor/AI tools and component tests. No publication occurs on save.
'''
        path = self.root / 'context' / f"{assets['comp_id']}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return assets, path
