"""Explicit editable-copy operation; ordinary lifecycle never writes the authoring tree."""
from __future__ import annotations

import ast
import copy
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from src.app.components.transactions import FileTransaction, assert_idle, workspace_lock
from src.app.core.provider_catalog import get_catalog, kind_name
from src.app.core.provider_loader import invalidate


def fork_package(paths, reference, new_id, name=None):
    if not new_id.isidentifier() or new_id.startswith('_'): raise ValueError("Editable copy ID must be a Python identifier starting with a letter")
    catalog = get_catalog(paths)
    owner = catalog.package(reference)
    if owner is None: raise ValueError(f"Unknown package {reference}")
    existing = catalog.package(new_id)
    if existing: raise ValueError(f"{'Package ID' if existing.id == new_id else 'Runtime identity'} already exists: {new_id}")
    if any(catalog.candidates(c.kind,new_id) for p in catalog.packages for c in p.capabilities):
        raise ValueError(f"Runtime identity already exists: {new_id}")
    kind = kind_name(owner.manifest.get('kind', owner.capabilities[0].kind if owner.capabilities else 'package'))
    target = paths.user(kind if kind in ('model','algorithm','datamodule','evaluator','env','plotter','objective','rules','tuner','plugin') else 'package') / new_id
    if target.exists(): raise ValueError(f"User destination already exists: {target}")
    primary = [c for c in owner.capabilities if c.kind == kind]
    identities = {c.id: (new_id if len(primary)==1 and c is primary[0] else f"{new_id}/{c.id}") for c in owner.capabilities}
    aliases = {old: identities[c.id] for c in owner.capabilities for old in c.aliases}
    selectors = {**identities, **aliases}
    declared_configs = dict(owner.manifest.get('payload', {}).get('configs', {}))
    declared_configs.update({dest:'profiles/'+dest.removeprefix('in/config/') for dest in owner.manifest.get('user_config_paths',[]) if dest not in declared_configs})
    destination_map = {}
    for dest in declared_configs:
        path = Path(dest)
        peers = [Path(value) for value in declared_configs if Path(value).parts[:3] == path.parts[:3]]
        if len(peers)==1 and path.stem==owner.id:
            destination_map[dest]=(Path(*path.parts[:3])/(new_id+path.suffix)).as_posix()
        else:
            import os
            common=Path(os.path.commonpath([str(value.parent) for value in peers]))
            destination_map[dest]=(Path(*path.parts[:3])/new_id/path.relative_to(common)).as_posix()
    def map_profile(value):
        for old,new in destination_map.items():
            if value==old: return new
            if value=='/'+old.removeprefix('in/config/').rsplit('.',1)[0]:
                return '/'+new.removeprefix('in/config/').rsplit('.',1)[0]
        return value
    def map_defaults(values):
        result=[]
        for value in values:
            if isinstance(value,str): result.append(map_profile(value));continue
            if not isinstance(value,dict): result.append(value);continue
            mapped={}
            for key,profile in value.items():
                group=key.removeprefix('override ').strip('/').split('@')[0]
                old='in/config/'+group+'/'+str(profile)+'.yaml'
                if old not in destination_map: mapped[key]=profile;continue
                new=Path(destination_map[old]).relative_to('in/config').with_suffix('')
                if '/' in group:
                    new_key=key.replace(group,new.parent.as_posix(),1)
                    mapped[new_key]=new.name
                else: mapped[key]=new.relative_to(group).as_posix()
            result.append(mapped)
        return result
    profiles = {}
    with workspace_lock(paths):
        assert_idle(paths, owner.id)
        with tempfile.TemporaryDirectory(prefix='theta-editable-') as temporary:
            stage = Path(temporary) / 'package'
            shutil.copytree(owner.root, stage, ignore=shutil.ignore_patterns('__pycache__','*.pyc','node_modules','.git'))
            for file in stage.rglob('*'):
                if file.is_file(): file.chmod(file.stat().st_mode | 0o200)
            # Only declared registration identities and known import namespaces are changed in source.
            for file in stage.rglob('*.py'):
                source = file.read_text()
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    module = node.module if isinstance(node,ast.ImportFrom) and not node.level else ''
                    imports = [v.name for v in node.names] if isinstance(node,ast.Import) else [module]
                    if any(name.startswith(('src.usr.','usr.','in.envs.','plot.')) for name in imports if name):
                        raise ValueError(f'Native copy requires package-relative imports: {file}')
                class Remap(ast.NodeTransformer):
                    def visit_Constant(self, node):
                        if isinstance(node.value,str) and node.value.startswith('package://' + owner.id + '/'):
                            node.value = 'package://' + new_id + '/' + node.value.split('/',3)[3]
                        if isinstance(node.value,str): node.value=map_profile(node.value)
                        return node
                    def visit_Call(self, node):
                        self.generic_visit(node)
                        func = node.func.id if isinstance(node.func, ast.Name) else ''
                        if isinstance(node.func,ast.Attribute) and node.func.attr=='__init__' and node.args:
                            argument=node.args[0]
                            if isinstance(argument,ast.Constant) and argument.value in {c.id for c in owner.capabilities if c.kind=='plotter'}:
                                argument.value=identities[argument.value]
                        if func.startswith('register_'):
                            for arg in node.args:
                                if isinstance(arg, ast.Constant) and arg.value in selectors: arg.value = selectors[arg.value]
                        return node
                changed = Remap().visit(tree)
                # Preserve comments and formatting if no identity/import changes were needed.
                if ast.dump(ast.parse(source)) != ast.dump(changed): file.write_text(ast.unparse(changed)+'\n')
            metadata = copy.deepcopy(owner.manifest)
            for field in ('source_hashes','authoring_hash','asset_hashes','config_hashes','config_backups','installed_from','installed_files','sha256','target_dir','aliases','provides','suite','hidden','source_manifest','saved_local_changes','user_config_paths'):
                metadata.pop(field, None)
            config_map = dict(owner.manifest.get('payload', {}).get('configs', {}))
            for dest in [*owner.manifest.get('config_paths', []), *owner.manifest.get('user_config_paths', [])]:
                if dest not in config_map and paths.config_file(dest).is_file():
                    config_map[dest] = 'profiles/' + dest.removeprefix('in/config/')
                    saved = stage / config_map[dest]
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(paths.config_file(dest), saved)
            if not config_map and kind=='model' and len(primary)==1:
                dest='in/config/model/'+new_id+'.yaml'
                relative='configs/model/'+new_id+'.yaml'
                (stage/relative).parent.mkdir(parents=True,exist_ok=True)
                (stage/relative).write_text(yaml.safe_dump({'architecture':new_id}))
                config_map[dest]=relative;destination_map[dest]=dest
            for dest, relative in config_map.items():
                old = paths.config_file(dest)
                relative = str(relative)
                if old.is_file(): shutil.copy2(old, stage / relative)
                profile_text = (stage / relative).read_text()
                data = yaml.safe_load(profile_text) or {}
                if kind=='model' and dest.startswith('in/config/model/') and len(primary)==1 and not any(k in data for k in ('architecture','name','type')):
                    data['architecture']=primary[0].id
                def remap(value):
                    if isinstance(value, dict):
                        return {k: (map_defaults(v) if k=='defaults' and isinstance(v,list) else selectors.get(v,v) if k in ('name','architecture','algorithm','component','type') and isinstance(v,str) else remap(v)) for k,v in value.items()}
                    if isinstance(value,str) and value.startswith('package://' + owner.id + '/'):
                        return 'package://' + new_id + '/' + value.split('/',3)[3]
                    if isinstance(value, list): return [remap(v) for v in value]
                    return map_profile(value) if isinstance(value,str) else value
                leading = []
                for line in profile_text.splitlines(keepends=True):
                    if line.strip() and not line.lstrip().startswith('#'): break
                    leading.append(line)
                new_dest=destination_map[dest]
                selected='configs/'+new_dest.removeprefix('in/config/')
                (stage/selected).parent.mkdir(parents=True,exist_ok=True)
                (stage/selected).write_text(''.join(leading) + yaml.safe_dump(remap(data), sort_keys=False))
                profiles[new_dest]=selected
            # A copy supplies defaults for its own selectors, never its parent's.
            for relative in config_map.values():
                if relative.startswith('configs/') and relative not in profiles.values():
                    (stage/relative).unlink(missing_ok=True)
            schema = metadata.get('parameter_schema', {})
            if isinstance(schema, dict):
                profile_schemas = schema.get('x-theta-config-schemas', {})
                schema['x-theta-config-schemas'] = {destination_map.get(key,key):value for key,value in profile_schemas.items()}
                for selected_schema in [schema, *profile_schemas.values()]:
                    for key in ('name','architecture','algorithm','component','type'):
                        domain = selected_schema.get('properties', {}).get(key)
                        if not isinstance(domain,dict): continue
                        for field in ('const','default'):
                            if isinstance(domain.get(field),str): domain[field] = selectors.get(domain[field],domain[field])
                        if isinstance(domain.get('enum'),list): domain['enum'] = [selectors.get(v,v) if isinstance(v,str) else v for v in domain['enum']]
            metadata = remap(metadata) if config_map else metadata
            metadata.update(id=new_id, name=name or new_id, version='0.1.0', origin='user',
                parent={'id':owner.id,'version':owner.version,'origin':owner.origin}, storage_version=2, api_version=1,
                capabilities=[{**c.metadata,'id':identities[c.id],'kind':c.kind,'entry_point':c.entry_point,'aliases':[]} for c in owner.capabilities],
                config_paths=list(profiles), config_path=next(iter(profiles), None),
                payload={**metadata.get('payload',{}),'configs':profiles})
            # Profile names are explicit capability selectors in the authored manifest.
            groups={'model':'model','algorithm':'agent','env':'env','datamodule':'data','evaluator':'eval','plotter':'plot'}
            for original,cap in zip(owner.capabilities,metadata['capabilities']):
                group=groups.get(original.kind)
                for dest,relative in profiles.items():
                    if not group or not dest.startswith('in/config/'+group+'/'): continue
                    data=yaml.safe_load((stage/relative).read_text()) or {}
                    selected=data.get('architecture') or data.get('type') or data.get('name') or data.get('algorithm')
                    if original.kind=='plotter':selected=Path(dest).stem
                    matches={cap['id'].lower(),original.id.lower(),*(a.lower() for a in original.aliases)}
                    if not isinstance(selected,str) or selected.lower() not in matches: continue
                    alias=dest.removeprefix('in/config/'+group+'/').rsplit('.',1)[0]
                    if alias.lower() != cap['id'].lower() and alias.lower() not in {v.lower() for v in cap['aliases']}:cap['aliases'].append(alias)
            if (stage / 'plugin.json').is_file():
                plugin = json.loads((stage / 'plugin.json').read_text());plugin.update(id=new_id,name=name or new_id,default_enabled=False)
                plugin.pop('core',None);plugin.pop('is_core',None)
                (stage/'plugin.json').write_text(json.dumps(plugin,indent=2)+'\n')
            (stage / '.theta_component.json').write_text(json.dumps(metadata,indent=2)+'\n')
            from src.app.components.archive import validate_native_package
            validate_native_package(metadata,stage)
            from src.app.services.local_components import LocalComponents
            local = LocalComponents(paths)
            retained = local.root / 'packages' / new_id
            if retained.exists() or new_id in local.state()['records']: raise ValueError(f'Component ID already exists: {new_id}')
            with FileTransaction(paths,'editable-copy',new_id) as transaction:
                destinations = [paths.config_file(dest) for dest in profiles]
                if any(p.exists() for p in destinations): raise ValueError('An editable config destination already exists')
                transaction.protect([target,*destinations,retained,local.state_path])
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(stage,target)
                retained.parent.mkdir(parents=True,exist_ok=True)
                shutil.copytree(stage,retained)
                state = local.state()
                state["records"][new_id] = {"package":str(retained),"source":"local"}
                local._write(state)
                for dest,relative in profiles.items():
                    active=paths.config_file(dest);active.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(stage/relative,active)
                local._authoring_failure_point("copy",new_id)
            invalidate(paths,target)
    return paths.config_file(next(iter(profiles))) if profiles else target / '.theta_component.json'
