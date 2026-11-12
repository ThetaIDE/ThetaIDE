"""Native declarations are the only evidence for discovery, installation and preflight."""
import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from src.app.components.archive import validate_native_package
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent,ReleaseInfo
from src.app.core.provider_catalog import ProviderCatalog,DuplicateProvider
from src.app.workspace import WorkspacePaths


def package(root, name='native', **changes):
    root.mkdir(parents=True,exist_ok=True)
    (root/'model.py').write_text('class Model: pass\n')
    meta={'id':name,'name':name,'kind':'model','version':'1.0.0','storage_version':2,'api_version':1,
          'capabilities':[{'id':name,'kind':'model','entry_point':'model.py:Model'}],
          'payload':{'implementations':['model.py'],'resources':[],'configs':{}},**changes}
    (root/'.theta_component.json').write_text(json.dumps(meta))
    return meta


@pytest.mark.parametrize('field,value', [('storage_version',None),('storage_version',1),('api_version',None),('legacy_paths',{}),('module_aliases',{}),('target_path','src/usr/models/native')])
def test_legacy_or_missing_contract_is_rejected_without_installation(tmp_path,field,value):
    source=tmp_path/'source';meta=package(source,**{field:value})
    archive=tmp_path/'native.zip'
    with ZipFile(archive,'w') as z:
        for file in source.iterdir():z.write(file,file.name)
    installer=HubInstaller(tmp_path/'workspace',tmp_path/'custom-state')
    comp=HubComponent.from_dict(meta);comp.releases={comp.version:ReleaseInfo(comp.version,archive.as_uri(),'')}
    with pytest.raises(ValueError):installer.install(comp,install_dependencies=False)
    assert not installer.paths.installed_components.exists()
    assert not installer.paths.research.exists()


def test_configs_registries_and_old_directories_do_not_establish_availability(tmp_path,monkeypatch):
    paths=WorkspacePaths(tmp_path)
    config=paths.config/'model/ghost.yaml';config.parent.mkdir(parents=True);config.write_text('architecture: ghost\n')
    source=paths.user('model')/'ghost';source.mkdir(parents=True)
    (source/'model.py').write_text("raise AssertionError('legacy discovery imported source')\n")
    old=tmp_path/'in/envs/ghost';old.mkdir(parents=True)
    (old/'env_vectorized.py').write_text("raise AssertionError('legacy discovery imported environment')")
    monkeypatch.setenv('THETA_WORKSPACE',str(tmp_path))
    from src.app.core.model_registry import register_model
    register_model('ghost')(type('Ghost',(),{}))
    from src.app.pipeline.validation import _model_exists,_env_exists
    assert not _model_exists('ghost') and not _env_exists('ghost')
    assert ProviderCatalog(paths).resolve('model','ghost') is None


def test_preflight_proves_selected_provider_loads(tmp_path,monkeypatch):
    root=WorkspacePaths(tmp_path).user('model')/'broken';package(root,'broken')
    (root/'model.py').write_text("import absent_native_dependency_123\nclass Model: pass\n")
    monkeypatch.setenv('THETA_WORKSPACE',str(tmp_path))
    from src.app.pipeline.validation import _model_exists
    from src.app.pipeline.exceptions import ConfigurationError
    with pytest.raises(ConfigurationError,match='absent_native_dependency_123'):_model_exists('broken')


def test_alias_collisions_are_diagnosed_and_unknown_qualified_selectors_are_not_guessed(tmp_path,monkeypatch):
    paths=WorkspacePaths(tmp_path)
    for name in ('one','two'):
        package(paths.user('model')/name,name,capabilities=[{'id':name,'kind':'model','entry_point':'model.py:Model','aliases':['shared']}])
    catalog=ProviderCatalog(paths)
    assert any('Duplicate capability model:shared' in d for d in catalog.diagnostics)
    with pytest.raises(DuplicateProvider):catalog.resolve('model','shared')
    from src.app.core.provider_loader import resolve_provider
    assert resolve_provider('model','undeclared/one',paths) is None


def test_export_refuses_to_infer_implementation_from_a_loose_config(tmp_path):
    paths=WorkspacePaths(tmp_path)
    config=paths.config/'model/loose.yaml';config.parent.mkdir(parents=True);config.write_text('architecture: loose\n')
    root=paths.user('model')/'loose';root.mkdir(parents=True)
    (root/'model.py').write_text("from src.app.core.model_registry import register_model\n@register_model('loose')\nclass Model: pass\n")
    from src.app.services.component_exporter import ComponentExporter,ComponentExportOptions
    output=tmp_path/'loose.zip'
    result=ComponentExporter(tmp_path).export(config,ComponentExportOptions(output_path=output))
    assert not result.success and 'native package' in result.error
    assert not output.exists()


def test_requested_release_version_is_checked_against_the_archive(tmp_path):
    source=tmp_path/'source';meta=package(source)
    archive=tmp_path/'old-release.zip'
    with ZipFile(archive,'w') as z:
        for file in source.iterdir():z.write(file,file.name)
    comp=HubComponent.from_dict({**meta,'version':'2.0.0'})
    comp.releases={'1.0.0':ReleaseInfo('1.0.0',archive.as_uri(),'')}
    installer=HubInstaller(tmp_path/'workspace',tmp_path/'state')
    installer.install(comp,version='1.0.0',install_dependencies=False)
    installed=ProviderCatalog(installer.paths).package('native')
    assert installed.version=='1.0.0'
    installed_meta=json.loads((installed.root/'.theta_component.json').read_text())
    assert not installer.has_local_changes(installed.root,installed_meta)
    installed_meta['capabilities'][0]['aliases']=['edited_selector']
    assert installer.has_local_changes(installed.root,installed_meta)


def test_explicit_native_config_mapping_supplies_packaged_default(tmp_path):
    source=tmp_path/'source'
    meta=package(source,payload={'implementations':['model.py'],'resources':[],
        'configs':{'in/config/model/native.yaml':'profile.yaml'}})
    (source/'profile.yaml').write_text('architecture: native\nwidth: 8\n')
    archive=tmp_path/'native.zip'
    with ZipFile(archive,'w') as z:
        for file in source.iterdir():z.write(file,file.name)
    comp=HubComponent.from_dict(meta);comp.releases={comp.version:ReleaseInfo(comp.version,archive.as_uri(),'')}
    installer=HubInstaller(tmp_path/'workspace',tmp_path/'state')
    installer.install(comp,install_dependencies=False)
    (installer.paths.config/'model/native.yaml').unlink()
    assert installer.paths.resolve_config('model/native.yaml')==installer.paths.installed_components/'native/configs/model/native.yaml'
    assert installer.paths.resolve_config('model/native.yaml').read_text()=='architecture: native\nwidth: 8\n'


def test_episode_outcomes_require_a_declared_runner_contract(tmp_path,monkeypatch):
    from src.app.pipeline.engine import runner_supports_episode_outcomes
    from src.app.core.provider_loader import invalidate
    paths=WorkspacePaths(tmp_path)
    from src.app.core.provider_loader import resolve_resource
    assert resolve_resource('src/app/train.py',paths)==paths.application/'src/app/train.py'
    assert resolve_resource('old/src/app/train.py',paths)==paths.root/'old/src/app/train.py'
    root=paths.user('algorithm')/'native'
    meta=package(root,kind='method',capabilities=[{'id':'native','kind':'algorithm',
        'entry_point':'model.py:Model'}])
    monkeypatch.setenv('THETA_WORKSPACE',str(tmp_path))
    assert not runner_supports_episode_outcomes({'paradigm':'online_rl','agent':'missing'})
    assert not runner_supports_episode_outcomes({'paradigm':'online_rl','agent':'native'})
    meta['capabilities'][0]['episode_outcomes']=True
    (root/'.theta_component.json').write_text(json.dumps(meta))
    invalidate(paths,root)
    assert runner_supports_episode_outcomes({'paradigm':'online_rl','agent':'native'})
