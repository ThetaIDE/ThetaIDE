"""App lifecycle ownership and cleanup without starting unrelated terminal/WebEngine panels."""
import json
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

pytest.importorskip('PyQt6')
from frontend.plugins.commands import CommandRegistry
from frontend.plugins.manager import PluginManager
from frontend.plugins.slots import ExtensionSlotRegistry
from frontend.settings import SettingsManager
from src.app.components.installer import HubInstaller
from src.app.components.models import HubComponent, ReleaseInfo


def test_config_tree_fallback_uses_selected_workspace(tmp_path, monkeypatch):
    from frontend.config_tree import find_config_root
    selected = tmp_path / 'selected'
    monkeypatch.setenv('THETA_WORKSPACE', str(selected))
    assert find_config_root() == selected / 'in/config'


def test_managed_app_plugin_uses_catalog_identity_and_releases_contributions(tmp_path):
    archive=tmp_path/'panel.zip'
    meta={'id':'my_panel','name':'My panel','kind':'plugin','version':'1.0.0','storage_version':2,'api_version':1,
          'dependencies':{'components':['sequential_spec']},
          'payload':{'implementations':['__init__.py','helper.py'],'resources':[],'configs':{}},
          'capabilities':[{'id':'my_panel','kind':'plugin','entry_point':'__init__.py:Panel'}]}
    with ZipFile(archive,'w') as z:
        z.writestr('.theta_component.json',json.dumps(meta))
        z.writestr('plugin.json',json.dumps({'id':'my_panel','entry_point':'Panel','core':True,'default_enabled':True}))
        z.writestr('helper.py','VALUE=7\n')
        z.writestr('__init__.py', '''from frontend.plugins.base import Plugin
from .helper import VALUE
class Panel(Plugin):
    def activate(self,context):
        self.context=context
        context.register_command('my_panel.run','Run',lambda:VALUE)
        context.set_setting('visits',context.get_setting('visits',0)+1)
    def deactivate(self): pass
''')
    component=HubComponent.from_dict(meta);component.releases={'1.0.0':ReleaseInfo('1.0.0',archive.as_uri(),'')}
    installer=HubInstaller(tmp_path/'workspace',tmp_path/'state');installer.install(component,install_dependencies=False)
    logs=[]
    window=SimpleNamespace(log=logs.append,store=SimpleNamespace(root=installer.paths.data_dir),command_registry=CommandRegistry(),extension_slots=ExtensionSlotRegistry())
    settings=SettingsManager(installer.paths.data_dir)
    manager=PluginManager(window,paths=installer.paths,settings_manager=settings)
    from src.app.components.development import ComponentDevelopment
    copied=ComponentDevelopment(installer.paths).copy('my_panel','my_panel_copy')
    assert copied.is_file()
    manager.discover()
    assert not manager.manifests['my_panel_copy'].default_enabled
    assert manager.manifests['my_panel_copy'].extra['origin']=='user'
    assert not manager.manifests['my_panel'].is_core
    assert manager.manifests['model_builder'].is_core
    assert manager.enable_plugin('my_panel'),logs
    assert window.command_registry.get_command('my_panel.run').callback()==7
    with pytest.raises(RuntimeError,match='in use'):installer.install(component,install_dependencies=False)
    assert manager.disable_plugin('my_panel')
    assert window.command_registry.get_command('my_panel.run') is None
    installer.install(component,install_dependencies=False)
    manager.discover();assert manager.enable_plugin('my_panel'),logs
    assert manager.contexts['my_panel'].get_setting('visits')==2
    assert manager.disable_plugin('my_panel')
    manager.uninstall_plugin('my_panel')
    assert not installer.resolve_target_dir(component).exists()
