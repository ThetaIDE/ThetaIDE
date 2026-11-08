"""Model lifecycle regression tests using real packages and the shared installer."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from frontend.hub.client import HubClient
from frontend.hub.installer import HubInstaller
from frontend.hub.models import HubComponent
from src.app.services.local_components import LocalComponents


@pytest.fixture
def model_workspace(tmp_path):
    root = tmp_path
    source = root / 'src/usr/models/parent'
    source.mkdir(parents=True)
    (root / 'pyproject.toml').write_text('version = "0.1.0"\n')
    (source / '__init__.py').write_text('from .model import Example\n')
    (source / 'model.py').write_text('from src.app.core.model_registry import register_model\n@register_model("parent", "legacy_parent")\nclass Example:\n    def __init__(self, **kwargs): pass\n    def forward(self): return 1\n')
    (source / 'rules.txt').write_text('owned asset')
    profiles = ['in/config/model/parent/default.yaml', 'in/config/model/parent/wide.yaml']
    for profile in profiles:
        path = root / profile
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('architecture: parent\nchild:\n  name: external_child\n')
    meta = dict(id='parent', name='Parent', kind='model', version='1.0.0', target_path='src/usr/models/parent', config_path=profiles[0], config_paths=profiles, aliases=['legacy_parent'], parameter_schema={'type': 'object', 'properties': {}, 'x-theta-config-schemas': {p: {'properties': {'width': {'type': 'integer'}}} for p in profiles}})
    meta.pop('target_path')
    meta.update(storage_version=2,api_version=1,capabilities=[{'id':'parent','kind':'model','entry_point':'model.py:Example','aliases':['legacy_parent']}],
        payload={'implementations':['__init__.py','model.py'],'resources':['rules.txt'],'configs':{v:'configs/'+v.removeprefix('in/config/') for v in profiles}})
    for dest,relative in meta['payload']['configs'].items():
        packaged=source/relative;packaged.parent.mkdir(parents=True,exist_ok=True);packaged.write_bytes((root/dest).read_bytes())
    (source / '.theta_component.json').write_text(json.dumps(meta))
    return root, root / profiles[0]


def runtime(workspace, component):
    # A new interpreter is the same boundary used by experiment launches.
    code = f"""from src.app.core.model_registry import build_model
print(build_model({component!r}).forward())
"""
    return subprocess.check_output([sys.executable, '-c', code], text=True,
        cwd=Path(__file__).parents[1], env={**os.environ, 'THETA_WORKSPACE': str(workspace)}).strip()



def test_independent_identity_direct_save_profiles_context_and_hide(model_workspace):
    ws, config = model_workspace
    service = LocalComponents(ws)
    copied = service.deep_copy(config, 'fork', 'My fork')
    assert copied == ws / 'in/config/model/fork/default.yaml'
    meta = json.loads((ws / 'src/usr/models/fork/.theta_component.json').read_text())
    assert set(meta['config_paths']) == {'in/config/model/fork/default.yaml', 'in/config/model/fork/wide.yaml'}
    assert set(meta['parameter_schema']['x-theta-config-schemas']) == set(meta['config_paths'])
    assert 'aliases' not in meta
    assert meta['parent']['id'] == 'parent'
    from src.app.workspace import WorkspacePaths
    copied.unlink()
    assert WorkspacePaths(ws).resolve_config('model/fork/default.yaml') == ws/'src/usr/models/fork/configs/model/fork/default.yaml'
    assert not (ws/'src/usr/models/fork/configs/model/parent/default.yaml').exists()
    copied.write_bytes(WorkspacePaths(ws).resolve_config('model/fork/default.yaml').read_bytes())
    assert (ws / 'src/usr/models/fork/rules.txt').read_text() == 'owned asset'
    assert yaml.safe_load(copied.read_text())['child']['name'] == 'external_child'
    assert runtime(ws, 'parent') == runtime(ws, 'fork') == '1'
    implementation = ws / 'src/usr/models/fork/model.py'
    implementation.write_text(implementation.read_text().replace('return 1', 'return 77'))
    assert runtime(ws, 'fork') == '77'
    assert runtime(ws, 'parent') == '1'
    parent = ws / 'src/usr/models/parent/model.py'
    parent.write_text(parent.read_text().replace('return 1', 'return 23'))
    assert runtime(ws, 'parent') == '23'
    assets, context = service.context(copied)
    assert assets['code_dir'] == implementation.parent
    assert str(copied) in context.read_text()
    service.set_hidden('fork')
    assert LocalComponents(ws).hidden('fork')
    assert copied in service.hidden_config_paths()
    assert runtime(ws, 'fork') == '77'
    from frontend.config_model import ConfigTree
    tree = ConfigTree(ws / 'in/config')
    assert 'fork/default' in tree.models
    assert 'fork/default' not in tree.model_names
    service.set_hidden('fork', False)
    assert 'fork/default' in ConfigTree(ws / 'in/config').model_names


def test_uninstall_reinstall_retains_saved_source_and_offline_catalog(model_workspace):
    ws, config = model_workspace
    service = LocalComponents(ws)
    service.deep_copy(config, 'fork')
    implementation = ws / 'src/usr/models/fork/model.py'
    implementation.write_text(implementation.read_text().replace('return 1', 'return 42'))
    component = service.entries()[0]
    installer = HubInstaller(ws, ws / '.thetaide')
    with pytest.raises(ValueError, match='user|local changes'):
        service.install(component, installer)
    service.archive_authored(component, installer)
    assert not implementation.exists()
    assert not (ws / 'in/config/model/fork/default.yaml').exists()
    assert Path(service.state()['records']['fork']['package']).is_dir()
    assert (ws / 'src/usr/models/parent/model.py').exists()
    service.restore_authored(component)
    assert runtime(ws, 'fork') == '42'
    index = ws / 'index.json'
    index.write_text(json.dumps({'components': [dict(id='remote', kind='model', name='Remote', version='1.0.0')]}))
    client = HubClient(ws, ws / '.thetaide', registry_url=index.as_uri())
    assert {c.id for c in client.fetch_index_sync()} == {'fork', 'remote'}
    client.local_components.set_hidden('fork')
    client.local_components.set_hidden('remote')
    assert client.search() == []
    assert {c.id for c in client.search(show_hidden=True)} == {'fork', 'remote'}
    assert [c.id for c in client.search(local_only=True, show_hidden=True)] == ['fork']
    assert json.loads(index.read_text())['components'][0]['id'] == 'remote'


def test_conflict_and_failed_install_roll_back(model_workspace, monkeypatch):
    ws, config = model_workspace
    service = LocalComponents(ws)
    with pytest.raises(ValueError, match='already exists'):
        service.deep_copy(config, 'parent')
    with pytest.raises(ValueError, match='Runtime identity already exists'):
        service.deep_copy(config, 'legacy_parent')
    original = config.read_bytes()
    def failed(self, boundary, identity):
        target = ws / 'src/usr/models/broken'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'partial.py').write_text('partial')
        raise OSError('interrupted')
    monkeypatch.setattr(LocalComponents, '_authoring_failure_point', failed)
    with pytest.raises(OSError, match='interrupted'):
        service.deep_copy(config, 'broken')
    assert not (ws / 'src/usr/models/broken').exists()
    assert 'broken' not in service.state()['records']
    assert not (service.root / 'packages/broken').exists()
    assert original == config.read_bytes()


def test_delete_moves_only_owned_paths_to_trash(model_workspace, monkeypatch, tmp_path):
    from PyQt6.QtCore import QFile
    ws, config = model_workspace
    service = LocalComponents(ws)
    service.deep_copy(config, 'fork')
    component = service.entries()[0]
    installer = HubInstaller(ws, ws / '.thetaide')
    experiment = ws / 'in/config/experiment/uses_fork.yaml'
    experiment.parent.mkdir(parents=True)
    experiment.write_text('model: fork/default\n')
    shared = ws / 'src/usr/rules/shared.txt'
    shared.parent.mkdir(parents=True)
    shared.write_text('shared dependency')
    for cid in ('parent', 'fork'):
        manifest = ws / f'src/usr/models/{cid}/.theta_component.json'
        data = json.loads(manifest.read_text())
        data.setdefault('installed_files', []).append('src/usr/rules/shared.txt')
        manifest.write_text(json.dumps(data))
    paths, references = service.deletion_details(component, installer)
    assert references == [experiment]
    trash = tmp_path / 'fake-trash'
    def move(path):
        Path(path).rename(trash)
        return True, str(trash)
    monkeypatch.setattr(QFile, 'moveToTrash', move)
    service.delete(component, installer, trash=QFile.moveToTrash)
    assert trash.is_dir()
    assert not any(path.exists() for path in paths)
    assert experiment.exists()
    assert shared.read_text() == "shared dependency"
    assert (ws / 'src/usr/models/parent/model.py').exists()
    assert service.entries() == []


def test_abandoned_copy_recovery(model_workspace, monkeypatch):
    ws, config = model_workspace
    service = LocalComponents(ws)
    source = ws / 'src/usr/models/interrupted'
    package = service.root / 'packages/interrupted'
    source.mkdir()
    package.mkdir(parents=True)
    service._write({'records': {'interrupted': {'source': 'local', 'package': str(package)}}, 'hidden': []})
    journal = service.root / 'operations/interrupted.json'
    journal.parent.mkdir()
    journal.write_text(json.dumps({'id': 'interrupted', 'pid': 99999999, 'paths': [str(source), str(package)]}))
    def dead_process(pid, signal):
        raise ProcessLookupError()
    monkeypatch.setattr(os, 'kill', dead_process)
    recovered = LocalComponents(ws)
    assert recovered.entries() == []
    assert not source.exists() and not package.exists() and not journal.exists()
    assert config.exists()


def test_hub_hide_controls_and_current_hidden_selector(model_workspace, monkeypatch):
    from types import SimpleNamespace

    from PyQt6.QtWidgets import QApplication, QComboBox

    import frontend.config_viewer as viewer_module
    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer
    from frontend.editor_service import EditorService
    from frontend.hub.dialog import HubView
    app = QApplication.instance() or QApplication([])
    ws, config = model_workspace
    local = LocalComponents(ws)
    copied = local.deep_copy(config, 'fork')
    client = HubClient(ws, ws / '.thetaide')
    client.refresh_installed_status()
    view = HubView(client)
    view._populate_cards(client.components)
    view.cards['fork'].btn_visibility.click()
    assert local.hidden('fork')
    assert view.cards['fork'].isHidden()
    view.show_hidden.setChecked(True)
    assert not view.cards['fork'].isHidden()
    monkeypatch.setattr(viewer_module, 'config_tree', lambda *args, **kwargs: ConfigTree(ws / 'in/config'))
    experiment = ws / 'in/config/experiment/current.yaml'
    experiment.parent.mkdir(parents=True)
    experiment.write_text('methods:\n  trial:\n    agent: missing_agent\n    model: fork/default\n')
    viewer = ConfigViewer()
    viewer.load_file(experiment, 'experiment/current.yaml')
    assert any(combo.currentText() == 'fork/default' for combo in viewer.findChildren(QComboBox))
    assert viewer.raw_data['methods']['trial']['model'] == 'fork/default'
    view.cards['fork'].btn_visibility.click()
    assert not local.hidden('fork')
    routed = []
    editor = EditorService(SimpleNamespace(default_text_editor='test'))
    editor.register('test', 'Test', lambda *args, **kwargs: True,
                    lambda root, **kwargs: routed.append((root, kwargs)) or True,
                    lambda: False)
    assert not editor.prepare_component_copy()
    assert editor.open_folder(ws / 'src/usr/models/fork', initial_file=ws / 'src/usr/models/fork/model.py', owned_paths=[copied])
    assert routed[0][1]['owned_paths'] == [copied]
    viewer.deleteLater()
    view.deleteLater()
    app.processEvents()


def test_profile_copy_rejects_old_imports_and_preserves_relative_imports(model_workspace):
    ws, config = model_workspace
    config.write_text('width: 16\n')
    source = ws / 'src/usr/models/parent'
    (source / 'helper.py').write_text('VALUE = 10\n')
    (source / '__init__.py').write_text('from src.usr.models.parent.model import Example\n')
    service=LocalComponents(ws)
    with pytest.raises(ValueError,match='package-relative imports'):service.deep_copy(config,'fork')
    assert not (ws/'src/usr/models/fork').exists()
    (source/'__init__.py').write_text('from . import helper as own_package\nfrom .model import Example\n')
    manifest=source/'.theta_component.json';meta=json.loads(manifest.read_text())
    meta['payload']['implementations'].append('helper.py');manifest.write_text(json.dumps(meta))
    copied=service.deep_copy(config,'fork')
    assert yaml.safe_load(copied.read_text())['architecture']=='fork'
    assert 'from . import helper as own_package' in (ws/'src/usr/models/fork/__init__.py').read_text()
    assert runtime(ws,'fork')=='1'


def test_schema_edits_are_protected_from_updates(model_workspace):
    ws, config = model_workspace
    service = LocalComponents(ws)
    service.deep_copy(config, 'fork')
    manifest = ws / 'src/usr/models/fork/.theta_component.json'
    meta = json.loads(manifest.read_text())
    meta['parameter_schema']['properties']['new_parameter'] = {'type': 'number'}
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='user|local changes'):
        service.install(service.entries()[0], HubInstaller(ws, ws / '.thetaide'))
    assert json.loads(manifest.read_text())['parameter_schema']['properties']['new_parameter'] == {'type': 'number'}


def test_package_owned_assets_are_copied_independently(model_workspace):
    ws,config=model_workspace
    source=ws/'src/usr/models/parent'
    asset=source/'resources/rules/logic.txt';asset.parent.mkdir(parents=True);asset.write_text('parent rules')
    manifest=source/'.theta_component.json';meta=json.loads(manifest.read_text())
    meta['payload']['resources'].append('resources/rules')
    meta['editor_actions']=[{'label':'Rules','path':'package://parent/resources/rules/logic.txt'}]
    manifest.write_text(json.dumps(meta))
    config.write_text(config.read_text()+'rules: package://parent/resources/rules/logic.txt\n')
    implementation=source/'model.py';implementation.write_text(implementation.read_text()+"\nRULES = 'package://parent/resources/rules/logic.txt'\n")
    local=LocalComponents(ws);copied=local.deep_copy(config,'fork')
    fork_asset=ws/'src/usr/models/fork/resources/rules/logic.txt'
    assert fork_asset.read_text()=='parent rules'
    assert yaml.safe_load(copied.read_text())['rules']=='package://fork/resources/rules/logic.txt'
    assert "RULES = 'package://fork/resources/rules/logic.txt'" in (ws/'src/usr/models/fork/model.py').read_text()
    meta=json.loads((ws/'src/usr/models/fork/.theta_component.json').read_text())
    assert meta['editor_actions'][0]['path']=='package://fork/resources/rules/logic.txt'
    fork_asset.write_text('fork rules');assert asset.read_text()=='parent rules'
    component=local.entries()[0];installer=HubInstaller(ws,ws/'.thetaide')
    local.archive_authored(component,installer);assert not fork_asset.exists()
    local.restore_authored(component);assert fork_asset.read_text()=='fork rules'


def test_saved_experiment_profile_runs_its_declared_identity_after_restart(model_workspace):
    """Use the experiment parser, command builder and Hydra profile composition."""
    ws, config = model_workspace
    local = LocalComponents(ws)
    local.deep_copy(config, 'fork')
    model_file = ws / 'src/usr/models/fork/model.py'
    model_file.write_text(model_file.read_text().replace('return 1', 'return 81'))
    config_root = ws / 'in/config'
    (config_root / 'config.yaml').write_text('defaults:\n  - _self_\n  - model: null\n  - agent: null\n  - paradigm: null\n')
    for kind in ('agent', 'paradigm', 'experiment'):
        (config_root / kind).mkdir(exist_ok=True)
    (config_root / 'agent/probe.yaml').write_text('algorithm: probe\n')
    (config_root / 'paradigm/online_rl.yaml').write_text('{}\n')
    experiment = config_root / 'experiment/check.yaml'
    experiment.write_text('methods:\n  parent_run:\n    agent: probe\n    model: parent/default\n  fork_run:\n    agent: probe\n    model: fork/default\n')
    code = f"""import json
import src.usr.models
src.usr.models.__path__.insert(0, {str(ws / 'src/usr/models')!r})
from src.app.core import model_registry
model_registry.__file__ = {str(ws / 'src/app/core/model_registry.py')!r}
from src.app.pipeline.config import parse_methods_dict
from src.app.pipeline.commands import build_method_overrides
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
experiment = OmegaConf.load({str(experiment)!r})
results = {{}}
for name, method in parse_methods_dict(experiment).items():
    overrides = build_method_overrides(name, method, cfg={{'paradigm': 'online_rl', 'experiment_name': 'check'}})
    with initialize_config_dir(version_base=None, config_dir={str(config_root)!r}):
        composed = compose(config_name='config', overrides=overrides)
    assert composed.model.architecture in ('parent', 'fork'), composed.model
    results[name] = model_registry.build_model(composed.model).forward()
print(json.dumps(results))
"""
    def run():
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
        return json.loads(subprocess.check_output([sys.executable, '-c', code], cwd=ws, env=env, text=True))
    assert run() == {'parent_run': 1, 'fork_run': 81}
    local.set_hidden('fork')
    assert LocalComponents(ws).hidden('fork')
    assert run() == {'parent_run': 1, 'fork_run': 81}


def test_hub_inline_deep_copy_and_edit(model_workspace):
    from types import SimpleNamespace

    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLineEdit, QWidget

    from frontend.editor_service import EditorService
    from frontend.hub.dialog import HubView
    app = QApplication.instance() or QApplication([])
    ws, config = model_workspace
    client = HubClient(ws, ws / '.thetaide')
    # Hub cards come from the registry, not a retained source snapshot.
    parent = HubComponent.from_dict(json.loads((ws / 'src/usr/models/parent/.theta_component.json').read_text()))
    client.components = [parent]
    client.refresh_installed_status()
    host = QWidget()
    host.hub_client = client
    host.editor_service = EditorService(SimpleNamespace(default_text_editor='test'))
    opened = []
    host.editor_service.register('test', 'Test', lambda *args, **kwargs: True,
        lambda root, **kwargs: opened.append((root, kwargs)) or True, lambda: True)
    view = HubView(client, parent=host)
    view._populate_cards(client.components)
    host.resize(1100, 800)
    host.show()
    view.show()
    app.processEvents()
    view.cards['parent'].btn_deep_copy.click()
    editor = view.cards['parent'].findChild(QLineEdit, 'componentCopyName')
    assert editor is not None and editor.text() == ''
    assert 'fork' not in client.local_components.state()['records']
    QTest.keyClicks(editor, 'Private Fork')
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert 'private_fork' in view.cards
    assert client.get_component('private_fork').name == 'Private Fork'
    assert client.get_component('private_fork').is_installed
    view.cards['private_fork'].btn_edit.click()
    assert opened[0][0] == ws / 'src/usr/models/private_fork'
    assert ws / 'in/config/model/private_fork/default.yaml' in opened[0][1]['owned_paths']
    client.local_components.archive_authored(client.get_component('private_fork'), client.installer)
    assert not view.cards['private_fork'].btn_edit.isEnabled()
    host.deleteLater()
    app.processEvents()


def test_failed_reinstall_rolls_back_and_retains_package(model_workspace, monkeypatch):
    ws, config = model_workspace
    local = LocalComponents(ws)
    local.deep_copy(config, 'fork')
    component = local.entries()[0]
    installer = HubInstaller(ws, ws / '.thetaide')
    local.archive_authored(component, installer)
    package = Path(local.state()['records']['fork']['package'])
    config = ws / 'in/config/model/fork/default.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('preexisting: keep me\n')
    def fail(boundary, identity):
        target = ws / 'src/usr/models/fork'
        target.mkdir(parents=True, exist_ok=True)
        (target / 'model.py').write_text('partial')
        config.write_text('overwritten')
        raise OSError('interrupted reinstall')
    monkeypatch.setattr(local, '_authoring_failure_point', fail)
    with pytest.raises(OSError, match='interrupted reinstall'):
        local.restore_authored(component)
    assert config.read_text() == 'preexisting: keep me\n'
    assert not (ws / 'src/usr/models/fork').exists()
    assert package.is_dir()
    assert LocalComponents(ws).entries()[0].id == 'fork'


def test_trash_failure_restores_all_owned_locations(model_workspace, monkeypatch):
    from PyQt6.QtCore import QFile
    ws, config = model_workspace
    local = LocalComponents(ws)
    local.deep_copy(config, 'fork')
    component = local.entries()[0]
    installer = HubInstaller(ws, ws / '.thetaide')
    paths, _ = local.deletion_details(component, installer)
    monkeypatch.setattr(QFile, 'moveToTrash', lambda path: (False, ''))
    with pytest.raises(OSError, match='Trash'):
        local.delete(component, installer, trash=QFile.moveToTrash)
    assert all(path.exists() for path in paths)
    assert LocalComponents(ws).entries()[0].id == 'fork'


def test_external_schema_and_visibility_changes_refresh_config_fingerprint(model_workspace):
    from frontend.config_model import ConfigTree, config_fingerprint
    ws, config = model_workspace
    before = config_fingerprint(ws / 'in/config')
    manifest = ws / 'src/usr/models/parent/.theta_component.json'
    meta = json.loads(manifest.read_text())
    meta['parameter_schema']['x-theta-config-schemas']['in/config/model/parent/default.yaml']['properties']['width']['maximum'] = 31
    manifest.write_text(json.dumps(meta))
    changed = config_fingerprint(ws / 'in/config')
    assert changed != before
    assert ConfigTree(ws / 'in/config').model_parameter_schema('parent/default')['properties']['width']['maximum'] == 31
    LocalComponents(ws).set_hidden('parent')
    assert config_fingerprint(ws / 'in/config') != changed


def test_retained_edits_remain_protected_after_reinstall(model_workspace):
    ws, config = model_workspace
    local = LocalComponents(ws)
    local.deep_copy(config, 'fork')
    component = local.entries()[0]
    implementation = ws / 'src/usr/models/fork/model.py'
    implementation.write_text(implementation.read_text().replace('return 1', 'return 82'))
    installer = HubInstaller(ws, ws / '.thetaide')
    local.archive_authored(component, installer)
    local.restore_authored(component)
    assert runtime(ws, 'fork') == '82'
    installed_meta = json.loads((implementation.parent / '.theta_component.json').read_text())
    assert installed_meta['saved_local_changes'] is True
    with pytest.raises(ValueError, match='user|local changes'):
        local.install(component, installer)
    assert runtime(ws, 'fork') == '82'


def test_install_release_then_copy_saved_implementation(model_workspace, tmp_path):
    import hashlib
    import shutil

    from frontend.hub.models import ReleaseInfo
    from src.app.services.component_exporter import ComponentExporter, ComponentExportOptions
    source_ws, config = model_workspace
    package = tmp_path / 'release'
    exported = ComponentExporter(source_ws).export(config, ComponentExportOptions(export_format='folder', output_path=package, generate_git_scaffold=False))
    assert exported.success
    archive = Path(shutil.make_archive(str(tmp_path / 'parent-release'), 'zip', package))
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    component = HubComponent.from_dict(exported.manifest)
    component.releases = {component.version: ReleaseInfo(component.version, archive.as_uri(), digest)}
    ws = tmp_path / 'active'
    ws.mkdir()
    installer = HubInstaller(ws, ws / '.thetaide')
    assert installer.install(component)
    from src.app.core.provider_catalog import ProviderCatalog
    installed, capability = ProviderCatalog(installer.paths).resolve('model', 'parent')
    parent = installed.resource(capability.entry_point.split(':')[0])
    parent.chmod(parent.stat().st_mode | 0o200)  # Simulate an installation edited before protection.
    parent.write_text(parent.read_text().replace('return 1', 'return 91'))
    assert runtime(ws, 'parent') == '91'
    local = LocalComponents(ws)
    local.deep_copy(ws / 'in/config/model/parent/default.yaml', 'fork')
    cloned_package, cloned_capability = ProviderCatalog(installer.paths).resolve('model', 'fork')
    fork = cloned_package.resource(cloned_capability.entry_point.split(':')[0])
    fork.write_text(fork.read_text().replace('return 91', 'return 92'))
    assert runtime(ws, 'fork') == '92'
    assert runtime(ws, 'parent') == '91'
    local.set_hidden('fork')
    assert LocalComponents(ws).hidden('fork')
    assert runtime(ws, 'fork') == '92'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest


def test_declared_asset_directory_keeps_its_new_ownership(model_workspace):
    ws,config=model_workspace
    source=ws/'src/usr/models/parent';rules=source/'resources/rules';rules.mkdir(parents=True)
    (rules/'one.txt').write_text('one');(rules/'two.txt').write_text('two')
    manifest=source/'.theta_component.json';meta=json.loads(manifest.read_text())
    meta['payload']['resources'].append('resources/rules');manifest.write_text(json.dumps(meta))
    local=LocalComponents(ws);local.deep_copy(config,'fork')
    copied=json.loads((ws/'src/usr/models/fork/.theta_component.json').read_text())
    assert 'resources/rules' in copied['payload']['resources']
    assert (ws/'src/usr/models/fork/resources/rules/one.txt').read_text()=='one'
    local.snapshot(ws/'in/config/model/fork/default.yaml',source='local')
    package=Path(local.state()['records']['fork']['package'])
    assert (package/'resources/rules/two.txt').read_text()=='two'
    assert not (package/'src/usr').exists()


def test_copy_preserves_hydra_directive_and_model_child_identity(model_workspace, monkeypatch):
    ws, config = model_workspace
    config.write_text('# @package model\n# Profile notes\narchitecture: parent\nmodel:\n  name: external_child\n')
    copied = LocalComponents(ws).deep_copy(config, 'fork')
    assert copied.read_text().startswith('# @package model\n# Profile notes\n')
    data = yaml.safe_load(copied.read_text())
    assert data['architecture'] == 'fork'
    assert data['model']['name'] == 'external_child'
    # An explicit architecture makes model a constructor parameter, rather
    # than replacing the selected parent with its child during factory lookup.
    from src.app.core.model_registry import build_model
    monkeypatch.setenv('THETA_WORKSPACE',str(ws))
    (ws/'src/usr/models/fork/model.py').write_text('class Example:\n    def __init__(self,model): self.child=model\n')
    assert build_model(data).child=={'name':'external_child'}


def test_code_only_model_can_edit_copy_uninstall_and_reinstall(model_workspace):
    ws, config = model_workspace
    source = ws / 'src/usr/models/parent'
    manifest = source / '.theta_component.json'
    meta = json.loads(manifest.read_text())
    for relative in meta['config_paths']:
        (ws / relative).unlink()
    meta.pop('config_path')
    meta['config_paths'] = []
    meta['payload']['configs'] = {}
    meta.pop('parameter_schema')
    manifest.write_text(json.dumps(meta))
    local = LocalComponents(ws)
    installer = HubInstaller(ws, ws / '.thetaide')
    component = HubComponent.from_dict(meta)
    assert local.installed_config(component, installer) == manifest
    assets, context = local.context(manifest)
    assert assets['code_dir'] == source
    assert context.is_file()
    copied = local.deep_copy(manifest, 'fork')
    assert yaml.safe_load(copied.read_text())['architecture'] == 'fork'
    assert runtime(ws, 'fork') == '1'
    local.archive_authored(component, installer)
    assert not source.exists()
    assert 'parent' in local.state()['records']
    assert any(entry.id == 'parent' for entry in local.entries())


def test_copy_remaps_owned_hydra_defaults_but_keeps_external_defaults(model_workspace):
    ws, config = model_workspace
    config.write_text('architecture: parent\ndefaults:\n  - /model/parent/wide\n  - /model: parent/wide\n  - /model/parent: wide\n  - /model: external/shared\n')
    source = ws / 'src/usr/models/parent/model.py'
    source.write_text(source.read_text() + "\nPROFILE = 'in/config/model/parent/wide.yaml'\n")
    copied = LocalComponents(ws).deep_copy(config, 'fork')
    data = yaml.safe_load(copied.read_text())
    assert data['defaults'] == ['/model/fork/wide', {'/model': 'fork/wide'}, {'/model/fork': 'wide'}, {'/model': 'external/shared'}]
    assert "PROFILE = 'in/config/model/fork/wide.yaml'" in (ws / 'src/usr/models/fork/model.py').read_text()
    assert '/model/parent/wide' in config.read_text()


def test_copy_updates_selector_schema_without_changing_child_schema(model_workspace):
    ws, config = model_workspace
    manifest = ws / 'src/usr/models/parent/.theta_component.json'
    meta = json.loads(manifest.read_text())
    schema = meta['parameter_schema']['x-theta-config-schemas']['in/config/model/parent/default.yaml']
    schema['properties']['architecture'] = {'type': 'string', 'const': 'parent', 'default': 'parent', 'enum': ['parent', 'external']}
    schema['properties']['child'] = {'properties': {'name': {'const': 'parent'}}}
    manifest.write_text(json.dumps(meta))
    LocalComponents(ws).deep_copy(config, 'fork')
    copied = json.loads((ws / 'src/usr/models/fork/.theta_component.json').read_text())
    schema = copied['parameter_schema']['x-theta-config-schemas']['in/config/model/fork/default.yaml']
    assert schema['properties']['architecture'] == {'type': 'string', 'const': 'fork', 'default': 'fork', 'enum': ['fork', 'external']}
    assert schema['properties']['child']['properties']['name']['const'] == 'parent'


def test_copy_model_with_owned_plot_package_and_plot_profile(model_workspace):
    ws,config=model_workspace
    source=ws/'src/usr/models/parent'
    auxiliary=source/'plotters';auxiliary.mkdir()
    (auxiliary/'__init__.py').write_text('from .routing import render\n')
    (auxiliary/'routing.py').write_text('def render(): return "routing"\n')
    code=source/'model.py';code.write_text(code.read_text()+'\ndef plot_module():\n    from .plotters import routing\n    return routing\n')
    flat=ws/'in/config/model/parent.yaml';flat.write_text('architecture: parent\nneural: {type: dnn}\nsymbolic: {type: nsfr}\n')
    plot=ws/'in/config/plot/parent/routing.yaml';plot.parent.mkdir(parents=True);plot.write_text('figsize: [8, 5]\n')
    for dest in (flat,plot):
        packaged=source/'configs'/dest.relative_to(ws/'in/config');packaged.parent.mkdir(parents=True,exist_ok=True);packaged.write_bytes(dest.read_bytes())
    manifest=source/'.theta_component.json';meta=json.loads(manifest.read_text())
    paths=[str(v.relative_to(ws)) for v in (flat,plot)]
    meta.update(config_path=paths[0],config_paths=paths)
    meta['payload']['configs']={v:'configs/'+v.removeprefix('in/config/') for v in paths}
    meta['payload']['implementations'].append('plotters')
    meta['capabilities'].append({'id':'routing','kind':'plotter','entry_point':'plotters/routing.py:render'})
    schema={'type':'object','properties':{'figsize':{'type':'array'}}}
    meta['parameter_schema']['x-theta-config-schemas']={paths[1]:schema};manifest.write_text(json.dumps(meta))
    copied=LocalComponents(ws).deep_copy(flat,'fork')
    assert copied==ws/'in/config/model/fork.yaml'
    data=yaml.safe_load(copied.read_text());assert data['architecture']=='fork' and data['neural']['type']=='dnn' and data['symbolic']['type']=='nsfr'
    new_plot=ws/'in/config/plot/fork/routing.yaml';assert yaml.safe_load(new_plot.read_text())==yaml.safe_load(plot.read_text())
    assert 'from .routing import render' in (ws/'src/usr/models/fork/plotters/__init__.py').read_text()
    assert 'from .plotters import routing' in (ws/'src/usr/models/fork/model.py').read_text()
    fork=json.loads((ws/'src/usr/models/fork/.theta_component.json').read_text())
    assert fork['parameter_schema']['x-theta-config-schemas']['in/config/plot/fork/routing.yaml']==schema
    assert (auxiliary/'routing.py').read_text()=='def render(): return "routing"\n'


def test_component_tree_copy_names_inline_and_opens_new_parameters(model_workspace, monkeypatch):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLineEdit, QMessageBox

    from frontend.components_panel import ComponentsPanel
    app = QApplication.instance() or QApplication([])
    ws, config = model_workspace
    panel = ComponentsPanel()
    panel.components_tree.root_dir = ws / 'in/config'
    panel.components_tree.populate()
    panel.on_component_selected(config, 'model/parent/default.yaml')
    panel.resize(1100, 700)
    panel.show()
    app.processEvents()
    def no_modal(*args, **kwargs):
        pytest.fail('Inline component copying must not open a modal dialog')
    monkeypatch.setattr(QMessageBox, 'critical', no_modal)
    monkeypatch.setattr(QMessageBox, 'warning', no_modal)
    panel.components_tree.prompt_deep_copy_component('model/parent/default.yaml')
    editor = panel.components_tree.findChild(QLineEdit, 'componentCopyName')
    assert editor.text() == ''
    assert editor.hasFocus()
    assert not (ws / 'src/usr/models/fork').exists()
    QTest.keyClicks(editor, 'fork')
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert panel.active_path == ws / 'in/config/model/fork/default.yaml'
    assert panel.viewer.raw_data['architecture'] == 'fork'
    assert panel.components_tree.current_rel_path == 'model/fork/default.yaml'
    assert config.exists()
    panel.deleteLater()
    app.processEvents()


def test_inline_copy_conflict_and_escape_do_not_create_files(model_workspace):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QLabel, QLineEdit

    from frontend.config_tree import ConfigTreeWidget
    app = QApplication.instance() or QApplication([])
    ws, config = model_workspace
    tree = ConfigTreeWidget(root_dir=ws / 'in/config', mode='components')
    tree.show()
    app.processEvents()
    tree.prompt_deep_copy_component('model/parent/default.yaml')
    editor = tree.findChild(QLineEdit, 'componentCopyName')
    QTest.keyClicks(editor, 'parent')
    QTest.keyClick(editor, Qt.Key.Key_Return)
    assert 'already exists' in tree.findChild(QLabel, 'componentCopyError').text()
    assert editor.hasFocus()
    QTest.keyClick(editor, Qt.Key.Key_Escape)
    app.processEvents()
    assert not LocalComponents(ws).state()['records']
    assert config.exists()
    assert tree._active_inline_item is None
    tree.deleteLater()
    app.processEvents()


def test_legacy_downloaded_copy_requires_native_source_declarations(tmp_path):
    """Native BlendRL positive coverage lives with its maintained package sources."""
    source=tmp_path/'src/usr/models/blendrl';source.mkdir(parents=True)
    manifest=source/'.theta_component.json';manifest.write_text(json.dumps({'id':'blendrl','name':'BlendRL','kind':'model','version':'1.0.0'}))
    config=tmp_path/'in/config/model/blendrl.yaml';config.parent.mkdir(parents=True);config.write_text('architecture: blendrl\n')
    before=manifest.read_bytes()
    with pytest.raises(ValueError,match='declared native package'):
        LocalComponents(tmp_path).deep_copy(config,'blendrl_fork')
    assert manifest.read_bytes()==before and not (source.parent/'blendrl_fork').exists()
