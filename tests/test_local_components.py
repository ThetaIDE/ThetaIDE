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
    (source / '__init__.py').write_text('from src.usr.models.parent.model import Example\n')
    (source / 'model.py').write_text('from src.app.core.model_registry import register_model\n@register_model("parent", "legacy_parent")\nclass Example:\n    def __init__(self, **kwargs): pass\n    def forward(self): return 1\n')
    (source / 'rules.txt').write_text('owned asset')
    profiles = ['in/config/model/parent/default.yaml', 'in/config/model/parent/wide.yaml']
    for profile in profiles:
        path = root / profile
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('architecture: parent\nchild:\n  name: external_child\n')
    meta = dict(id='parent', name='Parent', kind='model', version='1.0.0', target_path='src/usr/models/parent', config_path=profiles[0], config_paths=profiles, aliases=['legacy_parent'], parameter_schema={'type': 'object', 'properties': {}, 'x-theta-config-schemas': {p: {'properties': {'width': {'type': 'integer'}}} for p in profiles}})
    (source / '.theta_component.json').write_text(json.dumps(meta))
    return root, root / profiles[0]


def runtime(workspace, component):
    # A new interpreter is the same boundary used by experiment launches.
    code = f"""import src.usr.models
src.usr.models.__path__.insert(0, {str(workspace / 'src/usr/models')!r})
import src.app.core.model_registry as registry
registry.__file__ = {str(workspace / 'src/app/core/model_registry.py')!r}
print(registry.build_model({component!r}).forward())
"""
    return subprocess.check_output([sys.executable, '-c', code], text=True, cwd=Path(__file__).parents[1]).strip()


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
    with pytest.raises(ValueError, match='local changes'):
        service.install(component, installer)
    service.uninstall(component, installer)
    assert not implementation.exists()
    assert not (ws / 'in/config/model/fork/default.yaml').exists()
    assert Path(service.state()['records']['fork']['package']).is_dir()
    assert (ws / 'src/usr/models/parent/model.py').exists()
    service.install(component, installer)
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
    def failed(self, component, installer):
        target = ws / 'src/usr/models/broken'
        target.mkdir(parents=True)
        (target / 'partial.py').write_text('partial')
        raise OSError('interrupted')
    monkeypatch.setattr(LocalComponents, 'install', failed)
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
    shared = ws / 'in/rules/shared.txt'
    shared.parent.mkdir(parents=True)
    shared.write_text('shared dependency')
    for cid in ('parent', 'fork'):
        manifest = ws / f'src/usr/models/{cid}/.theta_component.json'
        data = json.loads(manifest.read_text())
        data.setdefault('installed_files', []).append('in/rules/shared.txt')
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


def test_implicit_profile_identity_and_imported_package(model_workspace):
    ws, config = model_workspace
    config.write_text('width: 16\n')
    source = ws / 'src/usr/models/parent'
    (source / 'helper.py').write_text('VALUE = 10\n')
    (source / '__init__.py').write_text('from src.usr.models import parent as own_package\nfrom src.usr.models.parent.model import Example\n')
    service = LocalComponents(ws)
    copied = service.deep_copy(config, 'fork')
    assert yaml.safe_load(copied.read_text())['architecture'] == 'fork'
    assert 'from src.usr.models import fork as own_package' in (ws / 'src/usr/models/fork/__init__.py').read_text()
    assert runtime(ws, 'fork') == '1'


def test_schema_edits_are_protected_from_updates(model_workspace):
    ws, config = model_workspace
    service = LocalComponents(ws)
    service.deep_copy(config, 'fork')
    manifest = ws / 'src/usr/models/fork/.theta_component.json'
    meta = json.loads(manifest.read_text())
    meta['parameter_schema']['properties']['new_parameter'] = {'type': 'number'}
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='local changes'):
        service.install(service.entries()[0], HubInstaller(ws, ws / '.thetaide'))
    assert json.loads(manifest.read_text())['parameter_schema']['properties']['new_parameter'] == {'type': 'number'}


def test_external_owned_assets_are_copied_and_shared_assets_stay_shared(model_workspace):
    ws, config = model_workspace
    source = ws / 'src/usr/models/parent'
    asset = ws / 'in/rules/parent/logic.txt'
    asset.parent.mkdir(parents=True)
    asset.write_text('parent rules')
    manifest = source / '.theta_component.json'
    meta = json.loads(manifest.read_text())
    meta['installed_files'] = ['in/rules/parent/logic.txt']
    meta['editor_actions'] = [{'label': 'Rules', 'path': 'in/rules/parent/logic.txt'}]
    manifest.write_text(json.dumps(meta))
    config.write_text(config.read_text() + 'rules: in/rules/parent/logic.txt\n')
    implementation = source / 'model.py'
    implementation.write_text(implementation.read_text() + "\nRULES = 'in/rules/parent/logic.txt'\n")
    local = LocalComponents(ws)
    copied = local.deep_copy(config, 'fork')
    fork_asset = ws / 'in/rules/fork/logic.txt'
    assert fork_asset.read_text() == 'parent rules'
    assert yaml.safe_load(copied.read_text())['rules'] == 'in/rules/fork/logic.txt'
    assert "RULES = 'in/rules/fork/logic.txt'" in (ws / 'src/usr/models/fork/model.py').read_text()
    meta = json.loads((ws / 'src/usr/models/fork/.theta_component.json').read_text())
    assert meta['editor_actions'][0]['path'] == 'in/rules/fork/logic.txt'
    fork_asset.write_text('fork rules')
    assert asset.read_text() == 'parent rules'
    component = local.entries()[0]
    installer = HubInstaller(ws, ws / '.thetaide')
    local.uninstall(component, installer)
    assert not fork_asset.exists()
    local.install(component, installer)
    assert fork_asset.read_text() == 'fork rules'


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
    LocalComponents(ws).snapshot(config)
    client = HubClient(ws, ws / '.thetaide')
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
    client.local_components.uninstall(client.get_component('private_fork'), client.installer)
    assert not view.cards['private_fork'].btn_edit.isEnabled()
    host.deleteLater()
    app.processEvents()


def test_failed_reinstall_rolls_back_and_retains_package(model_workspace, monkeypatch):
    ws, config = model_workspace
    local = LocalComponents(ws)
    local.deep_copy(config, 'fork')
    component = local.entries()[0]
    installer = HubInstaller(ws, ws / '.thetaide')
    local.uninstall(component, installer)
    package = Path(local.state()['records']['fork']['package'])
    config = ws / 'in/config/model/fork/default.yaml'
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text('preexisting: keep me\n')
    def fail(component, **kwargs):
        target = ws / 'src/usr/models/fork'
        target.mkdir(parents=True)
        (target / 'model.py').write_text('partial')
        config.write_text('overwritten')
        raise OSError('interrupted reinstall')
    monkeypatch.setattr(installer, 'install', fail)
    with pytest.raises(OSError, match='interrupted reinstall'):
        local.install(component, installer)
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
    local.uninstall(component, installer)
    local.install(component, installer)
    assert runtime(ws, 'fork') == '82'
    installed_meta = json.loads((implementation.parent / '.theta_component.json').read_text())
    assert installed_meta['saved_local_changes'] is True
    with pytest.raises(ValueError, match='local changes'):
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
    parent = ws / 'src/usr/models/parent/model.py'
    parent.chmod(parent.stat().st_mode | 0o200)  # Simulate an installation edited before protection.
    parent.write_text(parent.read_text().replace('return 1', 'return 91'))
    assert runtime(ws, 'parent') == '91'
    local = LocalComponents(ws)
    local.deep_copy(ws / 'in/config/model/parent/default.yaml', 'fork')
    fork = ws / 'src/usr/models/fork/model.py'
    fork.write_text(fork.read_text().replace('return 91', 'return 92'))
    assert runtime(ws, 'fork') == '92'
    assert runtime(ws, 'parent') == '91'
    local.set_hidden('fork')
    assert LocalComponents(ws).hidden('fork')
    assert runtime(ws, 'fork') == '92'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == digest


def test_declared_asset_directory_keeps_its_new_ownership(model_workspace):
    ws, config = model_workspace
    rules = ws / 'in/rules/parent'
    rules.mkdir(parents=True)
    (rules / 'one.txt').write_text('one')
    (rules / 'two.txt').write_text('two')
    manifest = ws / 'src/usr/models/parent/.theta_component.json'
    meta = json.loads(manifest.read_text())
    meta['asset_paths'] = ['in/rules/parent']
    manifest.write_text(json.dumps(meta))
    local = LocalComponents(ws)
    local.deep_copy(config, 'fork')
    meta = json.loads((ws / 'src/usr/models/fork/.theta_component.json').read_text())
    assert meta['asset_paths'] == ['in/rules/fork']
    assert (ws / 'in/rules/fork/one.txt').read_text() == 'one'
    local.snapshot(ws / 'in/config/model/fork/default.yaml', source='local')
    package = Path(local.state()['records']['fork']['package'])
    assert (package / 'in/rules/fork/two.txt').read_text() == 'two'
    assert not (package / 'in/rules/parent').exists()


def test_copy_preserves_hydra_directive_and_model_child_identity(model_workspace):
    ws, config = model_workspace
    config.write_text('# @package model\n# Profile notes\narchitecture: parent\nmodel:\n  name: external_child\n')
    copied = LocalComponents(ws).deep_copy(config, 'fork')
    assert copied.read_text().startswith('# @package model\n# Profile notes\n')
    data = yaml.safe_load(copied.read_text())
    assert data['architecture'] == 'fork'
    assert data['model']['name'] == 'external_child'
    # An explicit architecture makes model a constructor parameter, rather
    # than replacing the selected parent with its child during factory lookup.
    from src.app.core.model_registry import MODEL_REGISTRY, build_model
    previous = dict(MODEL_REGISTRY)
    class Parent:
        def __init__(self, model):
            self.child = model
    try:
        MODEL_REGISTRY['fork'] = Parent
        assert build_model(data).child == {'name': 'external_child'}
    finally:
        MODEL_REGISTRY.clear()
        MODEL_REGISTRY.update(previous)


def test_code_only_model_can_edit_copy_uninstall_and_reinstall(model_workspace):
    ws, config = model_workspace
    source = ws / 'src/usr/models/parent'
    manifest = source / '.theta_component.json'
    meta = json.loads(manifest.read_text())
    for relative in meta['config_paths']:
        (ws / relative).unlink()
    meta.pop('config_path')
    meta['config_paths'] = []
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
    local.uninstall(component, installer)
    assert not source.exists()
    retained = next(entry for entry in local.entries() if entry.id == 'parent')
    local.install(retained, installer)
    assert runtime(ws, 'parent') == '1'


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
    ws, config = model_workspace
    plot_config = ws / 'in/config/plot/parent/routing.yaml'
    plot_config.parent.mkdir(parents=True)
    plot_config.write_text('figsize: [8, 5]\nauthority_threshold: 0.9\n')
    auxiliary = ws / 'plot/parent'
    auxiliary.mkdir(parents=True)
    (auxiliary / '__init__.py').write_text('from plot.parent.routing import render\n')
    (auxiliary / 'routing.py').write_text('def render(): return "routing"\n')
    source = ws / 'src/usr/models/parent/model.py'
    source.write_text(source.read_text() + '\ndef plot_module():\n    import plot.parent.routing as routing\n    return routing\n')
    flat_config = ws / 'in/config/model/parent.yaml'
    flat_config.write_text('actor_mode: hybrid\nneural:\n  type: dnn\nsymbolic:\n  type: nsfr\n')
    manifest = source.parent / '.theta_component.json'
    meta = json.loads(manifest.read_text())
    meta.update(config_path='in/config/model/parent.yaml',
        config_paths=['in/config/model/parent.yaml', 'in/config/plot/parent/routing.yaml'],
        installed_files=['plot/parent/__init__.py', 'plot/parent/routing.py'],
        installed_components=['dnn', 'nsfr', 'neumann'],
        dependencies={'components': ['dnn', 'nsfr', 'neumann']})
    plot_schema = {'type': 'object', 'properties': {'figsize': {'type': 'array'}}}
    meta['parameter_schema']['x-theta-config-schemas'] = {'in/config/plot/parent/routing.yaml': plot_schema}
    manifest.write_text(json.dumps(meta))
    copied = LocalComponents(ws).deep_copy(flat_config, 'fork')
    assert copied == ws / 'in/config/model/fork.yaml'
    result = yaml.safe_load(copied.read_text())
    assert result['architecture'] == 'fork'
    assert result['neural']['type'] == 'dnn'
    assert result['symbolic']['type'] == 'nsfr'
    new_plot = ws / 'in/config/plot/fork/routing.yaml'
    assert yaml.safe_load(new_plot.read_text()) == yaml.safe_load(plot_config.read_text())
    assert 'from plot.fork.routing import render' in (ws / 'plot/fork/__init__.py').read_text()
    assert 'import plot.fork.routing as routing' in (ws / 'src/usr/models/fork/model.py').read_text()
    new_meta = json.loads((ws / 'src/usr/models/fork/.theta_component.json').read_text())
    assert new_meta['dependencies'] == meta['dependencies']
    assert not new_meta.get('installed_components')
    assert new_meta['parameter_schema']['x-theta-config-schemas']['in/config/plot/fork/routing.yaml'] == plot_schema
    assert not (ws / 'src/usr/models/dnn').exists()
    assert (auxiliary / 'routing.py').read_text() == 'def render(): return "routing"\n'


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


def test_installed_blendrl_copy_preserves_original_identity(tmp_path):
    """Exercise the downloaded package without changing the user's installation."""
    import shutil
    root = Path(__file__).resolve().parents[1]
    config = root / 'in/config/model/blendrl.yaml'
    manifest = root / 'src/usr/models/blendrl/.theta_component.json'
    if not config.is_file() or not manifest.is_file():
        pytest.skip('Downloaded BlendRL is not installed in this workspace')
    meta = json.loads(manifest.read_text())
    ws = tmp_path / 'workspace'
    shutil.copytree(manifest.parent, ws / 'src/usr/models/blendrl', ignore=shutil.ignore_patterns('__pycache__'))
    payload = list(dict.fromkeys(meta.get('config_paths', []) + meta.get('installed_files', [])))
    original = {relative: (root / relative).read_bytes() for relative in payload}
    for relative in payload:
        (ws / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, ws / relative)
    copied = LocalComponents(ws).deep_copy(ws / 'in/config/model/blendrl.yaml', 'blendrl_fork')
    assert copied == ws / 'in/config/model/blendrl_fork.yaml'
    cloned = json.loads((ws / 'src/usr/models/blendrl_fork/.theta_component.json').read_text())
    assert cloned['dependencies'] == meta['dependencies']
    assert (ws / 'plot/blendrl_fork/routing.py').exists()
    assert 'in/config/plot/blendrl_fork/blend_routing.yaml' in cloned['config_paths']
    assert yaml.safe_load((ws / 'in/config/plot/blendrl_fork/blend_routing.yaml').read_text()) == yaml.safe_load((ws / 'in/config/plot/blendrl/blend_routing.yaml').read_text())
    # Only registration/import isolation is tested here. The developer's optional
    # Captum/NSFR dependencies are stubbed; no training or dependency install runs.
    code = f"""import sys, types, importlib
import src.usr.models
src.usr.models.__path__.insert(0, {str(ws / 'src/usr/models')!r})
from src.app.core import model_registry as registry
registry.__file__ = {str(ws / 'src/app/core/model_registry.py')!r}
for name in ('captum', 'captum.attr', 'nsfr', 'nsfr.common', 'nsfr.utils', 'nsfr.utils.common'):
    sys.modules[name] = types.ModuleType(name)
for name in ('DeepLift', 'DeepLiftShap', 'GradientShap', 'IntegratedGradients', 'LayerConductance', 'NeuronConductance', 'NoiseTunnel'):
    setattr(sys.modules['captum.attr'], name, type(name, (), {{}}))
sys.modules['nsfr.common'].get_nsfr_model = lambda *args, **kwargs: None
sys.modules['nsfr.utils.common'].load_module = lambda *args, **kwargs: None
importlib.import_module('src.usr.models.blendrl')
original = registry.get_model_class('blendrl')
importlib.import_module('src.usr.models.blendrl_fork')
fork = registry.get_model_class('blendrl_fork')
assert original is not fork
assert registry.get_model_class('blender') is original
assert registry.get_model_class('blendrl') is original
assert fork.__module__.startswith('src.usr.models.blendrl_fork.')
print('independent')
"""
    assert subprocess.check_output([sys.executable, '-c', code], cwd=root, text=True).strip() == 'independent'
    assert all((root / relative).read_bytes() == contents for relative, contents in original.items())
