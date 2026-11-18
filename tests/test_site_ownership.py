import json
from pathlib import Path
import pytest
from src.app.sites.catalog import create, load, configure, defaults_path
from src.app.sites import profiles
from src.app.sites.transfer import connection
from src.app.workspace import WorkspacePaths


def component(root):
    create(root, 'example', {'scheduler': 'batch', 'resources': {'cores': 4}},
           {'cpu': {'resources': {'gpus': 0}}, 'gpu': {'resources': {'gpus': 1}}})
    configure(root, 'example/gpu', {'host': 'shared-alias', 'remote_root': '/scratch/theta'})


def test_site_owns_shared_connection_and_variants(tmp_path):
    component(tmp_path)
    assert connection(tmp_path, 'example/cpu')['host'] == 'shared-alias'
    assert connection(tmp_path, 'example/gpu')['host'] == 'shared-alias'
    assert load(tmp_path, 'example/gpu')['resources'] == {'cores': 4, 'gpus': 1}
    assert load(tmp_path, 'example/cpu')['resources']['gpus'] == 0
    assert not (tmp_path / '.thetaide/site_connections.json').exists()
    available = profiles(WorkspacePaths(tmp_path))
    assert 'example/gpu' in available and 'example/_defaults' not in available
    assert 'example_gpu' not in available
    assert {'local/default', 'example/cpu', 'example/gpu'} <= available.keys()


def test_native_profiles_compose_through_hydra(tmp_path):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    component(tmp_path)
    config = tmp_path / 'src/usr/config'
    config.mkdir(parents=True)
    (config / 'config.yaml').write_text('defaults:\n  - site: example/cpu\n  - _self_\n')
    paths = WorkspacePaths(tmp_path)
    search = '[' + ','.join('file://' + str(p) for p in paths.config_roots() if p != config) + ']'
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config), version_base=None):
        cfg = compose(config_name='config', overrides=['site=example/gpu', 'hydra.searchpath=' + search])
    assert OmegaConf.to_container(cfg.site) == load(tmp_path, 'example/gpu')


def test_components_tree_groups_site_profiles(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    from frontend.config_tree import ConfigTreeWidget
    app = QApplication.instance() or QApplication([])
    component(tmp_path)
    (tmp_path / 'src/usr/config').mkdir(parents=True)
    tree = ConfigTreeWidget(root_dir=tmp_path / 'src/usr/config', mode='components')
    def visit(node):
        for i in range(node.childCount()):
            child = node.child(i)
            yield child
            yield from visit(child)
    nodes = list(visit(tree.tree.invisibleRootItem()))
    site = next(node for node in nodes if node.text(0) == 'example')
    assert {site.child(i).text(0) for i in range(site.childCount())} == {'Defaults', 'default', 'cpu', 'gpu'}
    for i in range(site.childCount()):
        assert Path(site.child(i).data(0, Qt.ItemDataRole.UserRole)['path']).is_relative_to(tmp_path / 'src/usr/sites/example')
    tree.close()


def test_settings_selects_components_without_connection_editor(tmp_path):
    from PyQt6.QtWidgets import QApplication, QPushButton
    from frontend.site_launch import SiteSetup
    from component_fixtures import install_fixture
    app = QApplication.instance() or QApplication([])
    paths = WorkspacePaths(tmp_path)
    component(tmp_path)
    empty = SiteSetup(workspace=paths)
    assert not empty.scope.active  # no installed scheduler contributes site setup
    empty.close()
    install_fixture(paths, 'batch-scheduler', tmp_path.parent / (tmp_path.name + '-archives'))
    widget = SiteSetup(workspace=paths)
    setup = widget.layout().itemAt(0).widget()
    assert setup.site.findText('example/cpu') >= 0
    assert not setup.site.isEditable()
    assert not any('Configure' in button.text() for button in widget.findChildren(QPushButton))
    widget.close()
