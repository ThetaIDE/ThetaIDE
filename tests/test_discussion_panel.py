"""The Discussion tab (offscreen Qt): tree, + menu, parameters, figure updates and discussion text.

The WebEngine figure view is replaced by a recorder; tests/test_figure_contract.py covers the specs.
"""

import pytest
import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from discussion_fixtures import build_discussion_workspace
from frontend.discussion import model
from frontend.discussion.tree import NO_CANDIDATES, DiscussionTree
from src.app.workspace import WorkspacePaths


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def paths(tmp_path, monkeypatch):
    root = build_discussion_workspace(tmp_path / "workspace")
    monkeypatch.setenv("THETA_WORKSPACE", str(root))
    return WorkspacePaths(root)


def item_texts(tree, parent=None):
    parent = parent or tree.tree.invisibleRootItem()
    return [parent.child(i).text(0) for i in range(parent.childCount())]


def find(tree, rel):
    return tree._find(rel)


# ── Tree ─────────────────────────────────────────────────────────────────────


def test_tree_mirrors_the_experiment_tree_with_figure_leaves(app, paths):
    tree = DiscussionTree(paths)
    assert item_texts(tree) == ["figs", "other"]
    figs = find(tree, "figs")
    assert figs.data(0, Qt.ItemDataRole.UserRole)["is_group"]
    assert item_texts(tree, figs) == ["sweeps", "online"]
    assert item_texts(tree, find(tree, "figs/online")) == ["fixture_curve"]
    assert item_texts(tree, find(tree, "other/solo")) == []
    assert tree.mode == "discussion"
    groups = tree.header_widget.header_layout.groups
    assert [b.button_id for b in groups["row1_left"]] == ["reload", "collapse_all", "expand_all", "search"]
    assert [b.button_id for b in groups["row1_right"]] == ["plus", "regenerate"]
    assert all(b.width() == b.height() == 30 for g in groups.values() for b in g)


def test_selection_drives_the_add_and_regenerate_buttons(app, paths):
    tree = DiscussionTree(paths)
    figures, experiments = [], []
    tree.figure_selected.connect(figures.append)
    tree.experiment_selected.connect(experiments.append)
    assert not tree.btn_add.isEnabled() and not tree.btn_regenerate.isEnabled()

    tree.tree.setCurrentItem(find(tree, "figs/online"))
    assert tree.btn_add.isEnabled() and not tree.btn_regenerate.isEnabled()
    assert experiments[-1] == "figs/online" and figures[-1] is None

    figure = paths.discussion / "figs/online/fixture_curve.yaml"
    tree.select_figure(figure)
    assert tree.btn_add.isEnabled() and tree.btn_regenerate.isEnabled()
    assert figures[-1] == figure and experiments[-1] == "figs/online"

    tree.tree.setCurrentItem(find(tree, "figs/sweeps"))
    assert not tree.btn_add.isEnabled() and not tree.btn_regenerate.isEnabled()
    assert figures[-1] is None and experiments[-1] == ""


def test_add_menu_omits_blocked_candidates_and_marks_warned_ones(app, paths):
    tree = DiscussionTree(paths)
    tree.tree.setCurrentItem(find(tree, "figs/online"))
    actions = {a.text(): a for a in tree.build_add_menu().actions()}
    assert set(actions) == {"convergence", "fixture_curve"}
    warned = actions["fixture_curve"]
    assert warned.property("compatibilityLevel") == "warn"
    assert warned.toolTip() == "Fixture curves are tuned for DQN."
    assert not warned.icon().isNull()
    assert actions["convergence"].property("compatibilityLevel") is None
    assert actions["convergence"].icon().isNull()

    tree.tree.setCurrentItem(find(tree, "figs/sweeps/offline"))
    assert [a.text() for a in tree.build_add_menu().actions()] == ["convergence"]


def test_add_menu_without_compatible_candidates_shows_one_disabled_item(app, paths):
    tree = DiscussionTree(paths, candidates=lambda rel: [])
    tree.tree.setCurrentItem(find(tree, "other/solo"))
    [action] = tree.build_add_menu().actions()
    assert action.text() == NO_CANDIDATES and not action.isEnabled()


def test_choosing_a_candidate_creates_and_selects_the_figure(app, paths):
    tree = DiscussionTree(paths)
    selected = []
    tree.figure_selected.connect(selected.append)
    tree.tree.setCurrentItem(find(tree, "other/solo"))
    actions = {a.text(): a for a in tree.build_add_menu().actions()}
    actions["convergence"].trigger()

    created = paths.discussion / "other/solo/convergence.yaml"
    assert yaml.safe_load(created.read_text()) == {"plotter": "convergence", "params": {}}
    assert tree.selected_figure() == created and selected[-1] == created
    assert item_texts(tree, find(tree, "other/solo")) == ["convergence"]
    assert find(tree, "other").isExpanded() and find(tree, "other/solo").isExpanded()

    tree.add_figure("other/solo", "convergence")
    assert item_texts(tree, find(tree, "other/solo")) == ["convergence", "convergence_2"]
    assert tree.selected_figure().name == "convergence_2.yaml"


def test_search_filters_figures_and_experiments(app, paths):
    tree = DiscussionTree(paths)
    tree.toggle_search(True)
    tree.search.setText("fixture")
    assert not find(tree, "figs/online").isHidden()
    assert find(tree, "other").isHidden() and find(tree, "figs/sweeps").isHidden()
    tree.toggle_search(False)
    assert not find(tree, "other").isHidden()
