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


# ── Parameters box ───────────────────────────────────────────────────────────


from PyQt6.QtWidgets import QComboBox, QLineEdit, QToolButton  # noqa: E402

from frontend.discussion.model import FigureDoc  # noqa: E402
from frontend.discussion.params import ParametersBox  # noqa: E402


def saved(path):
    return yaml.safe_load(path.read_text())["params"]


def box_for(paths, **kwargs):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    box = ParametersBox(doc, **kwargs)
    live, regenerate = [], []
    box.live_changed.connect(lambda name, value: live.append((name, value)))
    box.regenerate_needed.connect(lambda: regenerate.append(True))
    return box, doc, live, regenerate


def removes(box):
    return {b.property("parameterPath") for b in box.findChildren(QToolButton) if b.property("parameterRemove")}


def test_parameters_box_shows_defaults_as_inherited_values(app, paths):
    box, doc, _, _ = box_for(paths)
    assert set(box.editors) == {"color", "metric"}
    assert box.editors["color"].text() == "#d62728" and not box.editors["color"].property("inheritedDefault")
    assert box.editors["metric"].text() == "train/reward" and box.editors["metric"].property("inheritedDefault")
    assert removes(box) == {"color"}  # only explicit values can be removed


def test_live_edit_saves_and_emits_without_regenerating(app, paths):
    box, doc, live, regenerate = box_for(paths)
    box.editors["color"].setText("#2ca02c")
    assert saved(doc.path) == {"color": "#2ca02c"}
    assert live == [("color", "#2ca02c")] and regenerate == []


def test_regeneration_edit_saves_and_requests_a_rebuild(app, paths):
    box, doc, live, regenerate = box_for(paths)
    box.editors["metric"].setText("eval/reward")
    assert saved(doc.path) == {"color": "#d62728", "metric": "eval/reward"}
    assert live == [] and regenerate == [True]
    app.processEvents()
    assert removes(box) == {"color", "metric"}  # now explicit, so removable


def test_removing_an_explicit_value_restores_the_default(app, paths):
    box, doc, live, regenerate = box_for(paths)
    next(b for b in box.findChildren(QToolButton) if b.property("parameterPath") == "color").click()
    assert saved(doc.path) == {}
    assert live == [("color", "#1f77b4")]
    app.processEvents()
    assert box.editors["color"].text() == "#1f77b4" and removes(box) == set()


def test_invalid_drafts_leave_the_figure_file_unchanged(app, paths, monkeypatch):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    schema = doc.schema()
    patched = {**schema, "properties": {**schema["properties"],
                                        "alpha": {"type": "number", "minimum": 0, "maximum": 1, "default": 1}}}
    monkeypatch.setattr(doc, "schema", lambda: patched)
    box = ParametersBox(doc)
    before = doc.path.read_text()
    box.editors["alpha"].setText("2")
    box.editors["alpha"].editingFinished.emit()
    assert box.editors["alpha"].property("parameterInvalid")
    assert doc.path.read_text() == before


def test_plus_adds_parameters_without_defaults_and_metric_choices_are_offered(app, paths, monkeypatch):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    schema = doc.schema()
    patched = {**schema, "properties": {**schema["properties"], "label": {"type": "string"}}}
    monkeypatch.setattr(doc, "schema", lambda: patched)
    box = ParametersBox(doc, metric_choices=["eval/reward", "train/reward"])
    combo = box.editors["metric"]
    assert isinstance(combo, QComboBox)
    assert [combo.itemText(i) for i in range(combo.count())] == ["", "eval/reward", "train/reward"]
    assert combo.currentText() == "train/reward"

    add = next(b for b in box.findChildren(QToolButton) if b.property("parameterAdd"))
    assert [a.text() for a in add.menu().actions()] == ["label"]
    add.menu().actions()[0].trigger()
    app.processEvents()
    assert saved(doc.path) == {"color": "#d62728", "label": ""}
    assert isinstance(box.editors["label"], QLineEdit) and "label" in removes(box)


# ── Figure view and builder ──────────────────────────────────────────────────


import json  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

from PyQt6.QtCore import QEventLoop, QTimer  # noqa: E402

from frontend.discussion import view as view_module  # noqa: E402
from frontend.discussion.view import FigureBuilder, FigureView, embed_config  # noqa: E402


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(10, loop.quit)
        loop.exec()
    return predicate()


def bridge_calls(view):
    calls = []
    view.bridge.renderRequested.connect(lambda spec: calls.append(("render", json.loads(spec))))
    view.bridge.paramRequested.connect(lambda name, value: calls.append(("param", name, json.loads(value))))
    view.bridge.errorRequested.connect(lambda text: calls.append(("error", text)))
    view.bridge.busyRequested.connect(lambda on: calls.append(("busy", on)))
    return calls


def test_vega_builds_are_vendored_pinned_and_licensed():
    html = view_module.HTML_PATH.read_text()
    vendor = view_module.HTML_PATH.parent / "vendor"
    for name in ("vega", "vega-lite", "vega-embed"):
        assert f'src="vendor/{name}.min.js"' in html
        assert (vendor / f"{name}.min.js").stat().st_size > 10_000
        assert "BSD" in (vendor / f"LICENSE-{name}.txt").read_text() or "Redistribution" in (
            vendor / f"LICENSE-{name}.txt").read_text()
    assert "http" not in html.replace("https://vega.github.io", "")  # nothing is fetched


def test_figure_view_creates_no_web_engine_until_shown_and_replays_when_ready(app):
    view = FigureView()
    calls = bridge_calls(view)
    view.render({"params": [{"name": "w", "value": 1}]})
    view.setParam("w", 3)
    view.set_busy(True)
    assert view.web is None and calls == []  # nothing is sent before the page exists
    view._page_ready()
    assert calls == [("render", {"params": [{"name": "w", "value": 1}]}), ("param", "w", 3), ("busy", True)]
    calls.clear()
    view.setParam("w", 4)
    view.error("No runs")
    view.setParam("w", 5)  # no figure to update
    assert calls == [("param", "w", 4), ("error", "No runs")]


def test_embed_config_uses_theme_tokens():
    from frontend.theme import theme_color
    config = embed_config()
    assert config["axis"]["labelColor"] == theme_color("muted")
    assert config["title"]["color"] == theme_color("text")
    assert config["axis"]["gridColor"] == theme_color("border")


class Doc:
    plotter, experiment = "fixture_curve", "figs/online"

    def __init__(self, value):
        self.value = value

    def values(self):
        return {"metric": self.value}

    def live_params(self):
        return []


def test_builder_debounces_and_delivers_only_the_newest_request(app, paths):
    builds = []
    builder = FigureBuilder(paths, debounce_ms=50, build=lambda p, e, values, live: builds.append(values) or values)
    built, states = [], []
    builder.built.connect(built.append)
    builder.started.connect(lambda: states.append("started"))
    builder.finished.connect(lambda: states.append("finished"))
    for value in ("a", "b", "c"):
        builder.request(Doc(value))
    assert wait_until(lambda: built)
    assert builds == [{"metric": "c"}] and built == [{"metric": "c"}]
    assert states == ["started", "finished"]


def test_builder_drops_superseded_results_and_reports_component_errors(app, paths):
    release = threading.Event()

    def build(plotter, experiment, values, live):
        if values["metric"] == "slow":
            release.wait(5)
            return {"slow": True}
        if values["metric"] == "broken":
            raise ValueError("No run of figs/online logged 'broken'.")
        return {"fast": True}

    builder = FigureBuilder(paths, build=build)
    built, failed = [], []
    builder.built.connect(built.append)
    builder.failed.connect(failed.append)
    builder.request(Doc("slow"), immediate=True)
    builder.request(Doc("fast"), immediate=True)
    assert wait_until(lambda: built)
    release.set()
    wait_until(lambda: False, 0.2)
    assert built == [{"fast": True}] and not builder.busy

    builder.request(Doc("broken"), immediate=True)
    assert wait_until(lambda: failed)
    assert failed == ["No run of figs/online logged 'broken'."]


def test_builder_runs_the_selected_plot_component(app, paths):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    builder = FigureBuilder(paths)
    built = []
    builder.built.connect(built.append)
    builder.request(doc, immediate=True)
    assert wait_until(lambda: built)
    assert built[0]["params"] == [{"name": "color", "value": "#d62728"}]
    assert {row["method"] for row in built[0]["data"]["values"]} == {"a2c", "ppo"}
