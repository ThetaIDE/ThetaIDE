"""The Discussion tab (offscreen Qt): tree, + menu, parameters, figure updates and discussion text.

The WebEngine figure view is replaced by a recorder; tests/test_figure_contract.py covers the specs.
"""

import pytest
import yaml
from types import SimpleNamespace

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFrame, QLabel

from discussion_fixtures import build_discussion_workspace
from frontend.discussion import model
from frontend.discussion.picker import NO_CANDIDATES, FigurePicker
from frontend.discussion.tree import DiscussionTree
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
    assert [b.button_id for b in groups["row1_right"]] == ["plus"]
    assert all(b.width() == b.height() == 30 for g in groups.values() for b in g)


def test_selection_drives_the_add_button(app, paths):
    tree = DiscussionTree(paths)
    figures, experiments = [], []
    tree.figure_selected.connect(figures.append)
    tree.experiment_selected.connect(experiments.append)
    assert not tree.btn_add.isEnabled()

    tree.tree.setCurrentItem(find(tree, "figs/online"))
    assert tree.btn_add.isEnabled()
    assert experiments[-1] == "figs/online" and figures[-1] is None

    figure = paths.discussion / "figs/online/fixture_curve.yaml"
    tree.select_figure(figure)
    assert tree.btn_add.isEnabled()
    assert figures[-1] == figure and experiments[-1] == "figs/online"

    tree.tree.setCurrentItem(find(tree, "figs/sweeps"))
    assert not tree.btn_add.isEnabled()
    assert figures[-1] is None and experiments[-1] == ""


def picker_for(paths, experiment, **kwargs):
    picker = FigurePicker(experiment, kwargs.get("candidates") or (lambda rel: model.figure_candidates(paths, rel)))
    return picker, {card.selector: card for card in picker.cards}


def test_plus_requests_the_picker_of_the_selected_experiment(app, paths):
    tree = DiscussionTree(paths)
    requested = []
    tree.add_requested.connect(requested.append)
    tree.select_figure(paths.discussion / "figs/online/fixture_curve.yaml")
    tree.btn_add.click()
    assert requested == ["figs/online"]


def test_picker_omits_blocked_candidates_and_marks_warned_ones(app, paths):
    picker, cards = picker_for(paths, "figs/online")
    # The sweep figures are blocked: the target has no sweep.
    assert set(cards) == {"convergence", "final_performance", "performance_profile", "run_variability", "losses", "metric_table"} | {"fixture_curve"}
    warned = cards["fixture_curve"]
    assert warned.property("figureCard") and warned.property("configBox")
    [status] = [w for w in warned.findChildren(QLabel) if w.objectName() == "compatibilityStatus"]
    assert status.property("compatibilityLevel") == "warn"
    assert status.text() == "Off-label: Fixture curves are tuned for DQN."
    assert not [w for w in cards["convergence"].findChildren(QLabel) if w.objectName() == "compatibilityStatus"]
    texts = [w.text() for w in cards["convergence"].findChildren(QLabel)]
    assert "Learning curve" in texts and "16 parameters · 7 live" in texts

    _, cards = picker_for(paths, "figs/sweeps/offline")
    assert set(cards) == {"convergence", "final_performance", "performance_profile", "run_variability", "losses", "metric_table"}


def test_picker_without_compatible_candidates_says_so(app, paths):
    picker, cards = picker_for(paths, "other/solo", candidates=lambda rel: [])
    assert not cards
    assert NO_CANDIDATES in [w.text() for w in picker.findChildren(QLabel)]


def test_picker_cards_choose_on_click_and_keyboard_and_escape_cancels(app, paths):
    picker, cards = picker_for(paths, "other/solo")
    chosen, cancelled = [], []
    picker.chosen.connect(lambda e, s: chosen.append((e, s)))
    picker.cancelled.connect(lambda: cancelled.append(True))
    card = cards["convergence"]
    card.resize(200, 80)
    QTest.mouseClick(card, Qt.MouseButton.LeftButton, pos=card.rect().center())
    QTest.keyClick(card, Qt.Key.Key_Return)
    assert chosen == [("other/solo", "convergence")] * 2
    QTest.keyClick(picker, Qt.Key.Key_Escape)
    assert cancelled == [True]


def test_add_figure_creates_and_selects_the_figure(app, paths):
    tree = DiscussionTree(paths)
    selected = []
    tree.figure_selected.connect(selected.append)
    tree.add_figure("other/solo", "convergence")

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

    def capability(self):
        return SimpleNamespace(id="fixture_figures", version="1.0.0"), None

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
    built, failed = [], []
    builder.built.connect(built.append)
    builder.failed.connect(failed.append)
    builder.request(doc, immediate=True)
    # A fresh process imports the plotting stack in this worker for the first time.
    assert wait_until(lambda: built or failed, timeout=30)
    assert not failed, failed
    assert built[0]["params"] == [{"name": "color", "value": "#d62728"}]
    assert {row["method"] for row in built[0]["data"]["values"]} == {"a2c", "ppo"}


# ── Assembled panel ─────────────────────────────────────────────────────────


from PyQt6.QtWidgets import QWidget  # noqa: E402
from frontend.discussion import panel as panel_module  # noqa: E402


class RecordingView(QWidget):
    image_ready = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.calls = []

    def reset_view(self):
        self.calls.append(("reset",))

    def request_image(self, scale=2.0):
        self.calls.append(("image", scale))

    def render(self, spec):
        self.calls.append(("render", spec))

    def setParam(self, name, value):
        self.calls.append(("param", name, value))

    def error(self, text):
        self.calls.append(("error", text))

    def clear(self):
        self.calls.clear()

    def set_busy(self, busy):
        self.calls.append(("busy", busy))


@pytest.fixture
def panel(app, paths, monkeypatch):
    monkeypatch.setattr(panel_module, "FigureView", RecordingView)
    panel = panel_module.DiscussionPanel(paths)
    yield panel
    panel.shutdown()
    panel.deleteLater()
    app.processEvents()


def open_fixture(panel, paths):
    path = paths.discussion / "figs/online/fixture_curve.yaml"
    panel.tree.select_figure(path)
    assert wait_until(lambda: any(c[0] == "render" for c in panel.figure_view.calls), timeout=30)
    return path


def test_panel_empty_selection_and_open_figure(panel, paths):
    assert panel.parameter_stack.currentWidget().text() == "Select a figure"
    assert panel.figure_stack.currentWidget().text() == "Select a figure"
    assert not panel.discussion.isEnabled()
    path = open_fixture(panel, paths)
    assert panel.doc.path == path
    assert panel.discussion.toPlainText() == model.read_discussion(path)
    panel.tree.tree.setCurrentItem(find(panel.tree, "figs/online"))
    assert panel.doc is None and not panel.discussion.isEnabled()
    assert panel.parameter_stack.currentIndex() == panel.figure_stack.currentIndex() == 0


def test_panel_is_laid_out_like_the_experiment_tab(panel, paths):
    assert panel.title.text() == "No experiment selected"
    margins = panel.layout().contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (18, 14, 18, 14)
    assert panel.figure_box.property("configBox") and panel.discussion_box.property("configBox")
    assert panel.discussion_box.title_label.text() == "Discussion"
    assert panel.discussion_box.isAncestorOf(panel.discussion)
    open_fixture(panel, paths)
    assert panel.title.text() == "figs / online"
    assert panel.parameters.title_label.text() == "fixture_curve"  # the box names the figure
    holder = panel.parameter_scroll.widget()
    assert holder is not panel.parameters and holder.isAncestorOf(panel.parameters)
    margins = holder.layout().contentsMargins()
    assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (12, 10, 12, 16)


def test_panel_picker_replaces_the_parameters_box_until_chosen_or_closed(panel, paths):
    path = open_fixture(panel, paths)
    panel.tree.btn_add.click()
    assert panel.parameter_stack.currentIndex() == panel.PICKER_PAGE
    assert panel.picker.experiment == "figs/online" and panel.doc.path == path
    assert panel.picker.property("configBox") and panel.picker.title_label.text() == "Add figure"
    assert all(panel.picker.isAncestorOf(card) for card in panel.picker.cards)
    panel.picker.cancelled.emit()
    assert panel.picker is None and panel.parameter_stack.currentIndex() == panel.PARAMETERS_PAGE

    panel.tree.btn_add.click()
    panel.tree.tree.setCurrentItem(find(panel.tree, "other/solo"))  # selecting elsewhere closes it
    assert panel.picker is None and panel.parameter_stack.currentIndex() == 0

    panel.tree.btn_add.click()
    next(c for c in panel.picker.cards if c.selector == "convergence").clicked.emit()
    created = paths.discussion / "other/solo/convergence.yaml"
    assert panel.picker is None and panel.doc.path == created
    assert panel.parameter_stack.currentIndex() == panel.PARAMETERS_PAGE
    assert panel.parameters.title_label.text() == "convergence"


def test_panel_live_edits_and_debounced_regeneration(panel, paths):
    open_fixture(panel, paths)
    builds = []
    panel.builder._build = lambda p, e, values, live: builds.append(values) or {"params": []}
    panel.builder._timer.setInterval(40)
    panel.parameters.editors["color"].setText("purple")
    assert ("param", "color", "purple") in panel.figure_view.calls
    assert not builds and not panel.builder._timer.isActive()
    metric = panel.parameters.editors["metric"]  # the target's logged metrics
    assert isinstance(metric, QComboBox)
    metric.setCurrentText("eval/reward")
    assert wait_until(lambda: len(builds) == 1)
    assert builds[0]["metric"] == "eval/reward"
    assert wait_until(lambda: not panel.builder.busy)
    panel.figure_tools["regenerate"].click()
    assert wait_until(lambda: len(builds) == 2 and not panel.builder.busy)


def test_panel_discussion_autosaves_and_flushes_on_switch_and_close(panel, paths):
    path = open_fixture(panel, paths)
    panel._save_timer.setInterval(30)
    panel.discussion.setPlainText("Autosaved note\nwith a second line.")
    assert wait_until(lambda: model.read_discussion(path) == "Autosaved note\nwith a second line.")
    panel.discussion.setPlainText("Saved on selection")
    panel.tree.tree.setCurrentItem(find(panel.tree, "other/solo"))
    assert model.read_discussion(path) == "Saved on selection"
    open_fixture(panel, paths)
    assert panel.discussion.toPlainText() == "Saved on selection"
    panel.discussion.setPlainText("Saved on close")
    panel.close()
    assert model.read_discussion(path) == "Saved on close"


def test_panel_reapplies_live_edits_made_during_a_build(panel, paths):
    open_fixture(panel, paths)
    release = threading.Event()
    def build(*args):
        release.wait(3)
        return {}
    panel.builder._build = build
    panel.regenerate()
    panel.parameters.editors["color"].setText("orange")
    release.set()
    assert wait_until(lambda: not panel.builder.busy)
    assert panel.figure_view.calls[-2:] == [("param", "color", "orange"), ("busy", False)]


def test_panel_reports_invalid_figure_without_retaining_previous_editor(panel, paths):
    open_fixture(panel, paths)
    broken = paths.discussion / "figs/online/broken.yaml"
    broken.write_text("plotter: fixture_curve\nparams: {unknown: 1}\n")
    panel.select_figure(broken)
    assert panel.doc is None and panel.parameters is None
    assert not panel.discussion.isEnabled()
    assert "Unknown parameter" in panel.figure_view.calls[-1][1]


def test_stamped_run_selection_creates_a_figure_and_plots_only_that_run(panel, paths):
    from src.app.services.catalog import list_runs
    panel.tree.set_runs(list_runs(workspace=paths))
    target = "figs/online/20261116-120000"
    run = find(panel.tree, target)
    assert run is not None and run.text(0).startswith("20261116-120000")
    panel.tree.tree.setCurrentItem(run)
    assert panel.tree.selected_experiment() == target
    assert panel.tree.btn_add.isEnabled()
    panel.tree.btn_add.click()
    assert panel.parameter_stack.currentIndex() == panel.PICKER_PAGE
    next(c for c in panel.picker.cards if c.selector == "fixture_curve").clicked.emit()
    assert panel.parameter_stack.currentIndex() == panel.PARAMETERS_PAGE
    assert wait_until(lambda: any(c[0] == "render" for c in panel.figure_view.calls), timeout=30)
    assert panel.doc.path == paths.discussion / target / "fixture_curve.yaml"
    spec = next(c[1] for c in panel.figure_view.calls if c[0] == "render")
    assert [row["value"] for row in spec["data"]["values"]] == [101, 102]
    panel.discussion.setPlainText("This stamped run improved.")
    panel.tree.tree.setCurrentItem(find(panel.tree, "figs/online/20261116-130000"))
    assert model.read_discussion(paths.discussion / target / "fixture_curve.yaml") == "This stamped run improved."
    panel.tree.search.setText("120000")
    assert not find(panel.tree, target).isHidden()
    assert find(panel.tree, "figs/online/20261116-130000").isHidden()
    panel.tree.set_runs(list_runs(workspace=paths))
    assert find(panel.tree, target).childCount() == 1


def test_panel_loads_runs_lazily_through_the_results_backend(app, paths, monkeypatch):
    from src.app.services.catalog import list_runs
    monkeypatch.setattr(panel_module, "FigureView", RecordingView)
    class Backend:
        calls = 0

        def list_runs(self, callback, *, completed_only):
            self.calls += 1
            assert not completed_only
            callback({"runs": list_runs(workspace=paths)}, None)

    backend = Backend()
    panel = panel_module.DiscussionPanel(paths, backend=backend)
    assert backend.calls == 0
    panel.show()
    app.processEvents()
    assert backend.calls == 1
    assert find(panel.tree, "figs/online/20261116-120000") is not None
    panel.tree.btn_refresh.click()
    assert backend.calls == 2
    panel.close()
    panel.deleteLater()
    app.processEvents()


# ── Grouped standard figures: groups, Auto values and run choices ───────────


from PyQt6.QtGui import QShortcut  # noqa: E402
from PyQt6.QtWidgets import QMessageBox  # noqa: E402
from frontend.config_panel.editors import ChoiceListEditor  # noqa: E402


def convergence_box(paths, **kwargs):
    path = model.create_figure(paths, "figs/online", "convergence")
    doc = FigureDoc.load(path, paths=paths)
    return ParametersBox(doc, **kwargs), doc


def disclosure(box, group):
    caption = next(w for w in box.findChildren(QWidget) if w.property("parameterGroupCaption") == group)
    return next(b for b in caption.findChildren(QToolButton) if b.property("parameterDisclosure"))


def test_grouped_parameters_render_as_collapsible_blocks(app, paths):
    box, _ = convergence_box(paths, collapsed={"Style"})
    assert list(box.groups) == ["Data", "Axes", "Style"]
    assert box.groups["Data"].isAncestorOf(box.editors["metric"])
    assert box.groups["Style"].isAncestorOf(box.editors["line_width"])
    assert box.groups["Style"].isHidden() and not box.groups["Axes"].isHidden()
    toggled = []
    box.collapsed_changed.connect(toggled.append)
    disclosure(box, "Style").click()
    disclosure(box, "Axes").click()
    assert not box.groups["Style"].isHidden() and box.groups["Axes"].isHidden()
    assert toggled == [set(), {"Axes"}]


def test_auto_text_values_show_a_placeholder_and_clearing_restores_the_default(app, paths):
    box, doc = convergence_box(paths)
    title = box.editors["title"]
    assert title.placeholderText() == "Auto" and title.text() == ""
    assert box.editors["y_min"].placeholderText() == "Auto"
    box.set_param("title", "Reward")
    assert saved(doc.path) == {"title": "Reward"}
    box.set_param("title", "")
    assert saved(doc.path) == {}


def test_metric_and_method_parameters_offer_the_targets_logged_runs(app, paths):
    box, doc = convergence_box(paths, metric_choices=["eval/reward", "train/reward"], method_choices=["a2c", "ppo"])
    metric = box.editors["metric"]
    assert isinstance(metric, QComboBox) and metric.currentText() == "eval/reward"
    methods = box.editors["methods"]
    assert isinstance(methods, ChoiceListEditor) and methods.choices == ["a2c", "ppo"]
    methods.add("ppo")
    assert saved(doc.path) == {"methods": ["ppo"]}
    methods.remove("ppo")
    assert saved(doc.path) == {"methods": []}
    # Nothing logged yet: the metric is typed.
    assert isinstance(convergence_box(paths)[0].editors["metric"], QLineEdit)


def test_choice_list_editor_writes_the_whole_list_and_offers_remaining_choices(app):
    written = []
    editor = ChoiceListEditor(["a"], ["a", "b", "c"], written.append, noun="method")
    entries = [w for w in editor.findChildren(QFrame) if w.property("choiceEntry")]
    assert [e.property("choiceValue") for e in entries] == ["a"]
    add = next(b for b in editor.findChildren(QToolButton) if b.property("choiceAdd"))
    assert [a.text() for a in add.menu().actions() if a.text() and not a.isSeparator()][:2] == ["b", "c"]
    editor.add("c")
    editor.add("c")  # already present
    assert written == [["a", "c"]]
    app.processEvents()
    remove = next(b for b in editor.findChildren(QToolButton) if b.property("choiceRemove") == "a")
    remove.click()
    app.processEvents()
    assert written[-1] == ["c"]
    assert [w.property("choiceValue") for w in editor.findChildren(QFrame)
            if w.property("choiceEntry") and not w.isHidden()] == ["c"]


# ── Spec cache ───────────────────────────────────────────────────────────────


def test_builder_serves_unchanged_figures_from_the_cache_until_forced_or_changed(app, paths):
    doc = FigureDoc.load(paths.discussion / "figs/online/fixture_curve.yaml", paths=paths)
    builds = []
    builder = FigureBuilder(paths, build=lambda p, e, values, live: builds.append(dict(values)) or {"n": len(builds)})
    built = []
    builder.built.connect(built.append)
    builder.request(doc, immediate=True)
    assert wait_until(lambda: built and not builder.busy)
    builder.request(doc, immediate=True)
    assert len(builds) == 1 and built == [{"n": 1}, {"n": 1}]  # served at once, without a build

    doc.set_param("color", "#000000")  # live: applied to the drawn figure, not rebuilt
    builder.request(doc, immediate=True)
    assert len(builds) == 1

    builder.request(doc, immediate=True, force=True)
    assert wait_until(lambda: len(built) == 4 and not builder.busy) and len(builds) == 2

    doc.set_param("metric", "eval/reward")
    builder.request(doc, immediate=True)
    assert wait_until(lambda: len(builds) == 3 and not builder.busy)
    metrics = paths.results / "logs/figs/online"
    csv = next(metrics.glob("**/metrics.csv"))
    csv.write_text(csv.read_text() + "\n")  # new logged data changes the target's stamp
    builder.request(doc, immediate=True)
    assert wait_until(lambda: len(builds) == 4 and not builder.busy)


# ── Renaming, duplicating and deleting from the tree ─────────────────────────


def figure_item(tree, paths, name="fixture_curve"):
    tree.select_figure(paths.discussion / f"figs/online/{name}.yaml")
    return tree.tree.currentItem()


def shortcut(tree, key):
    return next(s for s in tree.findChildren(QShortcut) if s.key().toString() == key)


def test_figure_context_menu_and_shortcuts(app, paths, monkeypatch):
    tree = DiscussionTree(paths)
    assert tree.build_context_menu(find(tree, "figs/online")) is None
    menu = tree.build_context_menu(figure_item(tree, paths))
    assert [a.text() for a in menu.actions()] == ["Rename…", "Duplicate", "", "Delete…"]
    renames = []
    tree.rename_requested.connect(renames.append)
    menu.actions()[0].trigger()
    shortcut(tree, "F2").activated.emit()
    figure = paths.discussion / "figs/online/fixture_curve.yaml"
    assert renames == [figure, figure]

    menu.actions()[1].trigger()
    copy = figure.with_name("fixture_curve_copy.yaml")
    assert copy.is_file() and tree.selected_figure() == copy

    answers = iter([QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes])
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: next(answers))
    shortcut(tree, "Del").activated.emit()
    assert copy.is_file()  # declined
    shortcut(tree, "Del").activated.emit()
    assert not copy.exists() and tree.selected_experiment() == "figs/online" and tree.selected_figure() is None
    assert item_texts(tree, find(tree, "figs/online")) == ["fixture_curve"]


def test_panel_renames_the_open_figure_inline_and_reports_a_taken_name(panel, paths):
    path = open_fixture(panel, paths)
    panel.discussion.setPlainText("Kept through the rename")
    panel.tree.duplicate_figure(path)  # the copy takes "fixture_curve_copy"
    panel.tree.select_figure(path)
    panel.tree.rename_requested.emit(path)
    editor = panel.parameters.name_editor
    assert editor is not None and editor.text() == "fixture_curve" and panel.parameters.title_label.isHidden()

    editor.setText("fixture_curve_copy")
    editor.editingFinished.emit()
    assert path.exists() and editor.property("parameterInvalid")
    assert "already exists" in editor.accessibleDescription()

    editor.setText("Reward curve")
    editor.editingFinished.emit()
    renamed = path.with_name("Reward curve.yaml")
    assert renamed.is_file() and not path.exists()
    assert wait_until(lambda: panel.doc is not None and panel.doc.path == renamed)
    assert panel.parameters.title_label.text() == "Reward curve"
    assert model.read_discussion(renamed) == "Kept through the rename"


def test_panel_escape_cancels_a_rename(panel, paths):
    path = open_fixture(panel, paths)
    panel.parameters.start_rename()
    panel.parameters.name_editor.setText("Unwanted")
    QTest.keyClick(panel.parameters.name_editor, Qt.Key.Key_Escape)
    assert panel.parameters.name_editor is None and not panel.parameters.title_label.isHidden()
    assert path.is_file() and not path.with_name("Unwanted.yaml").exists()


# ── Figure toolbar and saved group state ─────────────────────────────────────


def test_figure_toolbar_resets_the_view_and_copies_the_image(panel, paths):
    assert not any(button.isEnabled() for button in panel.figure_tools.values())
    open_fixture(panel, paths)
    assert all(button.isEnabled() for button in panel.figure_tools.values())
    panel.figure_tools["reset"].click()
    panel.figure_tools["copy"].click()
    assert ("reset",) in panel.figure_view.calls and ("image", 2.0) in panel.figure_view.calls
    from PyQt6.QtGui import QImage
    image = QImage(4, 3, QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.red)
    panel.figure_view.image_ready.emit(image)
    assert QApplication.clipboard().image().size() == image.size()


class Settings:
    def __init__(self):
        self.values = {}

    def get(self, section, key, default=None):
        return self.values.get((section, key), default)

    def persist_key(self, section, key, value):
        self.values[(section, key)] = value


def test_collapsed_groups_are_remembered_per_figure_type(app, paths, monkeypatch):
    monkeypatch.setattr(panel_module, "FigureView", RecordingView)
    settings = Settings()
    panel = panel_module.DiscussionPanel(paths, settings)
    try:
        first = model.create_figure(paths, "figs/online", "convergence")
        second = model.create_figure(paths, "figs/online", "convergence")
        panel.tree.populate(select=first)
        disclosure(panel.parameters, "Axes").click()
        assert settings.values[("discussion", "collapsed_groups")] == {"convergence": ["Axes"]}
        panel.tree.select_figure(second)
        assert panel.parameters.groups["Axes"].isHidden() and not panel.parameters.groups["Data"].isHidden()
    finally:
        panel.shutdown()
        panel.deleteLater()
        app.processEvents()
