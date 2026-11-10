"""Cross-box contracts: new boxes must use the shared parameter presentation."""
import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
import yaml

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QCheckBox,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QWidget,
)

from frontend.config_model import ConfigTree, component_selection
from frontend.config_viewer import (
    NUMBER_FIELD_WIDTH,
    PARAM_FIELD_WIDTH,
    PARAM_LABEL_WIDTH,
    ConfigBox,
    ConfigViewer,
    IndentedHierarchyBlock,
    _RowHoverReveal,
    invalidate_config_tree,
    set_config_tree,
)
from frontend.theme import current_theme, stylesheet
from frontend.widgets import ExpandingTextEdit, HierarchicalComboBox, ToggleSlider

app = QApplication.instance() or QApplication([])


@pytest.fixture
def viewer(tmp_path):
    root = tmp_path / "config"
    (root / "paradigms").mkdir(parents=True)
    (root / "paradigms" / "online_rl.yaml").write_text("budget_unit: timesteps\nallowed_agents: [ppo]\n")
    (root / "env").mkdir()
    (root / "env" / "demo.yaml").write_text("env_id: Demo-v0\ncompatible_paradigms: [online_rl]\n")
    tree = ConfigTree(root)
    set_config_tree(tree)
    previous_style = app.styleSheet()
    app.setStyleSheet(stylesheet(current_theme()))
    widget = ConfigViewer()
    widget._component_choices = lambda kind: {"demo": {"scale": 1, "nested": {"depth": 2}}}
    yield widget, tree, tmp_path
    widget.close()
    app.setStyleSheet(previous_style)
    invalidate_config_tree()


def load(viewer, data, rel="experiment/demo/test.yaml"):
    widget, _, root = viewer
    path = root / "test.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    widget.load_file(path, rel)
    widget.resize(1050, 1000)
    widget.show()
    app.processEvents()
    return widget


def boxes(widget):
    return [widget.boxes_layout.itemAt(i).widget() for i in range(widget.boxes_layout.count())
            if isinstance(widget.boxes_layout.itemAt(i).widget(), ConfigBox)]


@pytest.mark.parametrize("rel,title", [
    ("experiment/demo/_base.yaml", "Group Defaults"),
    ("experiment/demo/nested/_base.yaml", "Experiment"),
])
def test_only_group_bases_use_the_group_defaults_box(viewer, rel, title):
    widget = load(viewer, {"paradigm": "online_rl", "seed": 1}, rel=rel)
    labels = [label.text() for label in boxes(widget)[0].findChildren(QLabel)]
    assert title in labels


def parameter_controller(widget, *captions):
    candidates = [(controller, [g.caption.text() for g in controller.rows.groups if isinstance(g.caption, QLabel)])
                  for controller in widget._parameter_reorders]
    matches = [controller for controller, labels in candidates if set(captions).issubset(labels)]
    assert matches, (captions, [labels for _, labels in candidates])
    return matches[0]


def move_parameter(widget, controller, caption, target, *, accept=True):
    source = next(g for g in controller.rows.groups if isinstance(g.caption, QLabel) and g.caption.text() == caption)
    destination = next(g for g in controller.rows.groups if isinstance(g.caption, QLabel) and g.caption.text() == target)
    start = source.caption.mapToGlobal(source.caption.rect().center())
    widget.scroll.ensureWidgetVisible(destination.caption)
    app.processEvents()
    position = controller.host.mapToGlobal(destination.rect(controller.host).center())
    controller.begin(source, start)
    controller.move(position)
    app.processEvents()
    controller.finish(accept)
    app.processEvents()


def test_parameter_drag_moves_live_editors_and_persists_without_config_overrides(viewer):
    from frontend.settings import SettingsManager

    widget, _, root = viewer
    widget.settings_manager = SettingsManager(root / "preferences")
    widget = load(viewer, {"alpha": 1, "beta": 2, "gamma": 3}, "model/test.yaml")
    before = widget.current_path.read_text()
    field = next(w for w in widget.findChildren(QAbstractSpinBox) if w.property("parameterPath") == "alpha")
    controller = parameter_controller(widget, "alpha", "beta", "gamma")
    move_parameter(widget, controller, "alpha", "gamma")
    assert not widget.is_dirty
    assert [g.caption.text() for g in controller.rows.groups] == ["Description", "beta", "gamma", "alpha"]
    assert field is next(w for w in widget.findChildren(QAbstractSpinBox) if w.property("parameterPath") == "alpha")
    field.setValue(9)
    assert widget.raw_data["alpha"] == 9
    assert widget.current_path.read_text() == before
    widget.load_file(widget.current_path, "model/test.yaml")
    # Reload settings from disk and use a new viewer, as after restarting the app.
    other = ConfigViewer(SettingsManager(root / "preferences"))
    other.load_file(widget.current_path, "model/test.yaml")
    assert [g.caption.text() for g in parameter_controller(other, "alpha", "beta").rows.groups] == [
        "Description", "beta", "gamma", "alpha"]
    assert not other.is_dirty
    other.close()


def test_nested_parameter_drag_keeps_children_and_rejects_other_nesting(viewer):
    widget = load(viewer, {"section": {"first": {"child": 4}, "second": 2,
                                            "third": {"other": 8, "last": 9}}}, "model/test.yaml")
    controller = parameter_controller(widget, "first", "second", "third")
    child = next(w for w in widget.findChildren(QAbstractSpinBox) if w.property("parameterPath") == "section.first.child")
    move_parameter(widget, controller, "first", "third")
    assert [g.caption.text() for g in controller.rows.groups] == ["second", "third", "first"]
    assert child.value() == 4 and child.isVisible()
    nested = parameter_controller(widget, "other", "last")
    source = nested.rows.groups[0]
    nested.begin(source, source.caption.mapToGlobal(QPoint(4, 4)))
    assert not nested._valid_position(controller.host.mapToGlobal(controller.rows.groups[-1].rect(controller.host).center()))
    nested.move(controller.host.mapToGlobal(controller.rows.groups[-1].rect(controller.host).center()))
    nested.finish(False)
    assert widget.raw_data["section"] == {"first": {"child": 4}, "second": 2, "third": {"other": 8, "last": 9}}
    move_parameter(widget, controller, "first", "second", accept=False)
    assert [g.caption.text() for g in controller.rows.groups] == ["second", "third", "first"]


def test_parameter_drag_threshold_escape_and_editing_gestures(viewer):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    widget = load(viewer, {"alpha": 1, "beta": 2}, "model/test.yaml")
    controller = parameter_controller(widget, "alpha", "beta")
    source = next(g for g in controller.rows.groups if g.caption.text() == "alpha")
    QTest.mousePress(source.caption, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    QTest.mouseMove(source.caption, QPoint(6, 5))
    assert controller.dragged is None
    QTest.mouseMove(source.caption, QPoint(35, 5))
    assert controller.dragged is source
    QTest.keyClick(widget, Qt.Key.Key_Escape)
    assert controller.dragged is None
    assert source.caption.isVisible()
    assert QApplication.overrideCursor() is None
    editor = next(w for w in widget.findChildren(QAbstractSpinBox) if w.property("parameterPath") == "alpha")
    QTest.mousePress(editor, Qt.MouseButton.LeftButton)
    QTest.mouseMove(editor, QPoint(40, 10))
    QTest.mouseRelease(editor, Qt.MouseButton.LeftButton)
    assert controller.dragged is None


def test_parameter_mouse_drop_commits_and_outside_release_cancels(viewer):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    widget = load(viewer, {"alpha": 1, "beta": 2}, "model/test.yaml")
    controller = parameter_controller(widget, "alpha", "beta")
    alpha, beta = [next(g for g in controller.rows.groups if g.caption.text() == caption)
                   for caption in ("alpha", "beta")]
    destination = beta.caption.mapToGlobal(beta.caption.rect().center())
    QTest.mousePress(alpha.caption, Qt.MouseButton.LeftButton, pos=QPoint(5, 5))
    QTest.mouseMove(alpha.caption, QPoint(40, 5))
    assert controller.dragged is alpha
    point = controller.host.mapFromGlobal(destination)
    QTest.mouseMove(controller.host, point)
    QTest.mouseRelease(controller.host, Qt.MouseButton.LeftButton, pos=point)
    assert controller.dragged is None
    assert [g.caption.text() for g in controller.rows.groups] == ["Description", "beta", "alpha"]
    controller.begin(alpha, alpha.caption.mapToGlobal(QPoint(4, 4)))
    outside = widget.header_bar.mapToGlobal(QPoint(5, 5))
    QTest.mouseRelease(controller.host, Qt.MouseButton.LeftButton, pos=controller.host.mapFromGlobal(outside))
    assert controller.dragged is None
    assert [g.caption.text() for g in controller.rows.groups] == ["Description", "beta", "alpha"]


def test_two_column_and_budget_reordering_preserves_fields(viewer):
    viewer[1].paradigms["online_rl"].raw["training"] = {"budget": "timesteps"}
    widget = load(viewer, {"methods": {}, "paradigm": "online_rl", "seed": 7, "total_timesteps": 1000, "eval_episodes": 10,
                           "variability": {"training_runs": 2, "starting_seed": 4, "evaluation_seed": 7}})
    controller = parameter_controller(widget, "Training runs per method", "Starting training seed", "Evaluation seed")
    move_parameter(widget, controller, "Training runs per method", "Evaluation seed")
    assert [g.caption.text() for g in controller.rows.groups][:3] == [
        "Starting training seed", "Evaluation seed", "Training runs per method"]
    budget = parameter_controller(widget, "Random Seed", "Total Timesteps")
    field = widget.spin_seed
    move_parameter(widget, budget, "Random Seed", "Total Timesteps")
    assert budget.rows.groups[0].caption.text() == "Random Seed"
    assert widget.spin_seed is field and field.value() == 7
    assert not widget.is_dirty


def test_dependent_sweep_controls_refresh_locally_for_boolean_dependencies(viewer, monkeypatch):
    from types import SimpleNamespace

    provider = SimpleNamespace(settings=lambda: [
        {"key": "advanced", "label": "Advanced", "type": "bool", "default": False},
        {"key": "depth", "label": "Depth", "type": "int", "default": 3, "when": {"advanced": True}},
    ])
    monkeypatch.setattr("src.app.pipeline.tuners.get_tuner", lambda **kwargs: provider)
    widget = load(viewer, {"methods": {}, "seed": 8, "tuning": {"component": "optuna"}})
    field = widget.spin_seed
    assert widget.findChild(QAbstractSpinBox, "sweep_depth") is None
    widget.findChild(ToggleSlider, "sweep_advanced").setChecked(True)
    app.processEvents()
    assert widget.findChild(QAbstractSpinBox, "sweep_depth").value() == 3
    assert widget.spin_seed is field
    widget.findChild(ToggleSlider, "sweep_advanced").setChecked(False)
    app.processEvents()
    assert widget.findChild(QAbstractSpinBox, "sweep_depth") is None
    assert widget.spin_seed is field and field.value() == 8


def test_sweep_is_inherited_and_can_be_masked_without_a_separate_switch(viewer):
    from src.app.pipeline.tuning_config import tuning_enabled

    widget, tree, _ = viewer
    group = tree.config_root / "experiment/demo"
    group.mkdir(parents=True)
    base = group / "_base.yaml"
    inherited = {"tuning": {"component": "optuna", "metric": "eval/reward", "direction": "maximize"}}
    base.write_text(yaml.safe_dump(inherited))
    widget = load(viewer, {"sweep": False, "methods": {"m": {"agent": "ppo", "model": "dnn", "tune": {
        "agent.lr": {"type": "float", "low": .001, "high": .01}}}}})
    assert "tuning" not in widget.raw_data
    assert not any(w.text().lower() == "sweep" for w in boxes(widget)[0].findChildren(QLabel))
    assert widget.findChild(ToggleSlider, "sweepEnabled") is None
    assert widget.findChild(QComboBox, "sweepType_m_agent.lr") is not None
    sweep_box = next(b for b in boxes(widget) if b.property("configSection") == "sweep")
    sweep_box.hide()
    assert tuning_enabled(widget._parameter_context_config())
    next(b for b in sweep_box.findChildren(QToolButton) if b.toolTip() == "Remove hyperparameter sweep").click()
    assert widget.raw_data["tuning"] is None
    assert not tuning_enabled(widget._parameter_context_config())
    assert not any(b.property("configSection") == "sweep" for b in boxes(widget))
    assert yaml.safe_load(base.read_text()) == inherited
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["tuning"] is None


@pytest.mark.parametrize("flag", [True, False])
def test_legacy_sweep_flag_does_not_show_a_slider_or_add_tuning(viewer, flag):
    from src.app.pipeline.tuning_config import tuning_enabled

    widget = load(viewer, {"methods": {}, "sweep": flag})
    assert not any(w.text().lower() == "sweep" for w in widget.findChildren(QLabel))
    assert not tuning_enabled(widget._parameter_context_config())
    assert not any(b.property("configSection") == "sweep" for b in boxes(widget))


def test_shared_method_parameter_drag_preserves_group_and_edit_callbacks(viewer):
    widget = load(viewer, {"methods": {"params": {"lr": 0.1, "nested": {"first": 1, "second": 2},
                                                    "batch_size": 32}}})
    controller = parameter_controller(widget, "lr", "[nested]", "batch_size")
    move_parameter(widget, controller, "[nested]", "lr")
    assert [g.caption.text() for g in controller.rows.groups if isinstance(g.caption, QLabel)] == [
        "[nested]", "lr", "batch_size"]
    nested = parameter_controller(widget, "first", "second")
    move_parameter(widget, nested, "first", "second")
    assert [g.caption.text() for g in nested.rows.groups if isinstance(g.caption, QLabel)] == [
        "second", "first"]
    # The child header remains first; all edits still address the original keys.
    assert nested.rows.groups[0].caption is None
    widget._on_universal_param_edited("nested.first", 7)
    assert widget.raw_data["methods"]["params"]["nested"] == {"first": 7, "second": 2}


def test_structural_refresh_retains_scroll_and_cancels_active_drag(viewer):
    widget = load(viewer, {"settings": {f"param_{i}": i for i in range(40)}}, "model/test.yaml")
    widget.resize(1050, 450)
    app.processEvents()
    widget.scroll.verticalScrollBar().setValue(350)
    app.processEvents()
    position = widget.scroll.verticalScrollBar().value()
    controller = parameter_controller(widget, "param_0", "param_1")
    source = controller.rows.groups[0]
    controller.begin(source, source.caption.mapToGlobal(QPoint(4, 4)))
    assert widget.scroll.verticalScrollBar().value() == position
    widget._render_boxes()
    app.processEvents()
    assert QApplication.overrideCursor() is None
    assert widget.scroll.verticalScrollBar().value() == position
    widget._on_generic_path_edited(("settings", "added"), 99)
    widget._render_boxes()
    app.processEvents()
    assert any(w.property("parameterPath") == "settings.added" and w.isVisible()
               for w in widget.findChildren(QAbstractSpinBox))


def test_component_open_action_uses_latest_value_and_preserves_form(viewer):
    from types import SimpleNamespace

    from frontend.config_viewer import _RowHoverReveal
    widget, tree, root = viewer
    target = root / "chosen.txt"
    target.write_text("rules")
    tree.component_parameter_schemas["model/test.yaml"] = {"type": "object", "properties": {
        "rules": {"type": "string", "x-theta-actions": [{"type": "open-in-editor", "path": str(target)}]}}}
    widget = load(viewer, {"rules": "old"}, "model/test.yaml")
    opened = []
    widget.editor_service = SimpleNamespace(open_file=lambda path: opened.append(path) or True)
    button = next(b for b in widget.findChildren(QToolButton) if b.property("parameterOpen"))
    assert button.findChild(_RowHoverReveal)
    assert button.sizePolicy().retainSizeWhenHidden()
    field = button.parentWidget().findChild(QLineEdit)
    assert field is not None
    field.setText("new")
    button.click()
    assert opened == [target]
    assert widget.raw_data["rules"] == "new"


def test_objective_uses_shared_nested_forms_and_opens_source(viewer):
    from types import SimpleNamespace

    from src.app.objectives.api import load_objective
    widget = load(viewer, {"methods": {}, "objective": {"component": "learning_curve", "params": {}},
                           "variability": {"training_runs": 20}, "eval_episodes": 10,
                           "tuning": {"direction": "maximize", "metric": "eval/reward"}})
    opened = []
    widget.editor_service = SimpleNamespace(open_file=lambda path: opened.append(path) or True)
    action = next(b for b in widget.findChildren(QToolButton)
                  if b.property("parameterOpen") and b.property("parameterPath") == "objective.component")
    action.click()
    assert opened == [load_objective("learning_curve").source]
    assert widget.findChild(QLineEdit, "sweepMetric") is None
    sweep = next(box for box in boxes(widget) if title(box) == "Hyperparameter Sweep")
    assert all(title(box) != "Objective" for box in boxes(widget))
    assert sweep.findChild(QToolButton) is not None
    assert action in sweep.findChildren(QToolButton)
    assert sweep.findChild(QComboBox, "sweepObjective").currentText() == "learning_curve"
    field = next(w for w in widget.findChildren(QLineEdit)
                 if w.property("parameterPath") == "objective.params.curve.half_life")
    assert field.property("inheritedDefault")
    field.setText("123")
    field.editingFinished.emit()
    assert widget.raw_data["objective"]["params"]["curve"]["half_life"] == 123
    assert not any(b.property("parameterRemove") and b.property("parameterPath") == "objective.params.curve.half_life"
                   for b in widget.findChildren(QToolButton))
    groups = {b.property("parameterPath"): b for b in widget.findChildren(IndentedHierarchyBlock)}
    assert "objective.params.curve" in groups
    assert widget.raw_data["variability"]["training_runs"] == 20
    widget.remove_block("objective")
    assert "objective" not in widget.raw_data


def test_sweep_selects_objectives_and_replaces_their_children(viewer):
    widget = load(viewer, {"methods": {}, "tuning": {"metric": "val/accuracy", "direction": "maximize"}})
    assert "objective" not in widget.raw_data
    assert widget.findChild(QLineEdit, "sweepMetric").text() == "val/accuracy"
    assert all(action.text() != "Objective" for action in widget.btn_add_block.menu().actions())
    def choose(name):
        widget.findChild(QComboBox, "sweepObjective").setCurrentText(name)
        app.processEvents()
    choose("learning_curve")
    assert widget.raw_data["objective"]["component"] == "learning_curve"
    assert any(w.property("parameterPath") == "objective.params.curve.half_life" for w in widget.findChildren(QLineEdit))
    choose("final_metric")
    assert widget.raw_data["objective"] == {"component": "final_metric", "params": {"metric": "val/accuracy"}}
    assert not any(w.property("parameterPath") == "objective.params.curve.half_life" for w in widget.findChildren(QLineEdit))
    choose("Reported metric")
    assert "objective" not in widget.raw_data
    assert widget.findChild(QLineEdit, "sweepMetric").text() == "val/accuracy"


def test_workspace_objective_opens_source_and_preserves_params(viewer):
    from types import SimpleNamespace

    from src.app.objectives.api import create_objective
    widget, tree, root = viewer
    component = create_objective("editable_ui", workspace=tree.config_root.parent.parent)
    widget = load(viewer, {"methods": {}, "objective": {"component": "editable_ui", "params": {"curve": {"half_life": 50}}}})
    opened = []
    widget.editor_service = SimpleNamespace(open_file=lambda path: opened.append(path) or True)
    action = next(button for button in widget.findChildren(QToolButton)
                  if button.property("parameterOpen") and button.property("parameterPath") == "objective.component")
    action.click()
    assert widget.raw_data["objective"] == {"component": "editable_ui", "params": {"curve": {"half_life": 50}}}
    assert opened == [component.source]


def test_objective_optional_children_add_remove_and_inherited_identity(viewer):
    import json

    from src.app.objectives.api import create_objective
    widget, tree, _ = viewer
    component = create_objective("optional_ui", workspace=tree.config_root.parent.parent)
    manifest_path = component.source.with_suffix(".json")
    metadata = json.loads(manifest_path.read_text())
    metadata["defaults"] = {"nested": {"keep": 1}}
    metadata["schema"] = {"type": "object", "properties": {"nested": {"type": "object", "required": ["keep"],
        "properties": {"keep": {"type": "integer"}, "extra": {"type": "integer", "default": 7}}}}}
    manifest_path.write_text(json.dumps(metadata))
    inherited = {"objective": {"component": "optional_ui", "params": {}}}
    widget._get_resolved_defaults = lambda: inherited
    widget = load(viewer, {"methods": {}, "objective": {"params": {"nested": {"keep": 2}}}})
    add = next(b for b in widget.findChildren(QToolButton)
               if b.property("parameterAdd") and b.property("parameterPath") == "objective.params.nested")
    next(a for a in add.menu().actions() if a.text() == "extra").trigger()
    app.processEvents()
    assert widget.raw_data["objective"]["params"]["nested"]["extra"] == 7
    remove = next(b for b in widget.findChildren(QToolButton)
                  if b.property("parameterRemove") and b.property("parameterPath") == "objective.params.nested.extra")
    assert remove.findChild(_RowHoverReveal)
    assert remove.sizePolicy().retainSizeWhenHidden()
    remove.click()
    assert "extra" not in widget.raw_data["objective"]["params"]["nested"]
    assert widget._effective_objective_spec(widget.raw_data, inherited)["component"] == "optional_ui"
    assert not any(b.property("parameterRemove") and b.property("parameterPath") == "objective.params.nested.keep"
                   for b in widget.findChildren(QToolButton))


def test_method_model_open_action_retains_component_owned_parameter_context(viewer):
    from types import SimpleNamespace
    widget, tree, root = viewer
    schema = {"type": "object", "properties": {"rules": {"type": "string", "x-theta-actions": [
        {"type": "open-in-editor", "path": str(root / "rules.txt")}]} }}
    tree.models["example"] = SimpleNamespace(name="example", parameter_schema=schema, parameters={})
    rows = widget._render_method_model_parameters("demo", "example", {"rules": "default"}, {})
    buttons = []
    for _, row in rows:
        for index in range(row.count()):
            child = row.itemAt(index).widget()
            if child is not None:
                buttons.extend(b for b in child.findChildren(QToolButton) if b.property("parameterOpen"))
    assert len(buttons) == 1
    assert buttons[0].property("parameterPath") == "methods.demo.model.rules"


def ruleset_menu_action(combo, *parts):
    combo._populate_menu()
    menu = combo.popup_menu
    for part in parts[:-1]:
        menu = next(a.menu() for a in menu.actions() if a.text() == part)
    return next(a for a in menu.actions() if a.text() == parts[-1])


def test_rules_control_browses_all_environments_and_opens_selected_folder(viewer, tmp_path):
    from types import SimpleNamespace
    widget, _, _ = viewer
    config = tmp_path / "workspace/in/config"
    profile = config / "env/gym/cartpole.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("env_id: CartPole-v1\n")
    root = config.parent.parent
    for env in ["cartpole", "mountaincar"]:
        for name in ["default", "anticipatory_v1"]:
            folder = root / f"src/usr/rules/{env}/{name}"
            folder.mkdir(parents=True)
            (folder / "clauses.txt").write_text("rules\n")
    tree = ConfigTree(config)
    tree.component_parameter_schemas["model/test.yaml"] = {"properties": {
        "rules": {"type": "string", "x-theta-domain": "ruleset", "x-theta-actions": [
            {"type": "open-in-editor", "handler": {"path": "obsolete_handler.py", "function": "unused"}}
        ]}}}
    set_config_tree(tree)
    widget = load(viewer, {"rules": "default", "defaults": [{"override /env": "gym/cartpole"}],
                           "env": {"num_envs": 1}}, "model/test.yaml")
    opened = []
    widget.editor_service = SimpleNamespace(open_folder=lambda path: opened.append(path) or True,
                                            open_file=lambda path: pytest.fail("Rules must open as a folder"))
    assert widget._ruleset_domain_choices() == ["cartpole/anticipatory_v1", "cartpole/default",
                                                "mountaincar/anticipatory_v1", "mountaincar/default"]
    button = next(b for b in widget.findChildren(QToolButton) if b.property("parameterOpen"))
    combo = widget.findChild(HierarchicalComboBox)
    assert not combo.isEditable()
    assert widget.raw_data["rules"] == "default"  # Rendering preserves legacy values.
    with patch("frontend.config_viewer.QMenu.exec") as menu:
        button.click()
        ruleset_menu_action(combo, "cartpole", "anticipatory_v1").trigger()
        button.click()
        assert not menu.called
    assert widget.raw_data["rules"] == "cartpole/anticipatory_v1"
    assert opened == [root / "src/usr/rules/cartpole/default", root / "src/usr/rules/cartpole/anticipatory_v1"]
    # Qualified resources stay selected when the simulator's symbolic name changes.
    widget.raw_data["env"]["symbolic_name"] = "mountaincar"
    button.click()
    assert opened[-1] == root / "src/usr/rules/cartpole/anticipatory_v1"
    assert len(widget._ruleset_domain_choices()) == 4
    ruleset_menu_action(combo, "mountaincar", "anticipatory_v1").trigger()
    button.click()
    assert opened[-1] == root / "src/usr/rules/mountaincar/anticipatory_v1"
    widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["rules"] == "mountaincar/anticipatory_v1"
    set_config_tree(tree)  # Saving invalidates the global catalog cache.
    # A component profile without any environment opens the catalog once.
    combo.setCurrentText("default")
    widget.raw_data = {"rules": "default"}
    widget._get_resolved_defaults = lambda: {}
    with patch("frontend.config_viewer.QMenu.exec") as menu:
        button.click()
        assert not menu.called
    assert opened[-1] == root / "src/usr/rules"


def test_ruleset_menu_handles_directory_levels_cancellation_and_duplicate_names(viewer):
    widget, _, _ = viewer
    widget._ruleset_domain_choices = lambda: ["classic/cartpole/default", "classic/mountaincar/default"]
    written = []
    combo = widget._schema_value_widget("classic/cartpole/default", {"x-theta-domain": "ruleset"}, written.append)
    action = ruleset_menu_action(combo, "classic", "cartpole", "default")
    assert action.isChecked()
    assert action.data() == "classic/cartpole/default"
    with patch.object(type(combo.popup_menu), "exec", return_value=None) as menu:
        combo.showPopup()
        assert menu.called
    assert written == []
    ruleset_menu_action(combo, "classic", "mountaincar", "default").trigger()
    assert written == ["classic/mountaincar/default"]
    assert combo.currentText() == "classic/mountaincar/default"
    assert combo.toolTip() == combo.currentText()


def test_empty_ruleset_catalog_preserves_existing_value_without_free_text_entry(viewer):
    widget, _, _ = viewer
    widget._ruleset_domain_choices = lambda: []
    written = []
    combo = widget._schema_value_widget("unavailable", {"type": "string", "x-theta-domain": "ruleset"}, written.append)
    assert isinstance(combo, HierarchicalComboBox)
    assert not combo.isEditable()
    assert combo.currentData() == "unavailable"
    assert written == []


def test_blendrl_child_ruleset_menu_edits_the_method_and_preserves_environment(viewer):
    widget, tree, root = viewer
    models = tree.config_root / "model"
    models.mkdir()
    for name, config in {
        "blendrl": {"symbolic": "nsfr", "parameter_schema": {"properties": {
            "symbolic": {"type": "string", "x-theta-domain": "model"}}}},
        "nsfr": {"rules": "default", "parameter_schema": {"properties": {
            "rules": {"type": "string", "x-theta-domain": "ruleset"}}}},
    }.items():
        (models / f"{name}.yaml").write_text(yaml.safe_dump(config))
    set_config_tree(ConfigTree(tree.config_root))
    widget._ruleset_domain_choices = lambda: ["cartpole/default", "mountaincar/default"]
    widget = load(viewer, {"env": {"name": "demo"}, "methods": {"BlendRL": {
        "model": {"name": "blendrl", "symbolic": {"name": "nsfr", "rules": "default"}}}}})
    combo = widget.findChild(HierarchicalComboBox)
    ruleset_menu_action(combo, "mountaincar", "default").trigger()
    assert widget.raw_data["env"] == {"name": "demo"}
    assert widget.raw_data["methods"]["BlendRL"]["model"]["symbolic"] == {
        "name": "nsfr", "rules": "mountaincar/default"}
    next(b for b in widget.findChildren(QToolButton) if b.toolTip() == "Remove 'rules'").click()
    assert widget.raw_data["methods"]["BlendRL"]["model"]["symbolic"] == {"name": "nsfr"}
    add = next(b for b in widget.findChildren(QToolButton) if b.toolTip() == "Add nsfr parameter")
    next(a for a in add.menu().actions() if a.text() == "rules").trigger()
    combo = widget.findChild(HierarchicalComboBox)
    ruleset_menu_action(combo, "cartpole", "default").trigger()
    assert widget.save_to_disk()
    saved = yaml.safe_load((root / "test.yaml").read_text())
    assert saved["methods"]["BlendRL"]["model"]["symbolic"] == {"name": "nsfr", "rules": "cartpole/default"}
    assert saved["env"] == {"name": "demo"}


def test_ruleset_add_dialog_uses_the_shared_nested_menu(viewer):
    from PyQt6.QtWidgets import QDialog
    widget, _, _ = viewer
    widget._ruleset_domain_choices = lambda: ["cartpole/default", "mountaincar/default"]

    def choose(dialog):
        combo = dialog.findChild(HierarchicalComboBox)
        ruleset_menu_action(combo, "mountaincar", "default").trigger()
        return QDialog.DialogCode.Accepted

    with patch("frontend.config_viewer.QDialog.exec", new=choose):
        selected = widget._prompt_component_parameter_value("rules", {"x-theta-domain": "ruleset"})
    assert selected == "mountaincar/default"


def title(box):
    return next(w.text() for w in box.findChildren(QLabel) if w.objectName() in ("heading", "cardTitle"))


@pytest.mark.parametrize("shape", ["named", "keyed", "scalar"])
def test_method_component_identity_and_nested_edits_round_trip(viewer, shape):
    widget, tree, root = viewer
    for folder in ["agent", "model"]:
        (tree.config_root / folder).mkdir(exist_ok=True)
    (tree.config_root / "agent/ppo.yaml").write_text("algorithm: ppo\nlearning_rate: 0.0003\nn_epochs: 10\nmax_grad_norm: 0.5\n")
    (tree.config_root / "model/example.yaml").write_text("architecture: example\nwidth: 16\nbackbone: {dim: 64}\n")
    set_config_tree(ConfigTree(tree.config_root))
    agent_params = {"learning_rate": .001, "n_epochs": 8}
    model_params = {"width": 32, "backbone": {"dim": 128}}
    if shape == "named":
        method = {"agent": {"name": "ppo", **agent_params}, "model": {"name": "example", **model_params}}
    elif shape == "keyed":
        method = {"agent": {"ppo": agent_params}, "model": {"example": model_params}}
    else:
        method = {"agent": "ppo", "model": "example", **agent_params, **model_params}
    widget = load(viewer, {"paradigm": "online_rl", "methods": {"demo": method}})
    assert widget.findChild(QComboBox, "methodAgent_demo").currentText() == "ppo"
    assert widget.findChild(QComboBox, "methodModel_demo").currentText() == "example"
    assert widget.raw_data["methods"]["demo"] == method

    def field_for(text):
        for form in widget.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item and isinstance(item.widget(), QLabel) and item.widget().text() == text:
                    field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                    if field and field.layout():
                        return field.layout().itemAt(0).widget()
        raise AssertionError(f"Missing parameter {text}")

    rate = field_for("learning_rate")
    rate.setValue(.002)
    width = field_for("width")
    if isinstance(width, QLineEdit):
        width.setText("48")
        width.editingFinished.emit()
    else:
        width.setValue(48)
    widget._on_method_nested_param_edited("demo", "backbone", "dim", 256)
    widget._add_method_param("demo", "max_grad_norm", .7)
    remove = next(b for b in widget.findChildren(QToolButton) if b.toolTip() == "Remove 'n_epochs'")
    assert remove.findChild(_RowHoverReveal)
    remove.click()
    widget.save_to_disk()
    saved = yaml.safe_load((root / "test.yaml").read_text())["methods"]["demo"]
    agent_name, agent = component_selection(saved["agent"])
    model_name, model = component_selection(saved["model"])
    if shape == "scalar":
        agent = model = saved
    assert (agent_name, model_name) == ("ppo", "example")
    assert agent["learning_rate"] == .002 and agent["max_grad_norm"] == .7
    assert "n_epochs" not in agent
    assert model["width"] == 48 and model["backbone"]["dim"] == 256
    if shape != "scalar":
        assert not {"learning_rate", "max_grad_norm", "width", "backbone"} & saved.keys()


def test_every_ordinary_box_has_title_adder_and_shared_field_widths(viewer):
    widget = load(viewer, {
        "paradigm": "online_rl", "env": {"name": "demo", "num_envs": 2}, "seed": 4,
        "trainer": {"accelerator": "cpu"}, "methods": {},
        "checkpoint": {"monitor": "val/loss"}, "tuning": {"metric": "val/loss", "seed": 4},
        "variability": {"training_runs": 2},
        "objective": {"component": "learning_curve", "params": {}},
        "data_module": {"name": "demo", "params": {"scale": 2}},
        "evaluator": {"name": "demo", "params": {"nested": {"depth": 2}}},
        "plots": ["demo"], "plot_overrides": {"demo": {"scale": 2}},
        "telemetry": {"metrics": ["eval/reward"]}, "custom": {"value": 7},
    })
    cards = boxes(widget)
    assert len(cards) >= 9
    headings = [next(w for w in box.findChildren(QLabel)
                     if w.objectName() in ("heading", "cardTitle")) for box in cards]
    assert len({(heading.font().pixelSize(), heading.font().weight()) for heading in headings}) == 1
    for box in cards:
        if title(box) == "Methods":
            assert box.property("specialParameterLayout")
            continue
        heading = next(w for w in box.findChildren(QLabel) if w.objectName() in ("heading", "cardTitle"))
        header = next(row for row in box.findChildren(QHBoxLayout)
                      if any(row.itemAt(i).widget() is heading for i in range(row.count())))
        index = next(i for i in range(header.count()) if header.itemAt(i).widget() is heading)
        add = header.itemAt(index + 1).widget()
        assert isinstance(add, QToolButton) and add.property("parameterAdd"), title(box)
        compact = box.property("parameterColumns") == 2
        field_width = NUMBER_FIELD_WIDTH if compact else PARAM_FIELD_WIDTH
        label_width = 120 if compact else PARAM_LABEL_WIDTH
        for field in box.findChildren((QLineEdit, QComboBox, QAbstractSpinBox)):
            if isinstance(field.parentWidget(), (QComboBox, QAbstractSpinBox)):
                continue
            if field.property("parameterExpands"):
                assert field.maximumWidth() > field_width
                continue
            assert field.minimumWidth() == field.maximumWidth() == field_width, (title(box), field.objectName())
        for button in box.findChildren(QToolButton):
            if button.property("parameterRemove"):
                assert button.property("parameterHoverBound"), (title(box), button.toolTip())
        for form in box.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item and isinstance(item.widget(), QLabel):
                    assert item.widget().width() == label_width
    roots = [widget.findChild(QAbstractSpinBox, "sweep_n_trials")]
    roots += [field for field in widget.findChildren(QLineEdit) + widget.findChildren(QAbstractSpinBox)
              if field.property("parameterPath") in ("checkpoint.monitor", "custom.value")]
    assert len(roots) == 3
    positions = []
    for field in roots:
        box = field.parentWidget()
        while not isinstance(box, ConfigBox):
            box = box.parentWidget()
        positions.append(field.mapTo(box, QPoint()).x())
    assert positions[1] == positions[2]


def test_recursive_groups_use_equal_indents_and_protect_required_fields(viewer):
    _, tree, _ = viewer
    tree.component_parameter_schemas["model/test.yaml"] = {
        "required": ["identity"], "properties": {
            "identity": {"type": "string"}, "optional": {"type": "integer"},
            "outer": {"type": "object", "required": ["keep"], "properties": {
                "keep": {"type": "integer"}, "child": {"type": "object", "properties": {
                    "leaf": {"type": "integer"}, "added": {"type": "integer", "default": 9}}}}}}}
    widget = load(viewer, {"identity": "demo", "optional": 1,
                           "outer": {"keep": 2, "child": {"leaf": 3, "grandchild": {"value": 4}}}}, "model/test.yaml")
    removes = {b.property("parameterPath"): b for b in widget.findChildren(QToolButton) if b.property("parameterRemove")}
    assert "identity" not in removes
    assert "outer.keep" not in removes
    assert "optional" in removes and "outer.child.leaf" in removes
    groups = {b.property("parameterPath"): b for b in widget.findChildren(IndentedHierarchyBlock)}
    child = groups["outer.child"]
    grandchild = groups["outer.child.grandchild"]
    child_x = child.mapTo(widget, QPoint()).x()
    grandchild_x = grandchild.mapTo(widget, QPoint()).x()
    assert 0 < grandchild_x - child_x < PARAM_LABEL_WIDTH / 2
    # Each nested block is a spanning row, so label width never adds indentation.
    item = child.form_layout.itemAt(child.form_layout.rowCount() - 1, QFormLayout.ItemRole.SpanningRole)
    assert item.widget() is grandchild
    add = next(b for b in widget.findChildren(QToolButton)
               if b.property("parameterAdd") and b.property("parameterPath") == "outer.child")
    next(a for a in add.menu().actions() if a.text() == "added").trigger()
    app.processEvents()
    assert widget.raw_data["outer"]["child"]["added"] == 9
    delete = next(b for b in widget.findChildren(QToolButton)
                  if b.property("parameterRemove") and b.property("parameterPath") == "outer.child.added")
    delete.click()
    assert "added" not in widget.raw_data["outer"]["child"]


def test_hover_removal_does_not_move_editors(viewer):
    widget = load(viewer, {"optional": 1}, "model/test.yaml")
    remove = next(b for b in widget.findChildren(QToolButton) if b.property("parameterPath") == "optional")
    field = next(w for w in widget.findChildren(QAbstractSpinBox) if w.property("parameterPath") == "optional")
    hover = remove.findChild(_RowHoverReveal)
    before = field.geometry()
    with patch("frontend.config_viewer.QCursor.pos", return_value=field.mapToGlobal(field.rect().center())):
        hover._update_visibility()
        assert remove.isVisible()
    with patch("frontend.config_viewer.QCursor.pos", return_value=QPoint(-1000, -1000)):
        hover._update_visibility()
        assert not remove.isVisible()
    app.processEvents()
    assert field.geometry() == before


def test_first_environment_child_preserves_selector_identity(viewer):
    widget = load(viewer, {"paradigm": "online_rl", "env": "demo", "methods": {}}, "experiment/other/test.yaml")
    add = next(b for b in widget.findChildren(QToolButton)
               if b.property("parameterAdd") and b.property("parameterPath") == "env")
    next(a for a in add.menu().actions() if a.text() == "num_envs").trigger()
    app.processEvents()
    assert widget.raw_data["env"] == {"num_envs": 4}
    assert widget._get_explicit_env() == "demo"


@pytest.mark.parametrize("rel", ["experiment/demo/test.yaml", "experiment/demo/_base.yaml"])
def test_experiment_description_spans_two_columns_with_environment_children_and_actions(viewer, rel):
    widget = load(viewer, {"paradigm": "online_rl", "defaults": [{"override /env": "demo"}], "env": {
        "num_envs": 2, "options": {"depth": 1}},
        "description": "An experiment description that can use the available card width.",
        "trainer": {"accelerator": "cpu"}, "methods": {}}, rel)
    block = next(b for b in widget.findChildren(IndentedHierarchyBlock) if b.property("parameterPath") == "env")
    env_label = "Environment" if rel.endswith("/_base.yaml") else "Environment / Task"
    controller = parameter_controller(widget, "Paradigm", env_label)
    assert controller.rows.layout.property("equalParameterColumns")
    env_group = next(group for group in controller.rows.groups if group.caption.text() == env_label)
    assert env_group.parts[1] is block
    form = env_group.parts[0]
    assert form.itemAt(0, QFormLayout.ItemRole.FieldRole).layout().itemAt(0).widget() is widget.combo_env
    assert widget.combo_paradigm.mapTo(widget, QPoint()).y() == widget.combo_env.mapTo(widget, QPoint()).y()
    assert widget.combo_env.mapTo(widget, QPoint()).x() > widget.combo_paradigm.mapTo(widget, QPoint()).x()
    assert widget.txt_description.width() > PARAM_FIELD_WIDTH * 2
    assert widget.txt_description.mapTo(widget, widget.txt_description.rect().bottomLeft()).y() < widget.combo_env.mapTo(widget, QPoint()).y()
    assert widget._get_explicit_env() == "demo"
    assert not widget.is_dirty

    widget.txt_description.setText("Edited description across the full card")
    assert widget.raw_data["description"] == "Edited description across the full card"

    field = next(w for w in block.findChildren(QAbstractSpinBox) if w.property("parameterPath") == "env.num_envs")
    assert field.mapTo(widget, QPoint()).y() > widget.combo_env.mapTo(widget, widget.combo_env.rect().bottomLeft()).y()
    assert field.mapTo(widget, QPoint()).x() > widget.combo_env.mapTo(widget, QPoint()).x()
    field.setValue(3)
    assert widget.raw_data["env"]["num_envs"] == 3
    remove = next(b for b in block.findChildren(QToolButton) if b.property("parameterPath") == "env.num_envs")
    assert remove.findChild(_RowHoverReveal)
    assert remove.sizePolicy().retainSizeWhenHidden()
    remove.click()
    assert "num_envs" not in widget.raw_data["env"]
    assert widget._get_explicit_env() == "demo"
    add = next(b for b in widget.findChildren(QToolButton)
               if b.property("parameterAdd") and b.property("parameterPath") == "env")
    next(a for a in add.menu().actions() if a.text() == "num_steps").trigger()
    app.processEvents()
    assert widget.raw_data["env"]["num_steps"] == 128
    assert widget.raw_data["env"]["options"] == {"depth": 1}
    assert widget._get_explicit_env() == "demo"
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["env"]["num_steps"] == 128


@pytest.mark.parametrize("rel", ["experiment/demo/test.yaml", "experiment/demo/_base.yaml"])
def test_description_wraps_grows_and_preserves_typed_line_breaks(viewer, rel):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QTextCursor
    from PyQt6.QtTest import QTest

    widget = load(viewer, {"description": "Short description", "paradigm": "online_rl", "methods": {}}, rel)
    editor = widget.txt_description
    assert isinstance(editor, ExpandingTextEdit)
    short_height = editor.height()
    long_description = "A longer description should wrap and keep all of its content visible. " * 20
    editor.setPlainText(long_description)
    QTest.qWait(10)  # Let the surrounding form relayout after the document grows.
    assert editor.height() > short_height * 2
    assert editor.verticalScrollBar().maximum() == 0
    assert editor.document().documentLayout().documentSize().height() <= editor.viewport().height()
    assert widget.raw_data["description"] == long_description.strip()
    assert "\n" not in widget.raw_data["description"]  # Visual wrapping does not alter YAML.
    assert widget.combo_env.mapTo(widget, QPoint()).y() > editor.mapTo(widget, editor.rect().bottomLeft()).y()

    editor.setPlainText("First line")
    editor.moveCursor(QTextCursor.MoveOperation.End)
    QTest.keyClick(editor, Qt.Key.Key_Return)
    QTest.keyClicks(editor, "Second line")
    app.processEvents()
    assert widget.raw_data["description"] == "First line\nSecond line"
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["description"] == "First line\nSecond line"
    widget.load_file(widget.current_path, rel)
    app.processEvents()
    assert widget.txt_description.toPlainText() == "First line\nSecond line"
    next(b for b in widget.findChildren(QToolButton) if b.toolTip() == "Clear description").click()
    app.processEvents()
    assert "description" not in widget.raw_data
    assert widget.txt_description.height() == short_height


def test_expanding_text_editor_reflows_on_resize_without_changing_its_value():
    text = "Descriptions reflow when the available width changes. " * 10
    editor = ExpandingTextEdit(text)
    editor.resize(600, editor.height())
    editor.show()
    app.processEvents()
    wide_height = editor.height()
    changed = []
    editor.textChanged.connect(lambda: changed.append(editor.toPlainText()))
    editor.resize(240, editor.height())
    app.processEvents()
    assert editor.height() > wide_height
    editor.resize(600, editor.height())
    app.processEvents()
    assert editor.height() == wide_height
    assert editor.toPlainText() == text
    assert changed == []
    editor.close()


def test_metric_picker_keeps_checkbox_behavior_with_shared_actions(viewer):
    from frontend.metric_picker import MetricPicker

    widget = load(viewer, {"methods": {}, "telemetry": {"metrics": ["custom/", "eval/reward"]}})
    picker = widget.findChild(MetricPicker)
    assert picker.section_boxes["custom"].isChecked()
    assert picker.metric_boxes["eval/reward"].isChecked()
    add = next(button for button in picker.findChildren(QToolButton) if button.toolTip() == "Add custom/ metric")
    with patch("frontend.config_viewer.QInputDialog.getText", return_value=("new", True)):
        next(action for action in add.menu().actions() if action.text() == "+ Custom...").trigger()
    app.processEvents()
    assert set(widget.raw_data["telemetry"]["metrics"]) == {"custom/new", "eval/reward"}
    picker = widget.findChild(MetricPicker)
    assert not picker.section_boxes["custom"].isChecked()
    assert picker.metric_boxes["custom/new"].isChecked()
    delete = next(button for button in picker.findChildren(QToolButton) if button.toolTip() == "Remove custom/new")
    delete.click()
    assert widget.raw_data["telemetry"]["metrics"] == ["eval/reward"]


def test_monitors_use_shared_forms_and_preserve_inherited_exclusions(viewer, monkeypatch):
    from src.app.telemetry.monitors import MonitorSpec

    monkeypatch.setattr("src.app.telemetry.monitors.available_monitors", lambda _, **kwargs: [
        MonitorSpec("blend/", "Routing diagnostics", "Model"),
        MonitorSpec("blend/train/entropy", "Routing entropy", "Model"),
        MonitorSpec("eval/reward", "Evaluation reward", "Runner")])
    widget, _, root = viewer
    inherited = {"telemetry": {"monitors": {"disabled": ["eval/reward"]}}}
    widget._get_resolved_defaults = lambda: inherited
    widget = load(viewer, {"methods": {}})
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    assert title(box) == "Monitors"
    assert box.objectName() == "card"
    assert box.property("configBox") is True
    assert not box.findChildren(QCheckBox)
    assert all(isinstance(field, ToggleSlider) for key, field in widget.field_widgets.items()
               if key.startswith("telemetry.monitors."))
    assert not any("Checked outputs" in item.text() for item in box.findChildren(QLabel))
    assert box.findChildren(IndentedHierarchyBlock)
    # Showing defaults never materializes an override.
    assert "telemetry" not in widget.raw_data
    assert not widget.field_widgets["telemetry.monitors.eval/reward"].isChecked()

    widget.field_widgets["telemetry.monitors.blend/"].setChecked(False)
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"]["disabled"] == ["eval/reward", "blend/"]
    assert not widget.field_widgets["telemetry.monitors.blend/train/entropy"].isEnabled()
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    add = next(button for button in box.findChildren(QToolButton) if button.toolTip() == "Add monitor")
    # Header add is immediately after the title (drag handle may precede it).
    header = next(layout for layout in box.findChildren(QHBoxLayout) if layout.indexOf(add) >= 0)
    assert header.indexOf(add) > 0
    remove = next(button for button in box.findChildren(QToolButton) if button.toolTip() == "Remove exclusion for blend/")
    assert remove.findChild(_RowHoverReveal)
    remove.click()
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"]["disabled"] == ["eval/reward"]
    widget.field_widgets["telemetry.monitors.blend/train/entropy"].setChecked(False)
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"]["disabled"] == ["eval/reward", "blend/train/entropy"]
    widget.save_to_disk()
    saved = yaml.safe_load((root / "test.yaml").read_text())
    assert saved["telemetry"]["monitors"]["disabled"] == ["eval/reward", "blend/train/entropy"]
    widget.field_widgets["telemetry.monitors.enabled"].setChecked(False)
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"]["enabled"] is False
    assert widget.raw_data["telemetry"]["monitors"]["disabled"] == ["eval/reward", "blend/train/entropy"]
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    next(button for button in box.findChildren(QToolButton) if button.toolTip() == "Remove monitors and stream all metrics").click()
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"] is None
    assert not any(box.property("configSection") == "monitors" for box in boxes(widget))


def test_config_box_background_survives_section_identity():
    standard = ConfigBox()
    named = ConfigBox(section="monitors")
    named.setObjectName("differentSection")
    parent = QWidget()
    layout = QHBoxLayout(parent)
    layout.addWidget(standard)
    layout.addWidget(named)
    parent.resize(500, 160)
    previous = app.styleSheet()
    app.setStyleSheet(stylesheet(current_theme()))
    parent.show()
    app.processEvents()
    try:
        expected = standard.grab().toImage().pixelColor(20, 20)
        actual = named.grab().toImage().pixelColor(20, 20)
        assert actual == expected
        assert actual.name() != parent.grab().toImage().pixelColor(0, 0).name()
    finally:
        parent.close()
        app.setStyleSheet(previous)


def test_monitor_box_is_optional_and_deletion_preserves_other_telemetry(viewer):
    widget = load(viewer, {"methods": {}, "telemetry": {"sinks": ["csv", "stream"], "metrics": ["eval/"]}})
    assert not any(box.property("configSection") == "monitors" for box in boxes(widget))
    action = next(action for action in widget.btn_add_block.menu().actions() if action.text() == "Monitors")
    assert action.isEnabled()
    action.trigger()
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"] == {"enabled": True, "disabled": []}
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    remove = next(button for button in box.findChildren(QToolButton)
                  if button.toolTip() == "Remove monitors and stream all metrics")
    assert remove.findChild(_RowHoverReveal)
    remove.click()
    app.processEvents()
    assert not any(box.property("configSection") == "monitors" for box in boxes(widget))
    assert widget.raw_data["telemetry"] == {"sinks": ["csv", "stream"], "metrics": ["eval/"]}


def test_monitor_group_rows_and_header_bypass_preserve_choices(viewer, monkeypatch):
    from src.app.telemetry.monitors import MonitorSpec, monitors_from_config
    monkeypatch.setattr("src.app.telemetry.monitors.available_monitors", lambda _, **kwargs: [
        MonitorSpec("train/a", "A", "Model", group="Training"),
        MonitorSpec("train/b", "B", "Model", group="Training"),
        MonitorSpec("eval/c", "C", "Model", group="Evaluation")])
    widget = load(viewer, {"methods": {}, "telemetry": {
        "monitors": {"enabled": True, "disabled": ["train/b", "eval/c"]}}})

    def box():
        return next(box for box in boxes(widget) if box.property("configSection") == "monitors")

    def group(title):
        return next(toggle for toggle in box().findChildren(ToggleSlider) if toggle.property("monitorGroup") == title)

    assert group("Training").isChecked()
    assert not group("Evaluation").isChecked()
    # Group titles use normal parameter labels, followed by spanning child blocks.
    caption = next(label for label in box().findChildren(QLabel) if label.text() == "Training")
    assert caption.objectName() != "heading"
    header_toggle = widget.field_widgets["telemetry.monitors.enabled"]
    assert header_toggle.mapTo(box(), QPoint()).x() < box().title_label.mapTo(box(), QPoint()).x()
    group_label = caption.parentWidget()
    assert group_label.property("parameterLabel")
    assert any(form.getWidgetPosition(group_label)[1] == QFormLayout.ItemRole.LabelRole
               for form in box().findChildren(QFormLayout) if form.getWidgetPosition(group_label)[0] >= 0)
    child_block = widget.field_widgets["telemetry.monitors.train/a"].parentWidget()
    while child_block is not None and not isinstance(child_block, IndentedHierarchyBlock):
        child_block = child_block.parentWidget()
    assert child_block is not None
    assert child_block.mapTo(box(), QPoint()).y() > caption.mapTo(box(), QPoint()).y()

    group("Training").setChecked(False)
    app.processEvents()
    assert not group("Training").isChecked()
    assert not widget.field_widgets["telemetry.monitors.train/a"].isChecked()
    assert not widget.field_widgets["telemetry.monitors.train/b"].isChecked()
    group("Training").setChecked(True)
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"]["disabled"] == ["eval/c"]
    widget.field_widgets["telemetry.monitors.train/a"].setChecked(False)
    app.processEvents()
    assert group("Training").isChecked()  # B remains enabled
    widget.field_widgets["telemetry.monitors.train/b"].setChecked(False)
    app.processEvents()
    assert not group("Training").isChecked()
    widget.field_widgets["telemetry.monitors.train/a"].setChecked(True)
    app.processEvents()
    assert group("Training").isChecked()

    selected = list(widget.raw_data["telemetry"]["monitors"]["disabled"])
    widget.field_widgets["telemetry.monitors.enabled"].setChecked(False)
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"] == {"enabled": False, "disabled": selected}
    assert monitors_from_config(widget.raw_data).disabled == ()
    assert widget.field_widgets["telemetry.monitors.train/a"].isChecked()
    assert not widget.field_widgets["telemetry.monitors.train/b"].isChecked()
    assert not widget.field_widgets["telemetry.monitors.train/a"].isEnabled()
    widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["telemetry"]["monitors"] == {"enabled": False, "disabled": selected}
    widget.field_widgets["telemetry.monitors.enabled"].setChecked(True)
    app.processEvents()
    assert widget.raw_data["telemetry"]["monitors"] == {"enabled": True, "disabled": selected}
    assert group("Training").isChecked()
    assert monitors_from_config(widget.raw_data).keeps("train/a")
    assert not monitors_from_config(widget.raw_data).keeps("train/b")
    # Multiple changes before the deferred redraw must compose, not overwrite.
    group("Evaluation").setChecked(True)
    group("Training").setChecked(False)
    app.processEvents()
    assert set(widget.raw_data["telemetry"]["monitors"]["disabled"]) == {"train/a", "train/b"}


def test_real_component_monitors_show_individual_toggles_before_a_run(tmp_path):
    from pathlib import Path
    from src.app.telemetry.monitors import available_monitors

    tree = ConfigTree(Path(__file__).resolve().parents[1] / "in/config")
    set_config_tree(tree)
    data = {"telemetry": {"monitors": {"enabled": True, "disabled": []}},
            "methods": {"demo": {"agent": "sb3-dev/ppo", "model": {
        "name": "blendrl-dev", "neural": {"name": "dnn"}, "symbolic": {"type": "nsfr"}}}}}
    widget = ConfigViewer()
    widget.current_path = tmp_path / "unrun.yaml"
    widget.current_rel_path = "experiment/new/unrun.yaml"
    widget.raw_data = data
    widget._block_updates = False
    widget._render_monitors_box(data, {})
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    expected = {spec.name for spec in available_monitors(data, include_shared=False)}
    fields = {key.removeprefix("telemetry.monitors."): value for key, value in widget.field_widgets.items()
              if key.startswith("telemetry.monitors.") and key != "telemetry.monitors.enabled"}
    assert set(fields) == expected
    assert len(fields) > 40
    assert not any(name.endswith("/") for name in fields)
    assert all(isinstance(field, ToggleSlider) and field.isChecked() for field in fields.values())
    assert not box.findChildren(QCheckBox)
    assert "blend/train/logic_0/coverage" in fields
    assert "blend/eval/agreement/logic_0__neural_1/js_divergence" in fields
    assert "diagnostics/learning_rate" in fields
    assert "losses/train/value_loss" in fields
    assert widget.raw_data["telemetry"]["monitors"] == {"enabled": True, "disabled": []}
    widget.close()
    invalidate_config_tree()


def test_sweep_and_variability_use_two_columns_without_seed_summary(viewer):
    widget = load(viewer, {"methods": {}, "tuning": {"direction": "maximize"},
                           "variability": {"training_runs": 2, "starting_seed": 4,
                                           "evaluation_seed": 7}, "eval_episodes": 10})
    for name in ("Hyperparameter Sweep", "Training and Evaluation Variability"):
        box = next(box for box in boxes(widget) if title(box) == name)
        assert box.property("parameterColumns") == 2
        grid = next(grid for grid in box.findChildren(QGridLayout)
                    if any(grid.getItemPosition(i)[1] == 1 for i in range(grid.count())))
        assert any(grid.getItemPosition(i)[1] == 0 for i in range(grid.count()))
        widget.resize(1400, 1000)
        app.processEvents()
        left, right = grid.cellRect(0, 0), grid.cellRect(0, 1)
        assert abs(left.width() - right.width()) <= 1
        assert grid.columnCount() == 2
    variance = next(box for box in boxes(widget) if title(box) == "Training and Evaluation Variability")
    assert not any(label.text().startswith("Training seeds:") for label in variance.findChildren(QLabel))


def test_variability_parallel_limit_uses_shared_optional_parameter_row(viewer):
    widget = load(viewer, {"methods": {}, "variability": {"training_runs": 4, "parallel_runs": 1}})
    field = widget.findChild(QAbstractSpinBox, "variabilityParallelRuns")
    assert field is not None and field.minimum() == 1 and field.value() == 1
    assert 'sequentially' in field.toolTip()
    field.setValue(2)
    assert widget.raw_data['variability']['parallel_runs'] == 2
    button = next(b for b in widget.findChildren(QToolButton)
                  if b.property('parameterRemove') and b.property('parameterPath') == 'variability.parallel_runs')
    assert button.sizePolicy().retainSizeWhenHidden()
    button.click()
    app.processEvents()
    assert 'parallel_runs' not in widget.raw_data['variability']


def test_sweep_parameters_fill_space_beside_objective_children(viewer):
    widget = load(viewer, {"methods": {}, "tuning": {"direction": "maximize"},
                           "objective": {"component": "learning_curve", "params": {}}})
    sweep = next(box for box in boxes(widget) if title(box) == "Hyperparameter Sweep")
    group = next(group for group in sweep.findChildren(IndentedHierarchyBlock)
                 if group.property("parameterPath") == "objective.params")
    direction = sweep.findChild(QComboBox, "sweepDirection")
    tuner = sweep.findChild(QComboBox, "sweepComponent")
    group_top = group.mapTo(sweep, QPoint()).y()
    group_bottom = group_top + group.height()
    assert direction.mapTo(sweep, QPoint()).x() > group.mapTo(sweep, QPoint()).x()
    assert group_top <= tuner.mapTo(sweep, QPoint()).y() < group_bottom


def test_monitor_components_use_shared_forms_and_required_component_selector(viewer):
    widget = load(viewer, {"methods": {}, "monitors": {"features": {
        "component": "activation-evolution", "target": "network", "enabled": True,
        "probes": {"count": 16}, "params": {"max_features": 32}}}})
    box = next(box for box in boxes(widget) if title(box) == "Monitors")
    assert box.property("configBox")
    assert not box.findChildren(QCheckBox)
    assert not any(title(card) == "Monitor Components" for card in boxes(widget))
    assert not any(action.text() == "Monitor Components" for action in widget.btn_add_block.menu().actions())
    enabled = next(w for w in box.findChildren(ToggleSlider) if w.property("parameterPath") == "monitors.features.enabled")
    assert any(button.accessibleName() == "Expand or collapse Monitor · features"
               for button in box.findChildren(QToolButton))
    enabled.setChecked(False)
    assert widget.raw_data["monitors"]["features"]["enabled"] is False
    assert widget._parameter_remove(("monitors", "features", "component")) is None
    assert any(w.property("parameterPath") == "monitors.features.params" for w in box.findChildren(IndentedHierarchyBlock))
    widget._on_generic_path_edited(("monitors", "features", "interval_updates"), 8)
    widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["monitors"]["features"]["interval_updates"] == 8
    widget._remove_generic_path(("monitors", "features", "params", "max_features"))
    assert "max_features" not in widget.raw_data["monitors"]["features"]["params"]


@pytest.mark.parametrize("width", [960, 1050, 1400])
def test_nested_monitor_settings_keep_editors_clear_of_labels(viewer, monkeypatch, width):
    from src.app.telemetry.monitors import MonitorSpec

    monkeypatch.setattr("src.app.telemetry.monitors.available_monitors", lambda cfg, **kwargs: [
        MonitorSpec(f"interp/{name}/score", "Score", f"Monitor · {name}")
        for name in cfg["monitors"]])
    widget = load(viewer, {"methods": {}, "monitors": {
        "features": {"component": "activation-evolution", "target": "network", "enabled": True},
        "network": {"component": "network-structure", "target": "network",
                    "interval_updates": 100, "max_snapshots": 16,
                    "probes": {"count": 128, "steps": 512, "seed": 0},
                    "params": {"max_layers": 16, "max_units": 64, "max_edges": 4096}}}})
    widget.resize(width, 1000)
    app.processEvents()

    def check_geometry():
        box = next(box for box in boxes(widget) if title(box) == "Monitors")
        paths = set()
        for form in box.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                caption_item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                field_item = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                caption = caption_item.widget() if caption_item else None
                field_row = field_item.layout() if field_item else None
                if not isinstance(caption, QLabel) or field_row is None:
                    continue
                for index in range(field_row.count()):
                    field = field_row.itemAt(index).widget()
                    if field is None or not field.property("parameterPath"):
                        continue
                    path = field.property("parameterPath")
                    paths.add(path)
                    label_right = caption.mapTo(box, QPoint()).x() + caption.width()
                    field_left = field.mapTo(box, QPoint()).x()
                    assert field_left >= label_right + form.horizontalSpacing(), path
                    assert field_left + field.width() <= box.width(), path
        assert {"monitors.network.component", "monitors.network.interval_updates",
                "monitors.network.probes.count", "monitors.network.params.max_layers"} <= paths

    check_geometry()
    count = next(field for field in widget.findChildren(QAbstractSpinBox)
                 if field.property("parameterPath") == "monitors.network.probes.count")
    count.setValue(64)
    assert widget.raw_data["monitors"]["network"]["probes"]["count"] == 64
    widget.field_widgets["telemetry.monitors.interp/network/score"].setChecked(False)
    app.processEvents()
    # The toggle schedules a rebuild; lay out its newly created widgets too.
    app.processEvents()
    check_geometry()


def test_monitor_components_can_be_added_without_changing_stream_visibility(viewer):
    widget = load(viewer, {"methods": {}})
    widget.add_block("monitor_components")
    assert widget.raw_data["monitors"] == {}
    assert "telemetry" not in widget.raw_data
    assert any(title(box) == "Monitors" for box in boxes(widget))
    assert not any(title(box) == "Monitor Components" for box in boxes(widget))


def test_monitor_sets_collapse_remove_restore_and_reload(viewer, monkeypatch):
    from src.app.telemetry.monitors import MonitorSpec, monitors_from_config
    monkeypatch.setattr("src.app.telemetry.monitors.available_monitors", lambda _, **kwargs: [
        MonitorSpec("train/a", "A", "Model", group="Training"),
        MonitorSpec("train/b", "B", "Model", group="Training"),
        MonitorSpec("eval/c", "C", "Model", group="Evaluation")])
    widget, _, _ = viewer
    widget._get_resolved_defaults = lambda: {"telemetry": {"monitors": {"disabled": ["train/b"]}}}
    widget = load(viewer, {"methods": {}})

    def box():
        return next(box for box in boxes(widget) if box.property("configSection") == "monitors")

    def disclosure():
        return next(button for button in box().findChildren(QToolButton)
                    if button.accessibleName() == "Expand or collapse Training")

    child = widget.field_widgets["telemetry.monitors.train/a"]
    arrow = disclosure()
    assert arrow.mapTo(box(), QPoint()).x() < next(
        label for label in box().findChildren(QLabel) if label.text() == "Training").mapTo(box(), QPoint()).x()
    arrow.click()
    assert not child.isVisible()
    assert "telemetry" not in widget.raw_data
    widget.field_widgets["telemetry.monitors.eval/c"].setChecked(False)
    app.processEvents()
    assert not disclosure().isChecked()  # redraw retains disclosure state
    disclosure().click()
    app.processEvents()
    assert widget.field_widgets["telemetry.monitors.train/a"].isVisible()

    remove = next(button for button in box().findChildren(QToolButton)
                  if button.toolTip() == "Remove monitor set Training")
    assert remove.findChild(_RowHoverReveal)
    assert remove.sizePolicy().retainSizeWhenHidden()
    group = next(toggle for toggle in box().findChildren(ToggleSlider) if toggle.property("monitorGroup") == "Training")
    assert remove.mapTo(box(), QPoint()).x() > group.mapTo(box(), QPoint()).x()
    remove.click()
    app.processEvents()
    assert "telemetry.monitors.train/a" not in widget.field_widgets
    assert "telemetry.monitors.eval/c" in widget.field_widgets
    assert widget.raw_data["telemetry"]["monitors"]["disabled"] == ["train/b", "eval/c"]
    assert widget.raw_data["telemetry"]["monitors"]["removed"] == ["train/a", "train/b"]
    assert not monitors_from_config(widget.raw_data).keeps("train/a")
    path, rel = widget.current_path, widget.current_rel_path
    widget.save_to_disk()
    widget.load_file(path, rel)
    app.processEvents()
    assert "telemetry.monitors.train/a" not in widget.field_widgets
    add = next(button for button in box().findChildren(QToolButton) if button.toolTip() == "Add monitor")
    next(action for action in add.menu().actions() if action.text() == "Restore Training").trigger()
    app.processEvents()
    assert widget.field_widgets["telemetry.monitors.train/a"].isChecked()
    assert not widget.field_widgets["telemetry.monitors.train/b"].isChecked()
    assert not widget.field_widgets["telemetry.monitors.eval/c"].isChecked()


def test_monitor_title_plus_explicitly_adds_installed_provider(viewer, monkeypatch):
    from src.app.core.provider_catalog import Capability, Package, ProviderCatalog
    from src.app.telemetry.monitors import MonitorSpec
    from src.app.workspace import WorkspacePaths
    widget, _, root = viewer
    catalog = ProviderCatalog.__new__(ProviderCatalog)
    catalog.paths, catalog.overrides = WorkspacePaths(root), {}
    catalog.packages = [Package("optional", root, "user", "1.0.0", {}, [
        Capability("optional-probes", "monitor", "unused.py:Monitor")])]
    monkeypatch.setattr("src.app.core.provider_catalog.get_catalog", lambda *args: catalog)

    def discover(cfg, **kwargs):
        assert kwargs.get("include_shared") is False
        return [MonitorSpec("train/loss", "Loss", "Algorithm")] + [
            MonitorSpec(f"interp/{name}/score", "Score", f"Monitor · {name}")
            for name in cfg.get("monitors", {})]

    monkeypatch.setattr("src.app.telemetry.monitors.available_monitors", discover)
    widget = load(viewer, {"methods": {}, "telemetry": {"monitors": {"disabled": []}}})
    assert "monitors" not in widget.raw_data
    assert set(widget.field_widgets).intersection({"telemetry.monitors.interp/optional-probes/score"}) == set()

    def adder():
        box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
        button = next(button for button in box.findChildren(QToolButton) if button.toolTip() == "Add monitor")
        header = next(layout for layout in box.findChildren(QHBoxLayout) if layout.indexOf(button) >= 0)
        assert header.itemAt(header.indexOf(button) - 1).widget() is box.title_label
        return button

    assert [action.text() for action in adder().menu().actions()] == ["optional-probes"]
    adder().menu().actions()[0].trigger()
    app.processEvents()
    assert widget.raw_data["monitors"]["optional-probes"] == {
        "component": "optional-probes", "enabled": True, "target": ""}
    assert "telemetry.monitors.interp/optional-probes/score" in widget.field_widgets
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    component = next(field for field in box.findChildren(QWidget)
                     if field.property("parameterPath") == "monitors.optional-probes.component")
    assert box.isAncestorOf(component)
    stream = next(toggle for toggle in box.findChildren(ToggleSlider)
                  if toggle.property("monitorGroupSource") == "Monitor · optional-probes")
    stream.setChecked(False)
    app.processEvents()
    box = next(box for box in boxes(widget) if box.property("configSection") == "monitors")
    assert any(field.property("parameterPath") == "monitors.optional-probes.component"
               for field in box.findChildren(QWidget))
    assert widget.raw_data["monitors"]["optional-probes"]["enabled"] is True
    widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["monitors"] == widget.raw_data["monitors"]
    remove = next(button for button in box.findChildren(QToolButton)
                  if button.toolTip() == "Remove monitor set Monitor · optional-probes")
    remove.click()
    app.processEvents()
    assert "monitors" not in widget.raw_data
    assert "telemetry.monitors.interp/optional-probes/score" not in widget.field_widgets
    adder().menu().actions()[0].trigger()
    app.processEvents()
    assert set(widget.raw_data["monitors"]) == {"optional-probes"}
    assert widget.field_widgets["telemetry.monitors.interp/optional-probes/score"].isChecked()
    # A hidden instance saved by the older UI is restored, not duplicated.
    widget.raw_data["telemetry"]["monitors"]["removed"] = ["interp/optional-probes/"]
    widget._render_boxes()
    adder().menu().actions()[0].trigger()
    app.processEvents()
    assert set(widget.raw_data["monitors"]) == {"optional-probes"}
    assert widget.raw_data["telemetry"]["monitors"]["removed"] == []
    adder().menu().actions()[0].trigger()
    app.processEvents()
    assert set(widget.raw_data["monitors"]) == {"optional-probes", "optional-probes_2"}
    widget.remove_block("monitors")
    assert "monitors" not in widget.raw_data
    assert not any(card.property("configSection") == "monitors" for card in boxes(widget))
    inherited = {"monitors": {"optional-probes": {
        "component": "optional-probes", "target": "network", "enabled": True}}}
    widget._get_resolved_defaults = lambda: inherited
    widget._remove_standalone_monitor("optional-probes")
    assert widget.raw_data["monitors"]["optional-probes"] == {"enabled": False}
    widget._add_standalone_monitor("optional-probes")
    app.processEvents()
    assert set(widget.raw_data["monitors"]) == {"optional-probes"}
    assert widget.raw_data["monitors"]["optional-probes"]["target"] == "network"
    assert widget.raw_data["monitors"]["optional-probes"]["enabled"] is True
    assert inherited["monitors"]["optional-probes"]["enabled"] is True


def test_removing_invalid_monitor_instance_clears_warning_and_preserves_component_defaults(viewer):
    from pathlib import Path
    from src.app.telemetry.monitors import available_monitors

    set_config_tree(ConfigTree(Path(__file__).resolve().parents[1] / "in/config"))
    data = {"telemetry": {"monitors": {"enabled": True, "disabled": []}},
            "methods": {"demo": {"agent": "sb3-dev/ppo", "model": {
                "name": "blendrl-dev", "neural": {"name": "dnn"}, "symbolic": {"type": "nsfr"}}}}}
    expected = {"telemetry.monitors." + spec.name for spec in available_monitors(data, include_shared=False)}
    assert len(expected) > 40
    data["monitors"] = {"broken": {"component": "missing-provider"}}
    widget = load(viewer, data)

    def box():
        return next(card for card in boxes(widget) if card.property("configSection") == "monitors")

    assert expected <= widget.field_widgets.keys()
    assert any("missing-provider" in caption.text() for caption in box().findChildren(QLabel, "configError"))
    remove = next(button for button in box().findChildren(QToolButton)
                  if button.toolTip() == "Remove monitor set Monitor · broken")
    remove.click()
    app.processEvents()
    assert "monitors" not in widget.raw_data
    assert not box().findChildren(QLabel, "configError")
    assert expected <= widget.field_widgets.keys()
    widget.save_to_disk()
    widget.load_file(widget.current_path, widget.current_rel_path)
    app.processEvents()
    assert not box().findChildren(QLabel, "configError")
    assert expected <= widget.field_widgets.keys()


def test_data_loader_box_edits_method_bindings_and_preserves_inherited_selection(viewer):
    widget = load(viewer, {
        'paradigm': 'online_rl', 'data_module': {'name': 'demo', 'params': {'scale': 2}},
        'symbolic_state_loader': {'name': 'demo', 'params': {'scale': 3}},
        'methods': {'policy': {'agent': 'ppo', 'model': 'basics/dnn',
                               'symbolic_state_loader': {'name': 'other', 'params': {'depth': 7}}}},
    })
    box = next(box for box in boxes(widget) if title(box) == 'Data Loaders')
    assert widget.field_widgets['methods.policy.data_module'].currentData() == ''
    assert widget.field_widgets['methods.policy.symbolic_state_loader'].currentData() == 'other'
    editor = next(w for w in box.findChildren(QAbstractSpinBox)
                  if w.property('parameterPath') == 'methods.policy.symbolic_state_loader.params.depth')
    assert isinstance(editor.parentWidget(), QWidget)
    editor.setValue(9)
    assert widget.raw_data['methods']['policy']['symbolic_state_loader']['params']['depth'] == 9
    widget._set_loader_binding(('methods', 'policy', 'symbolic_state_loader'), 'demo')
    app.processEvents()
    assert widget.raw_data['methods']['policy']['symbolic_state_loader'] == {
        'name': 'demo', 'params': {'scale': 1, 'nested': {'depth': 2}}}
    widget._set_loader_binding(('methods', 'policy', 'symbolic_state_loader'), '')
    app.processEvents()
    assert 'symbolic_state_loader' not in widget.raw_data['methods']['policy']
    assert widget.raw_data['symbolic_state_loader'] == {'name': 'demo', 'params': {'scale': 3}}
    assert not next(card for card in boxes(widget) if title(card) == "Data Loaders").findChildren(QCheckBox)


@pytest.mark.parametrize('rel', ['experiment/demo/test.yaml', 'experiment/demo/_base.yaml'])
def test_site_always_above_runtime_settings_without_creating_override(viewer, rel):
    widget = load(viewer, {'paradigm': 'online_rl', 'trainer': {'max_epochs': 3}}, rel)
    site = widget.field_widgets['site']
    assert site.currentText() == 'local (default)'
    assert widget._site_selection(widget.raw_data) is None
    site_label = next(item for item in boxes(widget)[0].findChildren(QLabel) if item.text() == 'Site')
    trainer_label = next(item for item in boxes(widget)[0].findChildren(QLabel) if item.text() == 'trainer')
    assert site_label.mapTo(widget, QPoint()).y() < trainer_label.mapTo(widget, QPoint()).y()
    assert site.findText('ncshare/common') >= 0
    site.setCurrentIndex(site.findData('ncshare/common'))
    assert widget.raw_data['defaults'][-1] == {'override /site': 'ncshare/common'}
    assert 'site=ncshare/common' in widget.get_overrides()
    widget._set_experiment_site(None)
    assert widget._site_selection(widget.raw_data) is None
    assert widget.field_widgets['site'].currentText() == 'local (default)'


def test_site_resources_use_shared_nested_controls(viewer):
    widget = load(viewer, {'defaults': [{'override /site': 'ncshare'}], 'resources': {'cores': 8, 'memory': '16G'}})
    block = next(block for block in widget.findChildren(IndentedHierarchyBlock)
                 if any(label.text() == 'cores' for label in block.findChildren(QLabel)))
    assert boxes(widget)[0].isAncestorOf(block)
    editor = next(editor for editor in block.findChildren(QAbstractSpinBox) if editor.property('parameterPath') == 'resources.cores')
    editor.setValue(12)
    assert widget.raw_data['resources']['cores'] == 12
    widget._set_experiment_site('local')
    assert widget.raw_data['resources'] == {'cores': 12, 'memory': '16G'}


def test_ncshare_cpu_component_large_limits_render_and_round_trip(viewer):
    from pathlib import Path
    from frontend.widgets import IntegerField
    project = Path(__file__).resolve().parents[1]
    data = yaml.safe_load((project / 'src/usr/sites/ncshare/configs/site/ncshare/common.yaml').read_text())
    widget = load(viewer, data, 'site/ncshare/common.yaml')
    assert not widget.is_dirty
    limits = data['execution']
    for name in ('max_pending_bytes', 'max_published_bytes', 'max_input_bytes', 'max_runtime_bytes'):
        field = next(field for field in widget.findChildren(IntegerField)
                     if field.property('parameterPath') == 'execution.' + name)
        assert field.value() == limits[name]
        assert field.text() == str(limits[name])
    # Values that fit Qt but exceeded the old 100-million UI limit stay intact too.
    field = next(field for field in widget.findChildren(QAbstractSpinBox)
                 if field.property('parameterPath') == 'execution.diagnostic_bytes')
    assert field.value() == 134217728
    limit = next(field for field in widget.findChildren(IntegerField)
                 if field.property('parameterPath') == 'execution.max_published_bytes')
    exact = 2**60 + 1  # cannot round-trip through a double
    limit.setText(str(exact)); limit.editingFinished.emit()
    assert widget.raw_data['execution']['max_published_bytes'] == exact
    assert type(widget.raw_data['execution']['max_published_bytes']) is int
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())['execution']['max_published_bytes'] == exact
    widget.load_file(widget.current_path, 'site/ncshare/common.yaml')
    assert not widget.is_dirty


def test_schema_integer_controls_preserve_large_values_and_validate_bounds(viewer):
    from frontend.widgets import IntegerField
    widget, tree, _ = viewer
    tree.component_parameter_schemas['model/test.yaml'] = {'type': 'object', 'properties': {
        'budget': {'type': 'integer', 'minimum': 0, 'maximum': 2**64},
        'signed': {'type': 'integer', 'minimum': -(2**64), 'maximum': 2**64}}}
    widget = load(viewer, {'budget': 2**60 + 1, 'signed': -(2**60 + 1)}, 'model/test.yaml')
    assert not widget.is_dirty
    field = next(field for field in widget.findChildren(IntegerField) if field.property('parameterPath') == 'budget')
    for invalid in ('1.5', '-1', str(2**64 + 1), ''):
        field.setText(invalid); field.editingFinished.emit()
        assert widget.raw_data['budget'] == 2**60 + 1 and not widget.is_dirty
    field.setText('17179869184'); field.editingFinished.emit()
    assert widget.raw_data['budget'] == 17179869184 and widget.is_dirty
