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
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QToolButton,
    QWidget,
)

from frontend.config_model import ConfigTree
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


def test_rules_control_opens_selected_folder_with_effective_environment(viewer, tmp_path):
    from types import SimpleNamespace
    widget, _, _ = viewer
    config = tmp_path / "workspace/in/config"
    profile = config / "env/gym/cartpole.yaml"
    profile.parent.mkdir(parents=True)
    profile.write_text("symbolic_name: cartpole\n")
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
    assert widget._ruleset_domain_choices() == ["anticipatory_v1", "default"]
    button = next(b for b in widget.findChildren(QToolButton) if b.property("parameterOpen"))
    combo = next(c for c in widget.findChildren(QComboBox)
                 if {"anticipatory_v1", "default"}.issubset({c.itemText(i) for i in range(c.count())}))
    with patch("frontend.config_viewer.QMenu.exec") as menu:
        button.click()
        combo.setCurrentText("anticipatory_v1")
        button.click()
        assert not menu.called
    assert opened == [root / "src/usr/rules/cartpole/default", root / "src/usr/rules/cartpole/anticipatory_v1"]
    # Explicit overrides must also affect discovery and opening.
    widget.raw_data["env"]["symbolic_name"] = "mountaincar"
    button.click()
    assert opened[-1] == root / "src/usr/rules/mountaincar/anticipatory_v1"
    # A component profile without any environment opens the catalog once.
    widget.raw_data = {"rules": "default"}
    widget._get_resolved_defaults = lambda: {}
    with patch("frontend.config_viewer.QMenu.exec") as menu:
        button.click()
        assert not menu.called
    assert opened[-1] == root / "src/usr/rules"


def title(box):
    return next(w.text() for w in box.findChildren(QLabel) if w.objectName() in ("heading", "cardTitle"))


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
