"""Cross-box contracts: new boxes must use the shared parameter presentation."""
import os
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
import yaml

pytest.importorskip("PyQt6")
from PyQt6.QtCore import QPoint, QRect
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
from frontend.widgets import DoubleSpinBox, ExpandingTextEdit, FlowLayout, HierarchicalComboBox, SpinBox, ToggleSlider

app = QApplication.instance() or QApplication([])


def test_long_parameter_label_reveals_only_over_its_original_hover_area(viewer):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from frontend.config_panel.controls import ParameterLabel

    key = "a_very_long_parameter_name_that_extends_across_the_editor"
    widget = load(viewer, {key: 123}, "model/test.yaml")
    caption = next(w for w in widget.findChildren(ParameterLabel) if w.text() == key)
    field = next(w for w in widget.findChildren(QAbstractSpinBox) if w.property("parameterPath") == key)
    geometry = field.geometry()
    assert not caption.wordWrap() and not caption.hasHeightForWidth()
    assert caption.sizeHint().height() <= caption.fontMetrics().height() + 2
    QTest.mouseMove(field, field.rect().center())
    app.processEvents()
    assert getattr(caption, "_hover_overlay", None) is None or caption._hover_overlay.isHidden()
    QTest.mouseMove(caption, caption.rect().center())
    app.processEvents()
    overlay = caption._hover_overlay
    assert overlay.isVisible() and overlay.text() == key
    assert overlay.width() > caption.width()
    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
    assert field.geometry() == geometry
    QTest.mouseMove(field, field.rect().center())
    app.processEvents()
    assert overlay.isHidden() and field.value() == 123 and not widget.is_dirty


def test_hidden_configure_sweep_button_collapses_its_spacing(viewer):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"params": {"batch_size": 32}}})
    button = widget.findChild(QToolButton, "sweepToggle_params_agent.batch_size")
    assert button is not None and not button.sizePolicy().retainSizeWhenHidden()
    hover = button.findChild(_RowHoverReveal)
    row = hover._row()
    hover._set_visible(False)
    collapsed = row.sizeHint().width()
    hover._set_visible(True)
    expanded = row.sizeHint().width()
    assert expanded == collapsed
    assert row.indexOf(button) == -1


def test_floating_action_hides_with_its_owner(viewer):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"params": {"batch_size": 32}}})
    button = widget.findChild(QToolButton, "sweepToggle_params_agent.batch_size")
    hover = button.findChild(_RowHoverReveal)
    hover._set_visible(True)
    assert button.isVisible()
    owner = hover._parent()
    owner.hide()
    assert button.isHidden()
    with patch("frontend.config_viewer.QCursor.pos", return_value=button.mapToGlobal(button.rect().center())):
        hover._update_visibility()
    assert button.isHidden() and not widget.is_dirty


def test_pointer_can_cross_the_gap_and_click_a_floating_action(viewer):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"agent.custom": {"type": "choice", "values": [11111]}}}}})
    edit = widget.findChild(QLineEdit, "sweep_choice_m_agent.custom_0")
    entry = edit.parentWidget()
    button = next(w for w in widget.findChildren(QToolButton) if w.toolTip() == "Remove value"
                  and w.findChild(_RowHoverReveal)._parent() is entry)
    edit.clearFocus()
    before = entry.geometry()
    QTest.mouseMove(edit, edit.rect().center())
    app.processEvents()
    assert button.isVisible()
    start = widget.mapFromGlobal(edit.mapToGlobal(edit.rect().center()))
    end = widget.mapFromGlobal(button.mapToGlobal(button.rect().center()))
    for x in range(start.x(), end.x() + 1):
        QTest.mouseMove(widget, QPoint(x, end.y()))
        app.processEvents()
        assert button.isVisible()
        assert entry.geometry() == before
    QTest.mouseMove(widget, end + QPoint(0, 60))
    app.processEvents()
    assert button.isHidden()
    QTest.mouseMove(edit, edit.rect().center())
    app.processEvents()
    QTest.mouseMove(button, button.rect().center())
    app.processEvents()
    assert button.isVisible()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    assert widget.raw_data["methods"]["m"]["tune"]["agent.custom"]["values"] == []


@pytest.mark.parametrize("width", [450, 620, 850])
def test_floating_sweep_actions_never_reflow_during_repeated_hover(viewer, width):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"agent.custom": {"type": "choice", "values": [11111, 22222, 33333]}}}}})
    widget.resize(width, 1000)
    app.processEvents()
    edit = widget.findChild(QLineEdit, "sweep_choice_m_agent.custom_0")
    entry = edit.parentWidget()
    content = widget.findChild(QWidget, "sweepInline_m_agent.custom")
    remove = next(w for w in widget.findChildren(QToolButton) if w.toolTip() == "Remove value"
                  and w.findChild(_RowHoverReveal)._parent() is entry)
    hover = remove.findChild(_RowHoverReveal)
    before = (entry.sizeHint(), entry.geometry(), content.geometry(), content.minimumHeight())
    for _ in range(50):
        with patch("frontend.config_viewer.QCursor.pos", return_value=edit.mapToGlobal(edit.rect().center())):
            hover._update_visibility()
            app.processEvents()
        assert remove.isVisible()
        assert (entry.sizeHint(), entry.geometry(), content.geometry(), content.minimumHeight()) == before
        hover._set_visible(False)
        app.processEvents()
        assert (entry.sizeHint(), entry.geometry(), content.geometry(), content.minimumHeight()) == before
    remove.click()
    assert widget.raw_data["methods"]["m"]["tune"]["agent.custom"]["values"] == [22222, 33333]


@pytest.mark.parametrize("dimension", [
    {"type": "int", "low": 1, "high": 5},
    {"type": "choice", "values": [1, 2]},
])
def test_active_method_sweep_remove_stays_beside_its_contents(viewer, dimension):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn", "custom": 1,
        "tune": {"agent.custom": dimension}}}})
    content = widget.findChild(QWidget, "sweepInline_m_agent.custom")
    remove = widget.findChild(QToolButton, "sweepRemove_m_agent.custom")
    assert content is not None and remove is not None
    hover = remove.findChild(_RowHoverReveal)
    for width in (1050, 1400):
        widget.resize(width, 1000)
        app.processEvents()
        hover._set_visible(True)
        app.processEvents()
        last = content.layout().itemAt(content.layout().count() - 1)
        right = content.mapTo(widget, QPoint()).x() + last.geometry().right() + 1
        assert remove.mapTo(widget, QPoint()).x() - right == hover._row().spacing()


def test_parameter_reveal_hover_is_limited_to_its_column():
    from frontend.config_panel.controls import ParameterForm, ParameterRow, _make_subtle_tool_button
    from frontend.config_panel.layout import two_column_form
    host = QWidget()
    form = ParameterForm()
    fields, buttons = [], []
    for key in ("left", "right"):
        field = QLineEdit(key)
        button = _make_subtle_tool_button("close", "Remove " + key)
        form.add_parameter((key,), key, ParameterRow(field, remove=button))
        fields.append(field)
        buttons.append(button)
    host.setLayout(two_column_form(form, identity="fixture.hover"))
    host.resize(1000, 200)
    host.show()
    app.processEvents()
    for field in fields:
        field.clearFocus()
    for index in (0, 1):
        with patch("frontend.config_viewer.QCursor.pos", return_value=fields[index].mapToGlobal(fields[index].rect().center())):
            for button in buttons:
                button.findChild(_RowHoverReveal)._update_visibility()
        assert buttons[index].isVisible()
        assert buttons[1 - index].isHidden()
    host.close()


@pytest.mark.parametrize("text", ["123", "false", "null", ""])
def test_typed_string_edits_save_and_reload_without_reinterpretation(viewer, text):
    widget, tree, _ = viewer
    tree.component_parameter_schemas["model/test.yaml"] = {"type": "object", "properties": {
        "value": {"type": "string"}}}
    widget = load(viewer, {"value": "before", "unknown": {"keep": False}}, "model/test.yaml")
    field = next(w for w in widget.findChildren(QLineEdit) if w.property("parameterPath") == "value")
    field.setText(text)
    assert widget.raw_data["value"] == text and type(widget.raw_data["value"]) is str
    assert widget.save_to_disk()
    saved = yaml.safe_load(widget.current_path.read_text())
    assert saved["value"] == text and saved["unknown"] == {"keep": False}
    widget.load_file(widget.current_path, "model/test.yaml")
    assert widget.raw_data["value"] == text and not widget.is_dirty


def test_array_and_number_drafts_do_not_write_invalid_values(viewer):
    widget, tree, _ = viewer
    tree.component_parameter_schemas["model/test.yaml"] = {"type": "object", "properties": {
        "values": {"type": "array", "items": {"type": "integer"}},
        "small": {"type": "number", "minimum": 0}}}
    widget = load(viewer, {"values": [1, 2], "small": 1e-9}, "model/test.yaml")
    fields = {w.property("parameterPath"): w for w in widget.findChildren(QLineEdit)
              if w.property("parameterPath") in ("values", "small")}
    assert float(fields["small"].text()) == 1e-9 and not widget.is_dirty
    for path, invalid in (("values", "[1,"), ("values", "[1, false]"), ("small", "1e"),
                          ("small", "nan"), ("small", "-1")):
        fields[path].setText(invalid)
        fields[path].editingFinished.emit()
        assert widget.raw_data == {"values": [1, 2], "small": 1e-9}
        assert fields[path].property("parameterInvalid") and not widget.is_dirty
    fields["small"].setText("1e100")
    fields["small"].editingFinished.emit()
    assert widget.raw_data["small"] == 1e100


def test_loaded_invalid_schema_values_are_visible_and_correctable(viewer):
    widget, tree, _ = viewer
    tree.component_parameter_schemas["model/test.yaml"] = {"type": "object", "properties": {
        "count": {"type": "integer", "minimum": 1}}}
    widget = load(viewer, {"count": 0}, "model/test.yaml")
    from frontend.widgets import IntegerField
    field = next(w for w in widget.findChildren(IntegerField) if w.property("parameterPath") == "count")
    assert field.text() == "0" and field.property("parameterInvalid") and not widget.is_dirty
    field.setText("3")
    field.editingFinished.emit()
    assert widget.raw_data["count"] == 3 and not field.property("parameterInvalid")


def test_runtime_integers_do_not_clamp_or_round_loaded_values(viewer):
    exact = 2**60 + 1
    widget = load(viewer, {"paradigm": "online_rl", "total_timesteps": exact, "seed": exact})
    assert widget.field_widgets["total_timesteps"].value() == exact
    assert widget.field_widgets["seed"].value() == exact
    assert not widget.is_dirty
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["total_timesteps"] == exact


def test_region_counts_child_slots_and_resizes_without_recreating_editors():
    from frontend.config_panel.controls import ParameterForm, ParameterRow
    from frontend.config_panel.layout import two_column_form, row_slots
    host = QWidget()
    form = ParameterForm()
    left, right, site = QLineEdit("paradigm"), QLineEdit("environment"), QLineEdit("site")
    child = IndentedHierarchyBlock()
    for index in range(4):
        child.form_layout.add_parameter(("env", str(index)), str(index), ParameterRow(QLineEdit(str(index))))
    form.add_parameter(("paradigm",), "Paradigm", ParameterRow(left))
    form.add_parameter(("env",), "Environment", ParameterRow(right), children=child)
    form.add_parameter(("site",), "Site", ParameterRow(site))
    region = two_column_form(form, identity="fixture.context", align_rows=True)
    host.setLayout(region)
    host.resize(1000, 500)
    host.show()
    app.processEvents()
    assert row_slots(child) == 4
    slot = region._parameter_rows.slots[id(child)]
    assert (slot.column, slot.row, slot.span) == (1, 1, 4)
    wide_order = list(region._parameter_rows.groups)
    host.resize(450, 800)
    app.processEvents()
    assert region._narrow
    host.resize(1200, 500)
    app.processEvents()
    assert not region._narrow and region._parameter_rows.groups == wide_order
    assert left.text() == "paradigm" and right.text() == "environment" and site.text() == "site"
    host.close()


def test_autosave_failure_keeps_dirty_document_and_prevents_switching_files(viewer, monkeypatch):
    widget = load(viewer, {"description": "before"}, "model/test.yaml")
    old_path = widget.current_path
    widget._on_field_edited("description", "edited")
    widget.auto_save = True
    def fail(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr("src.app.components.configuration.ConfigurationFiles.write", fail)
    assert not widget.save_to_disk()
    assert widget.is_dirty and widget.raw_data["description"] == "edited"
    other = old_path.with_name("other.yaml")
    other.write_text("description: other\n")
    widget.load_file(other, "model/other.yaml")
    assert widget.current_path == old_path and widget.is_dirty
    assert widget.raw_data["description"] == "edited"
    widget._autosave_timer.stop()
    widget.auto_save = False


def test_read_only_document_rejects_typed_edits_and_save(viewer, monkeypatch):
    widget = load(viewer, {"value": "false"}, "model/test.yaml")
    before = widget.current_path.read_text()
    monkeypatch.setattr(widget, "_hub_read_only", lambda: True)
    widget._on_generic_path_edited(("value",), "123")
    widget._on_field_edited("description", "new")
    widget._remove_generic_path(("value",))
    assert widget.raw_data == {"value": "false"} and not widget.is_dirty
    assert not widget.save_to_disk() and widget.current_path.read_text() == before


def test_model_schema_fixture_uses_shared_groups_and_preserves_child_identity(viewer):
    from frontend.config_model import Model
    widget, tree, _ = viewer
    schema = {"type": "object", "properties": {"options": {"type": "object", "required": ["kind"],
        "properties": {"kind": {"type": "string", "enum": ["a", "b"]},
                       "child": {"type": "string", "x-theta-domain": "model"}}}}}
    tree.models["root"] = Model("root", "root", parameter_schema=schema)
    tree.models["leaf"] = Model("leaf", "leaf", parameters={"depth": 2}, parameter_schema={
        "type": "object", "properties": {"depth": {"type": "integer", "minimum": 1}}})
    data = {"paradigm": "online_rl", "methods": {"m": {"model": {"root": {
        "options": {"kind": "a", "child": "leaf"}}}}}}
    widget = load(viewer, data)
    rows = parameter_controller(widget, "kind", "child").rows
    assert len([g for g in rows.groups if g.caption is not None]) == 2
    assert not any(b.property("parameterRemove") and b.property("parameterPath") == "methods.m.model.options.kind"
                   for b in widget.findChildren(QToolButton))
    widget._on_method_model_parameter_edited("m", "root", ("options", "child", "depth"), 4, "leaf", ("depth",))
    assert widget.raw_data["methods"]["m"]["model"]["root"]["options"]["child"] == {"name": "leaf", "depth": 4}
    before = yaml.safe_dump(widget.raw_data)
    widget._on_method_model_parameter_edited("m", "root", ("options", "child", "other"), "invalid", "leaf", ("depth",))
    assert yaml.safe_dump(widget.raw_data) == before


def test_legacy_display_order_migrates_without_overwriting_old_preferences(viewer):
    import json
    from frontend.settings import SettingsManager
    widget, _, root = viewer
    widget.settings_manager = SettingsManager(root / "preferences")
    legacy_scope = json.dumps([json.dumps(["Parameters", [], "[]"]), 0])
    legacy_order = [json.dumps([key, 1]) for key in ("Description", "beta", "optional", "alpha")]
    widget.settings_manager.persist_key("config_viewer", "parameter_orders", [{
        "path": str((root / "test.yaml").resolve()), "groups": {legacy_scope: legacy_order}}])
    widget = load(viewer, {"alpha": 1, "beta": 2}, "model/test.yaml")
    controller = parameter_controller(widget, "alpha", "beta")
    assert [g.caption.text() for g in controller.rows.groups] == ["Description", "beta", "alpha"]
    move_parameter(widget, controller, "alpha", "beta")
    saved = widget.settings_manager.get("config_viewer", "parameter_orders")[0]["groups"]
    assert saved[legacy_scope] == legacy_order
    assert any(json.loads(scope)[0] == "parameters" for scope in saved if scope != legacy_scope)
    assert not widget.is_dirty


@pytest.fixture
def viewer(tmp_path, optional_tuner_metadata):
    root = tmp_path / "in/config"
    from component_fixtures import install_optional
    install_optional(tmp_path, "optuna", tmp_path / "archives")
    install_optional(tmp_path, "slurm", tmp_path / "archives")
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


def test_form_load_and_rebuild_share_discovery_only_within_operation(viewer, monkeypatch):
    from collections import Counter
    from pathlib import Path
    from src.app.core.provider_discovery import current_snapshot

    original = Path.rglob
    walks = []

    def walk(root, pattern):
        if pattern == ".theta_component.json":
            walks.append(root)
        return original(root, pattern)

    monkeypatch.setattr(Path, "rglob", walk)
    widget, _, root = viewer
    path = root / "test.yaml"
    path.write_text("paradigm: online_rl\nseed: 1\n")
    widget.load_file(path, "experiment/demo/test.yaml")
    assert walks and max(Counter(walks).values()) == 1
    assert current_snapshot() is None
    walks.clear()
    widget.raw_data["seed"] = 42
    widget.is_dirty = True
    widget._render_boxes()
    assert walks and max(Counter(walks).values()) == 1
    assert current_snapshot() is None
    assert widget.is_dirty and widget.raw_data["seed"] == 42


def test_declared_loader_roles_skip_metadata_fallback(viewer, monkeypatch):
    widget, _, _ = viewer
    # The common fixture stubs choices; exercise the real filtering here.
    del widget._component_choices
    widget.raw_data = {"paradigm": "online_rl"}
    monkeypatch.setattr("frontend.config_viewer._component_profiles", lambda *args, **kwargs: {
        "train": {"role": "training"}, "state": {"role": "symbolic_state"},
    })
    monkeypatch.setattr("src.app.core.provider_metadata.selector_metadata", lambda *args: {})
    with patch("src.app.core.data_loaders.loader_metadata") as metadata:
        assert widget._component_choices("data_module") == {"train": {"role": "training"}}
    metadata.assert_not_called()


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
    controller = parameter_controller(widget, "lr", "nested", "batch_size")
    move_parameter(widget, controller, "nested", "lr")
    assert [g.caption.text() for g in controller.rows.groups if isinstance(g.caption, QLabel)] == [
        "nested", "lr", "batch_size"]
    nested = parameter_controller(widget, "first", "second")
    move_parameter(widget, nested, "first", "second")
    assert [g.caption.text() for g in nested.rows.groups if isinstance(g.caption, QLabel)] == [
        "second", "first"]
    # The parent header sits outside the child block; paths stay unchanged.
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
                           "tuning": {"component": "optuna", "direction": "maximize", "metric": "eval/reward"}})
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
    widget = load(viewer, {"methods": {}, "tuning": {"component": "optuna", "metric": "val/accuracy", "direction": "maximize"}})
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
        "checkpoint": {"monitor": "val/loss"}, "tuning": {"component": "optuna", "metric": "val/loss", "seed": 4},
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
            ancestor = field.parentWidget()
            in_column = bool(field.property("parameterColumn"))
            while ancestor is not None and ancestor is not box:
                if ancestor.property("parameterColumn"):
                    in_column = True
                    break
                ancestor = ancestor.parentWidget()
            row_width = NUMBER_FIELD_WIDTH if in_column else field_width
            assert field.maximumWidth() == row_width, (title(box), field.objectName())
            assert field.minimumWidth() == row_width
        for button in box.findChildren(QToolButton):
            if button.property("parameterRemove"):
                assert button.property("parameterHoverBound"), (title(box), button.toolTip())
        for form in box.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                item = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                if item and isinstance(item.widget(), QLabel):
                    assert item.widget().width() == (120 if getattr(form, "compact", compact) else label_width)
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
    site = widget.field_widgets["site"]
    grid = controller.rows.layout
    site_group = next(group for group in controller.rows.groups if group.caption.text() == "Site")
    assert grid.getItemPosition(grid.indexOf(env_group.parts[0]))[:2] == (0, 1)
    assert grid.getItemPosition(grid.indexOf(block))[:2] == (1, 1)
    assert grid.getItemPosition(grid.indexOf(site_group.parts[0]))[:2] == (1, 0)
    assert site.mapTo(widget, QPoint()).x() == widget.combo_paradigm.mapTo(widget, QPoint()).x()
    assert site.mapTo(widget, QPoint()).y() < block.mapTo(widget, block.rect().bottomLeft()).y()
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
def test_input_row_reordering_preserves_site_and_environment_children(viewer, rel):
    widget = load(viewer, {"paradigm": "online_rl", "env": {"num_envs": 2},
                           "resources": {"cores": 8}, "methods": {}}, rel)
    env_label = "Environment" if rel.endswith("/_base.yaml") else "Environment / Task"
    controller = parameter_controller(widget, "Paradigm", env_label, "Site")
    site = widget.field_widgets["site"]
    env = widget.combo_env
    data = yaml.safe_dump(widget.raw_data)
    move_parameter(widget, controller, "Site", "Paradigm")
    assert controller.rows.groups[0].caption.text() == "Site"
    assert widget.field_widgets["site"] is site
    assert widget.combo_env is env
    assert yaml.safe_dump(widget.raw_data) == data
    assert not widget.is_dirty
    env_group = next(group for group in controller.rows.groups if group.caption.text() == env_label)
    assert env_group.parts[1].property("parameterPath") == "env"
    site_group = controller.rows.groups[0]
    assert isinstance(site_group.parts[1], IndentedHierarchyBlock)


def test_plotting_opt_in_survives_toggle_save_and_launch(viewer):
    widget = load(viewer, {"no_plot": True, "methods": {}})
    widget.chk_no_plot.setChecked(False)
    assert widget.raw_data["no_plot"] is False
    assert "no_plot=false" in widget.get_overrides()
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())["no_plot"] is False


def test_runtime_booleans_wrap_as_complete_controls_and_still_edit(viewer):
    widget = load(viewer, {"paradigm": "online_rl", "seed": 7, "save_dataset": True,
                           "recover": True, "no_plot": True, "custom_flag": False, "methods": {}})
    flow = widget.findChild(FlowLayout, "runtimeParameters")
    assert flow is not None
    controls = [flow.itemAt(i).widget() for i in range(flow.count())]
    box = boxes(widget)[0]
    rendered = box.grab().toImage()
    background = rendered.pixelColor(4, box.height() // 2)
    for control in controls:
        point = control.mapTo(box, QPoint(0, control.height() - 1))
        assert rendered.pixelColor(point) == background
    assert any(control.property("parameterPath") == "custom_flag" for control in controls)
    width = sum(control.sizeHint().width() for control in controls) + flow.spacing() * (len(controls) - 1)
    flow.setGeometry(QRect(0, 0, width, 100))
    assert len({control.y() for control in controls}) == 1
    flow.setGeometry(QRect(0, 0, width // 2, 100))
    assert len({control.y() for control in controls}) > 1
    for control in controls:
        caption = control.findChild(QLabel)
        toggle = control.findChild(ToggleSlider)
        assert caption is not None and toggle is not None
        assert toggle.parentWidget() is caption.parentWidget()
    controller = parameter_controller(widget, "Run methods in parallel", "Timestamp experiment")
    move_parameter(widget, controller, "Timestamp experiment", "Run methods in parallel")
    assert not widget.is_dirty
    widget.chk_timestamp_experiment.setChecked(True)
    assert widget.raw_data["timestamp_experiment"] is True
    custom = next(control for control in controls if control.property("parameterPath") == "custom_flag")
    custom.findChild(ToggleSlider).setChecked(True)
    assert widget.raw_data["custom_flag"] is True
    assert not any(button.text() == "Random" for button in widget.findChildren(QToolButton))
    for field in widget.findChildren(QAbstractSpinBox):
        assert field.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons


@pytest.mark.parametrize("factory", [SpinBox, DoubleSpinBox])
def test_shared_numeric_inputs_hide_arrows_and_keep_numeric_editing(factory):
    field = factory()
    field.setRange(0, 100)
    assert field.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.NoButtons
    field.lineEdit().setText("42")
    field.interpretText()
    assert field.value() == 42


@pytest.mark.parametrize("rel", ["experiment/demo/test.yaml", "experiment/demo/_base.yaml"])
@pytest.mark.parametrize("width", [960, 1050, 1400])
def test_environment_and_site_children_do_not_overlap_labels(viewer, rel, width):
    widget = load(viewer, {"paradigm": "online_rl", "env": {
        "num_envs": 4, "options": {"depth": 2}}, "resources": {
        "cores": 8, "memory": "16G", "time": "01:00:00"}, "methods": {}}, rel)
    widget.resize(width, 1000)
    app.processEvents()
    paths = set()
    for block in boxes(widget)[0].findChildren(IndentedHierarchyBlock):
        form = block.form_layout
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
                paths.add(field.property("parameterPath"))
                caption_rect = QRect(caption.mapTo(block, QPoint()), caption.size())
                field_rect = QRect(field.mapTo(block, QPoint()), field.size())
                assert not caption_rect.intersects(field_rect), field.property("parameterPath")
                assert field_rect.right() < block.width()
    assert {"env.num_envs", "env.options.depth", "resources.cores", "resources.memory", "resources.time"} <= paths
    assert not widget.is_dirty


@pytest.mark.parametrize("rel", ["experiment/demo/test.yaml", "experiment/demo/_base.yaml"])
@pytest.mark.parametrize("resources", [{}, {"cores": 8, "memory": "16G"}])
def test_explicit_site_stays_beside_its_label_with_both_actions(viewer, rel, resources):
    widget = load(viewer, {"paradigm": "online_rl", "env": {"num_envs": 1},
                           "defaults": [{"override /site": "ncshare/common"}],
                           "resources": resources, "methods": {}}, rel)
    widget.resize(960, 1000)
    app.processEvents()
    box = boxes(widget)[0]
    caption = next(item for item in box.findChildren(QLabel) if item.text() == "Site")
    field = widget.field_widgets["site"]
    caption_rect = QRect(caption.mapTo(box, QPoint()), caption.size())
    field_rect = QRect(field.mapTo(box, QPoint()), field.size())
    assert field_rect.left() > caption_rect.right()
    assert abs(field_rect.center().y() - caption_rect.center().y()) <= 4
    assert field_rect.x() == widget.combo_paradigm.mapTo(box, QPoint()).x()
    assert field.currentText() == "ncshare/common"
    assert any(button.toolTip() == "Use inherited site" for button in box.findChildren(QToolButton))
    assert not widget.is_dirty


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
    widget = load(viewer, {"methods": {}, "tuning": {"component": "optuna", "direction": "maximize"},
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
    widget = load(viewer, {"methods": {}, "tuning": {"component": "optuna", "direction": "maximize"},
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


def test_external_dataset_browse_and_save_use_project_relative_path(viewer, tmp_path, monkeypatch):
    from pathlib import Path
    from frontend.config_viewer import QFileDialog
    widget = load(viewer, {'paradigm': 'online_rl', 'seed': 1})
    monkeypatch.chdir(tmp_path)
    box = ConfigBox('Dataset')
    widget._add_dataset_source_fields(box, {}, {}, required=True, paradigm='apprenticeship')
    source = tmp_path.parent.parent / f'external-{tmp_path.name}.csv'
    source.write_text('artificial fixture')
    monkeypatch.setenv('THETA_TEST_DATASET', str(source))
    widget.combo_dataset_path.setEditText('${oc.env:THETA_TEST_DATASET}')
    assert any(label.text().startswith('✓ Exists') for label in box.findChildren(QLabel))
    assert widget.raw_data['dataset_path'] == '${oc.env:THETA_TEST_DATASET}'
    with patch.object(QFileDialog, 'getOpenFileName', return_value=(str(source), 'CSV')):
        widget._browse_dataset_path()
    import os
    from frontend.config_viewer import _dataset_workspace
    expected = Path(os.path.relpath(source, _dataset_workspace().root)).as_posix()
    assert expected.startswith('../')
    assert widget.combo_dataset_path.currentText() == expected
    assert widget.raw_data['dataset_path'] == expected
    assert any(label.text().startswith('✓ Exists') for label in box.findChildren(QLabel))
    assert widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())['dataset_path'] == expected
    box.close()


def test_sweep_bounds_are_inline_with_type_below_caption(viewer):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn", "lr": 0.001,
        "tune": {"agent.lr": {"type": "float", "low": 0.0001, "high": 0.01}}}}})
    picker = widget.findChild(QComboBox, "sweepType_m_agent.lr")
    low = widget.findChild(QAbstractSpinBox, "sweep_low_m_agent.lr")
    high = widget.findChild(QAbstractSpinBox, "sweep_high_m_agent.lr")
    inline = widget.findChild(QWidget, "sweepInline_m_agent.lr")
    assert picker.isVisible() and low.isVisible() and high.isVisible()
    caption = picker.parentWidget().findChild(QLabel)
    assert caption is not None
    assert picker.mapTo(widget, QPoint()).y() > caption.mapTo(widget, QPoint()).y()
    assert low.parentWidget() is high.parentWidget()
    assert inline.isAncestorOf(low)
    assert [w.text() for w in low.parentWidget().findChildren(QLabel)] == ["(", ",", ")"]
    assert abs(low.mapTo(widget, QPoint()).y() - caption.mapTo(widget, QPoint()).y()) < 16
    low.setValue(0.0002)
    assert widget.raw_data["methods"]["m"]["tune"]["agent.lr"]["low"] == 0.0002
    assert widget.raw_data["methods"]["m"]["lr"] == 0.001
    widget.findChild(QToolButton, "sweepAddStep_m_agent.lr").click()
    widget.findChild(QLineEdit, "sweep_step_m_agent.lr").setText("0.0001")
    assert widget.raw_data["methods"]["m"]["tune"]["agent.lr"]["step"] == 0.0001
    remove = next(w for w in widget.findChildren(QToolButton) if w.toolTip() == "Remove step parameter")
    remove.click()
    assert "step" not in widget.raw_data["methods"]["m"]["tune"]["agent.lr"]
    widget.findChild(QToolButton, "sweepRemove_m_agent.lr").click()
    assert "tune" not in widget.raw_data["methods"]["m"]


def test_sweep_lists_resize_wrap_and_add_remove_values(viewer):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"model.hidden_sizes.0": {"type": "choice", "values": [32, 64, 128, 256]}}}}})
    target = "model.hidden_sizes.0"
    inline = widget.findChild(QWidget, f"sweepInline_m_{target}")
    edit = widget.findChild(QLineEdit, f"sweep_choice_m_{target}_0")
    width = edit.width()
    edit.setText("a longer choice")
    app.processEvents()
    assert edit.width() > width
    assert inline.layout().heightForWidth(150) > inline.layout().heightForWidth(800)
    assert [w.text() for w in inline.findChildren(QLabel)] == [",", ",", ","]
    add = widget.findChild(QToolButton, f"sweepAddChoice_m_{target}")
    assert inline.layout().itemAt(inline.layout().count() - 1).widget() is add
    add.click()
    values = widget.raw_data["methods"]["m"]["tune"][target]["values"]
    assert values == ["a longer choice", 64, 128, 256, ""]
    edit = widget.findChild(QLineEdit, f"sweep_choice_m_{target}_1")
    next(w for w in edit.parentWidget().findChildren(QToolButton) if w.toolTip() == "Remove value").click()
    assert values == ["a longer choice", 128, 256, ""]


def test_sweep_choices_reuse_parameter_dropdown_options(viewer):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn"}}})
    widget.raw_data["methods"]["m"]["tune"] = {"model.activation": {"type": "choice", "values": ["relu", "tanh"]}}
    row = QHBoxLayout()
    original = QComboBox()
    original.addItems(["relu", "tanh", "sigmoid"])
    row.addWidget(original)
    form = QFormLayout()
    form.addRow("Activation", row)
    widget.container.layout().addLayout(form)
    widget._add_sweep_toggle(row, "m", "model.activation", "relu")
    widget._place_sweep_subtitles()
    choice = widget.findChild(QComboBox, "sweep_choice_m_model.activation_0")
    assert [choice.itemText(i) for i in range(choice.count())] == ["relu", "tanh", "sigmoid"]
    choice.setCurrentText("sigmoid")
    assert widget.raw_data["methods"]["m"]["tune"]["model.activation"]["values"][0] == "sigmoid"


@pytest.mark.parametrize("kind,low,high", [
    ("int", 2**63, 2**63 + 100),
    ("float", -0.5, 0.5),
    ("log_float", 1e-20, 1e-5),
])
def test_inline_sweep_numbers_preserve_precision_and_validate_edits(viewer, kind, low, high):
    from PyQt6.QtTest import QTest
    from PyQt6.QtCore import Qt
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"agent.custom": {"type": kind, "low": low, "high": high}}}}})
    inline = widget.findChild(QWidget, "sweepInline_m_agent.custom")
    edit = next(w for w in inline.findChildren(QWidget) if w.objectName() == "sweep_low_m_agent.custom")
    assert edit.value() == low
    assert widget.raw_data["methods"]["m"]["tune"]["agent.custom"]["low"] == low
    value = 2**63 + 1 if kind == "int" else 2e-20
    edit.setFocus()
    if isinstance(edit, QAbstractSpinBox):
        edit.lineEdit().selectAll()
    else:
        edit.selectAll()
    QTest.keyClicks(edit, str(value))
    QTest.keyClick(edit, Qt.Key.Key_Return)
    assert edit.value() == value
    dimension = widget.raw_data["methods"]["m"]["tune"]["agent.custom"]
    assert dimension["low"] == value
    widget._edit_sweep_dimension("m", "agent.custom", "low", "not a number")
    assert dimension["low"] == value
    widget._edit_sweep_dimension("m", "agent.custom", "low", "nan")
    assert dimension["low"] == value


def test_sweep_list_removal_collapses_until_hover(viewer):
    from PyQt6.QtCore import QEvent
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"agent.custom": {"type": "choice", "values": ["one", "two"]}}}}})
    edit = widget.findChild(QLineEdit, "sweep_choice_m_agent.custom_0")
    entry = edit.parentWidget()
    remove = next(w for w in widget.findChildren(QToolButton) if w.toolTip() == "Remove value"
                  and w.findChild(_RowHoverReveal)._parent() is entry)
    hover = remove.findChild(_RowHoverReveal)
    assert not remove.sizePolicy().retainSizeWhenHidden()
    with patch("frontend.config_viewer.QCursor.pos", return_value=entry.mapToGlobal(QPoint(-100, -100))):
        hover._update_visibility()
    app.processEvents()
    assert remove.isHidden()
    collapsed_width = entry.sizeHint().width()
    with patch("frontend.config_viewer.QCursor.pos", return_value=edit.mapToGlobal(edit.rect().center())):
        hover.eventFilter(edit, QEvent(QEvent.Type.Enter))
        app.processEvents()
    assert remove.isVisible()
    assert entry.sizeHint().width() == collapsed_width
    assert hover._row().indexOf(remove) == -1
    assert remove.mapToGlobal(QPoint()).x() >= entry.mapToGlobal(QPoint(entry.width(), 0)).x()
    with patch("frontend.config_viewer.QCursor.pos", return_value=remove.mapToGlobal(remove.rect().center())):
        hover._update_visibility()
    assert remove.isVisible()  # The overlay remains clickable outside the entry.
    with patch("frontend.config_viewer.QCursor.pos", return_value=entry.mapToGlobal(QPoint(-100, -100))):
        hover.eventFilter(edit, QEvent(QEvent.Type.Leave))
        app.processEvents()
    assert remove.isHidden()
    assert entry.sizeHint().width() == collapsed_width


@pytest.mark.parametrize("width", [850, 1050])
def test_sweep_additions_stay_in_their_wrapping_row(viewer, width):
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn", "lr": 0.001, "gamma": 0.99,
        "tune": {"agent.lr": {"type": "float", "low": 0.0000001, "high": 0.1234567},
                 "agent.gamma": {"type": "choice", "values": ["long choice", "another choice", "third choice"]}}},
        "other": {"agent": "ppo", "model": "dnn"}}})
    widget.resize(width, 1000)
    app.processEvents()
    add_step = widget.findChild(QToolButton, "sweepAddStep_m_agent.lr")
    low = widget.findChild(QAbstractSpinBox, "sweep_low_m_agent.lr")
    assert abs(add_step.mapTo(widget, add_step.rect().center()).y() -
               low.mapTo(widget, low.rect().center()).y()) <= 1
    add_step.click()
    app.processEvents()
    for _ in range(3):
        widget.findChild(QToolButton, "sweepAddChoice_m_agent.gamma").click()
        app.processEvents()
    step = widget.findChild(QLineEdit, "sweep_step_m_agent.lr")
    bounds = widget.findChild(QWidget, "sweepInline_m_agent.lr")
    choices = widget.findChild(QWidget, "sweepInline_m_agent.gamma")
    assert bounds.isAncestorOf(step)
    assert bounds.rect().contains(QRect(step.mapTo(bounds, QPoint()), step.size()))
    assert bounds.mapTo(widget, bounds.rect().bottomLeft()).y() < choices.mapTo(widget, QPoint()).y()
    assert choices.height() >= choices.heightForWidth(choices.width())
    for index in range(6):
        edit = widget.findChild(QLineEdit, f"sweep_choice_m_agent.gamma_{index}")
        assert choices.isAncestorOf(edit)
        assert choices.rect().contains(QRect(edit.mapTo(choices, QPoint()), edit.size()))
    add = widget.findChild(QToolButton, "sweepAddChoice_m_agent.gamma")
    last = widget.findChild(QLineEdit, "sweep_choice_m_agent.gamma_5")
    assert abs(add.mapTo(widget, add.rect().center()).y() -
               last.mapTo(widget, last.rect().center()).y()) <= 1
    assert widget.raw_data["methods"]["m"]["tune"]["agent.lr"]["step"] is None
    assert "tune" not in widget.raw_data["methods"]["other"]
    subtitle = widget.findChild(QComboBox, "sweepType_m_agent.lr")
    assert "background: transparent" in subtitle.styleSheet()
    assert "border: none" in subtitle.styleSheet()


@pytest.mark.parametrize("kind,text", [("int", "10000"), ("float", ".00001"),
                                      ("float", ".00000000000000000000000000000000000000000000000001")])
@pytest.mark.parametrize("key", ["low", "high", "step"])
def test_sweep_numeric_fields_fit_the_text_while_typing(viewer, kind, text, key):
    from PyQt6.QtTest import QTest
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"agent.custom": {"type": kind, "low": 1, "high": 2, "step": 1}}}}})
    inline = widget.findChild(QWidget, "sweepInline_m_agent.custom")
    control = next(w for w in inline.findChildren(QWidget) if w.objectName() == f"sweep_{key}_m_agent.custom")
    editor = control.lineEdit() if isinstance(control, QAbstractSpinBox) else control
    initial_width = control.width()
    control.setFocus()
    app.processEvents()
    editor.selectAll()
    QTest.keyClicks(control, text)
    app.processEvents()
    app.processEvents()
    assert control.value() == (int(text) if kind == "int" else float(text))
    displayed = editor.displayText()
    assert control.width() > initial_width
    assert editor.fontMetrics().horizontalAdvance(displayed) + 4 <= editor.width()
    assert editor.cursorPositionAt(QPoint(0, editor.height() // 2)) == 0


def test_sweep_list_numbers_fit_as_they_grow(viewer):
    from PyQt6.QtTest import QTest
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn",
        "tune": {"agent.custom": {"type": "choice", "values": [1, 2]}}}}})
    edit = widget.findChild(QLineEdit, "sweep_choice_m_agent.custom_0")
    width = edit.width()
    edit.selectAll()
    QTest.keyClicks(edit, "10000000000000000000000000000000000000000")
    app.processEvents()
    assert edit.width() > max(width, 220)
    assert edit.fontMetrics().horizontalAdvance(edit.text()) + 14 <= edit.width()


def test_sweep_hover_only_reveals_actions_for_the_live_row(viewer):
    from PyQt6.QtCore import QEvent
    widget = load(viewer, {"tuning": {"component": "optuna"}, "methods": {"m": {"agent": "ppo", "model": "dnn", "lr": 0.001, "gamma": 0.99,
        "tune": {"agent.lr": {"type": "float", "low": 0.0001, "high": 0.01},
                 "agent.gamma": {"type": "choice", "values": [0.9, 0.99]}}}}})
    low = widget.findChild(QAbstractSpinBox, "sweep_low_m_agent.lr")
    choice = widget.findChild(QLineEdit, "sweep_choice_m_agent.gamma_0")
    remove = widget.findChild(QToolButton, "sweepRemove_m_agent.lr")
    hover = remove.findChild(_RowHoverReveal)
    with patch("frontend.config_viewer.QCursor.pos", return_value=low.mapToGlobal(low.rect().center())):
        hover._update_visibility()
    assert remove.isVisible()
    with patch("frontend.config_viewer.QCursor.pos", return_value=choice.mapToGlobal(choice.rect().center())):
        hover._update_visibility()
    assert remove.isHidden()
    hover.eventFilter(choice, QEvent(QEvent.Type.FocusIn))
    assert remove.isHidden()
    entry_remove = next(w for w in widget.findChildren(QToolButton) if w.toolTip() == "Remove value"
                        and w.findChild(_RowHoverReveal)._parent() is choice.parentWidget())
    entry_hover = entry_remove.findChild(_RowHoverReveal)
    with patch("frontend.config_viewer.QCursor.pos", return_value=choice.mapToGlobal(choice.rect().center())):
        entry_hover._update_visibility()
    assert entry_remove.isVisible()
    # Reveal must lay the collapsing button out immediately, before painting.
    assert entry_remove.mapToGlobal(QPoint()).x() >= choice.mapToGlobal(QPoint(choice.width(), 0)).x()


def test_absent_tuner_hides_new_controls_and_preserves_saved_recipe(viewer):
    import copy
    from src.app.workspace import WorkspacePaths
    widget, _, root = viewer
    widget._loader_paths = lambda: WorkspacePaths(root / 'absent-workspace', application=root / 'empty-application')
    widget = load(viewer, {'methods': {'params': {'batch_size': 32}}})
    assert not any(a.text() == 'Hyperparameter Sweep' for a in widget.btn_add_block.menu().actions())
    widget.add_block('tuning')
    assert 'tuning' not in widget.raw_data
    data = {'tuning': {'component': 'missing-tuner'}, 'methods': {'params': {
        'batch_size': 32, 'tune': {'agent.batch_size': {'type': 'int', 'low': 1, 'high': 4}}}}}
    widget = load(viewer, copy.deepcopy(data))
    assert widget.raw_data == data
    notice = widget.findChild(QLabel, 'tunerReadinessNotice')
    assert notice is not None and 'install a compatible provider' in notice.text()
    assert not any(b.objectName().startswith('sweepToggle_') for b in widget.findChildren(QToolButton))


@pytest.mark.parametrize('selector', [None, '', 3])
def test_unselected_saved_tuner_requires_explicit_choice_and_preserves_ranges(viewer, selector):
    import copy
    import shutil
    from pathlib import Path
    from src.app.workspace import WorkspacePaths
    widget, _, root = viewer
    paths = WorkspacePaths(root, application=root / 'empty-application')
    shutil.copytree(Path(__file__).parent / 'fixtures/contribution_tuner', root / 'src/usr/tuners/lattice')
    widget._loader_paths = lambda: paths
    tuning = {'direction': 'maximize', 'options': {'lattice': {'token': 'configured'}}}
    if selector is not None:
        tuning['component'] = selector
    data = {'tuning': tuning, 'methods': {'params': {'batch_size': 32,
            'tune': {'agent.batch_size': {'type': 'int', 'low': 16, 'high': 64}}}}}
    before = copy.deepcopy(data)
    widget = load(viewer, copy.deepcopy(data))
    assert widget.raw_data == before and not widget.is_dirty
    assert not widget._tuner_ready()
    assert not any(b.objectName().startswith('sweepToggle_') for b in widget.findChildren(QToolButton))
    notice = widget.findChild(QLabel, 'tunerReadinessNotice')
    assert 'tuning.component' in notice.text()
    combo = widget.findChild(QComboBox, 'sweepComponent')
    assert combo.currentText() == 'Select a tuner' and combo.currentData() is None
    combo.setCurrentIndex(combo.findData('lattice'))
    app.processEvents()
    assert widget.raw_data['tuning']['component'] == 'lattice' and widget.is_dirty
    assert widget.raw_data['tuning']['options'] == before['tuning']['options']
    assert widget.raw_data['methods'] == before['methods'] and widget._tuner_ready()
    assert widget.findChild(QComboBox, 'sweepType_params_agent.batch_size') is not None
    assert not any(w.objectName() == 'tunerReadinessNotice' and w.isVisible() for w in widget.findChildren(QLabel))
    widget.save_to_disk()
    assert yaml.safe_load(widget.current_path.read_text())['tuning']['component'] == 'lattice'


def test_independent_tuner_setup_readiness_defaults_and_required_children(viewer):
    import shutil
    from pathlib import Path
    from src.app.workspace import WorkspacePaths
    from src.app.core.contributions import ContributionRegistry
    widget, _, root = viewer
    paths = WorkspacePaths(root, application=root / 'empty-application')
    # The independent provider must be the only tuner: remove the fixture's optional Optuna component.
    from src.app.components.installer import HubInstaller
    from src.app.components.models import HubComponent
    from src.app.core.provider_catalog import get_catalog
    installed = get_catalog(paths).package('optuna')
    assert HubInstaller(paths.root, paths.data_dir).uninstall(HubComponent.from_dict(installed.manifest))
    shutil.copytree(Path(__file__).parent / 'fixtures/contribution_tuner', root / 'src/usr/tuners/lattice')
    widget._loader_paths = lambda: paths
    widget = load(viewer, {'methods': {'params': {'batch_size': 32}}})
    action = next(a for a in widget.btn_add_block.menu().actions() if a.text() == 'Hyperparameter Sweep')
    action.trigger()
    assert widget.raw_data['tuning']['component'] == 'lattice'
    assert widget.raw_data['tuning']['options']['lattice']['attempts'] == 3
    assert 'n_trials' not in widget.raw_data['tuning']
    assert widget.findChild(QLabel, 'tunerReadinessNotice') is not None
    assert not widget._tuner_ready()
    widget.findChild(QLineEdit, 'sweep_token').setText('configured')
    app.processEvents()
    assert widget._tuner_ready()
    assert widget.findChild(QLabel, 'tunerReadinessNotice') is None
    assert any(b.objectName().startswith('sweepToggle_') for b in widget.findChildren(QToolButton))
    widget.findChild(QComboBox, 'sweep_mode').setCurrentText('resume')
    app.processEvents()
    assert widget.findChild(QLineEdit, 'sweep_reference') is not None
    assert widget.findChild(QToolButton, 'sweepRemoveOption_reference') is None
    before = dict(widget.raw_data['tuning'])
    ContributionRegistry(paths).set_enabled('tuner', 'lattice', False)
    widget._render_boxes()
    app.processEvents()
    from PyQt6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert widget.raw_data['tuning'] == before
    assert 'disabled' in widget.findChild(QLabel, 'tunerReadinessNotice').text()
    assert not any(b.objectName().startswith('sweepToggle_') for b in widget.findChildren(QToolButton))
