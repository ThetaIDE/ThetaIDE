"""ConfigViewer editing of composite models, method cards and group defaults.

Runs against the fixture workspace: ``blendrl-dev`` is the composite model (its
``neural`` and ``symbolic`` parameters select models), ``sb3-dev``/``d3rlpy-dev``
supply the algorithms and ``test-models`` supplies ``basics/dnn``.
"""
from pathlib import Path

import pytest
import yaml

from PyQt6.QtWidgets import QApplication
from frontend.config_viewer import ConfigViewer, config_tree, invalidate_config_tree
from frontend.config_panel.controls import ConfigBox

COMPOSITE = "blendrl-dev"
OFFLINE = """# @package _global_
defaults:
  - /paradigms: offline_rl
  - /env: mimic
  - _self_
paradigm: offline_rl
eval_episodes: 0
methods:
  cql_dnn:
    agent: d3rlpy-dev/cql
    model: basics/dnn
  cql_blend_symbolic:
    agent: d3rlpy-dev/cql
    model: blendrl-dev
    symbolic:
      type: basics/dnn
  cql_blend_neural:
    agent: d3rlpy-dev/cql
    model: blendrl-dev
    neural:
      name: basics/dnn
"""
ONLINE = """# @package _global_
defaults:
  - /paradigms: online_rl
  - /env: cartpole
  - _self_
paradigm: online_rl
eval_episodes: 10
methods:
  ppo_dnn:
    agent: sb3-dev/ppo
    model: basics/dnn
  ppo_blend:
    agent: sb3-dev/ppo
    model: blendrl-dev
"""


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def workspace(fixture_workspace):
    invalidate_config_tree()
    yield fixture_workspace
    invalidate_config_tree()


def open_recipe(tmp_path, text, rel_path):
    path = tmp_path / Path(rel_path).name
    path.write_text(text, encoding="utf-8")
    viewer = ConfigViewer()
    viewer.load_file(path, rel_path)
    return viewer, path


def offline(tmp_path):
    return open_recipe(tmp_path, OFFLINE, "experiment/composite/offline.yaml")


def online(tmp_path):
    return open_recipe(tmp_path, ONLINE, "experiment/composite/online.yaml")


def model_domains(model):
    properties = config_tree().models[model].parameter_schema.get("properties", {})
    return {key for key, schema in properties.items() if isinstance(schema, dict) and schema.get("x-theta-domain")}


def labels(viewer):
    from PyQt6.QtWidgets import QLabel
    return [label.text() for label in viewer.findChildren(QLabel)]


def test_offline_recipe_edits_composite_subcomponents(qapp, tmp_path):
    viewer, path = offline(tmp_path)
    assert {"cql_dnn", "cql_blend_symbolic"} <= set(viewer.raw_data["methods"])

    from frontend.config_viewer import get_method_candidate_hyperparameters
    tree = config_tree()
    assert "blend_ent_coef" not in get_method_candidate_hyperparameters("d3rlpy-dev/cql", "basics/dnn", tree)
    assert "hidden_sizes" in get_method_candidate_hyperparameters("d3rlpy-dev/cql", "basics/dnn", tree)
    assert "blend_ent_coef" in get_method_candidate_hyperparameters("d3rlpy-dev/cql", COMPOSITE, tree)

    method = "cql_blend_symbolic"
    viewer._add_method_model_parameter(method, COMPOSITE, ("neural",))
    assert "neural" in viewer.raw_data["methods"][method]
    viewer._on_method_model_parameter_edited(method, COMPOSITE, ("symbolic", "rules"), "custom_rules")
    assert viewer.raw_data["methods"][method]["symbolic"]["rules"] == "custom_rules"
    viewer._remove_method_model_parameter(method, COMPOSITE, ("symbolic", "rules"))
    assert "rules" not in viewer.raw_data["methods"][method]["symbolic"]

    viewer.save_to_disk()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"cql_dnn", "cql_blend_symbolic", "cql_blend_neural"} <= set(saved["methods"])
    assert "neural" in saved["methods"][method]


def test_online_recipe_candidates_follow_agent_and_model(qapp, tmp_path):
    online(tmp_path)
    from frontend.config_viewer import get_method_candidate_hyperparameters
    tree = config_tree()
    dnn = get_method_candidate_hyperparameters("sb3-dev/ppo", "basics/dnn", tree)
    assert "clip_range" in dnn and "blend_ent_coef" not in dnn
    blend = get_method_candidate_hyperparameters("sb3-dev/ppo", COMPOSITE, tree)
    assert {"blend_ent_coef", "logic_lr"} <= set(blend)


def test_scoped_candidate_hyperparameters(qapp):
    from frontend.config_viewer import (
        get_agent_candidate_hyperparameters,
        get_model_candidate_hyperparameters,
        get_training_candidate_hyperparameters,
    )
    tree = config_tree()
    cql = get_agent_candidate_hyperparameters("d3rlpy-dev/cql", tree)
    assert "n_steps" in cql
    assert "hidden_sizes" not in cql and "blend_ent_coef" not in cql
    ppo = get_agent_candidate_hyperparameters("sb3-dev/ppo", tree)
    assert "clip_range" in ppo and "hidden_sizes" not in ppo

    dnn = get_model_candidate_hyperparameters("basics/dnn", tree)
    assert "hidden_sizes" in dnn and "clip_range" not in dnn and "blend_ent_coef" not in dnn
    blend = get_model_candidate_hyperparameters(COMPOSITE, tree)
    assert {"blend_ent_coef", "blend_q_values", "logic_lr"} <= set(blend)
    assert "clip_range" not in blend

    assert {"lr", "batch_size", "gamma", "epochs"} <= set(get_training_candidate_hyperparameters())


def test_indented_hierarchy_blocks_in_gui(qapp, tmp_path):
    from frontend.config_viewer import IndentedHierarchyBlock
    viewer, _ = offline(tmp_path)
    blocks = viewer.findChildren(IndentedHierarchyBlock)
    assert blocks, "Expected IndentedHierarchyBlock instances in rendered GUI"
    assert "border-left: 2px solid #504945" in blocks[0].styleSheet()


def test_add_and_remove_parameters_in_hierarchy(qapp, tmp_path):
    viewer, _ = offline(tmp_path)
    method = viewer.raw_data["methods"]["cql_dnn"]
    viewer._add_method_param("cql_dnn", "n_steps", 1000)
    assert method["n_steps"] == 1000
    # A model parameter starts from the model's declared default.
    viewer._add_method_param("cql_dnn", "hidden_sizes")
    assert method["hidden_sizes"] == config_tree().model_parameters("basics/dnn")["hidden_sizes"]
    viewer._remove_method_param("cql_dnn", "n_steps")
    viewer._remove_method_param("cql_dnn", "hidden_sizes")
    assert "n_steps" not in method and "hidden_sizes" not in method


def section_titles(viewer):
    return {box.title_label.text() for box in viewer.findChildren(ConfigBox) if box.title_label is not None}


def test_top_box_defaults_hidden_until_added(qapp, tmp_path):
    viewer, _ = online(tmp_path)
    assert viewer.run_logger_toggles == {}
    assert (viewer.chk_save_ds, viewer.chk_recover, viewer.chk_no_plot) == (None,) * 3

    viewer.raw_data["logging"] = {"backends": {"tensorboard": {"enabled": True}}}
    viewer._render_boxes()
    assert viewer.run_logger_toggles["tensorboard"].isChecked()
    assert (viewer.chk_save_ds, viewer.chk_recover, viewer.chk_no_plot) == (None,) * 3
    assert "++logging.backends.tensorboard.enabled=true" in viewer.get_overrides()

    viewer.raw_data.pop("logging", None)
    viewer._render_boxes()
    assert viewer.run_logger_toggles == {}


def test_run_logger_toggle_edits_its_backend_switch(qapp, tmp_path):
    """Each run-logger toggle owns logging.backends.<id>.enabled; other logging settings keep their section."""
    viewer, _ = online(tmp_path)
    viewer.raw_data["logging"] = {"backends": {"tensorboard": {"enabled": True}, "wandb": {"enabled": True}}}
    viewer._render_boxes()
    assert list(viewer.run_logger_toggles) == ["tensorboard"]  # wandb has no provider in this workspace
    assert "logging" in section_titles(viewer)  # so it is still rendered generically

    viewer.run_logger_toggles["tensorboard"].setChecked(False)
    assert viewer.raw_data["logging"] == {"backends": {"wandb": {"enabled": True}}}
    assert "tensorboard" not in viewer.raw_data

    viewer.raw_data["logging"] = {"backends": {"tensorboard": {"enabled": True}}}
    viewer._render_boxes()
    assert "logging" not in section_titles(viewer)
    viewer.run_logger_toggles["tensorboard"].setChecked(False)
    assert "logging" not in viewer.raw_data


def test_method_card_duplicate_and_header_plus_icons(qapp, tmp_path):
    from PyQt6.QtWidgets import QToolButton
    viewer, _ = online(tmp_path)

    buttons = viewer.findChildren(QToolButton)
    assert len([b for b in buttons if "Duplicate method" in b.toolTip()]) >= 2
    assert len([b for b in buttons if b.toolTip() == "Add new empty method"]) == 1

    copy = viewer.duplicate_method("ppo_dnn")
    assert copy == "ppo_dnn_copy"
    assert viewer.raw_data["methods"][copy]["agent"] == viewer.raw_data["methods"]["ppo_dnn"]["agent"]
    assert viewer.duplicate_method("ppo_dnn") == "ppo_dnn_copy_2"


def test_add_new_method_flow_empty_and_invalid(qapp, tmp_path):
    from frontend.config_viewer import InlineTitleEditor
    viewer, _ = online(tmp_path)

    viewer._start_add_new_method()
    assert viewer._pending_new_method is True
    editors = [e for e in viewer.findChildren(InlineTitleEditor) if e.objectName() == "newMethodEditor"]
    assert len(editors) == 1
    assert editors[0].text() == ""
    # The placeholder stays generic when the group has no default method.
    assert editors[0].placeholderText() == "Method name..."

    existing = set(viewer.raw_data["methods"])
    assert viewer._validate_method_name("", existing) == "Method name cannot be empty."
    assert viewer._validate_method_name("   ", existing) == "Method name cannot be empty."
    assert "params" in viewer._validate_method_name("params", existing)
    assert "already exists" in viewer._validate_method_name("ppo_dnn", existing)
    for name in ("invalid name with spaces", "bad/name", "bad:name"):
        assert "invalid characters" in viewer._validate_method_name(name, existing)

    viewer._cancel_new_method()
    qapp.processEvents()
    assert viewer._pending_new_method is False
    assert not viewer.findChildren(InlineTitleEditor, "newMethodEditor")


def test_add_new_method_valid_empty_method(qapp, tmp_path):
    from frontend.widgets import ComboBox
    viewer, _ = online(tmp_path)

    viewer._start_add_new_method()
    viewer.raw_data.setdefault("methods", {})["custom_empty_method"] = {}
    viewer._pending_new_method = False
    viewer._render_boxes()
    assert viewer.raw_data["methods"]["custom_empty_method"] == {}

    combos = viewer.findChildren(ComboBox)
    agents = [c for c in combos if c.placeholderText() in ("Select algorithm...", "Select agent...")]
    models = [c for c in combos if c.placeholderText() == "Select model..."]
    assert any(c.currentIndex() == -1 for c in agents)
    assert any(c.currentIndex() == -1 for c in models)


def test_installed_algorithms_are_selectable(qapp, tmp_path):
    from frontend.widgets import ComboBox
    viewer, _ = online(tmp_path)

    texts = labels(viewer)
    assert "Algorithm" in texts and "Model" in texts

    combos = [c for c in viewer.findChildren(ComboBox) if c.placeholderText() == "Select algorithm..."]
    assert combos
    items = [combos[0].itemText(i) for i in range(combos[0].count())]
    assert {"sb3-dev/a2c", "sb3-dev/dqn", "sb3-dev/ppo", "sb3-dev/sac"} <= set(items)

    combos[0].setCurrentText("sb3-dev/dqn")
    qapp.processEvents()
    assert viewer.raw_data["methods"]["ppo_dnn"]["agent"] == "sb3-dev/dqn"


def test_composite_model_parameters_render_only_when_added(qapp, tmp_path):
    from frontend.widgets import ComboBox
    viewer, _ = online(tmp_path)
    method = "ppo_blend"
    assert viewer.raw_data["methods"][method]["model"] == COMPOSITE

    # Parameters with declared defaults are not inserted until the user adds them.
    assert "symbolic" not in labels(viewer) and "neural" not in labels(viewer)
    assert {"symbolic", "neural"} <= model_domains(COMPOSITE)
    assert "blender" not in model_domains(COMPOSITE)

    viewer._add_method_model_parameter(method, COMPOSITE, ("symbolic",))
    qapp.processEvents()
    assert "symbolic" in labels(viewer)
    assert "neural" not in labels(viewer) and "blender" not in labels(viewer)

    # The declared model domain is edited with the generic model selector, which keeps
    # the profile's default (a model this workspace does not provide) visible.
    selected = viewer.raw_data["methods"][method]["symbolic"]["type"]
    selectors = [c for c in viewer.findChildren(ComboBox) if c.currentText() == selected
                 and "basics/dnn" in {c.itemText(i) for i in range(c.count())}]
    assert len(selectors) == 1, "Expected a model selector for the declared model parameter"
    selectors[0].setCurrentText("basics/dnn")
    qapp.processEvents()
    symbolic = viewer.raw_data["methods"][method]["symbolic"]
    assert symbolic == "basics/dnn" or symbolic.get("type") == "basics/dnn" or symbolic.get("name") == "basics/dnn"

    viewer._on_method_model_parameter_edited(method, COMPOSITE, ("symbolic", "rules"), "custom_rules")
    assert viewer.raw_data["methods"][method]["symbolic"]["rules"] == "custom_rules"
    viewer._remove_method_model_parameter(method, COMPOSITE, ("symbolic", "rules"))
    assert "rules" not in viewer.raw_data["methods"][method]["symbolic"]

    viewer._remove_method_model_parameter(method, COMPOSITE, ("symbolic",))
    qapp.processEvents()
    assert "symbolic" not in labels(viewer)
    assert "symbolic" in set(config_tree().model_parameter_keys(COMPOSITE)) - set(viewer.raw_data["methods"][method])

    viewer._add_method_model_parameter(method, COMPOSITE, ("neural",))
    qapp.processEvents()
    assert "neural" in labels(viewer)
    viewer._on_method_model_parameter_edited(method, COMPOSITE, ("neural", "hidden_sizes"), [128, 128])
    assert viewer.raw_data["methods"][method]["neural"]["hidden_sizes"] == [128, 128]
    viewer._remove_method_model_parameter(method, COMPOSITE, ("neural", "hidden_sizes"))
    assert "hidden_sizes" not in viewer.raw_data["methods"][method]["neural"]
    viewer._remove_method_model_parameter(method, COMPOSITE, ("neural",))
    qapp.processEvents()
    assert "neural" not in labels(viewer)

    # A nested configuration group is added and removed the same way.
    viewer._add_method_param(method, "blender")
    qapp.processEvents()
    assert "blender" in labels(viewer)
    viewer._remove_method_model_parameter(method, COMPOSITE, ("blender",))
    qapp.processEvents()
    assert "blender" not in labels(viewer)


def test_composite_params_and_rules_round_trip(qapp, tmp_path):
    viewer, path = online(tmp_path)
    method = "ppo_blend"
    spec = viewer.raw_data["methods"][method]

    # Added model parameters start from the model's defaults and are then edited.
    viewer._add_method_param(method, "blend_q_values")
    viewer._add_method_param(method, "logic_lr")
    assert spec["blend_q_values"] is True and spec["logic_lr"] is None
    viewer._on_method_model_parameter_edited(method, COMPOSITE, ("logic_lr",), 0.001)
    assert spec["logic_lr"] == 0.001

    viewer._on_method_model_parameter_edited(method, COMPOSITE, ("symbolic", "rules"), "variant1")
    assert spec["symbolic"]["rules"] == "variant1"
    viewer._on_method_model_parameter_edited(method, COMPOSITE, ("symbolic", "rules"), "custom_rules")
    assert spec["symbolic"]["rules"] == "custom_rules"
    viewer._remove_method_model_parameter(method, COMPOSITE, ("symbolic", "rules"))
    assert "rules" not in spec["symbolic"]

    viewer._remove_method_param(method, "blend_q_values")
    assert "blend_q_values" not in spec

    viewer.save_to_disk()
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert saved["methods"][method]["logic_lr"] == 0.001


_GROUP_BASE_HEAD = (
    "# @package _global_\n"
    "defaults:\n"
    "  - override /env: cartpole\n"
    "paradigm: online_rl\n"
    "seed: 42\n"
    "methods:\n"
    "  params:\n"
    "    lr: 0.0003\n"
)


def test_group_defaults_keep_their_one_default_method(qapp, tmp_path):
    """A group's _base.yaml may declare one default method, which new experiments inherit.

    Opening the base used to strip it in memory and saving then deleted it from disk."""
    from PyQt6.QtWidgets import QLabel, QToolButton

    base_file = tmp_path / "_base.yaml"
    base_file.write_text(_GROUP_BASE_HEAD + "  default_ppo:\n    agent: ppo\n    model: dnn\n", encoding="utf-8")

    viewer = ConfigViewer()
    viewer.load_file(base_file, "experiment/test_group/_base.yaml")

    # 1. Shown as the group's default method, not as "none configured"
    assert "default_ppo" in viewer.raw_data["methods"]
    labels = [lbl.text() for lbl in viewer.findChildren(QLabel)]
    assert not any("No default method configured" in text for text in labels)

    # 2. At most one: a second cannot be added, nor this one duplicated
    assert viewer.add_method(agent="ppo", model="dnn") is None
    assert viewer.duplicate_method("default_ppo") is None
    assert not [b for b in viewer.findChildren(QToolButton) if "duplicate" in b.toolTip().lower()]

    # 3. Saving keeps it, and the universal params beside it
    viewer.save_to_disk()
    saved = yaml.safe_load(base_file.read_text(encoding="utf-8"))
    assert saved["methods"]["default_ppo"] == {"agent": "ppo", "model": "dnn"}
    assert saved["methods"]["params"]["lr"] == 0.0003


def test_a_default_method_added_to_a_group_base_survives_saving(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel

    base_file = tmp_path / "_base.yaml"
    base_file.write_text(_GROUP_BASE_HEAD, encoding="utf-8")

    viewer = ConfigViewer()
    viewer.load_file(base_file, "experiment/test_group/_base.yaml")
    labels = [lbl.text() for lbl in viewer.findChildren(QLabel)]
    assert any("No default method configured" in text for text in labels)

    name = viewer.add_method(agent="ppo", model="dnn")
    assert name is not None
    viewer.save_to_disk()

    reopened = ConfigViewer()
    reopened.load_file(base_file, "experiment/test_group/_base.yaml")
    assert reopened.raw_data["methods"][name]["agent"] == "ppo"
    assert reopened.raw_data["methods"]["params"]["lr"] == 0.0003


def test_experiment_environment_saved_to_defaults_override(qapp, tmp_path):
    exp_file = tmp_path / "exp.yaml"
    exp_file.write_text(
        "# @package _global_\n"
        "defaults:\n"
        "  - test_group/_base\n"
        "methods:\n"
        "  PPO:\n"
        "    agent: ppo\n"
        "    model: dnn\n",
        encoding="utf-8"
    )

    viewer = ConfigViewer()
    viewer.load_file(exp_file, "experiment/test_group/exp.yaml")

    # Set environment override
    viewer._set_experiment_env("atari/pong")
    assert viewer.raw_data.get("env") is None
    defaults = viewer.raw_data.get("defaults", [])
    assert any(isinstance(d, dict) and d.get("override /env") == "atari/pong" for d in defaults)

    # Save to disk
    viewer.save_to_disk()
    saved_text = exp_file.read_text(encoding="utf-8")
    assert "- override /env: atari/pong" in saved_text
    assert "\nenv: " not in saved_text

    # Re-read YAML
    saved = yaml.safe_load(saved_text)
    assert any(isinstance(d, dict) and d.get("override /env") == "atari/pong" for d in saved["defaults"])
    assert "env" not in saved

    # Clear override
    viewer._set_experiment_env(None)
    viewer.save_to_disk()
    saved_text_cleared = exp_file.read_text(encoding="utf-8")
    assert "override /env" not in saved_text_cleared


def test_experiment_methods_start_from_the_group_default_method(qapp, tmp_path):
    """A group base may name one default method; a method made in an experiment of that
    group, by add_method or through the + button's name box, starts with its parameters
    and keeps them on save. Without a default, a new method starts empty."""
    from frontend.config_viewer import InlineTitleEditor

    group = config_tree().config_root / "experiment" / "probe_group"
    group.mkdir(parents=True)
    base = group / "_base.yaml"
    default = {"agent": "ppo", "model": "dnn", "lr": 0.0003, "clip_coef": 0.1}
    base.write_text(
        "# @package _global_\nparadigm: online_rl\nseed: 1\nmethods:\n  ppo_default:\n"
        "    agent: ppo\n    model: dnn\n    lr: 0.0003\n    clip_coef: 0.1\n",
        encoding="utf-8",
    )
    experiment = group / "child.yaml"
    experiment.write_text(
        "# @package _global_\ndefaults:\n  - probe_group/_base\nexperiment_id: child\n", encoding="utf-8"
    )

    viewer = ConfigViewer()
    viewer.load_file(experiment, "experiment/probe_group/child.yaml")
    added = viewer.add_method()
    assert viewer.raw_data["methods"][added] == default

    viewer._start_add_new_method()
    editor = next(e for e in viewer.findChildren(InlineTitleEditor) if e.objectName() == "newMethodEditor")
    assert editor.placeholderText() == "Method name (e.g. ppo_default)..."
    editor.setText("my_variant")
    editor.returnPressed.emit()
    assert viewer.raw_data["methods"]["my_variant"] == default

    viewer.save_to_disk()
    saved = yaml.safe_load(experiment.read_text(encoding="utf-8"))["methods"]
    assert saved["my_variant"] == default and saved["ppo_default"] == default
    assert yaml.safe_load(base.read_text(encoding="utf-8"))["methods"] == {"ppo_default": default}

    # A base without a default method: new methods start empty
    base.write_text("# @package _global_\nparadigm: online_rl\nseed: 1\n", encoding="utf-8")
    fresh = ConfigViewer()
    fresh.load_file(experiment, "experiment/probe_group/child.yaml")
    added = fresh.add_method()
    assert fresh.raw_data["methods"][added] == {}


def test_model_component_parameter_domains(tmp_path):
    from frontend.config_model import Model

    assert COMPOSITE in config_tree().models
    assert model_domains(COMPOSITE) == {"neural", "symbolic"}
    assert model_domains("basics/dnn") == set()

    # Child model choices are declared by the model component's parameter schema.
    custom_model_yaml = tmp_path / "custom_composite.yaml"
    custom_model_yaml.write_text(
        "architecture: custom_composite\n"
        "parameter_schema:\n"
        "  type: object\n"
        "  properties:\n"
        "    backbone:\n"
        "      type: string\n"
        "      x-theta-domain: model\n"
        "    classifier_head:\n"
        "      type: string\n"
        "      x-theta-domain: model\n",
        encoding="utf-8",
    )
    custom_model = Model.from_file(custom_model_yaml)
    assert set(custom_model.parameter_schema["properties"]) == {"backbone", "classifier_head"}
    assert custom_model.parameter_schema["properties"]["backbone"]["x-theta-domain"] == "model"
