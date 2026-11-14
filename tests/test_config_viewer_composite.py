"""Tests for ConfigViewer composite subcomponents and dynamic hierarchies."""
import shutil
from pathlib import Path
import pytest
import yaml

from PyQt6.QtWidgets import QApplication
from frontend.config_viewer import ConfigViewer, config_tree


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _require_plugins():
    tree = config_tree()
    if not Path("src/usr/config/experiment/cartpole/final_cartpole.yaml").is_file() or "blendrl" not in tree.models:
        pytest.skip("cartpole experiment or blendrl model plugin uninstalled; skipping composite viewer tests")


def test_config_viewer_loads_cql_benchmark(qapp, tmp_path):
    # Copy mimic cql_benchmark.yaml to tmp_path
    src_file = Path("src/usr/config/experiment/mimic/cql_benchmark.yaml")
    assert src_file.is_file()

    dest_file = tmp_path / "cql_benchmark.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/mimic/cql_benchmark.yaml")

    # Check raw_data loaded
    data = viewer.raw_data
    methods = data.get("methods", {})
    assert "cql_dnn" in methods
    assert "cql_blendrl_cew_dueling_resnet" in methods

    # Verify tensorboard is False or not injected as true
    assert data.get("tensorboard") is not True

    # Test candidate parameters for cql_dnn (pure DNN)
    tree = config_tree()
    from frontend.config_viewer import get_method_candidate_hyperparameters
    dnn_cand = get_method_candidate_hyperparameters("cql", "dnn", tree)
    assert "blend_ent_coef" not in dnn_cand
    assert "hidden_sizes" in dnn_cand

    # Test candidate parameters for blendrl
    blend_cand = get_method_candidate_hyperparameters("cql", "blendrl", tree)
    assert "blend_ent_coef" in blend_cand

    # Test subcomponent addition and editing
    method_name = "cql_blendrl_human_rigid"
    # This method has symbolic, but not neural yet
    viewer._add_method_model_parameter(method_name, "blendrl", ("neural",))
    rigid_spec = viewer.raw_data["methods"][method_name]
    if isinstance(rigid_spec.get("model"), dict):
        assert "neural" in rigid_spec["model"]["blendrl"]
    else:
        assert "neural" in rigid_spec

    # Edit parameter on symbolic subcomponent
    viewer._on_method_model_parameter_edited(method_name, "blendrl", ("symbolic", "rules"), "custom_rules")
    if isinstance(rigid_spec.get("model"), dict):
        sym_dict = rigid_spec["model"]["blendrl"]["symbolic"]
    else:
        sym_dict = rigid_spec["symbolic"]
    assert sym_dict["rules"] == "custom_rules"

    # Remove parameter
    viewer._remove_method_model_parameter(method_name, "blendrl", ("symbolic", "rules"))
    assert "rules" not in sym_dict

    # Save to disk and check YAML
    viewer.save_to_disk()
    saved_yaml = yaml.safe_load(dest_file.read_text(encoding="utf-8"))
    assert "cql_blendrl_cew_dueling_resnet" in saved_yaml["methods"]
    assert "tensorboard" not in saved_yaml or saved_yaml["tensorboard"] is not True


def test_config_viewer_loads_cartpole_final(qapp, tmp_path):
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    assert src_file.is_file()

    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    tree = config_tree()
    from frontend.config_viewer import get_method_candidate_hyperparameters

    # ppo_final_cp has agent ppo, model dnn
    dnn_cand = get_method_candidate_hyperparameters("ppo", "dnn", tree)
    assert "blend_ent_coef" not in dnn_cand
    assert "clip_coef" in dnn_cand

    # ppo_blendrl_final_cp has agent ppo, model blendrl
    blend_cand = get_method_candidate_hyperparameters("ppo", "blendrl", tree)
    assert "blend_ent_coef" in blend_cand
    assert "logic_lr" in blend_cand


def test_scoped_candidate_hyperparameters(qapp):
    from frontend.config_viewer import (
        config_tree,
        get_agent_candidate_hyperparameters,
        get_model_candidate_hyperparameters,
        get_training_candidate_hyperparameters,
    )
    tree = config_tree()

    # Agent candidates
    cql_agent_cand = get_agent_candidate_hyperparameters("cql", tree)
    assert "cql_alpha" in cql_agent_cand
    assert "hidden_sizes" not in cql_agent_cand
    assert "blend_ent_coef" not in cql_agent_cand
    assert "lr" not in cql_agent_cand

    ppo_agent_cand = get_agent_candidate_hyperparameters("ppo", tree)
    assert "clip_coef" in ppo_agent_cand
    assert "cql_alpha" not in ppo_agent_cand

    # Model candidates
    dnn_model_cand = get_model_candidate_hyperparameters("dnn", tree)
    assert "hidden_sizes" in dnn_model_cand
    assert "cql_alpha" not in dnn_model_cand
    assert "blend_ent_coef" not in dnn_model_cand

    blend_model_cand = get_model_candidate_hyperparameters("blendrl", tree)
    assert "blend_ent_coef" in blend_model_cand
    assert "blend_q_values" in blend_model_cand
    assert "logic_lr" in blend_model_cand
    assert "cql_alpha" not in blend_model_cand

    # Training candidates
    train_cand = get_training_candidate_hyperparameters()
    assert "lr" in train_cand
    assert "batch_size" in train_cand
    assert "gamma" in train_cand
    assert "epochs" in train_cand


def test_indented_hierarchy_blocks_in_gui(qapp, tmp_path):
    from frontend.config_viewer import IndentedHierarchyBlock

    src_file = Path("src/usr/config/experiment/mimic/cql_benchmark.yaml")
    dest_file = tmp_path / "cql_benchmark.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/mimic/cql_benchmark.yaml")

    # Find hierarchy blocks rendered in the widget tree
    blocks = viewer.findChildren(IndentedHierarchyBlock)
    assert len(blocks) > 0, "Expected IndentedHierarchyBlock instances in rendered GUI"

    # Verify subtle vertical border is styled
    first_block = blocks[0]
    assert "border-left: 2px solid #504945" in first_block.styleSheet()


def test_add_and_remove_parameters_in_hierarchy(qapp, tmp_path):
    src_file = Path("src/usr/config/experiment/mimic/cql_benchmark.yaml")
    dest_file = tmp_path / "cql_benchmark.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/mimic/cql_benchmark.yaml")

    method_name = "cql_dnn"
    # Add agent parameter
    viewer._add_method_param(method_name, "cql_alpha", 2.5)
    assert viewer.raw_data["methods"][method_name]["cql_alpha"] == 2.5

    # Add model parameter
    viewer._add_method_param(method_name, "hidden_sizes", [128, 128])
    assert viewer.raw_data["methods"][method_name]["hidden_sizes"] == [128, 128]

    # Remove parameters
    viewer._remove_method_param(method_name, "cql_alpha")
    assert "cql_alpha" not in viewer.raw_data["methods"][method_name]

    viewer._remove_method_param(method_name, "hidden_sizes")
    assert "hidden_sizes" not in viewer.raw_data["methods"][method_name]


def test_top_box_defaults_hidden_until_added(qapp, tmp_path):
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # None of the 4 flags are in final_cartpole.yaml -> none should be rendered
    assert viewer.chk_tb is None
    assert viewer.chk_save_ds is None
    assert viewer.chk_recover is None
    assert viewer.chk_no_plot is None

    # Add option via raw_data or adder
    viewer.raw_data["tensorboard"] = True
    viewer._render_boxes()

    # Now tensorboard is rendered as a visible checkbox
    assert viewer.chk_tb is not None
    assert viewer.chk_tb.isChecked() is True
    # The others remain unrendered
    assert viewer.chk_save_ds is None
    assert viewer.chk_recover is None
    assert viewer.chk_no_plot is None

    # Remove option
    viewer.raw_data.pop("tensorboard", None)
    viewer._render_boxes()
    assert viewer.chk_tb is None


def test_method_card_duplicate_and_header_plus_icons(qapp, tmp_path):
    from PyQt6.QtWidgets import QToolButton
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # 1. Verify old add method dropdown and button are completely removed
    assert not hasattr(viewer, "combo_new_method")
    assert not hasattr(viewer, "btn_add_method")

    # 2. Duplicate stays beside delete on each card; method creation is a
    # single, explicit control in the Methods header.
    tool_buttons = viewer.findChildren(QToolButton)
    copy_buttons = [b for b in tool_buttons if "Duplicate method" in b.toolTip()]
    plus_buttons = [b for b in tool_buttons if b.toolTip() == "Add new empty method"]
    assert len(copy_buttons) >= 2, "Expected duplicate buttons on method cards"
    assert len(plus_buttons) == 1, "Expected one Methods-header add button"

    # 3. Test duplicate_method
    initial_methods = list(viewer.raw_data["methods"].keys())
    assert "ppo_final_cp" in initial_methods
    dup_name = viewer.duplicate_method("ppo_final_cp")
    assert dup_name == "ppo_final_cp_copy"
    assert dup_name in viewer.raw_data["methods"]
    assert viewer.raw_data["methods"][dup_name]["agent"] == viewer.raw_data["methods"]["ppo_final_cp"]["agent"]

    # Duplicate again generates incremental suffix
    dup2 = viewer.duplicate_method("ppo_final_cp")
    assert dup2 == "ppo_final_cp_copy_2"
    assert dup2 in viewer.raw_data["methods"]


def test_add_new_method_flow_empty_and_invalid(qapp, tmp_path):
    from frontend.config_viewer import InlineTitleEditor
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # Start adding new method
    viewer._start_add_new_method()
    assert viewer._pending_new_method is True

    # Find the editor for the new method
    editors = [e for e in viewer.findChildren(InlineTitleEditor) if e.objectName() == "newMethodEditor"]
    assert len(editors) == 1
    new_editor = editors[0]
    assert new_editor.text() == ""
    # The placeholder stays generic when the group has no default method.
    assert new_editor.placeholderText() == "Method name..."

    # Validate name rules:
    existing_keys = set(viewer.raw_data["methods"].keys())
    # Empty name
    assert viewer._validate_method_name("", existing_keys) == "Method name cannot be empty."
    assert viewer._validate_method_name("   ", existing_keys) == "Method name cannot be empty."
    # Reserved name 'params'
    assert "params" in viewer._validate_method_name("params", existing_keys)
    # Duplicate name
    assert "already exists" in viewer._validate_method_name("ppo_final_cp", existing_keys)
    # Invalid characters (spaces, punctuation)
    assert "invalid characters" in viewer._validate_method_name("invalid name with spaces", existing_keys)
    assert "invalid characters" in viewer._validate_method_name("bad/name", existing_keys)
    assert "invalid characters" in viewer._validate_method_name("bad:name", existing_keys)

    # Cancel pending method: empty name creates no method
    viewer._cancel_new_method()
    qapp.processEvents()
    assert viewer._pending_new_method is False
    assert len(viewer.findChildren(InlineTitleEditor, "newMethodEditor")) == 0


def test_add_new_method_valid_empty_method(qapp, tmp_path):
    from frontend.widgets import ComboBox
    from frontend.config_viewer import IndentedHierarchyBlock
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # Start adding new method
    viewer._start_add_new_method()
    # Add valid method name
    viewer.raw_data.setdefault("methods", {})["custom_empty_method"] = {}
    viewer._pending_new_method = False
    viewer._render_boxes()

    assert "custom_empty_method" in viewer.raw_data["methods"]
    assert viewer.raw_data["methods"]["custom_empty_method"] == {}

    # Inspect the rendered card for custom_empty_method
    combos = viewer.findChildren(ComboBox)
    # Check that placeholders "Select algorithm..." and "Select model..." are used with -1 index
    agent_combos = [c for c in combos if c.placeholderText() in ("Select algorithm...", "Select agent...")]
    model_combos = [c for c in combos if c.placeholderText() == "Select model..."]
    assert len(agent_combos) >= 1
    assert len(model_combos) >= 1
    # For custom_empty_method, nothing is selected
    assert any(c.currentIndex() == -1 for c in agent_combos)
    assert any(c.currentIndex() == -1 for c in model_combos)


def test_sb3_algorithms_selectable_and_labels_renamed(qapp, tmp_path):
    from PyQt6.QtWidgets import QLabel
    from frontend.widgets import ComboBox
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # Verify labels renamed from "Agent Algorithm" -> "Algorithm" and "Model Architecture" -> "Model"
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "Algorithm" in labels
    assert "Model" in labels
    assert "Agent Algorithm" not in labels
    assert "Model Architecture" not in labels

    # Verify SB3 algorithms are available in algorithm dropdown
    combos = viewer.findChildren(ComboBox)
    algo_combos = [c for c in combos if c.placeholderText() == "Select algorithm..."]
    assert len(algo_combos) >= 1
    items = [algo_combos[0].itemText(i) for i in range(algo_combos[0].count())]
    for sb3_algo in ("sb3/a2c", "sb3/dqn", "sb3/ppo", "sb3/sac"):
        assert sb3_algo in items

    # Select an SB3 algorithm and verify raw_data update
    algo_combos[0].setCurrentText("sb3/dqn")
    qapp.processEvents()
    assert viewer.raw_data["methods"]["ppo_final_cp"]["agent"] == "sb3/dqn"


def test_blendrl_model_block_and_logic_module_rendering(qapp, tmp_path):
    from frontend.widgets import ComboBox
    from frontend.config_viewer import IndentedHierarchyBlock
    from PyQt6.QtWidgets import QLabel

    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    # final_cartpole has ppo_blendrl_final_cp with model: blendrl
    assert "ppo_blendrl_final_cp" in viewer.raw_data["methods"]
    assert viewer.raw_data["methods"]["ppo_blendrl_final_cp"]["model"] == "blendrl"

    # Parameters with declared defaults are not inserted until the user adds them.
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "Neural Architecture" not in labels
    assert "Logic Module (Symbolic)" not in labels
    assert "Blender Mode" not in labels

    # Component-valued parameters are declared by the model schema.
    model_tree = config_tree()
    bp_items = {
        key for key, schema in model_tree.models["blendrl"].parameter_schema.get("properties", {}).items()
        if isinstance(schema, dict) and schema.get("x-theta-domain")
    }
    assert "symbolic" in bp_items
    assert "neural" in bp_items
    assert "blender" not in bp_items

    # 1. Add symbolic subcomponent via adder
    viewer._add_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("symbolic",))
    qapp.processEvents()

    # The generic nested parameter editor renders the declared component-valued field.
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "[symbolic]" in labels
    assert "[neural]" not in labels
    assert "[blender]" not in labels

    # The declared model domain is edited with the generic model selector.
    combos = viewer.findChildren(ComboBox)
    logic_combos = [
        c for c in combos
        if {"nsfr", "neumann", "cew"}.issubset({c.itemText(i) for i in range(c.count())})
    ]
    assert len(logic_combos) >= 1, "Expected a model selector for the declared model parameter"
    logic_combo = logic_combos[0]

    # Verify choices include nsfr, neumann, cew with default nsfr
    combo_items = [logic_combo.itemText(i) for i in range(logic_combo.count())]
    assert "nsfr" in combo_items
    assert "neumann" in combo_items
    assert "cew" in combo_items
    assert logic_combo.currentText() == "nsfr"

    # Change Logic Module to cew
    logic_combo.setCurrentText("cew")
    qapp.processEvents()

    # Verify raw_data was updated
    method_spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]
    assert method_spec.get("symbolic") == "cew" or (
        isinstance(method_spec.get("symbolic"), dict) and method_spec["symbolic"].get("type") == "cew"
    )

    # Change Logic Module to neumann
    logic_combo.setCurrentText("neumann")
    qapp.processEvents()
    method_spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]
    assert method_spec.get("symbolic") == "neumann" or (
        isinstance(method_spec.get("symbolic"), dict) and method_spec["symbolic"].get("type") == "neumann"
    )

    # 2. Add rule parameter under symbolic
    viewer._on_method_model_parameter_edited("ppo_blendrl_final_cp", "blendrl", ("symbolic", "rules"), "custom_rules")
    sym_spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]["symbolic"]
    assert (sym_spec.get("rules") if isinstance(sym_spec, dict) else sym_spec) == "custom_rules"

    # Remove rule parameter
    viewer._remove_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("symbolic", "rules"))
    sym_spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]["symbolic"]
    if isinstance(sym_spec, dict):
        assert "rules" not in sym_spec

    # 3. Remove symbolic subcomponent
    viewer._remove_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("symbolic",))
    qapp.processEvents()

    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "[symbolic]" not in labels

    # Verify symbolic is back in candidate blend parameters
    spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]
    present = set((spec.get("model") or {}).keys()) if isinstance(spec.get("model"), dict) else set()
    assert "symbolic" in set(config_tree().model_parameter_keys("blendrl")) - present

    # 4. Add neural subcomponent
    viewer._add_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("neural",))
    qapp.processEvents()
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "[neural]" in labels

    # Add hidden_sizes under neural
    viewer._on_method_model_parameter_edited("ppo_blendrl_final_cp", "blendrl", ("neural", "hidden_sizes"), [128, 128])
    neural_spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]["neural"]
    assert (neural_spec.get("hidden_sizes") if isinstance(neural_spec, dict) else neural_spec) == [128, 128]

    # Remove hidden_sizes
    viewer._remove_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("neural", "hidden_sizes"))
    neural_spec = viewer.raw_data["methods"]["ppo_blendrl_final_cp"]["neural"]
    if isinstance(neural_spec, dict):
        assert "hidden_sizes" not in neural_spec

    # Remove neural
    viewer._remove_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("neural",))
    qapp.processEvents()
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "[neural]" not in labels

    # 5. Add the nested blender configuration group
    viewer._add_method_param("ppo_blendrl_final_cp", "blender")
    qapp.processEvents()
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "[blender]" not in labels
    assert any("blender" in text.lower() for text in labels)

    # Remove blender
    viewer._remove_method_model_parameter("ppo_blendrl_final_cp", "blendrl", ("blender",))
    qapp.processEvents()
    labels = [l.text() for l in viewer.findChildren(QLabel)]
    assert "[blender]" not in labels


def test_blendrl_params_and_rules_editing(qapp, tmp_path):
    src_file = Path("src/usr/config/experiment/cartpole/final_cartpole.yaml")
    dest_file = tmp_path / "final_cartpole.yaml"
    shutil.copy(src_file, dest_file)

    viewer = ConfigViewer()
    viewer.load_file(dest_file, "experiment/cartpole/final_cartpole.yaml")

    method_name = "ppo_blendrl_final_cp"

    # 1. Add BlendRL top-level parameter (e.g. blend_q_values, logic_lr)
    viewer._add_method_param(method_name, "blend_q_values", True)
    assert viewer.raw_data["methods"][method_name]["blend_q_values"] is True

    viewer._add_method_param(method_name, "logic_lr", 0.001)
    assert viewer.raw_data["methods"][method_name]["logic_lr"] == 0.001

    # 2. Add rule parameter to Logic Module (symbolic)
    viewer._on_method_model_parameter_edited(method_name, "blendrl", ("symbolic", "rules"), "variant1")
    sym_spec = viewer.raw_data["methods"][method_name]["symbolic"]
    if isinstance(sym_spec, dict):
        assert sym_spec.get("rules") == "variant1"
    else:
        assert sym_spec == "variant1"

    # Edit rule parameter
    viewer._on_method_model_parameter_edited(method_name, "blendrl", ("symbolic", "rules"), "custom_rules")
    sym_spec = viewer.raw_data["methods"][method_name]["symbolic"]
    assert sym_spec["rules"] == "custom_rules"

    # 3. Remove rule parameter
    viewer._remove_method_model_parameter(method_name, "blendrl", ("symbolic", "rules"))
    assert "rules" not in viewer.raw_data["methods"][method_name]["symbolic"]

    # 4. Remove BlendRL parameter
    viewer._remove_method_param(method_name, "blend_q_values")
    assert "blend_q_values" not in viewer.raw_data["methods"][method_name]

    # Save to disk and verify round-trip
    viewer.save_to_disk()
    saved = yaml.safe_load(dest_file.read_text(encoding="utf-8"))
    assert "ppo_blendrl_final_cp" in saved["methods"]
    assert saved["methods"]["ppo_blendrl_final_cp"]["logic_lr"] == 0.001


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

    group = tmp_path / "experiment" / "probe_group"
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
    from frontend.config_model import ConfigTree, Model
    from frontend.config_viewer import config_tree

    tree = config_tree()
    assert "blendrl" in tree.models
    blendrl_domains = {
        key for key, schema in tree.models["blendrl"].parameter_schema.get("properties", {}).items()
        if isinstance(schema, dict) and schema.get("x-theta-domain")
    }
    assert set(blendrl_domains) == {"neural", "symbolic"}
    dnn_domains = {
        key for key, schema in tree.models["dnn"].parameter_schema.get("properties", {}).items()
        if isinstance(schema, dict) and schema.get("x-theta-domain")
    }
    assert dnn_domains == set()

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
