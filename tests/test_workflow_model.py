"""Unit tests for the string diagram workflow data model and presets."""

import pytest
from src.app.pipeline.workflow.model import (
    Port,
    PortDirection,
    PortType,
    WorkflowGraph,
    WorkflowNode,
    WorkflowString,
)
from src.app.pipeline.workflow.presets import (
    BUILTIN_PRESETS,
    create_model_distillation_preset,
    create_online_vs_offline_preset,
    create_reciprocal_refinement_preset,
    create_transfer_learning_preset,
    get_preset,
)
from src.app.pipeline.workflow.templates import (
    make_offline_rl_node,
    make_online_rl_node,
    make_supervised_node,
)


def test_port_type_colors_and_names():
    assert PortType.DATASET.color == "#06b6d4"
    assert PortType.CHECKPOINT.color == "#a855f7"
    assert PortType.VALUE_ESTIMATOR.color == "#f59e0b"
    assert PortType.METRICS.color == "#10b981"
    assert "Dataset" in PortType.DATASET.display_name


def test_type_compatibility():
    assert WorkflowGraph.is_type_compatible(PortType.DATASET, PortType.DATASET)
    assert not WorkflowGraph.is_type_compatible(PortType.DATASET, PortType.CHECKPOINT)
    assert WorkflowGraph.is_type_compatible(PortType.GENERIC, PortType.CHECKPOINT)
    assert WorkflowGraph.is_type_compatible(PortType.CHECKPOINT, PortType.GENERIC)


def test_graph_connect_and_validate():
    graph = WorkflowGraph(id="test_g", name="Test Graph")
    n1 = make_online_rl_node("n1", "Online RL")
    n2 = make_offline_rl_node("n2", "Offline RL")

    graph.add_node(n1)
    graph.add_node(n2)

    # Connecting dataset output to dataset input should succeed
    wire = graph.connect("n1", "dataset", "n2", "dataset")
    assert wire.source_node_id == "n1"
    assert wire.target_node_id == "n2"
    assert wire.port_type == PortType.DATASET
    assert len(graph.strings) == 1

    # Validation should pass
    issues = graph.validate()
    assert len(issues) == 0

    # Incompatible connection should raise TypeError
    with pytest.raises(TypeError):
        graph.connect("n1", "checkpoint", "n2", "dataset")


def test_graph_serialization_yaml():
    preset = create_online_vs_offline_preset()
    yaml_str = preset.to_yaml()
    assert "id: online" in yaml_str
    assert "id: offline" in yaml_str

    restored = WorkflowGraph.from_yaml(yaml_str)
    assert len(restored.nodes) == 3
    assert len(restored.strings) == 3
    assert restored.validate() == []


def test_topological_sort_levels():
    preset = create_online_vs_offline_preset()
    levels = preset.topological_levels()
    assert len(levels) >= 2
    # The online agent runs first; the offline agent depends on it
    assert "online" in levels[0]
    assert any("offline" in lvl for lvl in levels[1:])


def test_all_builtin_presets():
    for pid, factory in BUILTIN_PRESETS.items():
        g = factory()
        assert isinstance(g, WorkflowGraph)
        assert len(g.nodes) >= 2
        assert len(g.strings) >= 1
        # All builtin forward connections should be type valid
        issues = g.validate()
        assert len(issues) == 0, f"Preset {pid} had validation errors: {issues}"


def test_presets_name_no_experiment_and_refuse_to_run_until_assigned(monkeypatch):
    from src.app.pipeline.workflow import executor

    for pid, factory in BUILTIN_PRESETS.items():
        graph = factory()
        assert all(not node.experiment_ref for node in graph.nodes.values()), pid
        monkeypatch.setattr(executor, "load_workflow", lambda ref, graph=graph: graph)
        monkeypatch.setattr(executor, "_execute_node", lambda *args: pytest.fail("ran a node"))
        with pytest.raises(ValueError, match="choose an experiment for"):
            executor.run_workflow(pid, {}, {})
