"""Built-in workflow templates for multi-stage experiments."""

from __future__ import annotations

from collections.abc import Callable

from .model import Port, PortDirection, PortType, WorkflowGraph, WorkflowNode
from .templates import (
    make_dataset_source_node,
    make_distillation_node,
    make_feature_augmenter_node,
    make_offline_rl_node,
    make_online_rl_node,
    make_plot_evaluator_node,
    make_reward_shaper_node,
    make_supervised_node,
)


# Presets describe workflow shapes only. They name no research package or
# experiment: choose each node's experiment in the inspector before running.


def create_transfer_learning_preset() -> WorkflowGraph:
    """Transfer learning: a source run's checkpoint warm-starts a target run."""
    graph = WorkflowGraph(
        id="transfer_learning",
        name="Transfer Learning",
        description="Trains a source policy, then wires its checkpoint into a second experiment as initial weights.",
    )
    graph.add_node(make_online_rl_node("source", "Source Training", pos_x=60, pos_y=120))
    graph.add_node(make_online_rl_node("target", "Target Fine-Tuning", pos_x=440, pos_y=120))
    graph.connect("source", "checkpoint", "target", "initial_weights")
    return graph


def create_online_vs_offline_preset() -> WorkflowGraph:
    """Online vs. offline comparison: an online run's buffer trains an offline run."""
    graph = WorkflowGraph(
        id="online_vs_offline_comparison",
        name="Online vs. Offline RL Comparison",
        description="The buffer an online agent collects trains an offline agent, and both learning curves are compared.",
    )
    graph.add_node(make_online_rl_node("online", "Online Agent", pos_x=60, pos_y=80))
    graph.add_node(make_offline_rl_node("offline", "Offline Agent", pos_x=440, pos_y=80))
    graph.add_node(make_plot_evaluator_node("comparator", "Convergence Comparator", pos_x=800, pos_y=150))
    graph.connect("online", "dataset", "offline", "dataset")
    graph.connect("online", "metrics", "comparator", "metrics_a")
    graph.connect("offline", "metrics", "comparator", "metrics_b")
    return graph


def create_model_distillation_preset() -> WorkflowGraph:
    """Model distillation: a teacher's rollouts with soft targets train a student."""
    graph = WorkflowGraph(
        id="model_distillation",
        name="Model Distillation",
        description="A trained teacher generates rollouts with soft action targets that train a smaller student model.",
    )
    teacher = make_online_rl_node("teacher", "Teacher", pos_x=60, pos_y=120)
    generator = WorkflowNode(
        id="rollout_generator",
        label="Trajectory Rollout Generator",
        category="task",
        paradigm="rollout",
        pos_x=420,
        pos_y=120,
    )
    generator.add_input(
        Port(
            name="teacher_checkpoint",
            direction=PortDirection.INPUT,
            port_type=PortType.CHECKPOINT,
            description="Teacher model checkpoint",
            required=True,
        )
    )
    generator.add_output(
        Port(
            name="soft_dataset",
            direction=PortDirection.OUTPUT,
            port_type=PortType.DATASET,
            description="Dataset with soft targets and state visits",
        )
    )
    student = make_distillation_node("student", "Student", pos_x=780, pos_y=120)

    graph.add_node(teacher)
    graph.add_node(generator)
    graph.add_node(student)
    graph.connect("teacher", "checkpoint", "rollout_generator", "teacher_checkpoint")
    graph.connect("rollout_generator", "soft_dataset", "student", "dataset")
    return graph


def create_reciprocal_refinement_preset() -> WorkflowGraph:
    """Reciprocal refinement: a policy's values and a predictor's potential improve each other."""
    graph = WorkflowGraph(
        id="reciprocal_refinement",
        name="Reciprocal Refinement",
        description="An offline policy's state values augment a dataset for a supervised predictor, "
                    "whose output shapes the rewards the policy trains on.",
    )
    policy = make_offline_rl_node("policy", "Offline Policy", pos_x=60, pos_y=80)
    policy.add_output(
        Port(
            name="value_estimator",
            direction=PortDirection.OUTPUT,
            port_type=PortType.VALUE_ESTIMATOR,
            description="State-value estimates V(s) and action Q-values",
        )
    )
    dataset = make_dataset_source_node("dataset", "Transitions Dataset", pos_x=60, pos_y=300)
    augmenter = make_feature_augmenter_node("feature_augmenter", "V(s) Feature Augmenter", pos_x=440, pos_y=80)
    predictor = make_supervised_node("predictor", "Supervised Predictor", pos_x=800, pos_y=80)
    shaper = make_reward_shaper_node("reward_shaper", "Potential Reward Shaper", pos_x=440, pos_y=300)

    for node in (policy, dataset, augmenter, predictor, shaper):
        graph.add_node(node)
    graph.connect("dataset", "dataset", "feature_augmenter", "dataset")
    graph.connect("dataset", "dataset", "reward_shaper", "dataset")
    graph.connect("policy", "value_estimator", "feature_augmenter", "value_estimator")
    graph.connect("feature_augmenter", "augmented_dataset", "predictor", "dataset")
    graph.connect("predictor", "value_estimator", "reward_shaper", "potential_function")
    graph.connect("reward_shaper", "shaped_dataset", "policy", "dataset")
    return graph


BUILTIN_PRESETS: dict[str, Callable[[], WorkflowGraph]] = {
    "transfer_learning": create_transfer_learning_preset,
    "online_vs_offline": create_online_vs_offline_preset,
    "model_distillation": create_model_distillation_preset,
    "reciprocal_refinement": create_reciprocal_refinement_preset,
}


def get_preset(preset_id: str) -> WorkflowGraph | None:
    factory = BUILTIN_PRESETS.get(preset_id)
    return factory() if factory else None
