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


def create_transfer_learning_preset() -> WorkflowGraph:
    """1. Transfer Learning: Experiment 1 trains a policy, passed into Experiment 2 in a different env."""
    graph = WorkflowGraph(
        id="transfer_learning",
        name="Transfer Learning (CartPole -> MountainCar)",
        description="Trains a source policy on CartPole, then wires the checkpoint weights to warm-start fine-tuning on MountainCar.",
    )
    n1 = make_online_rl_node(
        "source_trainer", "CartPole PPO (Source)", experiment_ref="csc510/cartpole_demo", pos_x=60, pos_y=120
    )
    n2 = make_online_rl_node(
        "target_fine_tuner", "MountainCar PPO (Target)", experiment_ref="cartpole/quick_test", pos_x=440, pos_y=120
    )
    n2.overrides = ["env=gym_classic_control/mountaincar", "++train.freeze_backbone=false"]

    graph.add_node(n1)
    graph.add_node(n2)
    graph.connect("source_trainer", "checkpoint", "target_fine_tuner", "initial_weights")
    return graph


def create_online_vs_offline_preset() -> WorkflowGraph:
    """2. Online vs Offline RL Comparison: Online agent's explored buffer feeds offline agent."""
    graph = WorkflowGraph(
        id="online_vs_offline_comparison",
        name="Online vs. Offline RL Comparison",
        description="Exploration buffer collected by online PPO is piped to offline IQL, and both performance curves are compared.",
    )
    n1 = make_online_rl_node(
        "online_ppo", "CartPole PPO (Online)", experiment_ref="csc510/cartpole_demo", pos_x=60, pos_y=80
    )
    n2 = make_offline_rl_node(
        "offline_iql", "CartPole IQL (Offline)", experiment_ref="cartpole/offline_cartpole", pos_x=440, pos_y=80
    )
    n3 = make_plot_evaluator_node("comparator", "Convergence Comparator", pos_x=800, pos_y=150)

    graph.add_node(n1)
    graph.add_node(n2)
    graph.add_node(n3)

    graph.connect("online_ppo", "dataset", "offline_iql", "dataset")
    graph.connect("online_ppo", "metrics", "comparator", "metrics_a")
    graph.connect("offline_iql", "metrics", "comparator", "metrics_b")
    return graph


def create_model_distillation_preset() -> WorkflowGraph:
    """3. Model Distillation: Teacher model generates dataset of soft rollouts to train a compact student model."""
    graph = WorkflowGraph(
        id="model_distillation",
        name="Model Distillation (NeSy Teacher -> Student MLP)",
        description="A large hybrid BlendRL teacher model generates rollout transitions with soft action logits to train a fast student MLP.",
    )
    teacher = make_online_rl_node(
        "teacher_model", "Pretrained BlendRL (Teacher)", experiment_ref="cartpole/quick_test", pos_x=60, pos_y=120
    )

    # Rollout generator node
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

    student = make_distillation_node("student_mlp", "Compact Student MLP", pos_x=780, pos_y=120)

    graph.add_node(teacher)
    graph.add_node(generator)
    graph.add_node(student)

    graph.connect("teacher_model", "checkpoint", "rollout_generator", "teacher_checkpoint")
    graph.connect("rollout_generator", "soft_dataset", "student_mlp", "dataset")
    return graph


def create_sepsis_reciprocal_preset() -> WorkflowGraph:
    """4. Sepsis Clinician Behavior -> Early Prediction (Reciprocal Co-Training)."""
    graph = WorkflowGraph(
        id="sepsis_reciprocal_refinement",
        name="Sepsis Clinician Behavior ↔ Early Prediction",
        description="CQL clinician policy outputs V(s) to augment MIMIC patient records for Early Prediction, whose shock risk Phi(s) shapes CQL rewards.",
    )
    cql = make_offline_rl_node(
        "cql_clinician", "MIMIC CQL Clinician Model", experiment_ref="mimic/cql_literature", pos_x=60, pos_y=80
    )
    # Add value_estimator output to CQL
    cql.add_output(
        Port(
            name="value_estimator",
            direction=PortDirection.OUTPUT,
            port_type=PortType.VALUE_ESTIMATOR,
            description="State-value estimates V(s) and action Q-values",
        )
    )

    mimic_data = make_dataset_source_node(
        "mimic_dataset", "MIMIC Patient Transitions", "src/usr/datasets/mimic", pos_x=60, pos_y=300
    )
    augmenter = make_feature_augmenter_node("feature_augmenter", "V(s) Feature Augmenter", pos_x=440, pos_y=80)
    ep = make_supervised_node(
        "ep_transformer",
        "Early Prediction (Transformer)",
        experiment_ref="early_prediction/quick_test",
        pos_x=800,
        pos_y=80,
    )
    shaper = make_reward_shaper_node("reward_shaper", "Potential Reward Shaper", pos_x=440, pos_y=300)

    graph.add_node(cql)
    graph.add_node(mimic_data)
    graph.add_node(augmenter)
    graph.add_node(ep)
    graph.add_node(shaper)

    # Base data flow
    graph.connect("mimic_dataset", "dataset", "feature_augmenter", "dataset")
    graph.connect("mimic_dataset", "dataset", "reward_shaper", "dataset")

    # Forward pipeline
    graph.connect("cql_clinician", "value_estimator", "feature_augmenter", "value_estimator")
    graph.connect("feature_augmenter", "augmented_dataset", "ep_transformer", "dataset")
    graph.connect("ep_transformer", "value_estimator", "reward_shaper", "potential_function")

    # Feedback loop string back to CQL
    graph.connect("reward_shaper", "shaped_dataset", "cql_clinician", "dataset")
    return graph


BUILTIN_PRESETS: dict[str, Callable[[], WorkflowGraph]] = {
    "transfer_learning": create_transfer_learning_preset,
    "online_vs_offline": create_online_vs_offline_preset,
    "model_distillation": create_model_distillation_preset,
    "sepsis_reciprocal": create_sepsis_reciprocal_preset,
}


def get_preset(preset_id: str) -> WorkflowGraph | None:
    factory = BUILTIN_PRESETS.get(preset_id)
    return factory() if factory else None
