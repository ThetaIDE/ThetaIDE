"""Pre-configured node templates for learning paradigms and composite tasks."""

from __future__ import annotations

from .model import Port, PortDirection, PortType, WorkflowNode


def make_online_rl_node(
    node_id: str, label: str, experiment_ref: str = "", pos_x: float = 0, pos_y: float = 0
) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="paradigm",
        paradigm="online_rl",
        experiment_ref=experiment_ref,
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="initial_weights",
            direction=PortDirection.INPUT,
            port_type=PortType.CHECKPOINT,
            description="Warm-start weights for fine-tuning / transfer learning.",
            param_binding="++model.checkpoint_path=${source.checkpoint_path}",
        )
    )
    node.add_output(
        Port(
            name="checkpoint",
            direction=PortDirection.OUTPUT,
            port_type=PortType.CHECKPOINT,
            description="Saved policy checkpoint (.ckpt)",
        )
    )
    node.add_output(
        Port(
            name="dataset",
            direction=PortDirection.OUTPUT,
            port_type=PortType.DATASET,
            description="Generated trajectory buffer chunk files (.pkl)",
        )
    )
    node.add_output(
        Port(
            name="metrics",
            direction=PortDirection.OUTPUT,
            port_type=PortType.METRICS,
            description="Online training log CSV metrics",
        )
    )
    return node


def make_offline_rl_node(
    node_id: str, label: str, experiment_ref: str = "", pos_x: float = 0, pos_y: float = 0
) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="paradigm",
        paradigm="offline_rl",
        experiment_ref=experiment_ref,
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="dataset",
            direction=PortDirection.INPUT,
            port_type=PortType.DATASET,
            description="Replay buffer dataset for offline training",
            param_binding="++mode.dataset_path=${source.dataset_path}",
            required=True,
        )
    )
    node.add_input(
        Port(
            name="initial_weights",
            direction=PortDirection.INPUT,
            port_type=PortType.CHECKPOINT,
            description="Warm-start or initialization weights",
            param_binding="++model.checkpoint_path=${source.checkpoint_path}",
        )
    )
    node.add_output(
        Port(
            name="checkpoint",
            direction=PortDirection.OUTPUT,
            port_type=PortType.CHECKPOINT,
            description="Offline-trained policy checkpoint",
        )
    )
    node.add_output(
        Port(
            name="metrics",
            direction=PortDirection.OUTPUT,
            port_type=PortType.METRICS,
            description="Offline training loss & Bellman error metrics",
        )
    )
    return node


def make_supervised_node(
    node_id: str, label: str, experiment_ref: str = "", pos_x: float = 0, pos_y: float = 0
) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="paradigm",
        paradigm="supervised",
        experiment_ref=experiment_ref,
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="dataset",
            direction=PortDirection.INPUT,
            port_type=PortType.DATASET,
            description="Tabular or sequential supervised training dataset",
            param_binding="++mode.dataset_path=${source.dataset_path}",
            required=True,
        )
    )
    node.add_output(
        Port(
            name="checkpoint",
            direction=PortDirection.OUTPUT,
            port_type=PortType.CHECKPOINT,
            description="Supervised model weights (.ckpt)",
        )
    )
    node.add_output(
        Port(
            name="value_estimator",
            direction=PortDirection.OUTPUT,
            port_type=PortType.VALUE_ESTIMATOR,
            description="Prediction function / shock risk estimator Phi(s)",
        )
    )
    node.add_output(
        Port(
            name="metrics",
            direction=PortDirection.OUTPUT,
            port_type=PortType.METRICS,
            description="AUPRC, AUROC, and cross-entropy metrics",
        )
    )
    return node


def make_distillation_node(node_id: str, label: str, pos_x: float = 0, pos_y: float = 0) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="task",
        paradigm="distillation",
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="teacher_checkpoint",
            direction=PortDirection.INPUT,
            port_type=PortType.CHECKPOINT,
            description="Pretrained teacher policy weights",
            param_binding="++distill.teacher_checkpoint=${source.checkpoint_path}",
            required=False,
        )
    )
    node.add_input(
        Port(
            name="dataset",
            direction=PortDirection.INPUT,
            port_type=PortType.DATASET,
            description="Trajectory dataset with soft teacher action targets",
            param_binding="++mode.dataset_path=${source.dataset_path}",
        )
    )
    node.add_output(
        Port(
            name="student_checkpoint",
            direction=PortDirection.OUTPUT,
            port_type=PortType.CHECKPOINT,
            description="Distilled compact student model weights",
        )
    )
    node.add_output(
        Port(
            name="metrics",
            direction=PortDirection.OUTPUT,
            port_type=PortType.METRICS,
            description="Student imitation loss & validation divergence",
        )
    )
    return node


def make_feature_augmenter_node(node_id: str, label: str, pos_x: float = 0, pos_y: float = 0) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="transform",
        paradigm="transform",
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="value_estimator",
            direction=PortDirection.INPUT,
            port_type=PortType.VALUE_ESTIMATOR,
            description="Estimated values V(s) or Q(s, a)",
            param_binding="++augment.value_model=${source.checkpoint_path}",
            required=True,
        )
    )
    node.add_input(
        Port(
            name="dataset",
            direction=PortDirection.INPUT,
            port_type=PortType.DATASET,
            description="Raw dataset to augment with value features",
            param_binding="++augment.input_dataset=${source.dataset_path}",
            required=True,
        )
    )
    node.add_output(
        Port(
            name="augmented_dataset",
            direction=PortDirection.OUTPUT,
            port_type=PortType.DATASET,
            description="Dataset augmented with clinician value state features",
        )
    )
    return node


def make_reward_shaper_node(node_id: str, label: str, pos_x: float = 0, pos_y: float = 0) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="transform",
        paradigm="transform",
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="potential_function",
            direction=PortDirection.INPUT,
            port_type=PortType.VALUE_ESTIMATOR,
            description="Potential function Phi(s) for Ng et al. shaping",
            param_binding="++reward_shaping.ep_model=${source.checkpoint_path}",
            required=True,
        )
    )
    node.add_input(
        Port(
            name="dataset",
            direction=PortDirection.INPUT,
            port_type=PortType.DATASET,
            description="Base dataset to reshape rewards for",
            param_binding="++reward_shaping.dataset_path=${source.dataset_path}",
            required=True,
        )
    )
    node.add_output(
        Port(
            name="shaped_dataset",
            direction=PortDirection.OUTPUT,
            port_type=PortType.DATASET,
            description="Dataset with r_shaped = r + lambda*(gamma*Phi' - Phi)",
        )
    )
    return node


def make_plot_evaluator_node(node_id: str, label: str, pos_x: float = 0, pos_y: float = 0) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="eval",
        paradigm="eval",
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_input(
        Port(
            name="metrics_a",
            direction=PortDirection.INPUT,
            port_type=PortType.METRICS,
            description="First method metrics CSV",
        )
    )
    node.add_input(
        Port(
            name="metrics_b",
            direction=PortDirection.INPUT,
            port_type=PortType.METRICS,
            description="Second method metrics CSV",
        )
    )
    node.add_output(
        Port(
            name="comparison_report",
            direction=PortDirection.OUTPUT,
            port_type=PortType.GENERIC,
            description="Generated comparative PNG plots and Markdown reports",
        )
    )
    return node


def make_dataset_source_node(
    node_id: str, label: str, dataset_path: str = "", pos_x: float = 0, pos_y: float = 0
) -> WorkflowNode:
    node = WorkflowNode(
        id=node_id,
        label=label,
        category="data",
        paradigm="dataset",
        experiment_ref=dataset_path,
        pos_x=pos_x,
        pos_y=pos_y,
    )
    node.add_output(
        Port(
            name="dataset",
            direction=PortDirection.OUTPUT,
            port_type=PortType.DATASET,
            description=f"Static offline dataset: {dataset_path or 'src/usr/datasets'}",
        )
    )
    return node
