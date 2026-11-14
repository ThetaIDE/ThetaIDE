"""Workflow executor: runs a named WorkflowGraph DAG by executing its nodes
in topological order, wiring artifact outputs between nodes.

This replaces the old reciprocal_task.py hard-coded orchestration.
Workflow YAML files live in in/config/workflow/<workflow_id>.yaml.

Each node declares a paradigm (online_rl, offline_rl, supervised, transform)
and an experiment_ref pointing to an existing experiment config. The executor
builds and runs each node in topological level order.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

log = logging.getLogger(__name__)

from src.app.workspace import WorkspacePaths


def load_workflow(workflow_ref: Any):
    """Load a WorkflowGraph from in/config/workflow/<workflow_id>.yaml or from dict/DictConfig."""
    from omegaconf import DictConfig, OmegaConf

    config_dir = WorkspacePaths.for_process().config / "workflow"

    from src.app.pipeline.workflow.model import WorkflowGraph

    if isinstance(workflow_ref, (dict, DictConfig)):
        w_id = workflow_ref.get("id") if hasattr(workflow_ref, "get") else getattr(workflow_ref, "id", None)
        if w_id and (config_dir / f"{w_id}.yaml").exists():
            return WorkflowGraph.load_yaml(config_dir / f"{w_id}.yaml")

        data = (
            OmegaConf.to_container(workflow_ref, resolve=False)
            if isinstance(workflow_ref, DictConfig)
            else workflow_ref
        )
        return WorkflowGraph.from_dict(cast(dict[str, Any], data))

    workflow_id = str(workflow_ref)
    yaml_path = config_dir / f"{workflow_id}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Workflow '{workflow_id}' not found. "
            f"Expected: {yaml_path}. "
            f"Available: {[p.stem for p in config_dir.glob('*.yaml')]}"
        )
    return WorkflowGraph.load_yaml(yaml_path)


def run_workflow(workflow_id: str, cfg: Any, context: dict) -> None:
    """Execute a named workflow DAG.

    Nodes are executed in topological level order (parallelism deferred to future).
    Transform nodes are no-ops at the pipeline orchestration layer — they are
    handled by the workflow's data flow bindings and executed as subprocesses.

    Args:
        workflow_id: Stem of the workflow YAML file (e.g. 'sepsis_reciprocal').
        cfg: Resolved Hydra experiment config.
        context: Pipeline execution context dict from run_pipeline.py.
    """
    graph = load_workflow(workflow_id)
    issues = graph.validate()
    if issues:
        log.warning("Workflow '%s' has validation issues:\n  %s", workflow_id, "\n  ".join(issues))

    levels = graph.topological_levels()
    is_interactive = context.get("is_interactive", True)

    print(f"\n{'=' * 60}")
    print(f"  WORKFLOW: {graph.name}")
    print(f"  {graph.description}")
    print(f"  Execution order: {levels}")
    print(f"{'=' * 60}\n")

    for level_idx, node_ids in enumerate(levels):
        print(f"--- Workflow Level {level_idx}: {node_ids} ---")
        for node_id in node_ids:
            node = graph.nodes[node_id]
            _execute_node(node, graph, cfg, context, is_interactive)

    print(f"\nWorkflow '{graph.name}' complete.")


def _execute_node(node, graph, cfg, context, is_interactive: bool) -> None:
    """Execute a single workflow node."""
    from src.app.pipeline.workflow.model import WorkflowNode

    category = node.category
    paradigm = node.paradigm
    experiment_ref = node.experiment_ref

    print(f"  [{node.id}] {node.label} (category={category}, paradigm={paradigm})")

    if category == "data":
        # Data source nodes — no execution needed, just an artifact reference
        log.info("Skipping data source node '%s' (artifact reference only).", node.id)
        return

    if category == "transform":
        # Transform nodes delegate to their bound experiment subprocess
        log.info("Transform node '%s': bindings will be applied by the engine in future iteration.", node.id)
        print(
            f"  [WARNING] Transform node '{node.id}' execution not yet automated. "
            f"Apply param bindings manually or implement WorkflowExecutor.execute_transform()."
        )
        return

    if category == "paradigm" and experiment_ref:
        # Paradigm nodes: run the referenced experiment
        _run_experiment_node(node, cfg, context, is_interactive)
        return

    log.warning("Node '%s' has unhandled category '%s' — skipping.", node.id, category)


def _run_experiment_node(node, cfg, context, is_interactive: bool) -> None:
    """Run a paradigm experiment node by invoking run_pipeline.py as a subprocess."""
    import subprocess

    from src.app.pipeline.runtime import get_python_executable

    site_cfg = cfg.get("site", None) if hasattr(cfg, "get") else None
    python_exe = get_python_executable(site_cfg)

    experiment_ref = node.experiment_ref
    overrides = list(node.overrides or [])
    forwarded_args = [
        a
        for a in (context.get("sanitized_extra_args") or [])
        if not a.startswith("workflow=")
        and not a.startswith("+experiment=")
        and not a.startswith("++experiment_name=")
        and not a.startswith("++experiment_id=")
        and not a.startswith("++group=")
    ]

    cmd = [python_exe, "-u", "run_pipeline.py", experiment_ref] + overrides + forwarded_args
    print(f"    CMD: {' '.join(cmd)}")

    if is_interactive:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            log.warning("Node '%s' exited with code %d.", node.id, result.returncode)
    else:
        log.info("[Scheduler mode] Workflow node '%s' scheduler submission not yet implemented.", node.id)
        print("    [Scheduler] Submission for workflow nodes is not yet automated.")
