"""Modular component interfaces for the paradigm system.

These ABCs define the extension API for adding new paradigms.
Concrete implementations live in src/core/paradigm_impls/.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataModule(ABC):
    """Owns data loading for a paradigm.

    Examples: RL replay buffer, cross-validation splits, streaming dataset.
    Implementations live in src/core/paradigm_impls/base/.
    """

    @abstractmethod
    def setup(self, stage: str | None = None, cfg: Any = None) -> None:
        """Initialize from config or stage (load dataset, build splits, etc.)."""

    @abstractmethod
    def train_dataloader(self):
        """Return the training data loader."""

    def val_dataloader(self):
        """Return validation data loader. Return None if not applicable."""
        return None


class BaseEvalProtocol(ABC):
    """Owns evaluation for a paradigm.

    Examples: episodic reward rollout, AUC-ROC on held-out fold, val/loss.
    """

    @abstractmethod
    def evaluate(self, agent, data_source, cfg) -> dict:
        """Run evaluation. Returns metric dict, e.g. {"eval/auc_roc": 0.82}."""


class BaseParadigmRunner(ABC):
    """Owns a single training loop for a base paradigm.

    Receives assembled components and executes the training regiment
    defined by the paradigm.
    """

    execution_contract = 1

    @abstractmethod
    def run(self, cfg: Any, context: Any):
        """Execute one resolved method and return a RunResult. Never dispatch methods."""


class BaseMetaPipeline(ABC):
    """Orchestrates multiple base paradigm runs with managed data flow.

    Unlike BaseParadigmRunner (one loop), BaseMetaPipeline sequences or
    iterates over multiple base paradigm runners and manages what data and
    artifacts pass between phases.

    Note: Meta-paradigm composition is deferred. This stub exists for
    forward compatibility.
    """

    @abstractmethod
    def run(self, cfg, phases: dict, context: dict) -> None:
        """Execute the multi-phase pipeline.

        Args:
            cfg: Full experiment config.
            phases: Instantiated runners keyed by phase name.
            context: Pipeline execution context.
        """
