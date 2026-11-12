"""Offline RL paradigm components.

Provides the shared offline policy-learning runner, data module, and evaluation
protocol for training on static transition datasets.
"""

from __future__ import annotations

import logging

from src.app.core.interfaces import BaseDataModule, BaseEvalProtocol, BaseParadigmRunner
from src.app.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


from src.app.data.rl_data_module import RLDataModule


@register_component("RLReplayBufferDataModule")
class RLReplayBufferDataModule(RLDataModule):
    """Wraps the existing RLDataModule for offline RL on chunked .pkl datasets."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is not None:
            self.setup()


@register_component("OfflineRLEvalProtocol")
class OfflineRLEvalProtocol(BaseEvalProtocol):
    """Evaluation protocol for offline RL: val/loss and Bellman error.

    Offline RL evaluation is handled by the Lightning module's validation_step;
    this protocol is a no-op placeholder for the component registry.
    Actual metrics are logged by the Lightning trainer callbacks.
    """

    def evaluate(self, agent, data_source, cfg) -> dict:
        # Offline RL metrics come from Lightning's validation_step / callback_metrics.
        return {}


@register_component("OfflineRLRunner")
class OfflineRLRunner(BaseParadigmRunner):
    """Runs the offline RL training loop for each (dataset, agent) pair.

    Executes one resolved method through its declared backend.
    """

    def run(self, cfg, context):
        from src.app.core.execution import execute_backend
        return execute_backend(cfg, context)
