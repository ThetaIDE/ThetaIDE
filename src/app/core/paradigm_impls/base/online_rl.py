"""Online RL paradigm components.

Wraps the existing online training loop (VectorizedEnv + Lightning) for
simulator-based RL agents.
"""

from __future__ import annotations

import logging

from src.app.core.interfaces import BaseDataModule, BaseEvalProtocol, BaseParadigmRunner
from src.app.core.paradigm_loader import register_component

log = logging.getLogger(__name__)


from src.app.data.rl_data_module import RLDataModule


@register_component("SimulatorDataModule")
class SimulatorDataModule(RLDataModule):
    """Data 'source' for online RL: environment rollouts drive training."""

    def __init__(self, cfg=None):
        super().__init__(cfg)
        if cfg is not None:
            self.setup()


@register_component("EpisodicRewardEvalProtocol")
class EpisodicRewardEvalProtocol(BaseEvalProtocol):
    """Evaluation protocol for online RL: fixed-episode rollouts, mean reward.

    Actual evaluation is handled by EnvironmentEvaluatorCallback inside
    the Lightning training loop. This is a registry placeholder.
    """

    def evaluate(self, agent, data_source, cfg) -> dict:
        # Online eval is handled by EnvironmentEvaluatorCallback during trainer.fit().
        return {}


@register_component("OnlineRLRunner")
class OnlineRLRunner(BaseParadigmRunner):
    """Runs the online RL training loop for simulator-based agents.

    Executes one resolved method through its declared backend.
    """

    def run(self, cfg, context):
        from src.app.core.execution import execute_backend
        return execute_backend(cfg, context)
