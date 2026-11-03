"""Hydra discovery shim; the Optuna implementation is loaded only for a sweep."""

from hydra.plugins.sweeper import Sweeper


class ThetaOptunaSweeper(Sweeper):
    def __init__(self, sampler, direction, storage, study_name, n_trials, n_jobs, search_space,
                 pruner="none", pruner_startup_trials=5, pruner_warmup_steps=0, timeout_seconds=0,
                 max_failed_trials=0):
        from src.app.tuners.optuna.sweeper import ThetaOptunaSweeper as Implementation

        self.impl = Implementation(sampler, direction, storage, study_name, n_trials, n_jobs, search_space,
                                   pruner, pruner_startup_trials, pruner_warmup_steps, timeout_seconds, max_failed_trials)

    def setup(self, *, hydra_context, task_function, config):
        self.impl.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments):
        return self.impl.sweep(arguments)
