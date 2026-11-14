"""Hydra discovery entry point; the implementation is the installed Optuna component's export.

Hydra discovers sweepers only under the ``hydra_plugins`` namespace. This class
forwards to the ``hydra_sweeper`` export of the enabled ``optuna`` tuner of the
process workspace, loaded only when a sweep starts.
"""

from hydra.plugins.sweeper import Sweeper


class ThetaOptunaSweeper(Sweeper):
    def __init__(self, sampler, direction, storage, study_name, n_trials, n_jobs, search_space,
                 pruner="none", pruner_startup_trials=5, pruner_warmup_steps=0, timeout_seconds=0,
                 max_failed_trials=0):
        Implementation = _implementation()

        self.impl = Implementation(sampler, direction, storage, study_name, n_trials, n_jobs, search_space,
                                   pruner, pruner_startup_trials, pruner_warmup_steps, timeout_seconds, max_failed_trials)

    def setup(self, *, hydra_context, task_function, config):
        self.impl.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments):
        return self.impl.sweep(arguments)


def _implementation():
    from src.app.core.contributions import ensure_enabled
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import provider_export
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths.for_process()
    selected = get_catalog(paths).resolve("tuner", "optuna")
    if selected is None:
        raise ModuleNotFoundError("The Optuna sweeper requires the optional Optuna component; "
                                  f"install it in the selected workspace ({paths.root})")
    package, capability = selected
    ensure_enabled(package, capability, paths=paths)
    return provider_export(package.id, "hydra_sweeper", paths=paths)
