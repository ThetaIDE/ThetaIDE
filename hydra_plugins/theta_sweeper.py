"""Hydra discovery entry point for every tuner's sweeper.

Hydra discovers sweepers only under the ``hydra_plugins`` namespace. A tuner's
sweeper profile targets this class with its own ``tuner`` ID; the class forwards
every other setting to that enabled tuner's ``hydra_sweeper`` export in the process
workspace, loaded only when a sweep starts.
"""

from hydra.plugins.sweeper import Sweeper


class ThetaSweeper(Sweeper):
    def __init__(self, tuner, **options):
        self.impl = _implementation(tuner)(**options)

    def setup(self, *, hydra_context, task_function, config):
        self.impl.setup(hydra_context=hydra_context, task_function=task_function, config=config)

    def sweep(self, arguments):
        return self.impl.sweep(arguments)


def _implementation(tuner):
    from src.app.core.contributions import ensure_enabled
    from src.app.core.provider_catalog import get_catalog
    from src.app.core.provider_loader import provider_export
    from src.app.workspace import WorkspacePaths
    paths = WorkspacePaths.for_process()
    selected = get_catalog(paths).resolve("tuner", str(tuner))
    if selected is None:
        raise ModuleNotFoundError(f"The sweeper of tuner {tuner!r} requires that tuner's component; "
                                  f"install it in the selected workspace ({paths.root})")
    package, capability = selected
    ensure_enabled(package, capability, paths=paths)
    return provider_export(package.id, "hydra_sweeper", paths=paths)
