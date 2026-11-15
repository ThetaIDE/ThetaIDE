"""Read-only access to the selected workspace's run results (``WorkspacePaths.results``).

``list_runs``      every run under the results logs, however it was launched, with agent summaries.
``run_metrics``    the rows of one agent's metrics, from ``since`` on, for following a live run.
``run_manifest``   every file a run produced, with its kind, agent and version.
``read_run_file``  one file a run's manifest lists, as bytes; nothing outside the run is readable.

Each takes an optional ``workspace`` (a ``WorkspacePaths`` or a root); the process workspace is the default.
Writing, renaming and deleting results stay with the host.
"""
from ._exports import lazy_exports

_EXPORTS = {
    "list_runs": "src.app.services.catalog:list_runs",
    "run_metrics": "src.app.services.catalog:run_metrics",
    "run_manifest": "src.app.services.catalog:run_manifest",
    "read_run_file": "src.app.services.catalog:read_run_file",
}
__all__ = sorted(_EXPORTS)
__getattr__, __dir__ = lazy_exports(__name__, _EXPORTS)
