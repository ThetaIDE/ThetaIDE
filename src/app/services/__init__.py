"""In-process services for the desktop app: config composition, run catalog, metrics, jobs.

These used to sit behind the FastAPI server (src/app/api). The GUI now imports them directly
and runs training as a QProcess (frontend/job_manager.py), so there is no server to start.
"""

from .component_exporter import (
    ComponentExporter,
    ComponentExportOptions,
    ComponentExportResult,
    find_component_assets,
    find_workspace_root,
)
from .plugin_scaffolder import PluginScaffolder, ScaffoldOptions, ScaffoldResult
from .test_suite import (
    TEST_SUITE_FORMAT,
    ComponentContract,
    ComponentContractResult,
    TestCase,
    TestCaseResult,
    TestSuite,
    TestSuiteManifestError,
    evaluate_component_contracts,
    evaluate_test_suite,
    load_test_suite_manifest,
    parse_test_suite_manifest,
    run_pipeline_case,
)


def configure_workspace(paths):
    """Select the process workspace for an entry point (desktop host, validation worker).

    Reusable services receive `workspace=` explicitly; omitted context resolves through
    `WorkspacePaths.for_process()`, which reads the selection made here.
    """
    import os
    os.environ.update(paths.process_environment())
    from .config_service import clear_compose_cache
    clear_compose_cache()
    return paths
