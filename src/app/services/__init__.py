"""In-process services for the desktop app: config composition, run catalog, metrics, jobs.

These used to sit behind the FastAPI server (src/app/api). The GUI now imports them directly
and runs training as a QProcess (frontend/job_manager.py), so there is no server to start.
"""

from pathlib import Path

from src.app.workspace import WorkspacePaths

PROJECT_ROOT = WorkspacePaths().root
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
JOBS_DIR = PROJECT_ROOT / "results" / "jobs"
TENSORBOARD_DIR = PROJECT_ROOT / "results" / "tensorboard"
EXPORTS_DIR = PROJECT_ROOT / "results" / "exports"

from .component_exporter import (  # noqa: E402
    ComponentExporter,
    ComponentExportOptions,
    ComponentExportResult,
    find_component_assets,
    find_workspace_root,
)
from .experiment_exporter import ExperimentExporter, ExportOptions, ExportResult  # noqa: E402
from .plugin_scaffolder import PluginScaffolder, ScaffoldOptions, ScaffoldResult  # noqa: E402
from .test_suite import (  # noqa: E402
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
