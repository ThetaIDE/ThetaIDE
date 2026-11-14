# Theta-IDE pipeline and component service interface

This document describes shared headless component/configuration services and the detached training runner. Configuration composition and validation run in-process; the runner owns training child processes. There is no HTTP server to start.

These are host-internal service interfaces for the desktop and CLI. Components and app plugins import host functionality only through the [Theta SDK](SDK.md) (`src.app.api`).

---

## 1. Quick Start for Team Members

### 1.1 Development Setup

```bash
# 1. Clone repository
git clone https://github.com/ThetaIDE/ThetaIDE.git Theta-IDE
cd Theta-IDE

# 2. Virtual environment setup
python3 -m venv venv
source venv/bin/activate

# 3. Install core dependencies + dev tools + the desktop app
pip install -e ".[dev,gui]"

# 4. Verify test suite
pytest tests/ -v
```

### 1.2 Checking an existing experiment

Install the experiment's components from the Hub and choose an actual recipe under `in/config/experiment/`.
The following names are placeholders for a recipe already installed in your checkout:

```bash
python run_pipeline.py group/experiment dry_run=true
python run_pipeline.py component validate-experiment group/experiment
```

Simulator recipes and offline-data recipes have different inputs. Obtain datasets required by the selected
recipe and configure their paths; historical MIMIC/Pyrenees groups and private preprocessing scripts are
not guaranteed to exist in a clean checkout. `results/` and local datasets are generally gitignored.

---

## 2. Running ThetaIDE

```bash
python -m frontend          # from the repository root, with the venv active
```

ThetaIDE runs from the same venv as the pipeline. It imports the pipeline's composition and validation code
directly and launches `run_pipeline.py` as a child process, so there is nothing else to start.

---

## 3. Interface Reference

Everything the GUI used to request over HTTP is now a direct call. The pure-Python services live in
`src/app/services/` and have no Qt dependency, so scripts and tests can use them too. The Qt side lives in
`frontend/job_manager.py`, `frontend/local_backend.py` and the built-in plugins. `frontend/tensorboard.py`
is a compatibility import for `src/plugins/app/bundled/tensorboard/`.

### 3.1 Services (`src/app/services/`)

| Function | Returns | Replaces |
|---|---|---|
| `config_service.config_schema()` | Health/catalog information for an available valid recipe, or a clean-setup composition check; it does not synthesize the editor's parameter forms | former `GET /api/config/schema` |
| `config_service.compose_report(experiment, overrides)` | `{valid, argv, config, config_yaml, methods, methods_yaml, notices, errors}`, composed exactly as `run_pipeline.py` would | `POST /api/config/compose` |
| `config_service.prepare_launch(experiment, overrides, overwrite, active_jobs)` | The job description (group, experiment ID, agents, effective timesteps); raises `LaunchError` (`.conflict` for existing results or a clashing job) | launch validation |
| `catalog.list_runs()` | Every `results/logs/<group>/<id>` run folder, including nested IDs, with config summary and per-agent timing/reward | `GET /api/runs` |
| `catalog.run_metrics(group, experiment_id, agent)` | All rows of the newest `metrics.csv`; raises `FileNotFoundError` | `GET /api/runs/.../metrics` |
| `catalog.list_plots(group, experiment_id)` | Plot file names under `results/plots/` | `GET /api/runs/.../plots` |
| `catalog.list_environments()`, `list_experiments()`, `list_methods()` | Config catalog | `GET /api/environments`, `/experiments`, `/methods` |
| `metrics.MetricsTail(path).read()` | `(new rows, reset)`: only rows appended since the last read; `reset` when Lightning rewrote the file | incremental telemetry |
| `processes.kill_process_tree(pid)` | Stops the pipeline and every training/plotting subprocess it started | cancel |
| `job_store.JobStore(path)` | SQLite persistence for jobs and queue order | unchanged |

Run identities preserve the recipe hierarchy: `experiment/group/path/dir/run.yaml`
resolves to group `group` and experiment ID `path/dir/run`. Logs, checkpoints,
plots, and execution records retain those subdirectories below their respective
`results/<kind>/group/` roots. A custom leaf `experiment_id` or timestamp changes
the final directory name while preserving the recipe's parent subdirectories;
an ID containing `/` is treated as an explicit path relative to the group.
This resolution is shared by GUI and CLI launches and forwarded to every runner,
including SB3. Existing result folders are discovered in place and are not moved.

### 3.2 Async calls from the GUI (`frontend/local_backend.py`)

`LocalBackend` runs `config_schema`, `compose`, `list_runs` and `run_metrics` on one background thread
(Hydra's global state is not thread-safe, so calls never overlap) and calls back on the GUI thread with
`(data, error)`, the same shape the old network client used.

### 3.3 Training jobs (`frontend/job_manager.py`)

`frontend/job_manager.py` is the desktop client of `frontend/job_runner.py`. The detached runner uses
`frontend/runner_manager.py` to own training `QProcess` instances, queue persistence and resource scheduling.
It launches the existing pipeline command from the project root. The client sends requests through
file-based IPC, polls durable runner state and tails logs/metrics; the runner also uses Qt process signals
and filesystem watching. Closing the desktop does not itself cancel training.

Resource admission happens before each training worker (one seed) starts. The worker keeps its memory
reservation until it exits; observed growth increases the reservation and updates estimates for future
starts. Soft CPU/RAM targets can delay queued workers, but never suspend or automatically restart an
admitted worker. Actual process failures are reported as failures, and explicit cancellation still stops
the worker. A multi-seed trial may have pending seeds while its admitted seeds continue training.

| Method | Behaviour |
|---|---|
| `launch(experiment, overrides, queue=False, overwrite=False)` | Validates via `prepare_launch`, then starts the job (or queues it). Raises `LaunchError`. |
| `cancel(job_id)` | Queued: removed from the queue. Running: kills the process tree; the job ends as `cancelled`. |
| `start_queue()` / `pause_queue()` | Starts/stops admitting queued work. Resource budgets determine concurrency; already running jobs continue when queue admission pauses. |
| `move(job_id, position)` | Reorders a queued job (raises `LaunchError` for a job that is not queued). |
| `queue_state()` | `{running, active, queued (with position), finished (ten most recent)}` |
| `job(job_id)`, `active_jobs()`, `is_busy()` | Current records. |
| `shutdown()` | Disconnects the desktop client; the detached runner and queue keep running. Use cancellation to stop jobs. |

| Signal | Arguments |
|---|---|
| `job_changed` | the job's public record after any status change (`pending` → `running` → `completed` / `failed` / `cancelled` / `error`) |
| `job_output` | `job_id`, new log lines (also written to `<data dir>/jobs/<job_id>.log`) |
| `job_metrics` | `job_id`, agent, new metric rows, `reset` |
| `job_hardware` | `job_id`, `{cpu_percent, memory_mb}`, sampled at most once a second while output arrives |
| `queue_changed` | the same dictionary as `queue_state()` |

Jobs belong to the detached runner. The desktop can reconnect to persisted jobs after reopening; queued
work is durable. On runner recovery, stale jobs whose processes are no longer active may be marked
`interrupted`. The runner can exit after its idle grace period when no work or clients remain.

### 3.4 TensorBoard (`frontend/tensorboard.py`)

`TensorBoardServer` runs `python -m tensorboard.main --logdir results/tensorboard` on a free local port. It
emits `ready(url)` when TensorBoard prints that it is serving and `stopped(error)` when it exits (`""` after
a deliberate `stop()`).

### 3.5 Headless component development

`src/app/workspace.py` defines `WorkspacePaths` for the current checkout layout. It does not create,
switch or migrate standalone research workspaces. Component implementations live under
`src/app/components/`; frontend installer/model/dependency imports are compatibility aliases.

| Interface | Purpose |
| --- | --- |
| `ComponentDevelopment.inspect(reference)` / `list()` | Installed identity, source, profiles, dependencies and static registration inspection |
| `ComponentDevelopment.copy(reference, new_id, name=None)` | Independent copy of a supported directory-based model |
| `ConfigurationFiles.read/copy/configure/write` | Scoped configuration I/O, exclusive copies, atomic saves and Hub ownership checks |
| `ComponentDevelopment.validate_profile(path)` | YAML/static model identity and opted-in schema checks |
| `ComponentDevelopment.validate_experiment(experiment, overrides)` | Pipeline preflight and effective method model source reporting |
| `HubInstaller` | Headless installation, update and uninstall |
| `LocalComponents` | Collection snapshots, context and visibility; system Trash is supplied by the desktop |

Run `python run_pipeline.py component --help` or `python -m src.app.components --help` for the terminal
adapter. See [Component development](../guides/COMPONENT_DEVELOPMENT.md) for concrete commands, copy limitations,
storage paths and lifecycle semantics. Hub originals are read-only; edit an independent local copy.

## Provider ownership and execution

`WorkspacePaths` separates distribution assets from selected-workspace installed/user content. `ProviderCatalog`/`get_catalog` list metadata and resolve kind-qualified capabilities; `resolve_provider`, `resolve_resource`, and `environment_resource` load selected implementations/resources. The canonical agent registry is `src.app.core.agent_registry`, with compatible shared-object re-exports. See [native storage, exports, leases and worker contracts](PLUGIN_STORAGE.md).

## Single-method learning runtime

See [execution contract v1](EXECUTION_CONTRACT.md) for registered runner invocation,
backend selection, execution features, selected-provider preflight/resource hooks,
cancellation/results, removed selectors and editable migration.
`BaseParadigmRunner.run(cfg, context)` executes one method; experiment and queue
orchestration remain in the existing coordinator services.

## Portable tuner execution

Tuning requires an explicit nonempty string `tuning.component` capability selector.
`tuning: null` disables tuning; an empty mapping remains an unselected, invalid
sweep. `tuner_name(cfg)` and `get_tuner(cfg, paths=paths)` never guess a provider,
even if exactly one is installed. `get_tuner(name=selector, paths=paths)` explicitly
loads an enabled implementation for setup. Missing, blank and non-string selectors
fail before provider resolution. Capability aliases and case use normal catalog
resolution; portable plans record the canonical capability ID without editing the
recipe. See the [contribution contract](EXTENSION_CONTRIBUTIONS.md) for saved-recipe
diagnostics and the [portable contract](TUNER_EXECUTION_CONTRACT.md) for the bounded
historical receipt adapter.

The optional remote extension of the local tuner API is
`PortableTuner.remote_api_version = 1`. Supported types and publication helpers
are exported by `src.app.pipeline.tuner_execution`; see the
[portable tuner execution contract](TUNER_EXECUTION_CONTRACT.md) for plan, binding,
state snapshot, continuation and monitor interfaces. Core transports and verifies
artifacts while the selected provider owns state formats and trial repair.

## Extension contributions

`src.app.core.contributions` provides metadata-only `ContributionRegistry`,
workspace enablement/readiness, explicit factory loading and `ContributionScope`
lifecycle reconciliation. See [extension contributions v1](EXTENSION_CONTRIBUTIONS.md)
for supported slots, setup/operational states, cleanup and current host coverage.
