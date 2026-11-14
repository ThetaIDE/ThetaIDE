# Scheduler component contract v1

Batch schedulers are optional native components. Slurm is one such component, maintained in `optional_components/slurm` and installed like any other package. Core never selects a scheduler by name. It owns receipts, source snapshots, publication, artifact ownership, continuation lineage and transport primitives (`src/app/sites/transport.py`). The selected scheduler owns submission, allocation validation, polling, cancellation, scheduler-specific headers/environment, setup commands and site templates.

## Selection

A site profile names its scheduler with `site.scheduler`: either `local`, or the capability ID of an installed `scheduler` capability. Selection rules:

- `site.scheduler: local` runs in the current process (also inside an allocation).
- Any other nonempty string is resolved through the provider catalog when it is used. Absent, disabled, unconfigured or incompatible schedulers raise an error naming the selector and workspace; there is no fallback.
- A non-local profile that omits `site.scheduler` is rejected. Before scheduler components existed it implied Slurm; Theta never runs it locally and never chooses an installed scheduler implicitly.
- `create(..., scheduler=…)` and `site configure SITE --scheduler ID` record the scheduler when a new site component is authored. Without one, the generated profile carries `scheduler: null` and launch reports the missing selection.

New receipts record `scheduler_component`. Receipts with `schema_version: 1` and no `scheduler_component` predate scheduler components and are read as `slurm` by a bounded reader in `src/app/sites/scheduler_api.py`. Other receipts without an explicit owner are rejected.

## Manifest

```json
{"id": "batch", "kind": "scheduler", "entry_point": "execution.py:Scheduler",
 "scheduler_api": 1, "contribution_api": 1,
 "readiness": {"config": ["site.connection.host", "site.connection.remote_root"]},
 "contributions": [
   {"id": "setup", "slot": "settings.site", "availability": "enabled", "entry_point": "settings.py:create_settings"},
   {"id": "launch", "slot": "commands.site", "availability": "ready"},
   {"id": "scheduler-monitor", "slot": "monitor.scheduler", "availability": "ready"}]}
```

`scheduler_api: 1` is required on the capability, and the loaded object (or instance of the loaded class) must have `api_version = 1`. Readiness and contributions follow [extension contributions v1](EXTENSION_CONTRIBUTIONS.md). Site profiles the component owns go in its `configs/site/…` payload; top-level forwarding files (a profile whose only content is `defaults: [/site/<site>/<variant>, _self_]`) are treated as compatibility aliases and hidden from selection lists.

## Runtime methods

| Method | Required | Purpose |
| --- | --- | --- |
| `validate_allocation(cfg, *, paths)` | yes | Reject unsupported combinations; return normalized resources |
| `submit(root, experiment, overrides, cfg)` | yes | Freeze/pin source, write a receipt with `scheduler_component`, submit |
| `run(root, execution_id)` | yes | Allocation-side worker supervision |
| `refresh_status(root, record)` / `refresh_many(root, records, *, minimum_seconds)` | yes | Reconcile scheduler state; batching is the provider's choice |
| `cancel(root, record)` | yes | Request cancellation; only reconciliation proves termination |
| `remote_requirements(site_config)` | no | `(command, None \| note)` pairs; `site check` requires each command with `None` and prints the note when an optional one is missing |
| `submission_command(config, experiment, site)` | no | Command copied by the desktop **Run** action |
| `launch_in_terminal(root, site, experiment)`, `worker_threads`, `accounting_details` | no | Provider helpers used by its own tooling |

Core dispatch lives in `src/app/sites/execution.py`. An independently named scheduler fixture uses the same path (`tests/test_optional_component_extraction.py`).

## Compatibility adapters

`src/app/sites/slurm.py`, `src/app/pipeline/slurm.py`, `src/app/sites/resources.py` and `execution.worker_threads/accounting_details` keep API-1 import names for existing scripts. They resolve only an installed, enabled `slurm` scheduler in the process workspace and raise an actionable error otherwise. They are listed in the [layer exception register](LAYER_BOUNDARIES.md#public-integration-surface-and-exceptions). Receipt storage keeps its historical `results/slurm/<execution>` directory for every scheduler, so existing jobs and synced results stay readable; a renamed layout needs a versioned receipt reader.

## Installing the Slurm component

Build archives from the maintained sources, then install them into the selected workspace:

```bash
venv/bin/python scripts/package_optional_components.py slurm optuna --output /tmp/theta-optional
venv/bin/python run_pipeline.py component --root WORKSPACE install /tmp/theta-optional/slurm-1.0.0.zip
venv/bin/python run_pipeline.py component --root WORKSPACE install /tmp/theta-optional/optuna-1.0.0.zip
```

Disable or re-enable without uninstalling with `component extension disable scheduler slurm` / `enable`. Uninstalling removes its contributions; usr overrides of its site templates remain.
