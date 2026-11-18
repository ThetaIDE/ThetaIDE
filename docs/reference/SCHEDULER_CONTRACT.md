# Scheduler component contract v2

Batch schedulers are native components installed from Theta Hub. Slurm is one such component; its source lives in the theta-plugins repository under `schedulers/slurm`. Core never selects a scheduler by name. It owns receipts, [bundles](#bundles), publication, artifact ownership, continuation lineage and transport primitives (`src/app/sites/transport.py`). The selected scheduler owns submission, allocation validation, polling, cancellation, scheduler-specific headers/environment and setup commands. Schedulers ship no site profiles: each site is a package the user authors under `src/usr/sites/SITE`.

## Selection

A site profile names its scheduler with `site.scheduler`: either `local`, or the capability ID of an installed `scheduler` capability. Selection rules:

- `site.scheduler: local` runs in the current process (also inside an allocation).
- Any other nonempty string is resolved through the provider catalog when it is used. Absent, disabled, unconfigured or incompatible schedulers raise an error naming the selector and workspace; there is no fallback.
- A non-local profile that omits `site.scheduler` is rejected. Theta never runs it locally and never chooses an installed scheduler implicitly.
- `create(..., scheduler=…)` and `site configure SITE --scheduler ID` record the scheduler when a new site component is authored. Without one, the generated profile carries `scheduler: null` and launch reports the missing selection.

Receipts record `scheduler_component`; a receipt without one is rejected.

## Manifest

```json
{"id": "batch", "kind": "scheduler", "entry_point": "execution.py:Scheduler",
 "scheduler_api": 2, "contribution_api": 1,
 "readiness": {"config": ["site.connection.host", "site.connection.remote_root"]},
 "contributions": [
   {"id": "setup", "slot": "settings.site", "availability": "enabled", "entry_point": "settings.py:create_settings"},
   {"id": "launch", "slot": "commands.site", "availability": "ready"},
   {"id": "scheduler-monitor", "slot": "monitor.scheduler", "availability": "ready"}]}
```

`scheduler_api: 2` is required on the capability, and the loaded object (or instance of the loaded class) must have `api_version = 2`. Version 2 replaced source snapshots with bundles; a version-1 scheduler is refused. Readiness and contributions follow [extension contributions v1](EXTENSION_CONTRIBUTIONS.md). Site profiles the component owns go in its `configs/site/<site>/…` payload; `_`-prefixed files there are shared defaults and are not offered for selection.

## Runtime methods

| Method | Required | Purpose |
| --- | --- | --- |
| `validate_allocation(cfg, *, paths)` | yes | Reject unsupported combinations; return normalized resources |
| `submit(root, experiment, overrides, cfg)` | yes | Build or pin the experiment's bundle, write a receipt with `scheduler_component` and `bundle_sha256`, submit |
| `run(root, execution_id)` | yes | Allocation-side worker supervision |
| `refresh_status(root, record)` / `refresh_many(root, records, *, minimum_seconds)` | yes | Reconcile scheduler state; batching is the provider's choice |
| `cancel(root, record)` | yes | Request cancellation; only reconciliation proves termination |
| `remote_requirements(site_config)` | no | `(command, None \| note)` pairs; `site check` requires each command with `None` and prints the note when an optional one is missing |
| `submission_command(config, experiment, site)` | no | Command copied by the desktop **Run** action |
| `launch_in_terminal(root, site, experiment)`, `worker_threads`, `accounting_details` | no | Provider helpers used by its own tooling |

Core dispatch lives in `src/app/sites/execution.py`. Independently named scheduler fixtures use the same path (`tests/test_scheduler_components.py`).

## Bundles

A remote job runs from a bundle built from its composed experiment (`src/app/sites/bundle.py`, schema `theta.bundle/v1`). `.thetaide/bundle.json` lists every file with its digest, plus the experiment selector, overrides and site; the application name, version and digest; each selected provider (ID, version, origin, digest, storage and API versions, capability contract versions, bundle path); each configuration file the composition resolved (selector, owner, origin, digest); the site policy's declared inputs; and requirements (`application` from the source tree's `pyproject.toml` or the installed distribution's metadata, `providers` from the selected packages' `dependencies.pip`; `.thetaide/requirements.txt` lists both, so `pip install -r` sets up a remote environment). `bundle_sha256` is the digest over the file map.

Contents: the application (`run_pipeline.py`, packaging metadata, `hydra_plugins/`, `src/app/`, the bundled component packages and inventory), the selected installed and authored packages with their declared component dependencies, the workspace configuration outside `experiment/` plus the experiment files the composition read, authored rules, and `.thetaide/{workspace,provider_overrides,extensions}.json`. Never included: `frontend/`, app plugins, `src/usr/results`, `src/usr/datasets`, other experiments, unselected packages and the rest of `.thetaide/`. Verification rejects any other schema, a changed or missing file, and a listed path in an excluded tree.

Schedulers use the SDK (`src.app.api.sites`): `prepare_bundle(root, experiment, overrides, destination)` composes, builds and checks that the experiment composes inside the bundle; `current_bundle(root)` is the bundle `site push` activated (archived under `BUNDLE_ARCHIVES` by digest); `verify_bundle(directory)` and `extract_bundle(archive, destination)` return a verified manifest. `BUNDLE_REQUIREMENTS` is the requirements file's bundle path.

## Receipts

Receipts live in `src/usr/results/sites/<execution>` for every scheduler. Each records the `scheduler_component` that owns it and its `tuner_execution` plan (or `null`); a receipt without either is refused. Scripts use the installed scheduler through these runtime methods; SSH shortcuts use `run_pipeline.py site ssh`.

## Installing a scheduler

Install **Slurm scheduler** from Components → Hub, or install a locally built archive (built in theta-plugins with `scripts/package_plugins.py --only slurm`):

```bash
venv/bin/python run_pipeline.py component --root WORKSPACE install slurm-3.0.0.zip
```

Then author a site that selects it, for example `run_pipeline.py site configure cluster --scheduler slurm --host ALIAS --remote-root PATH`. Disable or re-enable without uninstalling with `component extension disable scheduler slurm` / `enable`. Uninstalling removes its contributions; authored sites remain.

IDE tests exercise this contract with the fixture `batch-scheduler` and `template-site` packages in `tests/fixtures/`; Slurm's own tests live beside its source in theta-plugins.
