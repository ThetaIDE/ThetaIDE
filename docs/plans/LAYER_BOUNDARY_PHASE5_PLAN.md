# Phase 5 implementation plan: results in the user layer and explicit remote bundles

Status: 5a and 5b done 2026-11-15; 5c code stabilized and verified (2,636 passed, 52 skipped, 126 existing failures, no new failure IDs against 5b), the move of the user's `results/` waits for their yes; 5d–5e not started. This plan expands phase 5 of the [layer boundary plan](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md#phase-5--results-in-the-user-layer-and-remote-bundles); where they differ, its [confirmed decisions](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md#confirmed-decisions) win. Track progress in the [implementation status](../development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md).

## Goal

After phase 5, every run output (metrics, logs, checkpoints, plots, TensorBoard events, Hydra output, tuning state, site receipts and publications) lives under `src/usr/results` of the selected workspace, and every path to it comes from `WorkspacePaths.results`. A remote job runs from an explicit bundle: the application build, the providers the experiment selects with their dependencies, and snapshots of the authored sources and configuration it uses, described by one versioned manifest. Nothing is copied because it happens to sit in the checkout.

## Decisions

These follow the confirmed decisions; none needs a new user decision.

1. **One results root.** `WorkspacePaths.results` is `src/usr/results`. It is the only definition; the unused `results_dir` key in `config.yaml` is deleted. No code joins `results/` to a working directory, the application root or a guessed root.
2. **The working directory is never a results root.** Hydra's `run.dir` and `sweep.dir` resolve through a `theta_results` OmegaConf resolver backed by `WorkspacePaths.for_process()`. Lightning, telemetry, callbacks and runners receive the workspace explicitly or call `WorkspacePaths.for_process()`.
3. **No readers for the old location.** After the move, a leftover top-level `results/` is an arbitrary directory. Nothing detects it, warns about it or reads it.
4. **The user's `results/` moves completely** with the 4f method: inventory every file (ignored ones included), timestamped backup under `.thetaide/migrations/`, a dry run shown to the user, then the move with hash checks of every file. Conflicts stop the move; nothing picks a winner. Migration code does not ship (it runs from a scratch script).
5. **Bundles name their contents.** A bundle is built from the composed experiment, not from directory walks of the checkout. Unselected experiments, providers, datasets and results are not in it.
6. **One bundle manifest version.** `schema: "theta.bundle/v1"`. Extraction, execution and continuation accept exactly that schema. Older receipts, snapshots and jobs fail like any unsupported version.
7. **Requirements are declared.** Remote setup installs the application's requirements plus the `dependencies.pip` of the selected providers (scheduler and tuner included). It never adds a package because a particular tuner exists.
8. **Application and workspace stay separate.** Application files come from `WorkspacePaths.application`; authored packages, configuration and inputs come from `WorkspacePaths.root`. When they coincide (development checkout) the bundle still lists them in their own sections.

## Bundle manifest (`theta.bundle/v1`)

Written to `.thetaide/bundle.json` inside the bundle and recorded by digest in the execution receipt.

| Field | Content |
| --- | --- |
| `schema` | `"theta.bundle/v1"` |
| `bundle_sha256` | Digest over the sorted `files` map |
| `project_id` | `workspace_id(root)` |
| `created_at` | UTC timestamp |
| `experiment` | Selector and the launch overrides |
| `application` | Distribution name, version, and the digest of its file set |
| `providers` | One entry per selected package and dependency: `id`, `version`, `origin` (`bundled`, `hub`, `local_archive`, `user`), `digest`, `storage_version`, `api_version`, the selected capabilities with their contract versions (`execution_contract`, `scheduler_api`, tuner and contribution API) |
| `configs` | Each configuration file the composition read: selector, owner, origin, digest, path in the bundle |
| `inputs` | Each declared data input: name, kind, workspace-relative path, digest or remote mapping. Dataset bytes are not in the bundle |
| `requirements` | Python requirement strings from the application and the selected providers |
| `files` | Bundle-relative path to sha256 for every file |

## What a bundle includes and excludes

Included:

- The application build: `run_pipeline.py`, `hydra_plugins/`, `src/app/` (with `config_defaults/`), `src/plugins/inventory.json` and the bundled component packages the experiment selects. The desktop (`frontend/`) and app plugins are not needed by workers and are excluded; if a worker import proves otherwise, that import is the bug.
- Each selected installed package and its declared component dependencies, as installed under `src/plugins/components/installed/<id>`.
- Each selected authored package under `src/usr` and its dependencies.
- The `src/usr/config` files the composition read (experiment chain, group defaults, overrides of package profiles), plus `.thetaide/provider_overrides.json`, `.thetaide/workspace.json` and `.thetaide/extensions.json`.

Excluded: `src/usr/results`, `src/usr/datasets` (data has its own transfer and mappings), unselected providers and experiments, the rest of `.thetaide/`, virtual environments, caches, VCS data, tests and docs.

## Results paths

Every results path is `WorkspacePaths.results` or derived from it (unit 5b); portable strings use `WorkspacePaths.results_relative`. Site paths name the remote layout as well; the remote workspace is a workspace like any other, so its results are also `src/usr/results` and they move together with the bundle format.

## Units

| Unit | Scope | Exit check |
| --- | --- | --- |
| 5a | This plan; docs index | Plan listed in `docs/README.md` |
| 5b | Every results path through `WorkspacePaths.results`; Hydra dirs through `theta_results`; delete `results_dir` | No literal `results/` path in core, components, scripts or authored runners (a guard test enforces it) |
| 5c | `WorkspacePaths.results` becomes `src/usr/results`; ignores, the test guard, monitors, results browser, site publication and receipts follow; the user's `results/` moves (backup, hash check, dry run shown first, the one stop) | Fixture runs write only under `src/usr/results`; the user's files are byte-identical after the move |
| 5d | Explicit bundles and the `theta.bundle/v1` manifest replace `snapshot.freeze`'s directory walk; extraction and continuation verify the manifest | An external workspace prepares and runs a bundle through the local and fixture schedulers; unselected providers are absent; continuation verifies provenance |
| 5e | Declared requirements for remote setup from the selected providers | Setup for an experiment without a tuner installs no tuner dependency |

Exit gate (local, no cluster run): an external workspace runs a reproducible bundle through the local and fixture schedulers with the installed IDE, selected providers and declared data mappings; continuation verifies provenance; nothing writes results outside `src/usr/results`.
