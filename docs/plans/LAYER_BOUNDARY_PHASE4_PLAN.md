# Phase 4 implementation plan: retire `in/` and every compatibility adapter

Status: decisions settled 2026-11-14; implementation not started. This plan expands phase 4 of the [layer boundary plan](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md). Where they differ, that plan's [confirmed decisions](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md#confirmed-decisions) win. Track progress in the [implementation status](../development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md).

## Goal

After phase 4, `src/usr/config` holds editable configuration and `src/usr/datasets` holds dataset bytes. Packages declare config destinations as logical selectors. The codebase contains no trace of the old layout or of any compatibility adapter. The user's workspace has been moved completely: no file is left behind in `in/`.

## Decisions (settled with the user, 2026-11-14)

The user is the only user. Old patterns are not supported, partially migrated states are not wanted, and nothing is kept for behavior that no longer exists.

1. **Package format v3 only.** `storage_version: 3`. Every config destination is a logical selector relative to the config root (`model/mlp.yaml`); the same holds for `x-theta-config-schemas` keys, aliases and `user_config_paths`. v2 is removed rather than deprecated. A v2 manifest fails like any unsupported version, with no special message.
2. **No layout detection.** There is no layout version, startup check, error or dialog. After the cutover, `in/` means nothing to the IDE: a leftover `in/` is an arbitrary directory, like `arb/`.
3. **No adapter for released archives, and one Hub feed.** v2 is retired on the Hub as well. Every registry entry is rebuilt as v3 in the existing feed, the validator accepts only v3, and no frozen v2 feed is kept for older IDEs.
4. **Installed components are converted in place** while the user's workspace is migrated.
5. **The migration method is free, but nothing may be left behind.** Every file under `in/`, including ignored files and the external `mimic` symlink, is accounted for: moved, proven redundant, or confirmed junk. Every state record that names an old path is rewritten. Migration code does not ship in the product.
6. **Every compatibility adapter is retired, including the checkpoint aliases.** Checkpoints that pickled moved class paths will no longer load. Accepted.
7. **Historical results are left untouched,** with no replay guard. Replaying an old result fails wherever its old paths break.
8. **The same stance applies to later phases.** Phase 6 moves `src/app` to `src/ide` without forwarding modules, and phase 5 changes the bundle format without readers for old jobs.

## Starting point

| Area | What exists |
| --- | --- |
| `in/` in the research workspace | 107 files (26 tracked): 11 config groups plus `config.yaml` and `hydra/`; `datasets/` holds `breast_cancer`, `pyrenees`, the `prepare_tabular.py` generator and `mimic`, a symlink to an external NeSyRL dataset; `envs/` holds only `.gitkeep`. `in/datasets` is ignored by git |
| Paths | `WorkspacePaths.config` is `in/config` and `.datasets` is `in/datasets`; `user_config` (`src/usr/config`) and `user_inputs` (`src/usr/datasets`) already exist. `config_roots()` searches `src/usr/config` then `in/config` |
| Resolution | `ConfigResolver` precedence: `src/usr/config`, legacy `in/config`, selected provider profile, framework fallback. Several modules strip an `in/config/` prefix from manifest destinations |
| Package format | `storage_version: 2` manifests map `in/config/<group>/<name>.yaml` destinations to package files. `archive.validate_native_package` requires that prefix |
| Manifests on disk | 20 installed or bundled packages, 11 authored `src/usr` packages and both optional components |
| Hub | Production `ThetaIDE/ThetaHub` has 39 registry manifests (30 with `in/config` destinations). Its validator accepts only storage 2 / API 1, and its kinds exclude `tuner` and `scheduler`. Two app plugins there (AI Assistant, Theta MCP Server) are by another author |
| Reusable pieces | `src/app/workspace_inventory.py` (hashes, modes, tracked status, symlinks, intended destinations, dataset references) and `src/app/components/transactions.py` (`workspace_lock`, `assert_idle`, `FileTransaction`) |
| State keyed by old paths | `.thetaide/retained_configs`, `.thetaide/config_backups`, installer ownership records, `config_viewer.parameter_orders` and editor state in `settings.toml`, fork provenance |
| Consumers | About 45 source files reference `in/` paths. 61 test files contain 238 references, and 18 documents mention `in/config` |

## Work units

Each unit is one or more commits on `implement/layer-boundaries`. Units 4a–4e change the codebase. Unit 4f moves the user's workspace and must run in the same session as 4b, so the workspace and code never disagree for long. Unit 4g prepares the Hub. The branch merges to `main` together with the Hub publication, and only when the user asks.

### 4a — Package format v3

- `src/app/core/manifest_schema.py`: storage 3 only. Destinations are `<group>/<name>.yaml` selectors, so absolute paths, `..`, and `in/` or `src/` prefixes are rejected.
- `src/app/components/archive.py`: `validate_native_package` accepts storage 3 only. Remove every `in/config/` check from `adapt` and from the alias validation.
- Switch the consumers that strip the prefix to logical selectors: `config_resolution.py` (owner map and schema key), `core/data_loaders.py`, `components/forking.py`, `components/installer.py` (`resolve_config_path`, retained-config restore, `track_user_config`, uninstall), `components/development.py`, `components/configuration.py`, `services/component_exporter.py`, `services/experiment_exporter.py` and `services/plugin_scaffolder.py`.
- Convert the in-repo manifests: bundled packages, authored `src/usr` packages, `optional_components/*`, the fixtures under `tests/fixtures/`, and the manifest written by `scripts/verify_installed_distribution.py`. `scripts/package_optional_components.py` emits v3.

### 4b — Path cutover

- `src/app/workspace.py`: `config` becomes `src/usr/config` and `datasets` becomes `src/usr/datasets`. Remove every `in/` path from `config_roots()`, `config_file()` and `for_process()`; a workspace is recognized by `src/usr` or `.thetaide/workspace.json`. Remove the now-redundant `user_config` and `user_inputs` aliases, or keep one name per concept.
- `ConfigResolver` precedence becomes: `src/usr/config`, selected provider profile, framework fallback. Remove the "legacy workspace profile" origin.
- `hydra_plugins/theta_config_source.py`, the Hydra search path and `pipeline/datasets.py` (search order rooted at `src/usr/datasets`) drop every `in/` reference.
- `sites/policy.py`: declared inputs target `src/usr/datasets/`, and `in/inputs/` goes away. `sites/transfer.py` reads the new root. `sites/snapshot.py` `SOURCE_ROOTS` uses `src/usr/config`.
- Relocate `in/datasets/prepare_tabular.py` into the `tabular-data` provider or a bundled example package. Give it an explicit `--output` that defaults to the workspace dataset root, and update the messages in `core/paradigm_impls/base/tabular.py`.

### 4c — Retire the compatibility adapters

Delete each adapter and migrate its callers to the supported contract. Then remove the exception register's rows from [the contract](../reference/LAYER_BOUNDARIES.md), keeping only the product-scope and Hydra entries.

| Adapter | Removal |
| --- | --- |
| `src/app/core/import_aliases.py` and the `src/usr` shims | Delete. Saved Hydra targets in configs point at canonical modules. Old checkpoints that pickled moved paths stop loading (accepted) |
| Saved Optuna import targets: `src/app/tuners/optuna/`, `src/app/pipeline/{tuners_optuna,optuna_utils,optuna_sweeper}.py` | Delete. `hydra_plugins/theta_optuna_sweeper.py` stays as the Hydra discovery entry point, but loads the implementation from the installed Optuna component through the provider loader, not through `src.app.tuners` |
| API-1 Slurm adapters: `src/app/sites/slurm.py`, `src/app/pipeline/slurm.py`, `src/app/sites/resources.py`, `execution.worker_threads/accounting_details` | Delete. Callers use the installed scheduler component through the scheduler contract, and SSH shortcuts use `run_pipeline.py site ssh` |
| Schema-v1 receipt reader and `src/app/pipeline/tuner_execution_compat.py` | Delete. Receipts without `scheduler_component` or `tuner_execution` are unsupported |
| `results/slurm/<execution>` receipt directory | Rename to a scheduler-neutral `results/sites/<execution>` (`sites/execution.py`, `sites/continuation.py`, publication, sync and monitor readers). There is no reader for the old directory |
| Top-level site selector YAMLs (`arc.yaml`, `ncshare.yaml`, `ncshare_common.yaml`, `ncshare_gpu.yaml` in the Slurm component) and the alias derivation in core | Delete. User configs reference the nested profiles (`ncshare/common`); unit 4f rewrites any references |
| `provider_object(package, 'file.py:attr')` | Delete. Migrate callers to declared `exports` plus `provider_export`: `frontend/parameter_actions.py`, `frontend/tensorboard.py`, `src/app/core/provider_loader.py` internals, and the user's `blendrl-dev` package (`agents/blender_agent.py`, `symbolic_state.py`) with exports declared in its manifest |
| API-1 plotter no-argument constructors (the `BasePlotter` ContextVar scope) | Delete. Constructors take `paths`. Update `standard_figures` and any user plotters |
| `plot/` forwarding modules for saved plotting targets | Delete, and update saved targets in configs |
| `PluginContext.window`, direct `frontend.plugins.base`/`frontend.plugins.context` imports, and `frontend/plugins/core/` | Delete. Bump the SDK to **2.0**, and bundled plugins declare `>=2.0,<3`. Check the Hub's two third-party app plugins against SDK 2 in 4g |
| Legacy precedence and editor actions that target `in/` files | Delete with 4b |

Keep only these entries: Lightning and the four base paradigms (product scope); the `hydra_plugins` namespace, which Hydra requires; and the root launchers `run_pipeline.py` and `frontend/__main__.py`, which are the current entry points until phase 6 adds installed ones.

### 4d — Remaining consumers and documentation

- Sweep the remaining references: `frontend/app.py`, `config_model.py`, `config_tree.py`, `config_panel/{domain,catalogs,sections/methods}.py`, `scaffold_dialog.py`, `model.py`, `workflows/workflows_panel.py`, `pipeline/{config,compose,validation,component_validation,workflow/executor}.py`, `services/{rulesets,local_components}.py`, `sites/catalog.py`, `optional_components/slurm/execution.py`, `standard_figures/reports.py`, `sequential_spec`, `model_builder`, `arxiv_papers` and `scripts/{erase_experiment,clean_deleted}.py`. Update editor links and CLI help.
- Documentation: `PLUGIN_STORAGE.md` (v3 only), `LAYER_BOUNDARIES.md` (precedence and exception register), `SDK.md` (2.0), the 18 guides that mention `in/config`, the documentation panel content, `.gitignore` (ignore `src/usr/datasets`) and the docs index. Describe the current layout only; do not explain what used to exist.

### 4e — Tests

- One shared fixture builds a workspace in the `src/usr` layout. Port the 61 test files that build `in/config` layouts to it, in the same commit as 4b, so the suite never mixes layouts. Delete the tests that only exercised retired adapters.
- New tests:
  - v3 round trip (export, install, resolve, uninstall), and rejection of storage 2;
  - Hydra composition with root `config.yaml`, absolute defaults, interpolations and nested group references from `src/usr/config`;
  - dataset discovery under `src/usr/datasets`;
  - SDK 2 refusing plugins that declare `<2`;
  - receipts under `results/sites`.
- Distribution probe: `verify_installed_distribution.py` uses v3 and asserts that a clean external workspace never creates `in/`.
- Guard: a test fails if `in/config`, `in/datasets`, `in/envs` or `in/inputs` appears in tracked source outside tests.

### 4f — Move the user's workspace

The method is free (decision 5). Use a one-off script kept in the session scratchpad, not committed to the product. Run it in the same session as 4b. Show the user the dry-run summary and get a go-ahead before changing anything.

1. **Baseline.** Before changing anything, capture `workspace_inventory` and the composed config of every experiment.
2. **Plan every file under `in/`:**
   - **Move:** configs to `src/usr/config` with selectors unchanged, including root `config.yaml` and `hydra/`; datasets to `src/usr/datasets`, with `mimic` moved as the same symlink.
   - **Drop as redundant:** an unchanged materialized package default, only when its recorded provenance and hash match the provider file.
   - **Drop as junk:** `__pycache__`, `.DS_Store`, `in/envs/.gitkeep`.
   - **Conflict:** a destination that already exists with different bytes. A conflict stops the run for the user to decide.
   - Tracked files move with `git mv`.
3. **Backup.** Copy `in/` and every state file to be rewritten to `.thetaide/migrations/<UTC timestamp>/` (symlinks as links), and verify the hashes.
4. **Apply.** Move the files. Convert the 20 installed manifests to v3 and update their recorded hashes. Rewrite the state records named above. Rewrite site selector references to nested profiles, `results/slurm` receipts to `results/sites`, and structural dataset references in `src/usr/config`. Report arbitrary strings that mention old paths rather than guessing.
5. **Verify nothing is left behind:**
   - every inventory entry under `in/` maps to a destination with the same hash, a recorded redundant-default proof, or the junk list;
   - `in/` no longer exists;
   - `git grep` and a scan of `.thetaide` find no old paths;
   - composition and dataset selection match the baseline, apart from the documented path changes.
6. **Keep the backup** until the user deletes it.

### 4g — Hub republish preparation (no publication)

- **Reconcile the clones first.** Your local `../theta-hub` tracks the CameronEgb fork, has releases that production lacks (e.g. "Publish SB3 suite 1.0.3"), and has uncommitted feed changes. Agree with the user on the authoritative set before building.
- **Build on a branch** of a fresh clone of production `ThetaIDE/ThetaHub`:
  - make the validator accept only storage 3 and add the `tuner` and `scheduler` kinds;
  - rebuild all 39 entries, plus Optuna and Slurm, as v3 archives with new versions, replacing the v2 entries in the single feed;
  - run `validate_manifest.py` and `build_index.py`.
- **Third-party app plugins.** Check the two plugins by the other author against SDK 2. If they use `context.window` or private imports, tell the user before rebuilding them.
- **Test installs.** Install every rebuilt archive into an isolated workspace with the branch IDE.
- **Publish only on request,** following the [production release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md), together with the merge to `main`.

## Exit gate

- **Clean code.** No tracked source outside tests mentions `in/` paths, storage 2, or any retired adapter module; the guard test enforces this. The exception register lists only Lightning and the base paradigms, the Hydra namespace and the root launchers.
- **Moved workspace.** The user's workspace passes every check in 4f step 5 and no longer has `in/`.
- **Same results.** Composition and dataset selection match the baseline in the GUI, the CLI and workers, from an unrelated working directory.
- **User edits survive.** Edits in `src/usr/config` survive package update and uninstall.
- **Distribution.** The installed-distribution probe passes with v3.
- **Hub ready.** Every rebuilt Hub archive installs in an isolated workspace, waiting for the publication request.

## Risks

- **Hub timing.** The Hub feed and `main` must change together. Between the merge and the publication, the IDE can install nothing from the Hub.
- **Old artifacts.** Old checkpoints, receipts and unfinished cluster jobs stop working by decision. Before 4f, confirm that no site job is still running.
- **Wide test churn.** The fixture port touches 61 files. Land it with 4b in one commit.
- **The other author's app plugins.** SDK 2 may break them. Coordinate before rebuilding.
