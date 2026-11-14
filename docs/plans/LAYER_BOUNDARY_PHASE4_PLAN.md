# Phase 4 implementation plan: retire `in/`

Status: proposed 2026-11-14, not started. This plan expands phase 4 of the [layer boundary plan](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md). Where they differ, the scope decisions in that plan's [confirmed decisions](LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md#confirmed-decisions) win. Track progress in the [implementation status](../development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md).

## Goal

After phase 4, a workspace keeps editable configuration in `src/usr/config` and dataset bytes in `src/usr/datasets`, and nothing reads or writes `in/`. Packages declare config destinations as logical selectors rather than `in/config/...` paths. One command moves an existing workspace to the new layout.

## Scope decisions (2026-11-14)

- **Hard cutover.** After the cutover the IDE reads only `src/usr`. A workspace that still has the old layout fails at every entry point with one actionable error naming the migrate command. There are no compatibility reads or diagnostics for `in/`.
- **No adapter for released archives.** Storage-v2 archives are rejected with a "needs a newer release" error. Hub components are rebuilt in the new format and republished, only when a release is requested.
- **Backup, then manual restore.** The command writes a complete timestamped backup before it moves anything. There is no `--rollback`. The dry-run reports the full inventory and collisions and stops on any conflicting changed file, and an interrupted run is safe to rerun.
- **Historical results are records.** Saved results, receipts and run configs keep their `in/` paths. Replaying or resuming one stops with an error naming the old path.

## Starting point

| Area | What exists |
| --- | --- |
| `in/` in the research workspace | 107 files (26 tracked): 11 config groups plus `config.yaml` and `hydra/`; `datasets/` holds `breast_cancer`, `pyrenees`, the `prepare_tabular.py` generator and `mimic`, a symlink to an external NeSyRL dataset; `envs/` holds only `.gitkeep`. `in/datasets` is ignored by git |
| Paths | `WorkspacePaths.config` is `in/config` and `.datasets` is `in/datasets`; `user_config` (`src/usr/config`) and `user_inputs` (`src/usr/datasets`) already exist. `config_roots()` searches `src/usr/config` then `in/config` |
| Resolution | `ConfigResolver` precedence: `src/usr/config` override, legacy `in/config`, selected provider profile, framework fallback. `config_resolution.py`, `data_loaders.py` and schema lookup strip the `in/config/` prefix from manifest destinations |
| Package format | Manifests with `storage_version: 2` map `payload.configs` destinations `in/config/<group>/<name>.yaml` to package files. `x-theta-config-schemas` keys, aliases and `user_config_paths` use the same prefix. `archive.validate_native_package` requires that prefix and `storage_version == 2` |
| Manifests on disk | 20 installed or bundled packages, 11 authored `src/usr` packages and both optional components use `in/config` destinations |
| Hub | Production `ThetaIDE/ThetaHub` has 39 registry manifests, 30 of them with `in/config` destinations. Its validator only accepts storage 2 / API 1, and its kinds exclude `tuner` and `scheduler`. Each registry entry carries one top-level `storage_version`, and clients read `dist/index.json` |
| Primitives to reuse | `src/app/workspace_inventory.py` (hashes, modes, tracked status, symlinks, intended destinations, dataset references) and `src/app/components/transactions.py` (`workspace_lock`, `assert_idle`, `FileTransaction`) |
| State keyed by old paths | `.thetaide/retained_configs`, `.thetaide/config_backups`, installer ownership records, `config_viewer.parameter_orders` in `settings.toml`, fork provenance |
| Consumers | About 45 source files reference `in/` paths (`git grep -E "in/config\|in/datasets"`). 61 test files contain 238 references, and 18 documents mention `in/config` |

## Decisions to settle before coding

Ask the user about each of these; the recommendation comes first.

1. **Destination representation.** *Recommended:* `storage_version: 3`, where every config destination is a logical selector path relative to the config root (`model/mlp.yaml`). The same applies to `x-theta-config-schemas` keys, aliases and `user_config_paths`. The workspace layout version is stored separately, as `layout_version` in `.thetaide/workspace.json`. *Alternative:* keep version 2 with a new `config_destinations` field (two shapes at once, so it is not recommended).
2. **Installed v2 packages in a workspace being migrated.** *Recommended:* the migration translates the workspace's own installed copies in place: strip the prefix, set `storage_version: 3`, update the installer's recorded hashes, and include the originals in the backup. These are managed copies, not downloaded archives. *Alternative:* require reinstalling from the v3 Hub feed, which makes migration depend on the Hub being published first.
3. **Hub feed strategy.** *Recommended:* a new feed, `dist/v3/index.json`, that the new IDE's `DEFAULT_REGISTRY_URL` points to. The current `dist/index.json` stays frozen for already-released IDEs, so they keep installing v2 releases. The validator accepts storage 3 and the `tuner` and `scheduler` kinds only in the v3 registry. *Alternative:* a minimum-IDE-version field in one feed, which older clients ignore.
4. **Desktop behavior on an unmigrated workspace.** *Recommended:* a blocking dialog that explains the change, shows the dry-run report and offers **Back up and migrate**, with the CLI command shown for copying. *Alternative:* an error dialog with the command only.

## Work units

Each unit is one or more commits on `implement/layer-boundaries`. Units 4a–4g land before the branch merges to `main`, because the cutover ships as a whole. Unit 4h prepares the Hub releases without publishing them. Unit 4i runs only with the user's explicit approval.

### 4a — Package format v3 (logical config destinations)

- `src/app/core/manifest_schema.py`: storage 3 schema. Destinations are `<group>/<name>.yaml` selectors, so absolute paths, `..`, and `in/` or `src/` prefixes are rejected.
- `src/app/components/archive.py`: `validate_native_package` accepts only v3. For v2 it raises: "<id> <version> was built for the old workspace layout; install a newer release from Theta Hub." Remove the `in/config/` checks in `adapt` and the alias validation.
- Strip-prefix consumers switch to logical selectors: `config_resolution.py` (owner map and schema key), `core/data_loaders.py`, `components/forking.py`, `components/installer.py` (`resolve_config_path`, retained-config restore, `track_user_config`, uninstall), `services/component_exporter.py`, `services/experiment_exporter.py`, `services/plugin_scaffolder.py`, `components/development.py` and `components/configuration.py`.
- Convert the in-repo manifests: bundled packages, `src/usr` authored packages, `optional_components/*`, test fixtures under `tests/fixtures/`, and the manifest written by `scripts/verify_installed_distribution.py`. `scripts/package_optional_components.py` emits v3.
- Tests: v3 round trip (export, install, resolve, uninstall), v2 rejection with the message, and the schema-key lookup.

### 4b — Workspace layout v2 and path cutover

- `src/app/workspace.py`: `config` becomes `src/usr/config` and `datasets` becomes `src/usr/datasets`. Drop `in/config` from `config_roots()` and `config_file()`. `for_process()` recognizes a workspace by `src/usr` or `.thetaide/workspace.json`. Add `layout_version` (read from `workspace.json`; 1 when it is absent and `in/` exists) and `WorkspaceLayoutError`.
- Check the layout once at each entry point: the desktop host (with the decision 4 dialog), `run_pipeline.py`, the train, task and provider workers, the validation worker and the component CLI. Reusable services do not check; an entry point fails before any service runs.
- `ConfigResolver` precedence becomes: `src/usr/config` override, selected provider profile, framework fallback. The "legacy workspace profile" origin is removed.
- `hydra_plugins/theta_config_source.py` and the search path no longer mention `in/config`.

### 4c — Migration command

- `python run_pipeline.py workspace migrate --root PATH [--apply]`. Without `--apply` it is a dry run. Desktop decision 4 calls the same service.
- **Plan.** Build on `workspace_inventory.inventory(paths)`. Each file under `in/` gets an action: `move`, `dedupe` (an unchanged materialized package default whose recorded provenance and hash match the provider file), `drop` (`__pycache__`, `.DS_Store`, `in/envs/.gitkeep`) or `conflict`.
- **Conflicts.** A destination that already exists with different bytes, two sources for one destination, or an unrecorded or modified materialized default that would change effective resolution. Any conflict stops `--apply`. The report names both files and their hashes. The command never picks a winner.
- **Preconditions.** Hold `workspace_lock` and pass `assert_idle` (no running local jobs or active implementations). Refuse when an unfinished site job is recorded under `results/slurm`, listing the jobs so the user can finish or abandon them.
- **Backup.** Before any move, copy `in/` (including ignored files) and every state file the migration rewrites to `.thetaide/migrations/<UTC timestamp>/`. Symlinks are copied as links, never followed, and modes are preserved. Write `plan.json` (the actions, source and destination hashes) and `RESTORE.md` (manual restore steps) beside it. Verify the backup's hashes before continuing.
- **Apply and rerun.** Run the actions in plan order. Each action is idempotent: it is skipped when the destination already has the planned hash and the source is gone. A rerun continues from the stored `plan.json` when the inventory still matches it, and otherwise re-plans. Write `layout_version: 2` last, as the commit marker. Remove the empty `in/` only after the marker is written.
- **Report.** Human-readable output plus `--json`. The dry-run output must be identical across two runs.

### 4d — Configs, installed packages and state records

- Move `in/config/**` to `src/usr/config/**` with selectors unchanged. This includes root `config.yaml` and `hydra/`. Absolute Hydra defaults (`/model: x`), interpolations and nested group references need no rewriting because selectors are stable. Add tests for each.
- Deduplicate only the `dedupe` actions; intentional full-profile overrides stay as overrides.
- Translate installed package manifests to v3 (decision 2).
- Rewrite the keys of state records keyed by old paths: retained configs, config backups, installer ownership, `config_viewer.parameter_orders`, editor recent files and open tabs in `settings.toml`, and fork provenance. Unknown keys that start with `in/` are reported rather than guessed.

### 4e — Datasets and references

- Move `in/datasets/**` to `src/usr/datasets/**`. `mimic` moves as the same symlink to the same external target.
- `pipeline/datasets.py`: replace the search order with `src/usr/datasets/...`. The `in/inputs/` target goes away.
- Rewrite references structurally in YAML under `src/usr/config` for the known dataset keys that `workspace_inventory` already detects. Arbitrary strings containing `in/datasets` are listed for review and never rewritten.
- `sites/policy.py`: declared inputs target `src/usr/datasets/`. `sites/transfer.py` reads the new root. `sites/snapshot.py` `SOURCE_ROOTS` uses `src/usr/config` (record it for phase 5's bundle manifest).
- Relocate `in/datasets/prepare_tabular.py` into the `tabular-data` provider or a bundled example package, with an explicit `--output` defaulting to `paths.user_inputs`. Update the messages in `core/paradigm_impls/base/tabular.py`.

### 4f — Remaining consumers and documentation

- Sweep the remaining references: `frontend/app.py`, `config_model.py`, `config_tree.py`, `config_panel/{domain,catalogs,sections/methods}.py`, `scaffold_dialog.py`, `model.py`, `workflows/workflows_panel.py`, `pipeline/{config,compose,validation,component_validation,workflow/executor}.py`, `services/{rulesets,local_components}.py`, `sites/catalog.py`, `optional_components/slurm/execution.py`, `standard_figures/reports.py`, `sequential_spec`, `model_builder`, `arxiv_papers` and `scripts/{erase_experiment,clean_deleted}.py`. Update editor links and CLI help.
- Replay guard: replaying or resuming a saved result whose config references `in/` stops with a `ConfigurationError` naming the path. This covers loading a result into the builder, site continuation and tuner continuation.
- Documentation: `PLUGIN_STORAGE.md` (v3 destinations and layout v2), `LAYER_BOUNDARIES.md` (precedence; retire the `in/` compatibility sentences), the guides naming `in/config` (18 files), the documentation panel content, `.gitignore` (ignore `src/usr/datasets`) and the docs index.

### 4g — Tests

- Add one shared fixture that builds a layout-v2 workspace, and a separate one that builds a v1 workspace for the migration tests. Port the 61 test files that build `in/config` layouts to the v2 fixture.
- Migration suite:
  - the dry-run report is stable across runs;
  - apply, then rerun, does nothing the second time;
  - an interrupted apply resumes from `plan.json`;
  - a conflict stops before any move;
  - a symlink stays a link, and modes and bytes are preserved;
  - the backup is complete and hash-verified;
  - the restore steps in `RESTORE.md` reproduce the original tree;
  - running jobs refuse the migration;
  - Hydra composition and dataset selection match before and after on a populated fixture.
- Entry-point tests: each entry point raises `WorkspaceLayoutError` on a v1 workspace, and the error names the command.
- Distribution probe: `verify_installed_distribution.py` uses v3 manifests and asserts that a clean external workspace never creates `in/`.

### 4h — Hub republish preparation (no publication)

- In a fresh clone of production `ThetaIDE/ThetaHub`, on a branch: add the v3 registry and feed (decision 3), extend the validator to storage 3 and the `tuner` and `scheduler` kinds, rebuild the 30 affected packages and the Optuna and Slurm components as v3 archives with new versions, and run `validate_manifest.py` and `build_index.py`.
- Install every rebuilt archive into an isolated v2-layout workspace with the branch IDE.
- Your local `../theta-hub` clone tracks the CameronEgb fork, has releases production does not (e.g. "Publish SB3 suite 1.0.3"), and has uncommitted feed changes. Reconcile with the user which releases are authoritative before building.
- Publication follows the [production release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md), only when the user requests it, and in step with the IDE merge to `main`.

### 4i — Migrate the research workspace (user-approved)

Run the dry-run on the repository's research workspace and review the report with the user. Apply only after explicit approval, then rerun to confirm a no-op. Then compare composition and dataset selection with a pre-migration baseline captured by `workspace_inventory`.

## Exit gate

- **Migration safety.** On a clean and on a populated v1 fixture, dry-run then apply then rerun is stable. The backup restores the original bytes, modes and symlinks by following `RESTORE.md`.
- **Same results.** Config composition and dataset selection match the baseline in the GUI, the CLI and workers, from an unrelated working directory.
- **Old artifacts fail clearly.** v2 archives fail with the release message, unmigrated workspaces fail at every entry point with the command, and historical replays fail with the path.
- **User edits survive.** Edits in `src/usr/config` survive package update and uninstall.
- **Distribution.** The installed-distribution probe passes with v3 manifests.
- **Code is clean.** `git grep -E "in/(config|datasets|envs|inputs)"` outside tests, historical-record readers and the migration code returns nothing.

## Risks

- **Hub timing.** Merging the IDE before the v3 feed is live leaves new installs with nothing to install, and Optuna and Slurm unavailable. Merge and publish together.
- **In-flight cluster jobs.** The migration refuses to run while site jobs are unfinished. Their continuation does not survive the cutover by decision.
- **Wide test churn.** The fixture port touches 61 files. Land it with 4b in one commit, so the suite never mixes layouts.
- **Ignored and external data.** The inventory and backup must include ignored files and preserve the external `mimic` link without copying its target.
