# Layer boundary implementation status

Current state of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) on branch `implement/layer-boundaries`, as of 2026-11-15. This page describes only what is true now and what is left; replace its contents as work lands rather than appending dated sections. The dated evidence for phases 0–4 (commits, test counts, reasoning) is in the [implementation history](../reports/LAYER_BOUNDARY_HISTORY_20261115.md). The [handoff prompt](LAYER_BOUNDARY_HANDOFF.md) lists the next work in order.

## Progress

| Phase (delivery) | State | What is left |
| --- | --- | --- |
| 0 — Contract and baseline (A) | Done | — |
| 1 — Workspace and config ownership (B) | Done | — |
| 2 — Runner features, site/tuner extraction (C, D) | Done | — |
| 3 — App-plugin SDK (E) | Done | — |
| 4 — Retire `in/` and every adapter (F) | 4a–4h done except the TensorBoard flag; Hub releases prepared, not published | TensorBoard flag (below) |
| 5 — Results in `src/usr/results`, remote bundles (G) | Not started; tuner continuation artifacts and site dependencies landed in phase 2 | Plan document first; see the plan's phase 5 |
| 6 — Installed distribution (H) | Not started; core stays at `src/app` (no move) | See the plan's phase 6 |
| 7 — Enforcement (I) | Not started | See the plan's phase 7 |

Gates: the real cluster smoke run is not a gate and is not run (user decision, 2026-11-15). The desktop is accepted as working; the interactive items below are still unchecked.

## Repositories

- **IDE:** `implement/layer-boundaries` is pushed to `origin` (ThetaIDE/ThetaIDE). No PR. It merges to `main` only together with the Hub publication, and only when the user asks.
- **Hub releases:** worktrees `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`), local and unpushed. At publication, `ai_assistant`, `theta_mcp` and the five MIMIC packages are removed from the feed; until then the Hub validator fails on those seven storage-v2 entries. The user's own `../theta-plugins` and `../theta-hub` checkouts are never modified.
- **Repository workspace:** runs the prepared v3 releases (20 packages). Optuna and Slurm are not installed there (Slurm collides with the authored `ncshare-site` package, which owns `site/ncshare/*`). The 4f backup is in `.thetaide/migrations/20261114T163513Z/`.

## Unit 4h — phase 4 close-out

Done in commits `2898c2d`..`e3c98cd` (see `git log b7c13ac..`). Old inputs fail through ordinary validation, and no code names the old patterns. Covered: dead Slurm entry points, `episode_outcomes` and implicit Lightning features, `in.envs` hot reload, experiment export, generic workflow presets, `from_study` as recorded parameters, results readers (sibling grouping, unmarked summaries, old reports), core accessors, desktop aliases and `.plugins.json`, config forms (string sweep dimensions, short-name aliases, saved order keys), the catalog `LOGS_DIR` adapter and the `services` path constants, sites (shared-workspace executions now complete through their finalized receipt instead of waiting for a seal), Optuna study versions, paradigms that must declare `method.algorithm`, and the composite config-viewer tests (17 now run on the fixture workspace).

Kept as current behavior (wording only): scalar method selections and method-level keys, the reported-metric objective, inline plot options, capability `aliases`, `ActionResult` unpacking, opt-in schema validation, the internal stylesheet roles in `frontend/theme.py`, and `hotkeys.panes` (still written by the settings view).

### Tuning has one shape

Each tuning setting has exactly one home:

- `tuning.component`, `tuning.metric` (or an `objective` component) and `tuning.direction`;
- tuner settings under `tuning.options.<component>`;
- the search space under a method's `tune`.

Validation rejects any other `tuning` key and names where tuner settings go. Nothing reads tuning settings from paradigms, environments (`monitor_metric`), `hydra.sweeper.*`, flat `tuning.<option>` keys or a method's `search_space`. A new sweep block in the form seeds no metric or direction.

The Optuna component has one sweeper config, `hydra/sweeper/theta_optuna` (`optuna` is taken by the upstream Hydra plugin's schema). Every value in it is `???`. The tuner compiles all of them from the tuning block and rejects caller `hydra/sweeper` or `hydra.sweeper.*` arguments. Constructor defaults are gone from `ThetaOptunaSweeper`. The sweeper opens its study through `get_optuna_storage`, as study preparation does, so one storage URL serves both. Launch bindings may assign only `tuning.*` and `theta_tuning.*`. The tracked user-layer copies `src/usr/config/hydra/sweeper/optuna_{online,offline}.yaml` are deleted.

The user's `BlendRL/Cartpole/sweep_sustainedSolve.yaml` sets `tuning.n_trials: 64` and `n_jobs: 1` at the flat level, so it now fails validation until those keys move under `tuning.options.optuna` (user go-ahead needed).

Outside tuning, the checkpoint monitor still comes from paradigm or environment `monitor_metric` and `checkpoint.monitor` (`src/app/core/lightning_builder.py:376`–`:416`, `:690`). The bundled documentation (`src/plugins/app/bundled/documentation/content.py:666`, `:1073`) wrongly says `monitor_metric` is also the tuning target.

### Removed old top-level keys

The `online_methods`/`offline_methods`/`offline_datasets` rejection and the `task:` warning in `run_pipeline.py` are deleted, with the tests that copied them. The root `sweep` key is gone from both base configs, the form's key lists and the docs; tuning is the `tuning` block alone.

### Left in 4h: the TensorBoard flag

TensorBoard is a bundled app plugin (`src/plugins/app/bundled/tensorboard`) plus the `tensorboard` run-logger backend, enabled by `logging.backends.tensorboard.enabled`. A root `tensorboard: true` flag still exists as a second switch:

- core translation: `src/app/core/run_logger_plugins.py:31`–`:52` (`_configured_backends`, also reads `agent.tensorboard`);
- the experiment form's "Log to TensorBoard" checkbox writes the root key: `frontend/config_panel/sections/experiment.py:363`, `:428`; `frontend/config_panel/domain.py:215`, `:501`, `:521`, `:595`, `:1649`; `frontend/config_panel/catalogs.py:207`; `frontend/config_viewer.py:178`, `:2528`;
- `frontend/model.py` `Config.tensorboard` (`:37`, `:48`, `:63`, `:280`) and `frontend/app.py:1837`, `:2005`;
- `tests/fixtures/workspace_config/experiment/thetaide/_base.yaml:12`.

Delete the translation and the root key; the checkbox (shown only when the plugin is enabled) edits `logging.backends.tensorboard.enabled` through the shared form controls (`docs/reference/CONFIG_PARAMETER_FORMS.md`).

### Also left

- The resource pool's SQLite `ALTER TABLE ADD COLUMN` upgrade (`src/app/services/resource_budget.py`) stays: the running desktop may hold that database.
- Deferred tests (user decision): tests that read `~/.config/thetaide/settings.toml` (`frontend/runner_manager.py` `_load_resource_settings`) or QSettings, and `test_ncshare_hardening` loading Slurm from source instead of the installed component.
- The published `blendrl` 1.3.0 source in `theta-plugins-v3` declares `neural`/`symbolic` as strings and `logic_lr`/`blender_lr` as numbers, contradicting its profile defaults (fixed in `blendrl-dev`, `e3c98cd`). Fix it before the Hub publication.

## Results paths for phase 5, unit 1

Every literal `results/...` path moves to `WorkspacePaths.results`. Paths relative to the working directory (they work only because pipeline processes start in the workspace): `src/app/core/callbacks.py:76`, `lightning_builder.py` (`get_original_cwd()` at `:57`, `:309`; literals at `:627`, `:661`), the defaults in `src/app/telemetry/run.py:229`, `:246` and `src/app/core/contract_logger.py:32`, `src/app/pipeline/variability.py:111`–`:129`, `src/plugins/components/bundled/standard_figures/reports.py:44`–`:109`, `optional_components/optuna/utils.py` (`DEFAULT_OPTUNA_DB_URL`, `:128`, `:214`, `:267`, `:420`), `optuna/sweeper.py:71`, `optuna/component.py:383`, and `src/usr/algorithms/sb3-dev/sb3_runner.py`. Paths joined to a root but spelled `results/...`: `src/app/pipeline/tuning_monitor.py`, `execution_data.py`, `src/app/sites/*` (`execution.py`, `transfer.py`, `publication.py`, `continuation.py`, `maintenance.py`), `optional_components/optuna/component.py:354`, `optuna/remote.py:14`, `slurm/execution.py:325`, and `scripts/erase_experiment.py`. Site paths name the remote layout as well, so they move together with the remote bundle format.

## Tests

Run per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. Compare against `main` by test ID before calling a failure a regression.

- **Fixtures.** `tests/conftest.py`'s autouse `tracked_workspace` selects a temporary copy of the checkout's tracked `src/usr`; a session hook fails the run if tests create result groups in the checkout's `results/`. `fixture_workspace` adds `tests/fixtures/workspace_usr` and `tests/fixtures/workspace_config`. `optuna_workspace`, `component_fixtures.sweep_workspace(root)` and `install_optional` install the optional components. `tests/themes_fixtures.py` generates THEMES inputs. `LocalHub` and the authored-NSFR test read sibling checkouts only through `THETA_HUB_ROOT` / `THETA_PLUGINS_ROOT`.
- **Package integration** lives in `../theta-plugins-v3`: `../Theta-IDE/venv/bin/python -m pytest -q tests/test_native_integration.py` (71 passed).
- **Stale tests, deferred (user decision):** 119 failures in 27 files, none new against the session-start commit `b7c13ac` (file-by-file comparison by test ID, 2026-11-15; 2605 passed, 83 skipped): `test_results_panel` (32), `test_job_manager` (12), `test_queue_panel` (11), `test_experiment_pane`, `test_parallel_runs` (6 each), `test_tray`, `test_trial_seed_visibility` (5 each), `test_filetree_toolbar`, `test_monitor_live`, `test_objective_components`, `test_tuner_components` (4 each), `test_monitor_lifecycle`, `test_monitor_panel`, `test_panel_sections` (3 each), `test_component_plugins`, `test_component_validation_contract`, `test_launcher`, `test_tree_mouse_selection` (2 each), and one each in `test_cluster_error_logs`, `test_component_gui_enhancements`, `test_components_filetree`, `test_optuna_pruning`, `test_run_navigator`, `test_sidetabs_bins`, `test_splash`, `test_test_suite`, `test_tuning_objective`.
- **Coverage gap:** no end-to-end test of managed parallel trials with a Theta-owned bootstrap.

## Desktop items not yet checked interactively

The user opens the app and launches experiments. Not yet looked at since the changes:

1. Editing a profile owned by an installed package asks for an override, and the copy appears under `src/usr/config`; authored-package profiles edit in place.
2. The text editor, model builder, TensorBoard, documentation, arXiv and antigravity panels load, and survive disable and re-enable.
3. Figures plot for a finished run.
