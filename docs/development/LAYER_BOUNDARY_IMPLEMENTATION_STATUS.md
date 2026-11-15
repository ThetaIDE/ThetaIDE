# Layer boundary implementation status

Current state of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) on branch `implement/layer-boundaries`, as of 2026-11-15. This page describes only what is true now and what is left; replace its contents as work lands rather than appending dated sections. The dated evidence for phases 0–4 (commits, test counts, reasoning) is in the [implementation history](../reports/LAYER_BOUNDARY_HISTORY_20261115.md). The [handoff prompt](LAYER_BOUNDARY_HANDOFF.md) lists the next work in order.

## Progress

| Phase (delivery) | State | What is left |
| --- | --- | --- |
| 0 — Contract and baseline (A) | Done | — |
| 1 — Workspace and config ownership (B) | Done | Workflow presets name research packages (4h) |
| 2 — Runner features, site/tuner extraction (C, D) | Done | Historical readers it added and dead Slurm entry points (4h) |
| 3 — App-plugin SDK (E) | Done | — |
| 4 — Retire `in/` and every adapter (F) | 4a–4g done; Hub releases prepared, not published | Close-out unit 4h (below) |
| 5 — Results in `src/usr/results`, remote bundles (G) | Not started; tuner continuation artifacts and site dependencies landed in phase 2 | Plan document first; see the plan's phase 5 |
| 6 — `src/ide` consolidation (H) | Not started | See the plan's phase 6 |
| 7 — Enforcement (I) | Not started | See the plan's phase 7 |

Gates: the real cluster smoke run is not a gate and is not run (user decision, 2026-11-15). The desktop is accepted as working; the interactive items below are still unchecked.

## Repositories

- **IDE:** `implement/layer-boundaries` is pushed to `origin` (ThetaIDE/ThetaIDE). No PR. It merges to `main` only together with the Hub publication, and only when the user asks.
- **Hub releases:** worktrees `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`), local and unpushed. At publication, `ai_assistant`, `theta_mcp` and the five MIMIC packages are removed from the feed; until then the Hub validator fails on those seven storage-v2 entries. The user's own `../theta-plugins` and `../theta-hub` checkouts are never modified.
- **Repository workspace:** runs the prepared v3 releases (20 packages). Optuna and Slurm are not installed there (Slurm collides with the authored `ncshare-site` package, which owns `site/ncshare/*`). The 4f backup is in `.thetaide/migrations/20261114T163513Z/`.

## Unit 4h — phase 4 close-out

Each item is one or more commits. Old inputs must keep failing through ordinary validation after a targeted check is removed; where nothing would reject them, add a generic check (for example, unknown top-level config keys) rather than keep the named one.

1. **Legacy code to remove.** Candidates found by searching tracked source for "legacy", "historical", "compatibility" and "deprecated"; comments that use the word without implementing old behavior are only reworded.
   - Tuner and config forms: `legacy_flat` fields (`frontend/config_panel/sections/sweep.py`, `frontend/config_panel/domain.py:1312`, `optional_components/optuna/component.py:117`); flat method storage (`frontend/config_panel/bindings.py:57`); non-dict objective specs and inline options (`frontend/config_panel/sections/components.py`); the `sweep` activation flag (`frontend/config_panel/catalogs.py:209`, `src/app/pipeline/tuning_config.py:11`, `tuning_features.py:39`); scalar `env` (`frontend/config_panel/domain.py:127`); bare names and "Sweep (legacy)" rows (`frontend/config_viewer.py:1409`, `:2137`); unconstrained third-party components and short-name aliases (`frontend/config_model.py:195`, `:458`); legacy reorder keys (`frontend/parameter_reorder.py:195`).
   - Desktop: `.plugins.json` reading and migration and deprecated aliases (`frontend/plugins/manager.py`); the `hotkeys.panes` list (`frontend/hotkeys.py:497`); old theme colour names (`frontend/theme.py:78`); runner-manager legacy queries (`frontend/runner_manager.py:134`, `:247`); attribute aliases and re-exports (`frontend/app.py:439`, `:612`, `frontend/config_tab_manager.py:54`, `frontend/config_tree.py:44`, `:407`, `frontend/terminal/terminal_widget.py:768`, `frontend/config_viewer.py:75`, `frontend/widgets.py:11`, `frontend/component_actions.py:23`, `frontend/monitor/sweep_view.py:16`).
   - Pipeline and results: legacy method keys (`run_pipeline.py:249`, `src/app/core/plotting/base.py:378`); the `tensorboard` flag (`src/app/core/run_logger_plugins.py`); sibling-run grouping (`src/app/pipeline/result_paths.py:65`, `:82`); unmarked tuning summaries and exports without a component (`src/app/pipeline/tuning_monitor.py:11`, `tuner_execution.py:196`); the historical Hydra direction (`src/app/pipeline/commands.py:289`); the legacy-manifest method contract (`src/app/pipeline/validation.py:125`); old markdown reports (`src/plugins/components/bundled/standard_figures/reports.py:165`, `:242`).
   - Core: `in.envs.` and decorator modules in hot reload (`src/app/core/hot_reload.py:36`); the callable action count (`src/app/core/env_vectorized.py:13`); the `blend_entropy` accessor (`src/app/core/types.py:51`); named rejection of old manifest fields and `legacy_loose` (`src/app/components/archive.py:37`, `development.py:25`); the `legacy` catalog field (`src/app/core/provider_catalog.py:80`); omitted-context desktop adapter (`src/app/services/catalog.py:41`); resource-budget legacy signals (`src/app/services/resource_budget.py:208`, `:393`).
   - Sites and components: Slurm `publication_state: 'legacy'` (`optional_components/slurm/execution.py:155`); legacy executions and databases (`src/app/sites/result_state.py:8`, `maintenance.py:1`, `transfer.py:200`); Optuna `version_N` runs, old sweeps and legacy Hydra expressions (`optional_components/optuna/utils.py:193`, `:448`, `component.py:11`); legacy databases in `scripts/erase_experiment.py:241`.
   - Out of scope: authored research packages (`blendrl-dev`, `themes-dev`), whose "historical" code reproduces published methods, and `src/app/services/result_rename.py:103`, where "historical" means a saved snapshot.
2. **`episode_outcomes`.** `sb3-dev`, the installed `sb3-suite` and `theta-plugins-v3/methods/sb3` declare the flag both at the capability's top level and in `execution_features`. Drop the top-level copy, rebuild the archive (`scripts/package_plugins.py`; in `theta-hub-v3`, `scripts/add_releases.py`, `scripts/build_index.py`, `scripts/validate_manifest.py --check-archives`), reinstall `sb3-suite`, delete the reader (`src/app/core/execution_features.py:34`).
3. **Declared Lightning features.** Remove the implicit default for Lightning providers without `execution_features` (`execution_features.py:78`); declare features where a tracked Lightning provider relies on it.
4. **Remove experiment export:** `src/app/services/experiment_exporter.py`, `frontend/export_dialog.py`, the `run_pipeline.py export` command, `tests/test_experiment_exporter.py`, `tests/test_experiment_plugin_export_and_install.py`, and the export cases in other tests; update the guides.
5. **Workflow presets** (`src/app/pipeline/workflow/presets.py`): rewrite generically. They currently name BlendRL and MIMIC CQL and reference five experiments that do not exist.
6. **`from_study`** (`src/app/pipeline/config.py:79`): describe and implement it as reading a configuration recorded in the results, with no tuner name, resolved in the selected workspace.
7. **Dead code:** `src/app/pipeline/slurm_runner.py` and `tuner_finalize.py` (referenced only by tests).
8. **Tests:** extend `tests/test_layout_guard.py` to dotted module paths; give `test_config_viewer_composite` (17, skipped) fixtures; stop tests reading `~/.config/thetaide/settings.toml` and QSettings; load Slurm in `test_ncshare_hardening` through the installed component.
9. **Docs:** `RESULT_STORAGE.md` still describes legacy compatibility; the guides lose experiment export.

## Working-directory results paths (phase 5, unit 1)

`src/app/core/callbacks.py`, `src/app/core/lightning_builder.py` (`get_original_cwd()`), the defaults in `src/app/telemetry/run.py` and `src/app/core/contract_logger.py`, the best-params lookup in `src/app/pipeline/config.py`, `src/app/pipeline/tuning_monitor.py` and `src/usr/algorithms/sb3-dev/sb3_runner.py`. They work only because pipeline processes run with the workspace as working directory.

## Tests

Run per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. Compare against `main` by test ID before calling a failure a regression.

- **Fixtures.** `tests/conftest.py`'s autouse `tracked_workspace` selects a temporary copy of the checkout's tracked `src/usr`; a session hook fails the run if tests create result groups in the checkout's `results/`. `fixture_workspace` adds `tests/fixtures/workspace_usr` and `tests/fixtures/workspace_config`. `optuna_workspace`, `component_fixtures.sweep_workspace(root)` and `install_optional` install the optional components. `tests/themes_fixtures.py` generates THEMES inputs. `LocalHub` and the authored-NSFR test read sibling checkouts only through `THETA_HUB_ROOT` / `THETA_PLUGINS_ROOT`.
- **Package integration** lives in `../theta-plugins-v3`: `../Theta-IDE/venv/bin/python -m pytest -q tests/test_native_integration.py` (71 passed).
- **Stale tests, deferred (user decision):** 122 failures in 27 files, all failing on `main` too: `test_results_panel` (32), `test_job_manager` (14), `test_queue_panel` (11), `test_parallel_runs` (6), `test_experiment_pane` (6), `test_tuner_components` (5), `test_trial_seed_visibility` (5), `test_tray` (5), `test_objective_components` (4), `test_monitor_live` (4), `test_filetree_toolbar` (4), `test_panel_sections`, `test_monitor_panel`, `test_monitor_lifecycle` (3 each), `test_tree_mouse_selection`, `test_launcher`, `test_component_validation_contract`, `test_component_plugins` (2 each), and one each in `test_tuning_objective`, `test_test_suite`, `test_splash`, `test_sidetabs_bins`, `test_run_navigator`, `test_optuna_pruning`, `test_components_filetree`, `test_component_gui_enhancements`, `test_cluster_error_logs`.
- **Coverage gap:** no end-to-end test of managed parallel trials with a Theta-owned bootstrap.

## Desktop items not yet checked interactively

The user opens the app and launches experiments. Not yet looked at since the changes:

1. Editing a profile owned by an installed package asks for an override, and the copy appears under `src/usr/config`; authored-package profiles edit in place.
2. The text editor, model builder, TensorBoard, documentation, arXiv and antigravity panels load, and survive disable and re-enable.
3. Figures plot for a finished run.
