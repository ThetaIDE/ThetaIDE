# Layer boundary implementation status

Current state of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) on branch `implement/layer-boundaries`, as of 2026-11-15. This page describes only what is true now and what is left; replace its contents as work lands rather than appending dated sections. The dated evidence for phases 0–4 (commits, test counts, reasoning) is in the [implementation history](../reports/LAYER_BOUNDARY_HISTORY_20261115.md). The [handoff prompt](LAYER_BOUNDARY_HANDOFF.md) lists the next work in order.

## Progress

| Phase (delivery) | State | What is left |
| --- | --- | --- |
| 0 — Contract and baseline (A) | Done | — |
| 1 — Workspace and config ownership (B) | Done | — |
| 2 — Runner features, site/tuner extraction (C, D) | Done | — |
| 3 — App-plugin SDK (E) | Done | — |
| 4 — Retire `in/` and every adapter (F) | Done; Hub releases prepared, not published | — |
| 5 — Results in `src/usr/results`, remote bundles (G) | [Phase 5 plan](../plans/LAYER_BOUNDARY_PHASE5_PLAN.md); 5a/5b done, 5c code stabilized and verified | Moving the user's `results/` (dry run shown, waiting for a yes), 5d bundles, 5e requirements |
| 6 — Installed distribution (H) | Not started; core stays at `src/app` (no move) | See the plan's phase 6 |
| 7 — Enforcement (I) | Not started | See the plan's phase 7 |

Gates: the real cluster smoke run is not a gate and is not run (user decision, 2026-11-15). The desktop is accepted as working; the interactive items below are still unchecked.

## Repositories

- **IDE:** `implement/layer-boundaries` exists on `origin` (ThetaIDE/ThetaIDE); 5b and stabilized 5c are local and unpushed. No PR. It merges to `main` only together with the Hub publication, and only when the user asks.
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

### Run loggers have one switch

`logging.backends.<id>.enabled` is the only switch for a run logger; the root and `agent.tensorboard` flags and their translation are gone. The experiment form shows one toggle per available `run_logger` capability, labelled by the capability's `label` and hidden while its `activation_plugin` is disabled. The form names no backend.

### Legacy sweep

Old paths, shims and fallbacks found by a repository-wide search are gone (`547fd3f`): the ignored `is_sweep` argument, the Optuna `trial_worker` and `objectives.runner` shims, the `BaseMetaPipeline` stub, the site-profile forwarding filter, the second paradigm key per source, swallowed registry errors, the `blendrl-dev` null confidence floor, agent-level rates and comma-joined rules, and the `themes-dev` version-1 policy reader and missing `component_ids` fallback. What a search for legacy wording still finds is current: THEMES' "historical demo" (a research term), CodeMirror's `legacy-modes` package, bare ruleset names, and the `compatibility` manifest field.

Two settings still have two homes, and fixing them needs v3 package changes and a republish: paradigm compatibility (the profile's `compatible_paradigms` wins over the manifest's `paradigms`, `frontend/config_viewer.py`) and the data-module `role` (profile first, then capability metadata). The plugin scaffolder (`src/app/services/plugin_scaffolder.py`) generates agent code that imports the private `src.app.core.protocols` and `agent_registry`; phase 7 enforcement will catch it.

### Also left

- The resource pool's SQLite `ALTER TABLE ADD COLUMN` upgrade (`src/app/services/resource_budget.py`) stays: the running desktop may hold that database.
- Deferred tests (user decision): tests that read `~/.config/thetaide/settings.toml` (`frontend/runner_manager.py` `_load_resource_settings`) or QSettings, and `test_ncshare_hardening` loading Slurm from source instead of the installed component.
- The published `blendrl` 1.3.0 source in `theta-plugins-v3` declares `neural`/`symbolic` as strings and `logic_lr`/`blender_lr` as numbers, contradicting its profile defaults (fixed in `blendrl-dev`, `e3c98cd`). Fix it before the Hub publication.

## Phase 5

Every results path comes from `WorkspacePaths.results`, which is `src/usr/results` (5c). The checkout's own `results/` has not moved yet: until it does, the desktop and pipeline started from this checkout see no earlier runs.

- Pipeline processes (Lightning, telemetry, callbacks, sb3, Optuna) call `WorkspacePaths.for_process()`; nothing joins `results/` to the working directory or Hydra's original cwd. Hydra's `run.dir` and `sweep.dir` use the `${theta_results:}` resolver, which the Hydra plugin module registers. The unused `results_dir` key is gone.
- Portable paths (site listings, run manifests, receipts) are workspace-relative strings built from `WorkspacePaths.results_relative`; site code uses `sites.storage.results_root` / `results_relative`.
- Tuner artifacts are relative to the results directory (`optuna/<execution>/study.sqlite`); the host joins them to `WorkspacePaths(runtime).results`. `RunLoggerContext.base_root` became `results_root` (the results directory itself).
- The desktop job database has one home, `.thetaide/runs/jobs/jobs.db`; the `results/jobs` default is gone.
- Removed: `scripts/erase_experiment.py` and `scripts/clean_deleted.py` (the deletion-tracking and rsync-pull model they served is gone; the catalog deletes runs), the reports plotter's `results/logs/slurm` timing fallback, and the export dialog's working-directory fallback.
- `tests/test_layout_guard.py` fails on any literal results path outside `src/app/workspace.py` (verbatim THEMES reference source excepted).
- 5c: `.gitignore`, `MANIFEST.in`, the test leak check and the tests use `src/usr/results`. Walks of `src/usr` that look for source skip the results tree: remote source snapshots, the compose fingerprint, the workspace inventory and the implementation-edit lock.
- `src.app.api.results` (SDK 2.4) gives packages and app plugins read-only access to runs: `list_runs`, `run_metrics`, `run_manifest`, `read_run_file`.
- Stabilization: source snapshots prune results before traversal. Regression checks cover source inventory/cache stability, edit-lock exclusion, explicit SDK workspace selection and confined artifact reads. Desktop runner snapshots cannot roll back a newer RPC response, and jobs that finish before the next poll still have final output drained. Fake-pipeline tests explicitly substitute their application launcher and await asynchronous reconnect readiness.
- The move of the user's `results/` (15,590 files, 1.23 GB, no symlinks or conflicts) runs from a scratch script: dry run, timestamped backup under `.thetaide/migrations/`, one rename, every hash checked. It moves `results/jobs`, `slurm`, `development` and `exports` as they are, though nothing reads them now.

Session sequencing: stabilize 5c first, then ask for the user's specific UI changes. Vega design is deferred until the user provides their own design. The results move still requires approval.

## Tests

5c verification (2026-11-15), all 225 test files in separate processes: **2,636 passed, 52 skipped, 126 failed**. There are **zero new failure IDs** against the 140 recorded 5b failures, and no unresolved aborts or timeouts after rerunning sandbox-affected files outside the sandbox. The job-manager file passes **19 tests, with 2 real-provider prerequisite skips**; the three new results-workspace checks and all 64 SDK-boundary checks pass. [Evidence and exact remaining failure IDs](../reports/LAYER_BOUNDARY_PHASE5C_VERIFICATION_20261115.json).

Run per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. For 5c, compare against the saved 5b baseline (`9129f8b`) by test ID before calling a failure a regression.

- **Fixtures.** `tests/conftest.py`'s autouse `tracked_workspace` selects a temporary copy of the checkout's tracked `src/usr`; a session hook fails the run if tests create result groups in the checkout's `src/usr/results`. `fixture_workspace` adds `tests/fixtures/workspace_usr` and `tests/fixtures/workspace_config`. `optuna_workspace`, `component_fixtures.sweep_workspace(root)` and `install_optional` install the optional components. `tests/themes_fixtures.py` generates THEMES inputs. `LocalHub` and the authored-NSFR test read sibling checkouts only through `THETA_HUB_ROOT` / `THETA_PLUGINS_ROOT`.
- **Package integration** lives in `../theta-plugins-v3`: `../Theta-IDE/venv/bin/python -m pytest -q tests/test_native_integration.py` (71 passed).
- **Stale tests, deferred (user decision):** 126 failures in 27 files remain, all in the saved 5b baseline; exact IDs and file counts are in the evidence above. This includes 20 `test_paradigm_unsupervised` failures: its import-time check reads the real workspace catalog, while the hermetic fixture workspace has no autoencoder. The earlier 119-failure status predates the 5b comparison and is not the current total.
- **Coverage gap:** no end-to-end test of managed parallel trials with a Theta-owned bootstrap.

## Desktop items not yet checked interactively

The user opens the app and launches experiments. Not yet looked at since the changes:

1. Editing a profile owned by an installed package asks for an override, and the copy appears under `src/usr/config`; authored-package profiles edit in place.
2. The text editor, model builder, TensorBoard, documentation, arXiv and antigravity panels load, and survive disable and re-enable.
3. Figures plot for a finished run.
