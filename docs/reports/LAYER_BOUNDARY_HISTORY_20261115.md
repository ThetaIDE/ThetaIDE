# Layer boundary implementation history (2026-11-13 to 2026-11-15)

Historical record. This is the dated evidence written while phases 0–4 of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) were implemented, newest first. It is kept for its commits, test counts and reasoning; it does not describe current behavior or remaining work. Sections contradict each other where later work or decisions superseded them; in particular, the "bounded" compatibility readers and adapters described in the phase 1–3 sections were retired or are being retired under the no-legacy decision of 2026-11-14. For the current state see the [implementation status](../development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md).

## Hermetic tests and phase 4 leftovers — 2026-11-14/15

**Status:** the IDE suite no longer depends on the user's installed packages, ignored recipes, rules, datasets, `.thetaide` state or sibling checkouts. Every failure that remains also fails on `main` and comes from stale tests, not from user state. Nothing was published or pushed.

### Counts

Per-file runs in fresh worktrees (`git worktree add`, no `THETA_*` variables):

| Tree | Files | Failures |
| --- | --- | --- |
| `main` 4738cae | 208 | 306 |
| Branch 64a0f61 (start of this work) | 229 | 323 |
| Branch 06e8755 | 229 | 149 in 34 files; none fail that pass on `main` |
| HEAD (06e8755 plus the fixes after it) | — | 122 in 27 files (by test ID, not rerun as a whole) |

### Fixture conventions

- `tests/conftest.py`: the autouse `tracked_workspace` selects a temporary copy of the checkout's tracked `src/usr` (`checkout_workspace()`), so populated and fresh checkouts behave the same. A session hook fails the run if tests create new result groups in the checkout's `results/`.
- `build_fixture_workspace(root)` / `fixture_workspace`: the checkout workspace plus `tests/fixtures/workspace_usr` (fixture packages such as `test-models`, `probe_evaluators` and the `mountaincar/monitoring_default` ruleset) and `tests/fixtures/workspace_config` (recipes selecting `sb3-dev/ppo`, `d3rlpy-dev/cql`, `gym_classic_control_dev/cartpole`).
- `optuna_workspace` (session-built, read-only) and `component_fixtures.sweep_workspace(root)` install the optional Optuna tuner; `install_optional` does the same for Slurm.
- `tests/themes_fixtures.py` generates THEMES inputs with `themes-dev`'s `generate_time_csv` and `write_resources`; the parity fixture skips when its local `.npz` is absent.
- `LocalHub` and the authored-NSFR test read sibling checkouts only through `THETA_HUB_ROOT` / `THETA_PLUGINS_ROOT` and skip otherwise.
- Tests that need real research packages (SB3 policies with BlendRL, BlendRL stability) live in `theta-plugins-v3` `tests/test_native_integration.py` (71 passed).

### Regressions found and fixed

- `prepare_launch` compared the site to `local` while the choice is `local/default`, so local launches skipped the resource budget and the one-device guard (682dd21).
- `ConfigTree` dropped the workspace's own `experiment/` group folders (7cdcace).
- Execution manifests, worker logs, seed folders and cancellation used the working directory, not the selected workspace; the desktop's cancellation looked in the wrong place (6bee081).
- `resolve_experiment_config_name` passed unknown names through to Hydra when no experiments existed (f39292b).
- The shared editor could not edit a nullable (`[type, "null"]`) number; a plain number schema now flags a cleared draft (d4682bd).
- The tracked `fast_stable` objective was not a package and could not be selected (ac72ef0).
- Tests had leaked results into the checkout (`results/*/g`, `results/*/group`); they were moved to the session scratchpad.

### Leftovers retired

Stable machine identity for provider leases (ffcf82f); Slurm without the NCShare profiles (fec5a8a, Hub 0e5f929); sweep bootstrap from the application (f80e6ac); data-loader migration removed (9db5a2a); SDK 2.3 and authored packages on the SDK only (71d3070, 64a0f61); `migrate_execution` removed (ed4bca3); objective-runner trial readers and `LegacyExecutionData` removed (6f63c8d); monitor suggestions from installed capabilities instead of hard-coded package IDs (5847bf3). MIMIC packages are held back from v3 (Hub 5047598).

### Still open

- **122 stale tests**, failing on `main` too: `test_results_panel` (32; explorer moved to current-item selection on `main`), `test_job_manager` (14; the runner launches `APPLICATION_ROOT/run_pipeline.py`, so the fake pipeline needs a redesign), `test_queue_panel` (11), `test_parallel_runs` (6), `test_experiment_pane` (6), `test_tuner_components` (5), `test_trial_seed_visibility` (5) and `test_objective_components` (4) (removed `theta_execution.script` / `theta_objective_runner.script`), `test_tray` (5), `test_monitor_live` (4), `test_filetree_toolbar` (4), `test_panel_sections`, `test_monitor_panel`, `test_monitor_lifecycle` (3 each), `test_tree_mouse_selection`, `test_launcher`, `test_component_validation_contract`, `test_component_plugins` (2 each), and one each in `test_tuning_objective`, `test_test_suite`, `test_splash`, `test_sidetabs_bins`, `test_run_navigator`, `test_optuna_pruning`, `test_components_filetree`, `test_component_gui_enhancements`, `test_cluster_error_logs`. `test_config_viewer_composite` (17) skips because it needs the user's MIMIC/BlendRL recipes.
- **v1 `episode_outcomes` reader.** `execution_features.from_metadata` still reads a capability's top-level `episode_outcomes`, which `sb3-dev` and the prepared `sb3-suite` 2.0.0 declare. Move the flag into `execution_features` in both, rebuild the unpublished archive in `theta-hub-v3`, reinstall it in the repository workspace, then delete the reader.
- **Working-directory results paths in the pipeline:** `core/callbacks.py`, `core/lightning_builder.py` (`get_original_cwd()`), `telemetry/run.py` and `core/contract_logger.py` defaults, `pipeline/config.py` best-params lookup, `tuning_monitor.py`, `sb3-dev/sb3_runner.py`. They work because pipeline processes run with the workspace as working directory, but `LAYER_BOUNDARIES.md` forbids the guess; fix with phase 6.
- Experiment exports copy package-owned profiles (for example `configs/agent/<id>.yaml`) as well as pinning the package; capability selection makes the copy inert.
- `test_ncshare_hardening` imports `optional_components.slurm` directly. `~/.config/thetaide/settings.toml` and QSettings are user-level state that tests can still read.
- The `theta-hub-v3` validator fails on the seven storage-v2 entries (five MIMIC packages, `ai_assistant`, `theta_mcp`) until they are republished.
- Coverage gap: an end-to-end test of managed parallel trials with a Theta-owned bootstrap (the synthetic-script versions were removed in 44fc7a1 and 340bb41).

## Phase 4 follow-up: workspace update and decisions — 2026-11-14

**Status:** the repository workspace runs the prepared v3 releases, and the user settled the open 4g questions. Nothing was published or pushed.

### Decisions

- `ai_assistant` and `theta_mcp` stay as they are; the user informs their author.
- The SDK namespace keeps the documented plan: at phase 6 `src.app.api` moves to `src.ide.api` with `src.app.api` as a forwarding package. That is the only forwarding module allowed in phase 6 (recorded in the plan's confirmed decisions).
- No publishing yet.
- Tests must not depend on what the user has installed. Making the suite hermetic is the next unit of work.

### Workspace update

- All 20 installed packages in the repository workspace were updated from `../theta-hub-v3`, for example `sb3-suite` and `d3rlpy-suite` 2.0.0, `blendrl` 1.3.0 and `dnn` 1.2.0. No diagnostics, no retained packages. Optuna and Slurm are not installed there; Slurm collides with the authored `ncshare-site` package, which owns `site/ncshare/*`.
- The update was first refused because of stale implementation leases. `src/app/components/transactions.py` treats a lease from a different `socket.gethostname()` as live, and macOS changes the host name with the network. 83 leases whose processes no longer exist were moved to `.thetaide/stale-provider-leases-20261114T200025/`; the two live ones, from the running desktop and its validation worker, were left in place. The fix (a stable machine identity) is in the handoff.
- Rerun in the updated workspace: `test_mechanistic_monitors` 22 passed, `test_execution_contract` 20, `test_data_loader_bindings` 16, `test_wet_runs` 3 (3 opt-in skips), `test_sb3_component_installation` 1.
- Still failing, each because it reads user state: `test_ncshare_hardening` (2, need Optuna and Slurm in the workspace), `test_blendrl_rich_monitors::test_recipe_preserves_dnn...` (the ignored `newMonitors/onlineRL` recipe), `test_rulesets::test_authored_nsfr...` (`../theta-plugins` is still v2) and `test_sb3_theta_policy` (13, modules that no longer exist).

### MIMIC configurations

In `theta-plugins-v3`, the MIMIC experiment methods (`experiments/mimic/configs/experiment/mimic/*.yaml`, `methods/cql_*.yaml`) and the evaluators in `eval/mimic-early-prediction` and `eval/reward-shaping` select an agent named `cql` with parameters and models that `d3rlpy/cql` does not accept. They must be ported to `d3rlpy/cql` or removed before `mimic`, `mimic-early-prediction` and `reward-shaping` are published.

### Commits

- `theta-hub-v3` `0424204`: `scripts/add_releases.py`, the tool that added the v3 archives and registry entries.
- IDE: this section, the [handoff prompt](../development/LAYER_BOUNDARY_HANDOFF.md) (replacing the phase 4 prompt) and the plan's recorded exception.

## Phase 4 unit 4g — 2026-11-14

**Status:** every maintained Hub package is rebuilt as a storage-v3 release that imports only the Theta SDK, with Optuna and Slurm added. The releases are prepared on local branches and **not published or pushed**. Two third-party plugins wait for a user decision.

### Sources and decisions

- **Authoritative sources.** Production `ThetaIDE/ThetaHub` (`4263c16`) equals the `CameronEgb/theta-hub` fork, so the fork concern was stale. The uncommitted storage-v2 migration in `../theta-plugins` matched the live archives byte for byte, except unreleased work from 2026-11-09/10 (BlendRL symbolic-state split, Gym CartPole without `symbolic_observation_builder`, NEUMANN `lark`). Those files were reset to the published bytes and are not part of this release. The user's `../theta-plugins` and `../theta-hub` checkouts were not modified; the work is on worktrees `../theta-plugins-v3` and `../theta-hub-v3`, branch `v3-republish`.
- **antigravity.** Option (a): SDK 2.1 gained host services for a dock panel and bound command hotkeys, and the plugin uses them instead of the host window.
- **SB3 and d3rlpy.** The published suites still used script `entrypoint`s, which the execution contract removed, so they could not run on this branch. `sb3-suite` and `d3rlpy-suite` 2.0.0 are ported from `sb3-dev` and `d3rlpy-dev` (the release handoff in [EXECUTION_ABSTRACTIONS_STATUS](../development/EXECUTION_ABSTRACTIONS_STATUS.md)). The published d3rlpy hyperparameters are kept, and keys d3rlpy accepts (`gamma`, `expectile`, `weight_temp`, `alpha`) now reach it. `cql_alpha`, which no d3rlpy configuration accepts, is dropped. Bare aliases (`ppo`, `cql`, ...) are removed, so an unrelated `agent: cql` fails instead of silently running d3rlpy.
- **MuJoCo profiles.** The `env/mujoco/*` profiles moved from `robotics` to `robotics-mujoco`, the env provider that claims those selectors; the resolver rejected the old split.

### Commits

- IDE (`implement/layer-boundaries`):
  - `4b078fb`: SDK 2.1 (environment, data, evaluation, ruleset and telemetry modules, model protocols, docks and hotkeys).
  - `e6548ed`: `configured_export` for `export://` references.
  - `806fd27`: SDK 2.2 (`RunResult`, `TrainingCancelled`, `src.app.api.monitoring`, `selected_capability`). The SB3 post-update monitor hook moves from core to the SB3 adapter.
  - `9b6ffe5`: Optuna and Slurm 1.1.0 with Hub metadata and reproducible archives.
  - `cc71729` and `78acdb0`: tests that depended on the installed Hub SB3 release or pinned 1.0.0 archives.
- `theta-plugins-v3`:
  - `3fdd53e` and `c2ddfe2`: the migration, matched to the published releases.
  - `d8470e2`: merge of `origin/main`.
  - `e45dcf2`: package format v3; the builder rejects SDK violations.
  - `213b4ff`: SDK and execution-contract migration.
  - `cf4d4a7`: release versions and notes.
- `theta-hub-v3`:
  - `447d5d9`: the validator accepts only format 3.
  - `19a2959`: 39 release archives, registry entries listing only the current release, rebuilt feeds, and `releases/native-storage-v3-2026-11-14.md`.

### Evidence

- Regression: all 229 IDE test files were run one at a time in fresh worktrees of `78acdb0` and of the 4e baseline `abc2474`. Neither worktree has installed packages or research inputs, which explains the remaining failures. Compared by test ID, nothing fails at 4g that passed at the baseline: 289 failures against 292, and the 3 fixed are the stale execution-contract and monitor-preflight tests.
- `theta-plugins-v3` `tests/test_native_integration.py`: **57 passed** against this branch. It builds, installs and imports every archive, runs real SB3 (DNN, BlendRL) and d3rlpy workers through `src/app/train.py`, reloads checkpoints, makes editable copies, runs an Optuna trial with promotion, objective and replicate workers, and exercises the app plugins. antigravity's own tests: 8 passed.
- Package and Hub checks:
  - Archives are deterministic, and each Hub copy is byte-identical to a fresh build.
  - `validate_manifest.py --check-archives` passes for all 39 release entries. It fails only for `ai_assistant` and `theta_mcp`, which are still storage v2.
- IDE worktree with the 37 releases and Optuna installed from the Hub branch, which this phase was waiting for:
  - `test_baselines_and_suites`: 7 passed, with `THETA_HUB_ROOT` set to the Hub branch.
  - `test_rulesets`: 7 passed, including the authored NSFR ruleset test with `THETA_PLUGINS_ROOT`.
  - `test_data_loader_bindings`: 16 passed.
  - `test_wet_runs`: 3 passed, 3 opt-in skips.
  - `test_blendrl_rich_monitors`: the real BlendRL training-equivalence test passes.
  - `test_execution_contract`: 20 passed.
  - `test_mechanistic_monitors`: 22 passed.
  - `test_sb3_telemetry`: 9 passed.

### Still open

- **Publishing** (user): push `theta-hub-v3` and `theta-plugins-v3` and verify the client feed, following the [release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md). Registry `published_at` reads 2026-11-14T00:00:00Z; set it at publication. Then remove the last exception-register row.
- **`ai_assistant` and `theta_mcp`** (third-party, ccast44): they rely heavily on `context.window` and private `frontend` and `src.app.services` modules. Settled in the follow-up below: they stay as they are.
- **Slurm in this repository's workspace.** The authored `ncshare-site` package owns `site/ncshare/*`, so installing Slurm 1.1.0 here is refused as a profile collision. `test_ncshare_hardening`'s two real-pipeline cases need Slurm installed.
- **Pre-existing, not caused by 4g:**
  - Some MIMIC experiments and evaluators select an agent named `cql`, which no package provides; the only CQL agent is `d3rlpy/cql`. See the follow-up below.
  - `test_sb3_theta_policy` still imports `src.usr.models.blendrl` and decorator-registered models (13 failures on the baseline too).
  - `test_blendrl_rich_monitors::test_recipe_preserves_dnn...` reads the user's ignored `newMonitors/onlineRL` recipe, whose PPO method is now disabled.
- **Phase 6 conflict:** [SDK](../reference/SDK.md) and the [ownership contract](../reference/LAYER_BOUNDARIES.md) say `src.app.api` becomes `src.ide.api` "with forwarding". The confirmed decision is no forwarders in phase 6, so the rename would require republishing every package again. Settled in the follow-up below.

## Phase 4 units 4c, 4d and 4e — 2026-11-14

**Status:** every compatibility adapter is retired, SDK 2.0 is in place, the guides describe only the `src/usr` layout, and the installed-distribution probe passes. What still fails waits for the sibling packages to be republished as v3 (4g).

### 4c — adapters retired

- **Deleted.** `src/app/core/import_aliases.py`, the `src/usr` shims and markers, the `plot/` and `frontend/plugins/core/` forwarding trees, `frontend/tensorboard.py`, `src/app/objectives/builtins`, the Optuna alias modules, the API-1 Slurm adapters and `scheduler_object`, `tuner_execution_compat` and the schema-v1 receipt reader, `provider_object`, the plotter `ContextVar` (`create_plotter`) and every top-level site forwarding YAML (framework `site/local.yaml`, the Slurm component's four, the user's five). Also the three frontend forwarding modules `frontend/hub/installer.py`, `frontend/hub/models.py` and `frontend/plugins/dependencies.py`; callers import `src.app.components` directly.
- **Receipts.** Live in `results/sites/<execution>`. A receipt without `scheduler_component` or `tuner_execution` is refused. Test receipts now record both, with a shared `optuna_plan` helper in `tests/component_fixtures.py`; tests of the old readers were deleted.
- **Sites.** The default is `site: local/default`. Profiles are nested (`ncshare/common`, `ncshare/gpu`, `arc/default`); a bare site name still selects its `default` profile in the site CLI.
- **SDK 2.0.** Adds `src.app.api.plotting`, `.packages` (`provider_export`) and `.agents`. An app plugin without an `sdk` requirement is refused; bundled plugins declare `>=2.0,<3`. `PluginContext` no longer exposes the window.
- **Plotters.** Constructors take `paths`. `run_experiment_plots` resolves `WorkspacePaths.for_process()` when called without one, so `python -m src.app.core.plotting.manager` works. The `themes-dev` plotter reads its own workspace instead of the process one.
- **Hydra Optuna sweeper.** Loads the Optuna component's `hydra_sweeper` export.
- **`blendrl-dev`.** Loads nsfr and neumann helpers through `provider_export`, which fails until those packages declare exports (4g).
- **Exception register.** Keeps Lightning and the base paradigms, the `hydra_plugins` namespace, the root launchers, the SDK 1.1 direct imports in installed Hub packages (until 4g) and direct host imports in authored `src/usr` packages (until phase 7).
- **Test ports missed by 4b.** `test_experiment_delete_rename`, `test_folder_crud`, `test_filetree_toolbar`, `test_experiment_component_overrides` and `test_themes_source_port` built bare config roots; they now use `src/usr/config`.

### 4d — documentation

PLUGIN_STORAGE (v3, profile resolution, no import aliases), LAYER_BOUNDARIES, SDK (2.0), SCHEDULER_CONTRACT and TUNER_EXECUTION_CONTRACT (receipts), EXECUTION_CONTRACT, API, COMPONENT_PLUGIN_GUIDE (SDK imports, manifest capabilities, v3 export layout), COMPONENT_DEVELOPMENT, CLUSTER_JOBS, INSTALL and MECHANISTIC_MONITORS, the documentation panel, README, CONTRIBUTING and the docs index. `.gitignore` already ignores `src/usr/datasets`.

### 4e — tests and distribution

`tests/test_v3_workspace_layout.py` covers the v3 round trip, refusal of storage 2, composition from `src/usr/config` and dataset discovery; SDK refusal and `results/sites` receipts are covered by `test_app_plugin_sdk` and the site suites. The distribution probe now refuses any `src/usr` payload. Built with a fresh setuptools build base and installed with `--no-deps` into `/tmp/theta-boundary-baseline/env` after removing its stale `site-packages/src`, it passed: headless imports, an unchanged clean workspace, two-workspace agreement, relocated checkpoint reload, and no `in/`.

### Evidence

All 229 test files were run one at a time on the branch, and compared by test ID with the same files on a baseline worktree at `e3fdbcd` (227 files exist there). Totals: 2519 passed, 269 failed and 0 errors on the branch, against 2506 passed, 279 failed and 7 errors on the baseline. The only new failures wait for v3 sibling packages (4g):

- `test_baselines_and_suites` (nsfr/neumann and dl-basics archives are storage 2) and `test_rulesets::test_authored_nsfr_package_loads_ruleset_predicates_in_fresh_process` (builds from the v2 `../theta-plugins` source);
- `test_data_loader_bindings` (7) and `test_blendrl_rich_monitors::test_actual_blendrl_training_is_identical_with_and_without_rich_monitors`: nsfr does not yet export `get_nsfr_model`;
- `test_wet_runs::test_preflight_blocks_supervised_without_dataset`: the installed `mlp` package still imports `provider_object`.

Newly passing: `test_config_constraints` (8), `test_themes_reconstruction` (11), `test_experiment_snapshot` (4), `test_folder_crud` (4), and one each in `test_component_actions` and `test_experiment_component_overrides`. Other failures (the Optuna-dependent tuning suites, `test_job_manager`, `test_filetree_toolbar` and others) are unchanged from the baseline. `test_layout_guard` and `test_v3_workspace_layout` pass. Ruff reports no new undefined or unused names in the changed files.

### Still open

- **4g.** Reconcile `../theta-hub` and `../theta-plugins` with the user first: `../theta-plugins` has a large uncommitted storage-v2 migration (455 paths, dated 2026-11-03 to 11-07). Then rebuild the sibling packages as v3, declare exports for nsfr and neumann, move installed `provider_object` and `plot.base` users to SDK 2, and check the two third-party app plugins against SDK 2.
- **Not retired, outside the 4c list.** Behavior that still reads older data: the monitor's CSV fallback, `legacy_flat` tuner fields, markerless tuning summaries and older result-folder names. Shared-workspace site execution records `publication_state: 'legacy'`. Several desktop widgets keep attribute aliases (`btn_new_group`, `self.tree`, `current_viewer` proxies).

## Phase 4 unit 4f — 2026-11-14

**Status:** this checkout's research workspace moved from `in/` to `src/usr` with the user's go-ahead, after they reviewed the dry run. `in/` no longer exists.

- **Plan.** 107 files under `in/`. 42 moved: tracked files by `git mv`, the edited themes-dev experiment overrides, the user's experiments, the dataset NPZ files and the `mimic` dataset symlink (moved as a link, not followed). 62 were deleted as redundant: 58 are byte-identical to the profile their package provides. The other four are `agent/sb3/*.yaml`, which have the same YAML data as sb3-suite and differ only in key order; the user chose to drop them. 3 were junk (`.DS_Store`, a `.pyc`, `in/envs/.gitkeep`). There were no conflicts and nothing was unclassified.
- **State.** 22 manifests (installed packages and two retained development packages) were converted to v3, dropping `config_hashes`, `config_backups` and `user_config_paths`. Two `settings.toml` lines were rewritten to `src/usr/config`. The retired `.thetaide/retained_configs` (5 files) and `.thetaide/config_backups` (9) were removed, and the Hub index cache was cleared. Run history, workspace cleanups and site objects still mention the old paths and were left as they are.
- **Backup.** Everything removed or changed is in `.thetaide/migrations/20261114T163513Z/` (`in/` with links kept as links, the original manifests, state files and `plan.json`). Every file's hash was checked against the plan before anything moved.

### Evidence

- The migration's verify step passed: `in/` is gone, every moved file and the symlink match their recorded hashes, every redundant file's proof still holds, every manifest is v3 with no old paths, `git grep` finds no old paths outside `tests/`, `docs/` and `results/`, and no live state file mentions them.
- Every experiment of this workspace was composed with the old code before the move and the new code after it. The effective configs are identical for all 14 experiments that composed both times. Dataset errors differ only in the path they name. Two outcomes changed, both because the new code recognizes authored v3 packages. `BlendRL/Cartpole/sweep_sustainedSolve` now composes, because `site/ncshare/common` resolves from the authored `src/usr/sites/ncshare`. Its dataset step then stops because optuna is not installed. The themes-dev datasets now resolve to files that exist in the authored package. Both `BlendRL/MountainCar` experiments still reference the missing `experiment/Belndrl Bench/_base`, as before.
- The failures that depended on the workspace now pass: `test_data_loader_bindings` (16), `test_provider_organization` (31), `test_workspace_defaults`, `test_execution_contract::test_old_selected_hub_profile_fails` and the run guard in `test_baselines_and_suites`. What still fails in those files also fails on the baseline (`test_pipeline_sweep_invokes_worker_once_per_trial`, which needs optuna) or waits for the v3 sibling packages in 4g (nsfr/neumann, dl-basics).

## Phase 4 units 4a and 4b — 2026-11-14

**Status:** package format v3 and the path cutover are in one commit, with the shared test fixture and the ported tests. The user's workspace has not moved yet (unit 4f), so suites that read this checkout's research workspace fail until it does.

- **One commit for 4a, 4b and the 4e port.** The scaffolder, exporters and installer already write `src/usr/config`, so the v3 format could not pass its tests without the cutover. The plan also asks that the port land with 4b so the suite never mixes layouts.
- **Install model (user decision, 2026-11-14: "stop copying").** Package profiles resolve from the package by selector and are never copied into the workspace. `src/usr/config` holds only the user's own files and explicit overrides; uninstall and update never touch it. The hash, backup, retained-config and `user_config_paths` machinery is deleted. The config tree lists package profiles through the resolver, and editing a managed one requires an override (`config override`). Profiles of authored (writable) packages are edited in place. Needs the user's installed-GUI check.
- **Format v3.** `storage_version: 3` only. Config destinations, `x-theta-config-schemas` keys, `config_path(s)`, compatibility selectors and test-suite paths are `<group>/<name>.yaml` selectors; absolute paths, `..` and `in/` or `src/` prefixes are rejected. Every tracked manifest (bundled, authored, optional components, fixtures, the distribution probe, generated site and objective manifests) is v3.
- **Paths.** `WorkspacePaths.config` is `src/usr/config` and `.datasets` is `src/usr/datasets`; the `user_config`/`user_inputs` aliases are gone. A workspace is recognized by `src/usr` or `.thetaide/workspace.json`, and `from_config_root` rejects any other root. Resolver precedence: `src/usr/config`, selected provider profile, framework fallback. A user file now reports origin `workspace`, so it can no longer be confused with an authored package (origin `user`); the config tree had dropped authored package profiles because of that clash.
- **Removed with no replacement.** The `CONFIG_DIR` rebinding constants (`compose`, `catalog`, `config_service`), `in/rules` detection (`reject_legacy_rules_paths`, `require_canonical_rules_layout`), `in/` checks in archive validation, and the inventory's `in/` → `src/usr` destination mapping.
- **Consumers.** Site policy, snapshot and transfer use `src/usr/datasets` (snapshots exclude it); the Slurm executor links it; the worker snapshot hashes each declared selector's effective profile, so an override created after launch is detected; component context and validation resolve profiles by selector; the arXiv catalog lists effective profiles; the reports plotter reads agent profiles through the resolver; the launch guard checks profiles through the resolver. `prepare_tabular.py` moved next to the core NPZ data module (`python -m src.app.core.paradigm_impls.base.prepare_tabular`, `--output` defaulting to the workspace dataset root) rather than into a Hub package, so core messages name no package. The documentation panel, README, CONTRIBUTING, `.gitignore`, `MANIFEST.in`, `pyproject.toml` and `pytest.ini` describe the new layout; the guides under `docs/` are unit 4d.
- **Tests.** `fixture_workspace` builds an isolated `src/usr/config` from every non-bundled profile this checkout resolves (`copy_research_config`) and selects it with `THETA_WORKSPACE`. All test files that built `in/config` layouts were ported; tests for retired behavior (legacy rules locations, materialization, retained configs) were deleted or rewritten. New: `tests/test_layout_guard.py` fails if tracked source outside `tests/`, `docs/` and `results/` mentions `in/config`, `in/datasets`, `in/envs` or `in/inputs`.

### Evidence

Each of the 87 affected test files was run on its own, on the branch and on a baseline worktree at `e3fdbcd`, and the failures compared by test ID. Totals are 1084 passed and 130 failed on the branch, against 1092 passed and 129 failed on the baseline. Deleted tests for retired behavior account for the drop in passes. No failure is new except these, which are expected:

- **Until this workspace moves (4f):** `test_data_loader_bindings` (7, the BlendRL agents resolve through `src/usr/config`), `test_execution_contract::test_old_selected_hub_profile_fails`, `test_provider_organization::test_legacy_import_contract_identity_and_paradigms`, `test_workspace_defaults::test_checkout_does_not_track_component_or_experiment_configs` (tracked `.gitkeep` paths) and `test_baselines_and_suites::test_missing_component_run_guard`.
- **Until the sibling packages are v3 (4g):** `test_rulesets::test_authored_nsfr_package_loads_ruleset_predicates_in_fresh_process` and `test_baselines_and_suites` (nsfr/neumann and dl-basics archives are storage 2).

`test_layout_guard` passes. Failures that already occur on the baseline (`test_job_manager`, `test_experiment_pane`, `test_experiment_exporter` and others) are unchanged.

### Still open

- **4f.** The dry run is ready and awaits the user's go-ahead. Installed manifests are still v2 on disk, so the catalog ignores them until 4f converts them.
- **4g.** The Hub sources in `../theta-plugins` are v2, so the NSFR, Neumann and DL-basics install tests fail until they are rebuilt.

## Pre-phase-4 cleanup — 2026-11-14

**Status:** the phase 2 and 3 items that could be finished locally are done. The branch's work is now committed in reviewable units.

- **Commits.** The working tree was split by file into phases 0–1, 2 and 3, the desktop fixes, the config panel refactor and the documentation. Shared files went to one commit each, so intermediate commits are not individually tested; the final tree is the validated one.
- **SDK 1.1.** `src.app.api.models` exports `register_model`, `build_model` and `get_model_class`. `src.app.api.tuning` adds `ObjectiveResult`, `RunResult` and `TrialResults`. The bundled `final_metric`, `learning_curve` and `sequential_spec` packages, the scaffolder templates, the documentation panel and the plugin-author guides use them. `tests/test_sdk_boundary.py` now also guards `src/plugins/components/bundled` (62 cases). Installed Hub packages and authored `src/usr` packages still import the private modules directly; the exception register records this.
- **Third-party libraries.** `scripts/generate_third_party_libraries.py` regenerates `docs/reference/THIRD_PARTY_LIBRARIES.md` from `pyproject.toml`, the optional components' pip dependencies and installed metadata. Optuna and SQLAlchemy are listed under the Optuna component. `hydra-optuna-sweeper` is gone, because the component uses Theta's own sweeper. The `video` extra declares `pygame` (LGPL) and `vidmaker` (GPL-3.0), and no host code imports either.
- **Installed distribution.** A fresh wheel and sdist (separate build base), installed into a clone of the baseline environment with its stale `src` removed, passed `verify_installed_distribution.py`. Headless imports stayed free of Qt, Optuna and SQLAlchemy, a clean workspace was unchanged, two workspaces agreed in GUI metadata, composition and a real worker (31 and 32), and a checkpoint reloaded after relocation. Bundled packages exclude Optuna and Slurm, and the wheel requires neither Optuna nor SQLAlchemy. The baseline environment still has Optuna and SQLAlchemy 2.1.8 installed from earlier, so rebuild it from scratch before the phase 6 gate.
- **Tests.** The SDK, model-builder, sequential-spec, scaffolder, app-plugin, provider and documentation suites passed. Seven files with failures had identical failure sets with and without the change (38 failures from the categories recorded below). One `test_results_panel` case failed once and passed in three full reruns.
- **Phase 4 input.** The probe writes a storage-v2 manifest with `in/config` destinations. Under the hard cutover it must move to the new destination representation in the same change.

## Desktop regression fixes — 2026-11-14

**Status:** the desktop again opens as fast as before the refactor and no longer aborts after launch. The [regression report](STARTUP_REGRESSION_20261114.md) has the profile and the timings.

- **Crash and slow startup.** The site-sync availability checks added on this branch polled every second on the GUI thread. Each poll listed every site, and that built one `ConfigResolver` per site selector, about 1.9 s per poll. The poll also closed a menu that had been deleted along with a replaced button. Fixed by:
  - giving the control ownership of its menu;
  - checking availability on events instead of polling;
  - using one resolver per site listing;
  - sharing one resolver and listing within `provider_discovery_snapshot()`.

  First paint is now 5.3 s, against 66.8 s before the fix and 5.8 s at the 2026-11-12 baseline. Validation is ready at 13.3 s.
- **`ConfigTree.refresh_groups()`** now also invalidates the resolver's selector index, so a newly created group folder is visible to group inheritance.
- **Branch-only test failures closed.**
  - `test_startup_services`: the compose mock accepts `workspace=`.
  - The three `test_ui_adjustments` cases: the fixture uses a real `in/config` workspace layout, plus the `refresh_groups` fix above.
- **Frontend failures that also failed at HEAD, now fixed.** These tests predated storage v2 and the lazy settings and editor startup:
  - `test_plugin_system` installs its fixture as a native package.
  - The documentation, plugin-system and text-editor host tests build the settings pages, and show the window before waiting for the editor renderer.
  - `test_hotkeys` drops the Job queue pane removed in `eb7aad1`.
- **Evidence.** The same 63-file per-file run gave 799 passed and 49 failed, against 764 and 57 before. Every remaining failure is in a file that also fails at HEAD: research-workspace inputs, tuning and objective suites, and `test_parallel_runs`. That file still targets the removed queue pane and monitor switcher and synchronous runner readiness, and needs rewriting against the current Monitor. Also passing: `test_hotkeys` (23), `test_ui_adjustments` (12), the site and sync suites, and two new regression tests in `test_experiment_site_sync.py`. A 150-second diagnostics launch exited cleanly with a longest event-loop gap of 295 ms.

## Phase 3 SDK and extraction test debt — 2026-11-13 third follow-up

**Status:** phase 3's local exit gate is met. The bundled plugins, both optional components and an independent fixture plugin operate without importing private host modules or touching the host window, and headless catalogs need neither Qt nor plugin code. The installed-GUI desktop check and a rebuilt-wheel distribution probe were not run.

### Extraction test debt (step 0)

- All 21 branch-only failures from the second follow-up's per-file run are fixed. Recipes now name `tuning.component: optuna` and run in a workspace with the Optuna component installed (`test_sweep_preflight`, `test_tuning_objective`, `test_objective_components`). The synced monitor and sweeper tests install Optuna into the workspace they exercise. The capacity mock accepts `paths`, and the plotting test passes its workspace explicitly.
- A wider run found 7 more branch-only failures and two files that no longer collected (11 tests), all with the same kinds of cause. All are fixed. `test_site_launch` and `test_site_workflow` load the installed Slurm component instead of the API-1 adapter. The launch window has a selected workspace, and a composed site connection satisfies the launch contribution's readiness. `test_site_ssh_timeout` patches the generic transport module. `test_site_transfer_direction` configures a real site with an installed scheduler. `test_sweep_visibility` gives its viewers an Optuna workspace.
- In `test_parameter_forms`, the shared fixture installs Optuna and Slurm, so the "independent tuner" test now removes Optuna to leave its non-Optuna provider as the only tuner.

### SDK (steps 1–4)

- **Classification.** All 27 private modules the optional components used are generic host functionality: transport, receipts, snapshots, publication, policy, tuner plans, objectives, workers and resource admission. They are promoted rather than moved into the components. Three host process entry points that the components wrote as module-path strings are now named SDK commands: `provider_worker_command`, `task_worker_command` and `site_cli_command`. Slurm SSH shortcuts use a new generic `run_pipeline.py site ssh SITE` command. Shortcuts created earlier with `-m src.app.sites.slurm` keep working through the API-1 adapter.
- **Facade.** [`src.app.api`](../reference/SDK.md) (SDK 1.0) has the modules `workspace`, `errors`, `config`, `execution`, `tuning`, `sites` and `desktop`. Exports are canonical host objects resolved lazily without caching, so test monkeypatches on defining modules still apply. Headless modules load no Qt, Optuna or SQLAlchemy. The generated SBATCH script is unchanged.
- **Optional components.** Slurm and Optuna import only `src.app.api`.
- **Bundled panels.** The text editor, model builder, TensorBoard, documentation and arXiv panels use `src.app.api.desktop` and new `PluginContext` services. The services cover editor registration (`activated_explicitly` keeps the old default-editor rule), editor preferences, zoom, appearance listeners, the config catalog, notifications about changed config files, managed-file checks, the Hub index and its listeners, tree-toolbar persistence limited to its settings section, a dialog parent and startup marks. The context withdraws all of them on cleanup. arXiv's module-level `for_process()` config root and its access to the private `_window` are gone, and the documentation panel's plugin-author guidance teaches the SDK.
- **Compatibility checks.** Bundled `plugin.json` files declare `"sdk": ">=1.0,<2"`. `PluginManager` refuses a malformed or unsatisfied range with a logged reason before importing plugin code. Plugins without a declaration are treated as written for 1.0. `PluginContext.window` remains but is unsupported (see the exception register).

### Evidence

- New tests: `tests/test_sdk_boundary.py` (48 cases) and `tests/test_app_plugin_sdk.py` (6).
  - The boundary test parses every file in `optional_components/` and `src/plugins/app/bundled/` for private imports, including function-local imports, dynamic imports and module-path strings. It rejects `context.window`/`_window` access, checks that exports are canonical and that headless imports stay free of Qt, Optuna and SQLAlchemy, and covers the SDK range parser.
  - The fixture test installs `lattice_notes`, a plugin that imports only the SDK. It checks activation from an unrelated CWD using the selected workspace, sidebar, command, plugin settings, editor routing, the default-editor rule on explicit enabling versus restore, editor preferences, appearance listeners, and full withdrawal on disable. It checks that an incompatible or malformed SDK range is refused before the plugin module is imported (an import marker stays empty), and that headless catalog discovery reads its SDK range with neither Qt nor plugin code loaded.
- Final per-file run over 63 affected files (site, scheduler, tuner and sweep suites, app-plugin and panel suites, and the new tests): 764 passed, 57 failed. Each failing file was rerun against a detached worktree of `4738cae` with the research workspace's untracked inputs copied in. 55 of the 57 also fail at HEAD. Their causes are the categories recorded in the second follow-up, plus some that predate this branch: the plugin-system `sample_plugin` discovery, the documentation settings page, hotkey pane order, the `test_text_editor` host-window tests, missing research objectives and old parallel-run UI expectations. The other two were the `test_parameter_forms` fixture fixed above and `test_startup_services::test_obsolete_queued_compositions_never_execute_and_late_results_are_dropped`.
- Branch-only failures that predate this follow-up and remain open: that `test_startup_services` case and three `test_ui_adjustments` cases (config-form defaults and the BlendRL parameter adder). None involves the SDK or the plugin host; they exercise the local backend's compose queue and config-viewer form rendering, both changed earlier on this branch. They are not yet investigated.
- One run of `test_text_editor` aborted inside Qt WebEngine during a test that also fails at HEAD. Two reruns gave 23 passed and 2 failed, matching HEAD.
- `git diff --check`, compilation of every changed Python file, and a ruff undefined-name check (F821) over the SDK, plugin host, components and bundled plugins passed. The remaining ruff findings are unused imports that predate this work.
- Not run: a rebuilt wheel with `verify_installed_distribution.py` (`src.app.*` package discovery already includes `src.app.api`), and an installed-GUI desktop session.

### Still open

- Bundled component packages (`src/plugins/components/bundled`) still import `src.app.core.model_registry` and `src.app.objectives`. Authored packages are not checked. Both belong to phase 7 enforcement; registration APIs need SDK exports or declared exports first.
- `THIRD_PARTY_LIBRARIES.md` regeneration, the real cluster smoke run and the installed-GUI matrix carry over from phase 2.

## Phase 1 context cutover and phase 2 extraction — 2026-11-13 second follow-up

This section supersedes the open items in the checklist below where they conflict. Older entries remain historical evidence.

**Status:** the phase 1 host-context gap and the local phase 2 extraction gate are closed. Neither phase is declared complete: the installed-GUI desktop matrix, a real cluster smoke run and the phase 3 SDK (needed by the optional components themselves) remain open.

### Implemented

- **Optional components.** Slurm and Optuna are maintained in `optional_components/{slurm,optuna}`, excluded from the wheel/sdist and installed through the ordinary native lifecycle (`scripts/package_optional_components.py`, then `component install`). The bundled `optuna` and `site_templates` packages and the framework ARC/NCShare YAMLs are removed; the Slurm component owns those site templates and their forwarding selectors. The `[optuna]` extra is gone; Optuna declares `optuna`/`sqlalchemy` itself.
- **Scheduler contract v1** ([reference](../reference/SCHEDULER_CONTRACT.md)). Execution, status, cancel and submit dispatch through the selected `scheduler` capability. `site check` builds its remote probe from the scheduler's `remote_requirements` hook instead of hardcoded `sbatch`/`squeue`; the generated script for Slurm is byte-identical to before. Desktop site setup is the `settings.site` contribution slot (`site_launch.SiteSetup` no longer imports a provider by ID).
- **No silent scheduler fallback.** A non-local site profile without `site.scheduler` now raises an actionable error. Before extraction it implied Slurm; the in-progress extraction had silently turned it into a local run. New site components record only an explicit `--scheduler`/`scheduler=` (else `scheduler: null` with a comment). Core's NCShare alias table is replaced by aliases derived from provider forwarding files; authored variants that merely inherit their defaults stay selectable.
- **CLI regression fixed.** `run_pipeline.py` imported Slurm header helpers at module level, so `--help` and every run crashed in a workspace without the Slurm component. The import and the unreachable Slurm dispatch branch are removed; help text names no provider or institution.
- **Explicit workspace context.** `configure_workspace` records the entry point's selection in the process environment and no longer rebinds module constants. Desktop job/runner managers, the config viewer, monitors section, run-ID allocation, saved-config loading and the TensorBoard plugin use the selected `WorkspacePaths` instead of import-time `PROJECT_ROOT`/`LOGS_DIR`/`JOBS_DIR`/`TENSORBOARD_DIR`. Local runner, experiment snapshots, execution manifests and result renaming no longer fall back to `WorkspacePaths(Path.cwd())`; the local runner resolves the workspace once and passes it to every phase. `PluginContext.workspace_paths()` is the public accessor; `get_workspace_dir`/`get_data_dir` honor an external state directory instead of guessing from the store path or CWD.
- **Declared cross-package exports.** Manifest `exports` plus `provider_export(package, name)` validate the calling package's declared dependency and PEP 440 version, resolve in the workspace that loaded the caller, and reject undeclared names. Native archive validation checks export targets. `provider_object` remains the bounded legacy adapter ([plugin storage](../reference/PLUGIN_STORAGE.md#cross-package-exports)).
- Smaller fixes: a missing `os` import in `PluginContext` cleanup (present at HEAD) meant the fallback process kill silently never ran; the Optuna dependency error now names its missing libraries; provider-neutral wording in queue, variability and workflow messages.

### Evidence

- New tests: `test_workspace_entrypoints.py` (6: omitted context uses the process selection with a decoy workspace as CWD, constants are not rebound, desktop managers, plugin context, provider-free `--help`), `test_provider_exports.py` (9) and three site cases in `test_optional_component_extraction.py` (forwarding aliases, explicit-only scheduler on new sites, `site check` from an independently named scheduler's requirements, including malformed-requirement rejection).
- Legacy tests that imported bundled Optuna/Slurm paths or assumed preinstalled Slurm now install the optional component into their own workspace (`tests/component_fixtures.py: install_optional/optional_workspace/optional_module`). Updated fixtures declare `scheduler_component` on new-format receipts and `site.scheduler` on cluster configs. Focused results: sites/site ownership/Slurm headers/cluster logs/remote queue 57 passed and 1 failed; NCShare hardening 33 passed and 2 failed; sweep shutdown 4 passed; tuner selection/tuning features 57 passed; extraction 8 passed. The failures are pre-existing: the research workspace's `sb3-dev` runner has had no `main` since HEAD, and the installed `sb3-suite@1.1.1` profile still declares removed execution fields.
- **Per-file run, after the second follow-up.** 24 related test files, each run in its own process (`QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`). Eleven files had no failures: workspace entry points 6, provider exports 9, extraction 8, sites 23, site ownership 5, Slurm 4, monitor remote queue 3, sweep shutdown 4, sweep monitoring 8, study-name allocation 3, resource budget 24, queue cancellation 6 and parameter importance 6. The other thirteen had 56 failures in total. Each failing file was also run in a detached worktree of `4738cae` (HEAD, before any layer-boundary work). The research workspace's untracked `src/usr`, `in` and installed packages were copied into that worktree, along with `.thetaide` minus `runs`, `site-objects` and `workspace-cleanups`. The research workspace itself was not modified.
  - **35 also fail at HEAD**, so they predate this branch. Causes: missing `src/usr` runners and models (`sb3-suite/sb3_runner.py`, `blendrl`, `sb3_error_runner.main`), `sb3-suite@1.1.1` declaring removed execution fields, and direct `@hydra.main` calls against `in/config` that miss `site/local/default`. Some of these tests now fail with a different message on the branch (for example, the Optuna-component error), but they were already failing.
  - **21 fail only on the branch.** All 21 are test fixtures that this work's intentional behavior changes have not caught up with yet. None shows a behavior regression, but they are open test debt and not passing gates:
    - 13 tuning recipes have no `tuning.component`. Since explicit tuner selection, the selection error fires before the check the test expects: `test_sweep_preflight` (all 10), `test_tuning_objective` (`test_method_preview_includes_the_search_space`, `test_command_compiler_never_drops_a_method_search_space`) and `test_objective_components::test_validation_requires_data_and_rejects_pruning`.
    - 6 never install the optional Optuna component in their workspace: `test_synced_sweep_monitor` (5; summaries are empty because the tmp workspace has no ready tuner) and `test_tuner_components::test_optuna_sweeper_does_not_import_lightning_when_pruning_is_off` (the saved-target adapter raises the actionable `ModuleNotFoundError`). Fix with `tests/component_fixtures.py: optional_workspace`.
    - 1 mock lambda does not accept the explicit `paths` keyword: `test_capacity_recovery::test_training_holds_one_slot_until_completion_without_capacity_rechecks`.
    - 1 test expects plot output relative to its CWD: `test_tuning_trial_artifacts::test_plotman_reads_generic_tuning_summary_and_writes_table_and_figure`. Plotting now resolves against the selected workspace, so the test must pass one.
- Installed distribution: a fresh wheel/sdist built with a separate setuptools build base, installed into a copy-on-write clone of the earlier baseline environment after removing stale files that no RECORD covered. The original baseline environment still contains bundled `optuna`/`site_templates` leftovers from an earlier install and should not be reused. `verify_installed_distribution.py` passed: ten bundled packages (no Optuna, Slurm or site templates), headless imports without Qt/Optuna/SQLAlchemy, unchanged clean workspace, two-workspace GUI/coordinator/worker agreement and relocated checkpoint reload. Optuna-named files in the wheel are only the registered saved-target forwarding shims and the Hydra discovery shim. With the installed application only, both optional archives installed into an external workspace via `python -m run_pipeline component install`; the scheduler and tuner resolved as `local_archive`, `ncshare/common` was selectable with its alias hidden, and `src.app.tuners.optuna.component` loaded from the installation. Artifacts and the report are in the session scratch directory and were not published.
- `git diff --check`, compilation of every changed/new Python file, and an undefined-name lint over the changed trees passed.

### Still open

- **Phase 3 SDK dependency of the components.** The optional components import about 27 private core modules (for example `src.app.sites.storage`, `src.app.pipeline.tuner_execution`, `frontend.widgets`). Extraction removed core's dependency on them, not their dependency on core internals. This is the next blocker for a stable plugin boundary.
- Installed-GUI desktop interaction matrix; a real cluster smoke run (submission, interruption, publication, synced monitor, continuation) with installed optional components.
- `THIRD_PARTY_LIBRARIES.md` still lists Optuna, SQLAlchemy and `hydra-optuna-sweeper` as `pyproject.toml` dependencies. Regenerate it from installed metadata rather than hand-editing.
- The research workspace at the repository root has neither optional component installed, so sweeps and cluster sites are absent there until they are installed. Installing them is a workspace change and was left to the user.
- Phases 4–7 are unchanged (see below). No component was published, no cluster job submitted, no research data or editable configuration migrated.

## Phase 0/1 acceptance checklist — 2026-11-13 follow-up

This checklist supersedes the earlier outstanding-baseline statement; older validation entries below remain historical evidence.

1. **Implemented and verified:** isolated installed-wheel baseline; wheel/sdist payload inspection; clean external-workspace read-only discovery; GUI metadata/coordinator/worker agreement for two independent external workspaces from an unrelated directory; native checkpoint reload after relocation in a fresh installed process; local portable-tuner allocation/publication/continuation; override update/replacement, conflict, uninstall, failed publication and abandoned-journal recovery; effective-profile export/plotting and usr-only editor group metadata.
2. **Implemented but not fully verified:** desktop figure construction now receives workspace context; component panel/tree actions pass their workspace; editor helpers preserve the tree's application/state context. The complete desktop editor/figure/export interaction matrix has not passed in an installed GUI environment. Headless installed metadata verification is not a desktop launch.
3. **Missing implementation / locally open (closed by the second follow-up above):** remaining service adapters still use process-selected globals (`services.__init__`, catalog/results/job adapters and local-runner context propagation); their callers need a complete explicit-context cutover. Historical top-level site selector YAMLs remain small framework forwarding adapters. Standalone optional Slurm/Optuna extraction and scheduler settings/commands/monitor contribution wiring remain unimplemented. Neither phase is declared complete.
4. **External acceptance evidence:** no real cluster job was submitted. Real scheduler allocation, interruption, publication, synced monitor and continuation evidence still requires an authorized cluster smoke run using an installed application and selected providers. Installed desktop interaction verification additionally needs the GUI dependencies/display environment; current GUI checks use the existing frontend environment.

### Concrete changes and evidence in this follow-up

- Added `scripts/verify_installed_distribution.py`. It runs the probe with an isolated interpreter (`-I`), strips inherited Python/Theta environment variables, uses an unrelated temporary CWD and verifies application imports are inside that environment. There is no checkout import overlay or `--system-site-packages`. Wheel dependencies were installed normally into `/tmp/theta-boundary-baseline/env`.
- The probe verifies clean discovery without Qt/Optuna/SQLAlchemy imports or workspace writes, metadata/composition/real worker `--cfg job` values (31 and 32) in two usr-only workspaces, and real native-package install plus pickle reload after workspace relocation in a fresh installed process. Artifact hashes, dependency versions, bundled package inventory and payload counts are recorded in [the distribution evidence](LAYER_BOUNDARY_DISTRIBUTION_20261113.json). Full file lists/inventories and logs remain in `/tmp/theta-boundary-baseline/`.
- Installed artifacts contain twelve bundled packages, including the then-bundled Optuna provider and the transitional `site_templates` config provider (both since moved to optional components). Application assets/framework defaults and fixed usr compatibility shims are present. Installed/user research payloads, input bytes, state, virtual environments and npm dependency trees are excluded. The initial wheel incorrectly included `node_modules`; explicit setuptools package-data exclusions and a fresh build directory fixed it.
- Exporters/scaffolder accept `WorkspacePaths` without discarding application/state identity, and choose process context at construction only for the legacy omitted-context adapter. Experiment/group/template/workflow export and component config exports read effective resolver profiles, retaining storage-v2 archive selectors. Scaffolding writes new editable profiles to `src/usr/config`; provider copies stay in package `configs/`. No existing profile was moved.
- Plot configs, inherited defaults, saved run config and metrics/output paths resolve against the selected workspace. Built-in plotters accept explicit paths. The bounded API-1 `create_plotter` adapter scopes older no-argument BasePlotter constructors with a ContextVar, resets it on failure and does not change CWD or environment. Desktop figure sources and the plotting manager use it. Provider lookup, plotting subprocess environment and interpreter selection accept explicit context.
- Moved ARC/NCShare default payloads out of framework defaults into `site_templates` (since moved again, into the optional Slurm component); preserved selectors, bytes and forwarding YAMLs. Named site variants follow the selected site's provider for config/schema ownership. Managed template connection edits explicitly create a usr override; authored site settings remain user-owned. Unknown sites are created only through the existing explicit configure action.
- Development inspection/read/validation now use effective profile paths, including usr overrides; loose copied profiles still validate against their selected model provider schema. Source payloads are never rewritten by these reads/exports.
- Final focused headless acceptance run: 74 passed (workspace/provenance, lifecycle, plotting, portable continuation and sites). GUI document/scaffolding/site checks: 34 passed. Component development: seven passed, one legacy experiment fixture excluded after the unrestricted run rejected its algorithm's missing execution contract. Checkpoint suites: 13 passed. Runs overlap. `git diff --check` and changed tracked Python syntax checks passed.
- Broader checks exposed old experiment-export fixtures with undeclared `demo_env`/`atari/seaquest` providers and one config-viewer fixture error; these are not passing full-desktop/export gates. The schema-validation regression discovered in the unrestricted development run was fixed and its test passes. No import-overlay result is used as installed-distribution evidence.
- Input preservation comparison against the earlier private inventory checked 659 input/authored/installed files. 656 match hashes/modes; the three differing editable SB3/d3rlpy/THEMES manifests were already modified on entry and are recorded by the preceding delivery status. This follow-up did not edit them. No dataset bytes were moved and phase 4 is inactive.

Reproduce the installed probe after building wheel/sdist and installing the wheel into a fresh virtual environment:

```sh
venv/bin/python scripts/verify_installed_distribution.py \
  --python /tmp/theta-boundary-baseline/env/bin/python \
  --wheel /tmp/theta-boundary-baseline/theta_ide-0.1.0-py3-none-any.whl \
  --sdist /tmp/theta-boundary-baseline/theta_ide-0.1.0.tar.gz \
  --output /tmp/theta-boundary-baseline/report.json
```

The script's module documentation gives build/install commands. Use a fresh setuptools build directory for wheel builds to avoid stale package-data artifacts. Phase 2 extraction was not resumed while the remaining phase 1 host-context/desktop gates remain open. Existing tuner transport/contribution work is preserved; it is not standalone optional delivery.

## Implemented

Phase 0 foundation: ownership/import/configuration/compatibility contract, finite exception register, read-only reproducible inventory command. The filtered initial local inventory is at `/tmp/theta-layer-boundary-inventory-initial-filtered.json` (private machine artifact): 1,163 files, 653 tracked, 12,336,512 bytes, 43 catalog packages and 12 structured dataset references; catalog diagnostics were empty. The original raw report remains at `/tmp/theta-layer-boundary-inventory.json`; it also included the frontend's virtual environment. The command now recognizes `pyvenv.cfg` and explicitly records excluded interpreter environments. It includes ignored research files and the tracked `.npz`, source hashes, modes, selectors, destination proposals and ownership state. These are populated-workspace counts, not distribution guarantees. No research inputs were migrated or deleted.

A separate clean external-workspace source inventory is at `/tmp/theta-layer-boundary-clean-inventory.json`: 477 application files, 11 bundled packages, no catalog diagnostics and no frontend/Qt imports. This uses the development application, not an isolated wheel installation. It confirms that the current distribution still includes the tuner, which remains a phase 2 extraction gap.

Phase 1 implementation:

- Named user-config/input, framework-default and provider-config paths.
- A metadata-only `ConfigResolver` returns selector, package owner, origin, effective file, override state and selected schema. Native config payload mappings define package ownership. Duplicate owners fail explicitly; capability ownership overrides select profile values and schema together. Framework defaults remain fallbacks, with complete user profiles taking precedence.
- GUI `ConfigTree`, configuration catalogs, method profile/schema selection, in-process composition, CLI snapshots and worker Hydra discovery use the resolver. A scoped Hydra config source preserves package directives and defaults composition without generating or mutating package files.
- Composition, validation, tuner lookup, app config services, dataset discovery and tabular loading accept explicit workspace context. Runtime composition no longer chooses a workspace at module import. Legacy `CONFIG_DIR` constants remain compatibility surfaces for existing desktop adapters/tests.
- Explicit `component config override <selector>` creates a full editable copy under `src/usr/config`, journals publication, records origin/digest and never overwrites an existing copy. Provider uninstall leaves it intact. Failure during provenance publication restores files through the existing transaction mechanism.
- Metadata/preview fingerprints include workspace identity, provider config contents, user overrides and explicit provider selections. Draft previews keep their real workspace context.

Phase 2 delivery C implementation (runner features and resource hints):

- Added provider-neutral execution feature declarations for supported input selections, evaluation ownership, resume/reload modes, checkpoint formats, tuning/winner promotion and episode outcomes. Native manifest schemas validate declarations; missing features use conservative defaults, with the retained core Lightning implementation preserving its existing behavior.
- Replaced the execution backend allowlist and SB3/d3rlpy-specific rejection branches with feature checks. Arbitrarily named adapters dispatch through the selected capability. Optional package-relative, read-only selected-provider preflight hooks run in coordinator and worker validation.
- Removed SB3 substring/model-policy checks from core resource planning. Static hints and an optional selected-provider hook provide CPU, base-memory and accelerator demand; existing model/data estimates, learned usage and user admission budgets remain enforced. Launch previews and worker admission pass explicit workspace context. Relative dataset estimates resolve from that workspace.
- Updated only the editable SB3, d3rlpy and THEMES manifests to declare existing execution behavior; SB3 supplies a lightweight resource hook. No installed package, dataset bytes, package configuration or algorithm implementation was migrated. Historical top-level episode-outcome metadata remains readable.
- Added independent-provider fixtures for arbitrary backend dispatch, feature rejection/acceptance, read-only preflight, workspace isolation and resource policy enforcement. Updated the capacity fixture to select the migrated SB3 provider and a stale launch-test mock to accept the existing workspace keyword.

Phase 2 delivery D, portable tuner execution implementation:

- Added the optional `PortableTuner.remote_api_version = 1` contract and serialized `theta.tuner-execution/v1` plans. Local tuner API v1 remains compatible. Plans declare execution-scoped artifacts, concurrency/trial count, storage kind, dependency distributions and continuation notes; providers supply runtime bindings, consistent snapshots, tuning-block compatibility normalization, state validation/repair and optional monitor summaries.
- Submission, scratch and retained-workspace execution, artifact ownership, publication, continuation and synced study monitors now use the selected tuner's plan. Core no longer chooses Optuna paths/options, interprets Optuna trial states or hardcodes its dependency-version list in those paths. CPU/resource restrictions, publication queue/byte/cadence budgets, object hashes, source/scientific identity, lineage ownership and termination checks remain host policy. State plus checkpoint restore sizes are checked before writing.
- Moved SQLite storage bindings, native consistency checks and interrupted-trial repair into Optuna's provider. Added public generic publication helpers and migrated the provider's publisher hooks to them. Optuna's manifest declares its pip dependencies; generic site setup no longer unconditionally requests the application's `[optuna]` extra.
- Added a bounded metadata-only reader for old receipts without the new plan marker. It honors external storage, does not enable tuning for old fixed jobs, and preserves old local-only tuner overrides. New ordinary jobs explicitly record `null`, never resolve a tuner and receive no SQLite defaults. Existing receipt/publication paths remain readable; unknown legacy/portable schemas fail explicitly.
- An independent native JSON tuner fixture exercises real allocation worker bindings, publication, read-only monitor data and continuation with Optuna/SQLAlchemy imports blocked. Tests also cover retained-workspace execution, schema/path/binding rejection, legacy records, missing snapshot output and final publication of multiple artifacts through one queue slot.

This is the tuner state/transport portion of D, not the standalone extraction gate. At the time, the Optuna component was still bundled; see the second follow-up for its extraction and Slurm's. Explicit selection and the bounded historical receipt adapter are recorded below. Objective/pruning capabilities, contribution/readiness and tuner UI implementation are recorded below. See [portable tuner execution v1](../reference/TUNER_EXECUTION_CONTRACT.md).

Phase 2 delivery D, contribution/readiness implementation:

- Added metadata-only contribution API v1 with absent/incompatible/disabled/setup/ready states, declarative composed-config/distribution requirements, capability ownership alignment, payload-validated factories and explicit activation. Generic lifecycle scopes close removed/disabled/replaced contributions, retain unchanged mounts and roll back new mounts on activation failure.
- Added workspace-scoped extension enablement under the selected state directory, serialized by the existing lifecycle lock and active-package lease checks. The component CLI supports `extension status/enable/disable KIND SELECTOR`. Public capability loading rejects disabled/incompatible providers before importing them. Tuner execution and remote preparation enforce readiness; setup/settings remain available while unconfigured.
- The configuration editor now consumes tuner setup and parameter-range contributions. No compatible provider means no new tuning controls; saved unavailable recipes retain tuning/search spaces with diagnostics and read-only range values. Newly authored recipes explicitly select a provider and store required provider defaults in namespaced options. Field compatibility and conditional requirements are provider declarations, replacing Optuna ID branches in the form.
- Optuna declares its setup/range contributions and dependency readiness. Independent native site/command and tuner fixtures exercise metadata discovery without code imports, setup-to-ready transitions, disabling/removal, workspace isolation, lifecycle cleanup, conditional fields and preserved saved recipes. Floating shared hover actions detach when their owning row is destroyed.
- Added extension state to config/preview fingerprints and standard source snapshots. The app-plugin facade, site settings/commands/monitors, full desktop change-event wiring and optional package extraction remain outstanding. The generic registry/scope are not automatic filesystem watchers. See [extension contributions v1](../reference/EXTENSION_CONTRIBUTIONS.md).

Phase 2 delivery D, objective/pruning capability implementation:

- Added metadata-only tuner objective modes and pruning compatibility declarations, with conservative defaults and manifest/runtime validation. Selected providers expose a read-only boolean pruning query; core no longer interprets Optuna options in objective validation.
- Active pruning requires runner intermediate-value reporting and explicit tuner support for custom objectives or repeated runs. Coordinator validation checks each method; worker dispatch rechecks before importing the runner. Core Lightning and editable SB3 declarations preserve existing reporting behavior. Optuna owns historical flat/Hydra, namespaced and worker-runtime pruning precedence and retains unsupported-combination rejection.
- Objective validation now passes explicit workspace paths through method resolution, engine validation and episode-outcome checks. Ordinary objective scoring without tuning never resolves a tuner. Independent fixture tests cover unfamiliar option vocabulary, absent declarations, unavailable reporting, malformed queries, disablement and custom objective acceptance from an unrelated CWD.
- This supplies compatibility declarations and execution gates, not optional installation/extraction, full provider constraint migration or new intermediate-scoring implementations.

Phase 2 delivery D, live tuner monitor integration:

- Added the ready-only, host-rendered `monitor.tuning` contribution slot. The desktop no longer imports/constructs its study dashboard unconditionally. Selected recipes and study exports resolve provider ownership and aliases through the same registry as execution; unavailable saved providers show diagnostics without modifying recipes or result artifacts.
- The monitor's existing visible refresh/clock reconciles contribution lifecycle changes. Disablement, removal, readiness loss, package replacement and workspace changes release mounts; the last mount disconnects trial navigation, closes stale menus, clears/detaches the dashboard and returns to Overview. Panel close stops timers and prevents queued notifications from remounting; reopen resumes reconciliation. This is polling within the existing host refresh, not a general filesystem watcher or the app SDK event facade.
- Readiness uses the monitored receipt, worker telemetry or saved experiment configuration, including local runs whose desktop record only contains compact display fields. It never uses the currently edited recipe. Optuna and the independent lattice fixture now declare the monitor slot; metadata discovery does not import provider code.
- Moved normalized study-summary reading into a shared headless API and removed the queue's Optuna-ID restriction. Current exports take precedence over matching archive copies; active queues exclude earlier-launch files. Partial/malformed/future-schema exports are skipped; markerless historical exports and unknown provider state strings are preserved read-only. Queue process supervision and ordinary telemetry remain available without an installed/enabled tuner dashboard.
- Closed remote artifact decoding now checks readiness against the saved configuration before importing a provider, validates returned summaries and stamps identity from the execution plan. Existing summary reads need neither Qt nor Optuna. The old desktop summary-reader name remains a forwarding compatibility import.

Phase 2 delivery D, explicit tuner selection and historical replay:

- Removed implicit Optuna selection from ordinary tuner lookup and telemetry discovery. Tuning mappings require a nonempty string `tuning.component`; installed provider counts never select a runtime default. Invalid or absent selectors fail before provider resolution. Explicit setup lookup remains supported. Portable plans resolve aliases/case to the canonical capability ID, preserving the recipe's serialized selector.
- Saved unselected/invalid-selector recipes keep their options and ranges, show a shared Tuner selector with a placeholder and actionable diagnostic, and keep ranges read-only. Choosing a provider explicitly refreshes method controls as well as settings. No authored recipe, installed package or research data was automatically rewritten.
- Local command compilation passes its workspace to tuner resolution, matching preflight/remote paths from unrelated launch directories. Core direction notices now use the generic tuning configuration instead of an Optuna-specific notice.
- Added nonmutating `record_config`/`record_overrides` readers for old receipts without `tuner_execution`. Only old tuning mappings inherit the historical selector. Old fixed jobs remain fixed, other explicitly named tuners retain their meaning, and new plan/null records never inherit a selector. Replay makes the bounded choice explicit in worker arguments; contradictory saved selector overrides fail before worker preparation. Allocation finalization, snapshot readiness and continuation identity use the adapted config without rewriting receipt configuration.
- Added independent selector/workspace/telemetry tests, historical identity/conflict checks and real allocation replay. Updated generic form/compiler fixtures to select their intended provider explicitly; form layout fixtures supply dependency metadata without requiring optional training libraries. Physical extraction and scheduler host wiring remain pending.

## Validation and limits

Independent fixture providers cover authored and installed origins, GUI/CLI/worker agreement from unrelated CWDs, consecutive workspace selection, schema ownership overrides, ambiguous/missing providers, explicit user copies, uninstall preservation and rollback on a failed record write. They deliberately raise if metadata inspection executes their provider code.

The execution-contract suite passed all 20 tests, including real worker dispatch, alternate runners, repeated seeds, tuning trials, cleanup/cancellation and obsolete-provider diagnostics. The combined native/provenance/context/execution/provider organization/discovery run passed 84 tests. A separate config-model/provenance/context/dataset/tabular/plotting run passed 112 tests with 17 skips. These runs overlap; counts are not additive. Additional checkpoint, site ownership and incremental snapshot suites were exercised. Skips include absent optional research models; no expensive training was required.

The final focused provenance/context/config-protection run passed 16 tests, including a usr-only workspace, explicit root-config overrides and symlink rejection. `git diff --check` passed and all 165 core/changed entry-point Python files parsed successfully. Hash comparison against the initial inventory checked 664 existing input, authored/installed package and unrelated pending frontend files: none changed or disappeared.

Baseline comparison used an import overlay in `/tmp/theta_layer_baseline.py` to execute the original HEAD versions of this task's edited modules while retaining unrelated working-tree changes. This reproduced nine failures across config-compose, evaluator/plot override and selector metadata suites, plus three in component-schema/snapshot fixtures. Those failures involve old algorithm declarations, invalid manifest fixtures, unregistered evaluator/plotter fixtures and a direct Hydra caller bypassing composed search paths. They are not counted as passing gates. No cluster jobs were submitted.

Delivery C validation: the expanded execution/resource/native/THEMES/SB3-evaluation/provenance/workspace run passed 127 tests, with 13 failures confined to `test_sb3_theta_policy.py`. Those same 13 failures were reproduced using `/tmp/theta_execution_features_baseline.py`, a read-only import overlay of the HEAD versions of the five changed core execution/resource/manifest/archive/engine modules, retaining the existing workspace and manifest declarations. They involve undeclared legacy model imports, unregistered model fixtures and an absent legacy symbolic-observation module; they are not passing gates. A separate feature/package-organization/discovery/SB3-installation run passed 67 tests, including hook payload containment checks. Runs overlap and counts are not additive. The final feature-only run covers 33 cases, including invalid dynamic resource hints and native-policy CPU/GPU planning. Resource tests required process inspection outside the macOS sandbox; GUI fixtures used the offscreen Qt platform. `git diff --check` and parsing of all 12 changed Python files passed. No cluster jobs, publication or data migration was performed.

Delivery D portable-tuner validation: the final combined portable-tuner, hardening, synced-monitor, worker-dispatch, site ownership/incremental-upload/profile and Slurm-header run passed 116 tests, with two legacy real-CPU fixture cases deselected after their obsolete SB3 profile failures were reproduced against the original execution modules. A separate final shutdown/portable-contract run passed 26 tests; nine terminal launch regressions passed outside the sandbox after Qt WebEngine aborted inside it. Counts overlap. The independent fixture covers scratch and retained-workspace allocation execution and blocks Optuna/SQLAlchemy imports during its full state lifecycle.

A broader tuning/packaging/provenance run passed 145 tests and exposed 17 failures in old tuning/objective fixtures. All 17 reproduced using `/tmp/theta-tuner-baseline/sitecustomize.py`, which loads original HEAD site/Optuna/monitor modules in both pytest and subprocesses while retaining preceding delivery work. They involve removed launcher methods/selectors, missing historical source paths and direct Hydra callers bypassing source-aware search paths. Five additional old tuner-component fixture failures plus the two legacy CPU fixtures also reproduced with the original execution-module overlay; these are not counted as passing gates. `git diff --check` and syntax parsing passed. No actual cluster submission, publication, installed-package rewrite or research-data migration occurred.

Delivery D contribution/readiness validation: the focused contribution, config-form, config-viewer, sweep-visibility, portable-tuner and real-worker run passed 215 tests, with four legacy parametrized model-form cases excluded. An unrestricted form/viewer/contribution run passed 170 tests and failed three model-form cases: a missing nested research ruleset widget and two historical agent-parameter ownership round trips. The same three failures reproduced with `/tmp/theta_contribution_gate_comparison.py`, which bypasses contribution loading enforcement and restores the original sweep-activation checks (the fourth scalar variant passed). This is a gate-isolation comparison, not a whole-checkout baseline. An initial broader pane run also found component-panel/window expectations for old title/close/queue controls and research categories; those are not passing desktop gates. The obsolete mocked tuner-discovery fixture was migrated to a real independent native provider, and all 41 ConfigViewer tests pass.

The final small verification passed 19 tests covering metadata/lifecycle behavior, absent and unconfigured tuner forms, floating control teardown and operation-scoped discovery after dependency-read caching. `git diff --check` passed and all 21 changed Python files parsed. Validation counts overlap. No component was unbundled or published, and no live scheduler job or data migration was performed.

Delivery D objective/pruning validation: the final combined compatibility, execution-feature, contribution, portable-tuner, worker-contract, tuner-component and objective run passed 125 tests, with 11 failing legacy/environment cases deselected. Its focused subset passed all 88 tests, including 23 new compatibility cases. Counts overlap. The full objective suite initially passed 10 and failed four; all four reproduced using the original HEAD objective API and pre-change engine through `/tmp/theta-pruning-baseline/sitecustomize.py`. The broader tuner/worker run passed 27 and failed seven; those seven also reproduced using that overlay plus pre-change Optuna component and worker-dispatch modules. Failures involve undeclared/obsolete algorithm fixtures, removed scheduler helper mocks, existing independent-tuner fixtures and the environment's SQLAlchemy/Alembic `_BindParamClause` incompatibility. These import-overlay comparisons retain the preceding delivery changes; they are not clean-checkout baselines. Both changed native manifests validated, changed Python files parsed and `git diff --check` passed. No dependencies were installed, components extracted/published, cluster jobs submitted or data migrated.

Delivery D live-monitor validation: the final GUI/summary/monitor/queue run passed 149 tests, with three skips and seven deselected pre-existing failures. All 28 new cases passed within these runs (ten lifecycle/host cases and eighteen headless summary/decoder cases). A separate final headless summary, portable-tuner, contribution and compatibility run passed 73 tests, including the independent remote lifecycle with Optuna/SQLAlchemy imports blocked. Counts overlap. Six initial monitor/queue failures reproduced using original HEAD panel/runner-manager modules through `/tmp/theta-monitor-baseline/sitecustomize.py`. A broader run exposed eleven old queue-control/window expectations plus a missing Lightning dependency in the frontend environment; all twelve reproduced after adding the original HEAD window module to that overlay. These are host-module comparisons retaining prior delivery changes, not clean-checkout baselines. The Anaconda Qt runtime aborted during application initialization both inside and outside the sandbox; GUI verification used the existing `frontend/.venv` runtime. `git diff --check`, parsing of twelve changed Python files and validation of both changed native manifests passed. No dependency installation, extraction, publication, cluster submission or data migration occurred.

Delivery D explicit-selection validation: the final headless selection, portable-tuner, contribution, objective/pruning, monitor-data, shutdown, tuner-component and worker-contract run passed 138 tests, with eight legacy/environment cases deselected. The GUI form/visibility/viewer/monitor run passed 135 tests, skipped seventeen optional viewer cases and excluded four previously audited model-form variants (three known failures and one passing scalar variant). All 38 added cases passed across these runs: 34 selection/identity/receipt tests, three shared-form selection cases and one real historical allocation replay. Earlier unrestricted form checks reproduced the same three model-form failures recorded above; layout fixtures were made explicit and supplied optional dependency metadata rather than relying on unnamed provider selection or optional libraries being installed. An unrestricted shutdown run failed during Optuna storage initialization with the environment's previously observed SQLAlchemy/Alembic `_BindParamClause` import incompatibility, before the changed finalizer executes; this remains an excluded environment gate. The other seven excluded tuner/worker fixtures are the previously reproduced cases recorded above. `git diff --check` and parsing of seventeen changed Python files passed. No component extraction/publication, dependency installation, cluster submission or research/configuration migration occurred.

## Outstanding acceptance gates

The [progress outline](#progress-outline) is the current summary; older dated sections above remain historical evidence.

- **External.** A real cluster smoke run (allocation, interruption, publication, synced monitor and continuation) and an interactive check of the installed desktop (editor, figures, export and the migrated panels) need the user's environment and authorization.
- **Phase 4.** Versioned config destinations, the migration command (dry-run inventory and collisions, timestamped backup, idempotent rerun, workspace locks), config and dataset moves, reference and state rewriting, republished Hub archives and removal of `in/`. Dataset bytes still use `in/datasets`. No migration command exists.
- **Phase 5.** Selected-provider execution bundles, versioned manifests and readers for existing jobs.
- **Phases 6–7.** `src/ide` consolidation, installed CLI entry points, checkpoint forwarding adapters, installed desktop checks and boundary enforcement over every package. Existing aliases are retained. Lightning and the four base paradigms remain in core. No components have been released to the Hub.
