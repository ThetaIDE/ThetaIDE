# Layer boundary implementation status

Current state of the [layer boundary plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) on branch `implement/layer-boundaries`, as of 2026-11-16. This page describes only what is true now and what is left; replace its contents as work lands. Dated evidence for phases 0–4 is in the [implementation history](../reports/LAYER_BOUNDARY_HISTORY_20261115.md). The [handoff prompt](LAYER_BOUNDARY_HANDOFF.md) lists the next work in order.

## Progress

| Phase (delivery) | State | What is left |
| --- | --- | --- |
| 0 — Contract and baseline (A) | Done | — |
| 1 — Workspace and config ownership (B) | Done | — |
| 2 — Runner features, site/tuner extraction (C, D) | Done | — |
| 3 — App-plugin SDK (E) | Done | — |
| 4 — Retire `in/` and every adapter (F) | Done; Hub releases prepared, not published | Publication |
| 5 — Results in `src/usr/results`, remote bundles (G) | Done (`4d95ad5`); the user's results live in `src/usr/results` | — |
| 6 — Installed distribution (H) | Done (`8da3c4c`) | — |
| 7 — Enforcement (I) | Done (`252d3dc`) | Compatibility requirements plan, then publication |

Gates: the real cluster smoke run is not a gate and is not run (user decision, 2026-11-15). The desktop is accepted as working; the interactive items at the end are unchecked.

## Repositories

- **IDE:** `implement/layer-boundaries` is pushed to `origin` (ThetaIDE/ThetaIDE). No PR. It merges to `main` only together with the Hub publication, and only when the user asks.
- **Hub releases:** Optuna 1.1.1 is published ([release verification](../reports/OPTUNA_HUB_RELEASE_20261116.md)). The rest of the v3 batch in `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`) is local and unpushed. Since the last preparation: Slurm is 2.0.0 (scheduler contract v2, declares `packaging`), the app plugins `antigravity`, `doom-sim` and `obsidian` are 1.2.0 (SDK `>=3.0,<4`), `cew` 1.1.1 declares `igraph`, `neumann` 1.1.1 drops an unused perception chain. The Hub-v3 archives and registry entries must be rebuilt for these before publication. At publication, `ai_assistant`, `theta_mcp` and the five MIMIC packages leave the feed; reconcile the stale Optuna 1.1.0 listing.
- **Repository workspace:** runs the prepared v3 releases. Optuna 1.1.1 is installed there; Slurm is not (it collides with the authored `ncshare-site` package, which owns `site/ncshare/*`).

## Phase 5 — results and bundles

- Every results path comes from `WorkspacePaths.results` (`src/usr/results`); `tests/test_layout_guard.py` rejects literal results paths. The user's old `results/` was merged into `src/usr/results`. Four superseded 2026-11-14 files that could not merge (macOS treats `newMonitors` and `NewMonitors` as one directory) were deleted at the user's request.
- A remote job runs from a bundle (`src/app/sites/bundle.py`, `theta.bundle/v1`) built from one composed experiment: the application without the desktop, the selected packages and their component dependencies, the workspace configuration outside `experiment/` plus the experiment files it reads, authored rules and the workspace identity. It composes its experiment on its own before upload. `.thetaide/requirements.txt` lists the application's and the selected providers' requirements; remote setup is `pip install -r`. Contract: [scheduler contract v2](../reference/SCHEDULER_CONTRACT.md#bundles). SDK 3.0 replaced `freeze`/`extract` with the bundle exports.
- `site push SITE EXPERIMENT [OVERRIDES…]`; the remote workspace holds the last pushed bundle, and submission requires that bundle to be for the submitted experiment. Continuation requires the parent's bundle digest.

## Phase 6 — installed distribution

`theta` (`run_pipeline:cli`) and `theta-ide` (`frontend.__main__:launch`) are installed entry points. An installed application is never a workspace: it uses the working directory or `THETA_WORKSPACE`. [Distribution evidence](../reports/LAYER_BOUNDARY_PHASE6_DISTRIBUTION_20261116.json): a fresh wheel installed in a clean environment installed SB3 with `theta component install`, trained CartPole with `theta` from an external workspace, opened `theta-ide` on it and launched the experiment through the desktop's job manager.

## Phase 7 — enforcement

- `tests/test_sdk_boundary.py` checks every tracked package (optional, bundled, authored `src/usr`, fixtures) with `package_errors` (SDK-only host imports, no other package's modules by path, no window access) and `undeclared_imports` (third-party imports come from the application's requirements or `dependencies.pip`), and checks that core and the desktop import no package by path and name no optional or authored package ID. See [SDK enforcement](../reference/SDK.md#enforcement).
- The installer refuses, and the exporter will not export, package code that fails `package_errors`. The v3 package builder runs both checks.
- Behavior tests use differently named providers: scheduler `lattice-batch`, tuners `lattice` and `fixture-grid` (dispatch, forms, bundle selection and requirements).
- The scaffolder generates SDK-only code; a test scaffolds every kind and checks and imports the result.
- Removed: the orphaned Model Builder modules and their test (its panel was withdrawn in the UI batch). Archived: the project organization proposal, the pre-v3 component development status, the execution refactor status and the first Slurm hardening handoff.

## Still more than one home

Paradigm compatibility (profile `compatible_paradigms`, capability and package `paradigms`, paradigm `allowed_agents`), the data slot `role` (profile, then capability) and input contracts (capability, then package) overlap. The [compatibility requirements plan](../plans/COMPATIBILITY_REQUIREMENTS_PLAN.md) replaces them with one capability-level `requires`; it is the next work, before publication.

## Kept as current behavior

Tuning has one shape: `tuning.{component, metric, direction, options.<component>}` plus each method's `tune`; validation rejects any other key. Checkpoint ranking by paradigm/environment `monitor_metric` and `checkpoint.monitor` stays (user). A run logger's only switch is `logging.backends.<id>.enabled`. The resource pool's SQLite column upgrade stays (the running desktop may hold that database). `requires-python` says 3.10, but core already imports `tomllib` (3.11).

The user's `BlendRL/Cartpole/sweep_sustainedSolve.yaml` sets `tuning.n_trials`/`n_jobs` at the flat level and fails validation until they move under `tuning.options.optuna` (user knows).

## Tests

Full per-file run (2026-11-16, all 225 files): **2,637 passed, 52 skipped, 133 failed**, no aborted files. No failure is new from this session: eight IDs absent from the 5c list fail identically at `9a3870e` (left by the UI/BlendRL batch: removed Plots panel in `test_frontend_layout`, required `blender_model` in `test_config_model`, Monitor consolidation in `test_tuning_monitor_contributions`, a capitalized label in `test_remote_partial_results`). Fifteen earlier IDs now pass or were removed. [Evidence and IDs](../reports/LAYER_BOUNDARY_PHASE7_VERIFICATION_20261116.json). Package integration in `../theta-plugins-v3`: 70 passed. Real SB3 CPU bundle runs (`test_ncshare_hardening` with `THETA_HUB_ROOT=../theta-hub-v3`) pass.

Run per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. Compare failures by ID against the evidence before calling one a regression. Fixtures: `tests/conftest.py`'s autouse `tracked_workspace`, `fixture_workspace`, `optuna_workspace`, `component_fixtures` (`install_optional`, `sweep_workspace`, `LocalHub` via `THETA_HUB_ROOT`), `tests/bundle_fixtures.py` for fake-project bundles. Deferred stale tests (user decision) include those reading `~/.config/thetaide/settings.toml` or QSettings.

## Desktop items not yet checked interactively

1. Editing a profile owned by an installed package asks for an override, and the copy appears under `src/usr/config`; authored-package profiles edit in place.
2. The text editor, TensorBoard, documentation, arXiv and antigravity panels load, and survive disable and re-enable.
3. Results previews generated figures for a finished run in a populated workspace.
4. Settings → Terminal → Cluster sites: the experiment picker and the generated setup (`pip install -r .thetaide/requirements.txt`).
