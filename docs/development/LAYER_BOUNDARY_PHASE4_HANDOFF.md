# Phase 4 handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary work. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch is pushed to `origin` (ThetaIDE/ThetaIDE); it has no PR and must not be merged to `main` until phase 4 and the Hub republish are ready together.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_PHASE4_PLAN.md`: the plan you are implementing.
3. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`, especially "Confirmed decisions", which win over anything else.
4. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: the progress outline and test-failure categories.
5. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register) and `docs/reference/PLUGIN_STORAGE.md`.

**Where things stand (2026-11-14):**
- **Phases 0–3** are complete locally, including SDK 1.1 (now being bumped to 2.0 in 4c) (`src.app.api`). The only open items are external: a real cluster smoke run and an interactive check of the installed desktop. Both need the user.
- **Phase 4 is in progress.** Every decision is settled: no legacy support of any kind (v3 only, no `in/` handling, every compatibility adapter retired including checkpoint aliases, historical results untouched, no readers or forwarders in phases 5 and 6). The user also chose the "stop copying" install model: package profiles resolve from the package, and `src/usr/config` holds only user files and overrides.
  - **Committed:** `01df36f` (4a + 4b + the 4e test port) and `94d80a5` (4f: workspace moved, backup in `.thetaide/migrations/20261114T163513Z/`). Details and evidence are in the status doc.
  - **4c is uncommitted and partly done in the working tree.** Done so far:
    - deleted `import_aliases`, the `plot/` and `frontend/plugins/core/` forwarding trees, the `src/usr` shims and `__init__` markers, `frontend/tensorboard.py`, `src/app/objectives/builtins`, the Optuna alias modules, the Slurm adapters plus `scheduler_object`, `tuner_execution_compat`, the receipt schema-v1 reader, `provider_object`, the plotter ContextVar, and every top-level site forwarding YAML (framework `site/local.yaml`, the Slurm component's four, the user's five);
    - config default is now `site: local/default`;
    - receipts moved from `results/slurm` to `results/sites`;
    - SDK 2.0 adds `src.app.api.plotting`, `.packages` (`provider_export`) and `.agents`, and an absent `sdk` requirement is refused;
    - the Hydra Optuna sweeper loads the Optuna component's new `hydra_sweeper` export;
    - `blendrl-dev` now uses `provider_export`, which fails until nsfr and neumann declare exports in 4g.
  - **Next for 4c:**
    - add `tuner_execution: None` and `scheduler_component` to the test receipt fixtures (`test_sites`, `test_results_sync_regression`, `test_synced_sweep_monitor`, `test_ncshare_hardening`, `test_cluster_error_logs`);
    - delete the legacy-receipt tests in `test_tuner_execution`, `test_sites::test_legacy_live_sqlite_sync_is_rejected` and the regexes about "implied 'slurm'" and "owning compute node";
    - rerun every test file against the baseline (`scratchpad` runs `base_b1` at `e3fdbcd`), then update the `LAYER_BOUNDARIES.md` exception register (keep only Lightning/base paradigms, `hydra_plugins`, root launchers; the SDK 1.1 direct-import row goes with 4g) and commit 4c.
  - **Then:**
    - 4d guides (COMPONENT_PLUGIN_GUIDE, CLUSTER_JOBS, COMPONENT_DEVELOPMENT, INSTALL, MECHANISTIC_MONITORS, API, EXECUTION_CONTRACT, LAYER_BOUNDARIES, PLUGIN_STORAGE, SCHEDULER_CONTRACT, SDK, CUSTOM_OBJECTIVES);
    - commit the untracked `tests/test_v3_workspace_layout.py` (passes) with the rest of 4e, then run the distribution probe;
    - 4g: republish the sibling packages as v3, declaring exports for nsfr and neumann, moving `plot.base` and `provider_object` users to SDK 2, bumping the Optuna and Slurm versions, and checking the two third-party app plugins against SDK 2.

  Do not reintroduce compatibility code, deprecation paths or messages about old formats.

**Then implement** the plan's units in order (4a → 4e, with 4f in the same session as 4b, then 4g), committing each unit separately with the attribution trailer the session gives you. Keep the status doc's progress outline and a dated section current as units land. Add new documents to `docs/README.md`.

**Hard constraints:**
- Do not publish to Theta Hub, push to any Hub repository, or merge to `main` unless the user asks. Unit 4g prepares releases only.
- Before moving the repository's own research workspace (unit 4f), show the user the dry-run summary and get a go-ahead. Back it up first, and verify that nothing is left behind.
- Never pick a winner between conflicting files or follow the `mimic` symlink (move it as a link). Leave historical results in place.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through `src.app.api`.
- Configuration UI changes follow `docs/reference/CONFIG_PARAMETER_FORMS.md`.

**Testing:**
- Run tests per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. A whole-suite run in one process can die silently.
- About 49 failures predate this work (research-workspace inputs, tuning and objective suites, `test_parallel_runs`). Before calling a failure a regression, compare it with and without your change: `git stash push -u`, rerun the file, `git stash pop`.
- Optional components (Slurm, Optuna) are not installed in the root research workspace. Tests install them with `tests/component_fixtures.py`.
- For the installed-distribution probe, build with a fresh setuptools build base and follow `scripts/verify_installed_distribution.py`. Do not reuse `/tmp/theta-boundary-baseline/env` without removing its stale `site-packages/src`; it also has Optuna installed.

**When you stop,** report what landed (with commits), what was verified and how, what failed or was skipped, and the next unit. Then update this prompt.
