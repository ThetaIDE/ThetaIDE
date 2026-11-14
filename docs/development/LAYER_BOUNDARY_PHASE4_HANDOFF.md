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
- **Phases 0–3** are complete locally, with SDK 2.0 (`src.app.api`). The only open items are external: a real cluster smoke run and an interactive check of the installed desktop. Both need the user.
- **Phase 4 units 4a–4f have landed.** Every decision is settled: no legacy support of any kind (v3 only, no `in/` handling, every compatibility adapter retired including checkpoint aliases, historical results untouched, no readers or forwarders in phases 5 and 6). Package profiles resolve from the package; `src/usr/config` holds only user files and overrides. Details, commits and evidence are in the status doc.
- **Next: 4g, Hub republish preparation (no publication).**
  - **Ask the user first.** `../theta-plugins` is on `main` with a large uncommitted storage-v2 migration (455 paths, dated 2026-11-03 to 11-07), and `../theta-hub` tracks a fork with releases production lacks. Agree on the authoritative source before building anything.
  - Rebuild the sibling packages as v3 archives with new versions. nsfr and neumann must declare `exports` for what `blendrl-dev` loads (`get_nsfr_model`, the neumann helpers). Move installed `provider_object` users (`blendrl`, `mlp`, `neumann`) and `plot.base` users to SDK 2 (`src.app.api.packages`, `src.app.api.plotting`, `paths` in plotter constructors). Bump the Optuna and Slurm versions.
  - App plugins against SDK 2 (checked read-only on 2026-11-14 in `../theta-plugins/plugins`): none of `doom-sim`, `obsidian` or `antigravity` declares `sdk`, so SDK 2 refuses all three, and all import `frontend.plugins.base`/`.context` directly. `doom-sim` and `obsidian` need only `src.app.api.desktop` imports and `"sdk": ">=2.0,<3"`. `antigravity` (third-party author) uses `context.window`/`_window` to add a dock widget and read `settings_manager` shortcuts, which SDK 2 has no service for; ask the user whether to add a host service or change the plugin before touching it.
  - Install every rebuilt archive into an isolated workspace with the branch IDE. The tests waiting on this: `test_baselines_and_suites` (nsfr/neumann, dl-basics), `test_rulesets::test_authored_nsfr_package_loads_ruleset_predicates_in_fresh_process`, `test_data_loader_bindings` (BlendRL), `test_blendrl_rich_monitors` and `test_wet_runs::test_preflight_blocks_supervised_without_dataset`.
  - Then remove the register row for SDK 1.1 direct imports in installed packages.

  Do not reintroduce compatibility code, deprecation paths or messages about old formats.

**Then implement** 4g, committing separately with the attribution trailer the session gives you. Keep the status doc's progress outline and a dated section current as units land. Add new documents to `docs/README.md`.

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
