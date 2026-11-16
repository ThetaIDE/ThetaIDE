# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`, which is pushed to `origin` (ThetaIDE/ThetaIDE). There is no PR.

Phases 0–7 are implemented: bundles `4d95ad5`, installed entry points `8da3c4c`, enforcement `252d3dc`. Two jobs remain, in this order:

1. **Unify compatibility** following `docs/plans/COMPATIBILITY_REQUIREMENTS_PLAN.md`. Today paradigm compatibility has three homes, the data slot `role` two, and input contracts, execution input types, paradigm `allowed_agents` and the run-time-only `execution_preflight` overlap. The plan replaces them with one capability-level `requires` (plus `provides` and a validation-time `preflight` hook), evaluated by one core module in the form, validation, execution and bundle check. The user wants plugin authors to be able to enforce what their capability needs (a paradigm, a model or algorithm, a contract, specific configuration values) through their package alone. Convert every maintained package to the new declarations.
2. **Publish** (authorized by the user on 2026-11-16, once job 1 is done and verified). Follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`. Publication pushes the `-v3` branches, releases the v3 batch on the Hub and merges this branch to `main` together with it, as the layer boundary plan requires.

Uncommitted files from separate user tasks may be present (`frontend/components_panel.py`, `frontend/config_tree.py`, `tests/test_startup_performance.py`). Preserve them and never commit them with refactor work. Read `git status` first.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`. Its "Confirmed decisions" win over anything else.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: current state, repositories, and test conventions.
4. `docs/plans/COMPATIBILITY_REQUIREMENTS_PLAN.md`: the design and units for job 1.
5. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` (including Enforcement), `docs/reference/SCHEDULER_CONTRACT.md` (bundles) and `docs/reference/PLUGIN_STORAGE.md`.

Older dated evidence is in `docs/reports/`; it is history.

Vega awaits the user's design; do not infer one. BlendRL iteration stays local in `src/usr/models/blendrl-dev`, without repeated Hub publication.

## How the user wants you to work

- Work through the authorized items without stopping for routine confirmation. Do not ask about commits, pushes or other git steps mid-work; commit each unit, push `implement/layer-boundaries`, and report at the end.
- Remove legacy patterns whenever you find them. Do not ask whether to remove them.
- Every setting has exactly one place it is defined. No fallback chains (for example, tuning settings come only from the `tuning` block, never from paradigms, environments or `hydra.sweeper`).
- When a change makes one of the user's files out of date, say so in one line (file and what moves where). Do not edit it unless asked, and do not dramatize it.
- Keep reports short.

## Next work

1. **Compatibility requirements:** the plan's units 1–4 (core, forms, packages, verification). Each changed v3 package gets a new version and release note. Report any of the user's own files the change makes out of date in one line each (for example, authored profiles under `src/usr/config` that set `compatible_paradigms`).
2. **Publication checklist:**
   - Rebuild Hub-v3 archives and registry entries for every package changed since the last preparation: Slurm 2.0.0 (scheduler contract v2), `antigravity`/`doom-sim`/`obsidian` 1.2.0 (SDK 3), `cew` 1.1.1, `neumann` 1.1.1, and everything job 1 converts.
   - Reconcile the prepared BlendRL release with the dev copy (`src/usr/models/blendrl-dev`; last preparation 1.3.2, source `7af130a`).
   - Fix reward-shaping's cwd-relative checkpoint default and its config-environment fallback.
   - Remove `ai_assistant`, `theta_mcp` and the five MIMIC packages from the feed. Reconcile the stale Optuna 1.1.0 listing with the published 1.1.1 and keep the 1.1.1 archive.
   - Verify the clean client feed from an installed IDE (`scripts/verify_installed_distribution.py --hub-root`), set `published_at`, push, merge `implement/layer-boundaries` to `main`.
   - Reinstall the published packages in this repository workspace.
3. **UI follow-ups when the user asks:** the deferred sync-button/tree-navigation notes, and the eight UI-batch test failures in `docs/reports/LAYER_BOUNDARY_PHASE7_VERIFICATION_20261116.json` (`test_frontend_layout`, `test_config_model`, `test_tuning_monitor_contributions`, `test_remote_partial_results`).

Left alone (user): checkpoint ranking by `monitor_metric`/`checkpoint.monitor`, the deferred stale tests, `prepare_tabular.py`. The user's `BlendRL/Cartpole/sweep_sustainedSolve.yaml` still sets flat `tuning.n_trials`/`n_jobs` (they know).

## Hard constraints

- No legacy support. Old inputs fail through ordinary validation; no code recognizes, detects or names them, and there are no compatibility adapters, forwarders, deprecation paths or migration hints. A compatibility path missing from the exception register is a bug.
- Never pick a winner between conflicting files. Back up, hash-check and show a dry run before moving the user's files.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through the SDK (`src.app.api`).
- Do not run anything on a cluster.
- Publish, push the `-v3` worktrees and merge to `main` only as the publication step above, after compatibility is done and verified. Pushing `implement/layer-boundaries` is allowed and expected throughout.
- Configuration UI changes follow `docs/reference/CONFIG_PARAMETER_FORMS.md`.
- Do not modify `../theta-plugins` or `../theta-hub`; work in the `-v3` worktrees and commit there with `git -c user.name=Tacho2`.
- Leave `prepare_tabular.py` alone.

## Testing

- Run tests per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. Running many files in one process causes cross-file interference (for example, `test_study_name_allocation` fails after other files purge the Optuna component's `src.plugins._workspace` module).
- Before calling a failure a regression, compare by test ID against the stale list in the status doc, or with and without your change (`git stash push -u`, rerun, `git stash pop`).
- Tests select their workspace through fixtures (`fixture_workspace`, `optuna_workspace`, `sweep_workspace`, `themes_fixtures`); never `monkeypatch.chdir` alone. Tests must not depend on what the user has installed. Tests that launch Hydra sweeps directly use `component_fixtures.sweeper_args` and `OPTUNA_CONFIGS`.
- For the installed-distribution probe, build in a fresh environment with a fresh setuptools build base, install the wheel with its `gui` extra into a new venv (`python -I -m pip` from a neutral directory), and run `scripts/verify_installed_distribution.py ... --hub-root ../theta-hub-v3`.

## When you stop

Commit each unit separately with the attribution trailer the session gives you. Replace the status doc's contents to match the new state (it is not an append-only log). Add new documents to `docs/README.md`. Report briefly: what landed (commits), what was verified, what failed or was skipped, and the next item. Then update this prompt.
