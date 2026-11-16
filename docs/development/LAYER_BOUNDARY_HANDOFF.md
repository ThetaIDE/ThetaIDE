# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. Update it when the state it describes changes.

---

You are continuing the layer boundary work of Theta-IDE. The refactor, including the compatibility requirements plan, is complete and merged into `main` (also pushed as `implement/layer-boundaries`). The v3 package batch is published on Theta Hub (`570d30c`; [release report](../reports/V3_HUB_RELEASE_20261116.md)). Work on `main` unless the user names a branch.

The end state: components declare compatibility on their capabilities (`requires`, `provides`, `preflight`), paradigms describe only themselves, components own their sub-components through component-reference parameters, and the SDK (3.1) and IDE do nothing extra for any component.

Uncommitted files from separate user tasks may be present (`frontend/components_panel.py`, `frontend/config_tree.py`, `tests/test_startup_performance.py`). Preserve them and never commit them with refactor work. Read `git status` first.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`. Its "Confirmed decisions" win over anything else.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: current state, repositories, and test conventions.
4. `docs/reference/COMPATIBILITY_REQUIREMENTS.md`: how components declare compatibility.
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

1. **Workspace reinstall (needs the user's yes):** thirteen installed copies from the earlier v3 preparation no longer validate, and the installer will not overwrite them. The dry run is in the status doc's Repositories section. After a direct yes: back up and hash-check each one, move it to `.thetaide/retained_packages`, then install the published releases (`venv/bin/python run_pipeline.py component install <archive>`), dependencies first, then BlendRL 1.4.0, NEUMANN 1.1.1 and the DL basics suite.
2. **UI follow-ups when the user asks:** the deferred sync-button/tree-navigation notes, the UI-batch test failures listed in `docs/reports/COMPATIBILITY_REQUIREMENTS_VERIFICATION_20261116.json`, and the unchecked desktop items in the status doc.

Left alone (user): Vega, checkpoint ranking by `monitor_metric`/`checkpoint.monitor`, the deferred stale tests, `prepare_tabular.py`. The user's BlendRL and NewMonitors experiments still set `symbolic_state_loader` (now `model.symbolic_state`), and `BlendRL/Cartpole/sweep_sustainedSolve.yaml` sets flat `tuning.n_trials`/`n_jobs` (they know).

## Hard constraints

- No legacy support. Old inputs fail through ordinary validation; no code recognizes, detects or names them, and there are no compatibility adapters, forwarders, deprecation paths or migration hints. A compatibility path missing from the exception register is a bug.
- Never pick a winner between conflicting files. Back up, hash-check and show a dry run before moving the user's files.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through the SDK (`src.app.api`).
- Do not run anything on a cluster.
- Publish a Hub release only when the user asks; follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`.
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
