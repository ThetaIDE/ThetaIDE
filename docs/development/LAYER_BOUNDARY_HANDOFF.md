# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`, which is pushed to `origin` (ThetaIDE/ThetaIDE). There is no PR. It is merged to `main` together with the v3 Hub publication, only when the user asks.

Phases 0–7 are implemented. 5d/5e (bundles) are `4d95ad5`, phase 6 (installed entry points) is `8da3c4c`, phase 7 (enforcement) follows it. What remains is publication and the settings that still have more than one home; see the status document.

Uncommitted files from separate user tasks may be present (`frontend/components_panel.py`, `frontend/config_tree.py`, `tests/test_startup_performance.py`). Preserve them and never commit them with refactor work. Read `git status` first.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`. Its "Confirmed decisions" win over anything else.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: current state, what still has two homes, and test conventions.
4. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` (including Enforcement), `docs/reference/SCHEDULER_CONTRACT.md` (bundles) and `docs/reference/PLUGIN_STORAGE.md`.

Older dated evidence is in `docs/reports/`; it is history.

Vega awaits the user's design; do not infer one. BlendRL iteration stays local in `src/usr/models/blendrl-dev`, without repeated Hub publication.

## How the user wants you to work

- Work through the authorized items without stopping for routine confirmation. Do not ask about commits, pushes or other git steps mid-work; commit each unit, push `implement/layer-boundaries`, and report at the end.
- Remove legacy patterns whenever you find them. Do not ask whether to remove them.
- Every setting has exactly one place it is defined. No fallback chains (for example, tuning settings come only from the `tuning` block, never from paradigms, environments or `hydra.sweeper`).
- When a change makes one of the user's files out of date, say so in one line (file and what moves where). Do not edit it unless asked, and do not dramatize it.
- Keep reports short.

## Next work

1. **Publication, only when requested:** follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`. Rebuild the Hub-v3 archives and registry for the packages changed since the last preparation (Slurm 2.0.0, Optuna as published, `antigravity`/`doom-sim`/`obsidian` 1.2.0, `cew` 1.1.1, `neumann` 1.1.1); reconcile the prepared BlendRL release with the dev copy; fix reward-shaping's cwd-relative checkpoint default and config-environment fallback; remove `ai_assistant`, `theta_mcp` and the five MIMIC packages from the feed; reconcile the stale Optuna 1.1.0 listing; verify the clean client feed; set `published_at`.
2. **With publication:** give paradigm compatibility one home (capability `paradigms`; drop profile `compatible_paradigms` and package-level `paradigms`) and data-module `role` one home, in the v3 packages and the forms, then reinstall in this workspace.
3. **UI follow-ups as the user asks:** the deferred sync-button/tree-navigation notes, and the eight UI-batch test failures recorded in the phase 7 evidence (`test_frontend_layout`, `test_config_model`, `test_tuning_monitor_contributions`, `test_remote_partial_results`).

Left alone (user): checkpoint ranking by `monitor_metric`/`checkpoint.monitor`, the deferred stale tests, `prepare_tabular.py`.

## Hard constraints

- No legacy support. Old inputs fail through ordinary validation; no code recognizes, detects or names them, and there are no compatibility adapters, forwarders, deprecation paths or migration hints. A compatibility path missing from the exception register is a bug.
- Never pick a winner between conflicting files. Back up, hash-check and show a dry run before moving the user's files.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through the SDK (`src.app.api`).
- Do not run anything on a cluster.
- Do not publish to the Hub, push the `-v3` worktrees or merge to `main` until the user asks. Pushing `implement/layer-boundaries` is allowed and expected.
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
