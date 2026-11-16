# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch exists on `origin` (ThetaIDE/ThetaIDE), but 5b, stabilized 5c and the UI refinement batch are local and unpushed. There is no PR. It is merged to `main` together with the v3 Hub publication, only when the user asks.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`. Its "Confirmed decisions" win over anything else.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: current state, the TensorBoard flag item, "Tuning has one shape", the results paths for phase 5, and the test conventions.
4. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` and `docs/reference/PLUGIN_STORAGE.md`.

Older dated evidence is in `docs/reports/LAYER_BOUNDARY_HISTORY_20261115.md`. It is history: its "bounded" readers and adapters are being removed, not kept.

## Current session priority

5c was stabilized (`e0a09a2`), then the user supplied specific config and downstream UI requirements. That batch is implemented locally; read `docs/plans/UI_REFINEMENTS_20261115.md` and its linked verification evidence. Only the unclear sync-button/tree-navigation notes are deferred. Vega is for later: the user has a specific design and has not provided it. Do not infer or implement a Vega design. The results move remains unapproved.

The sweep-rendering follow-up keeps saved range/list controls visible but read-only when a tuner is unavailable; provider readiness still gates edits and new sweeps. Optuna is not installed in the repository workspace, so its saved controls remain read-only until it is installed and ready.

## How the user wants you to work

- Follow the current session priority above; when resuming the remaining architecture work, work through the authorized items without stopping for routine confirmation. Do not ask about commits, pushes or other git steps mid-work; commit each unit, push `implement/layer-boundaries`, and report at the end.
- Remove legacy patterns whenever you find them. Do not ask whether to remove them.
- Every setting has exactly one place it is defined. No fallback chains (for example, tuning settings come only from the `tuning` block, never from paradigms, environments or `hydra.sweeper`).
- When a change makes one of the user's files out of date, say so in one line (file and what moves where). Do not edit it unless asked, and do not dramatize it.
- Keep reports short.

## Where things stand (2026-11-15)

- Phases 0–4 are done. Phase 5a/5b are done (`9129f8b` is 5b); 5c code is stabilized and verified. The user's results move and 5d/5e remain open. No module move is planned.
- Tuning has one validated shape (`0f373a7`): `tuning.{component, metric, direction, options.<component>}` plus each method's `tune`. Optuna ships one sweeper config, `hydra/sweeper/theta_optuna`, compiled entirely from that block.
- The `online_methods`/`task:` readers and the root `sweep` key are deleted.
- **Core stays at `src/app` and `frontend`.** The `src/ide` move is dropped (user decision); phase 6 is only installed entry points and distribution verification. No forwarding modules exist or are allowed.
- The v3 Hub releases are prepared in `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`), local and unpublished.
- Full 5c comparison: 2,636 passed, 52 skipped, 126 existing failures, zero new failure IDs against the saved 5b baseline. All sandbox-affected files pass outside the sandbox. Job-manager checks: 19 passed, 2 prerequisite skips. Evidence is linked from the status document.
- The user's `src/usr/config/experiment/BlendRL/Cartpole/sweep_sustainedSolve.yaml` is out of date: `tuning.n_trials`/`n_jobs` belong under `tuning.options.optuna`. The user knows; leave it.

## Next work

1. **Review the implemented UI batch with the user.** Further changes follow their feedback; clarify the deferred sync/navigation notes when they want to address them. Vega remains deferred pending their design.
2. **Results move, only after approval.** The last dry run recorded 15,590 files (1.23 GB), no symlinks or conflicts. Revalidate before acting, create the timestamped backup, rename to `src/usr/results`, and verify every hash. The old location is not read by the app. Do not claim all of 5c is complete before the move.
3. **Resume 5d/5e when requested:** explicit `theta.bundle/v1` bundles and declared requirements, following the phase 5 plan. No cluster run.
4. **Phase 6:** installed CLI entry points and wheel/sdist verification. Core stays at `src/app` and `frontend`.
5. **Phase 7:** boundary enforcement against the combined implementation.
6. **Publication, only when requested:** follow `THETA_HUB_RELEASE_WORKFLOW.md`. Known prerequisites: publish the locally prepared BlendRL 1.3.1 archive/feed after its source and local installation checks, fix reward-shaping's cwd-relative checkpoint default/config-environment fallback, the native SB3 Optuna test bypassing pipeline tuner setup, and reinstall the corrected SB3 package in this workspace. The earlier package fix is `7b7a3ef` in `../theta-plugins-v3`, unpushed. Remove `ai_assistant`, `theta_mcp` and the five MIMIC packages from the v3 feed, verify the clean client feed, set `published_at`, and merge with the Hub changes only when asked.

Left alone for now (user): checkpoint ranking by `monitor_metric`/`checkpoint.monitor` (`src/app/core/lightning_builder.py:376`–`:416`), the deferred stale tests, `prepare_tabular.py`.

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
- For the installed-distribution probe, build in a fresh environment with a fresh setuptools build base and follow `scripts/verify_installed_distribution.py`.

## When you stop

Commit each unit separately with the attribution trailer the session gives you. Replace the status doc's contents to match the new state (it is not an append-only log). Add new documents to `docs/README.md`. Report briefly: what landed (commits), what was verified, what failed or was skipped, and the next item. Then update this prompt.
