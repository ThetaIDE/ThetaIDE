# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch is pushed to `origin` (ThetaIDE/ThetaIDE) and has no PR. It is merged to `main` together with the v3 Hub publication, only when the user asks.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`. Its "Confirmed decisions" win over anything else.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: current state, the TensorBoard flag item, "Tuning has one shape", the results paths for phase 5, and the test conventions.
4. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` and `docs/reference/PLUGIN_STORAGE.md`.

Older dated evidence is in `docs/reports/LAYER_BOUNDARY_HISTORY_20261115.md`. It is history: its "bounded" readers and adapters are being removed, not kept.

## How the user wants you to work

- Work through the items below to the end without stopping for confirmation. Do not ask about commits, pushes or other git steps mid-work; commit each unit, push `implement/layer-boundaries`, and report at the end.
- Remove legacy patterns whenever you find them. Do not ask whether to remove them.
- Every setting has exactly one place it is defined. No fallback chains (for example, tuning settings come only from the `tuning` block, never from paradigms, environments or `hydra.sweeper`).
- When a change makes one of the user's files out of date, say so in one line (file and what moves where). Do not edit it unless asked, and do not dramatize it.
- Keep reports short.

## Where things stand (2026-11-15)

- Phases 0–3 are done. Phase 4 (4a–4h) is done except the TensorBoard flag.
- Tuning has one validated shape (`0f373a7`): `tuning.{component, metric, direction, options.<component>}` plus each method's `tune`. Optuna ships one sweeper config, `hydra/sweeper/theta_optuna`, compiled entirely from that block.
- The `online_methods`/`task:` readers and the root `sweep` key are deleted.
- **Core stays at `src/app` and `frontend`.** The `src/ide` move is dropped (user decision); phase 6 is only installed entry points and distribution verification. No forwarding modules exist or are allowed.
- The v3 Hub releases are prepared in `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`), local and unpublished.
- About 120 stale test failures predate this branch (listed by file in the status doc) and are deferred by the user.
- The user's `src/usr/config/experiment/BlendRL/Cartpole/sweep_sustainedSolve.yaml` is out of date: `tuning.n_trials`/`n_jobs` belong under `tuning.options.optuna`. The user knows; leave it.

## Work to do next, in order

1. **TensorBoard flag (last 4h item).** Delete the root/agent `tensorboard` flag and its core translation; the form checkbox edits `logging.backends.tensorboard.enabled`. Every site is listed in the status doc under "Left in 4h: the TensorBoard flag".
2. **Legacy sweep.** Search for remaining old patterns and delete them: `git grep -niE "legacy|deprecat|compat|historical|old (format|key|layout)"` across `src`, `frontend`, `optional_components`, `scripts`, `run_pipeline.py` and `tests`. Known lead: `docs/reference/SCHEDULER_CONTRACT.md:28` documents hiding "forwarding" site profiles as compatibility aliases; find the code behind it (site profile listing in the Slurm component or `src/app/sites`) and remove it with the doc sentence. Anything legitimately current gets reworded, not kept as "legacy".
3. **Phase 5.** Write the phase 5 plan document first (bundle manifest, application/workspace resolution, inclusions and exclusions) and add it to `docs/README.md`. Then implement the plan's phase 5 units: every results path through `WorkspacePaths` (the full list is in the status doc), results moved to `src/usr/results` (moving the user's `results/` needs a backup, hash check and a dry run shown to the user, as in 4f; that is the one stop), explicit bundles, declared requirements, one versioned bundle manifest.
4. **Phase 6**: installed CLI entry points and the wheel/sdist verification (`scripts/verify_installed_distribution.py`). No module moves.
5. **Phase 7**: enforcement, as the plan describes.
6. **Publication, only when the user asks:** follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`. Fix the published `blendrl` 1.3.0 schema types in `../theta-plugins-v3/models/blendrl` first (status doc, "Also left"). Remove `ai_assistant`, `theta_mcp` and the five MIMIC packages from the v3 feed, push `theta-hub-v3` and `theta-plugins-v3`, verify the client feed from a clean workspace, set `published_at`, merge to `main` together with the Hub change, and remove the "pre-v3 Hub releases" row from the exception register.

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
