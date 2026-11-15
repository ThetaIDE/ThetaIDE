# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch is pushed to `origin` (ThetaIDE/ThetaIDE) and has no PR. It must not be merged to `main` until the v3 Hub releases are published together with it, and only when the user asks.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`. Its "Confirmed decisions" win over anything else.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: the current state, the one held 4h decision, the results paths for phase 5 and the test conventions.
4. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` and `docs/reference/PLUGIN_STORAGE.md`.

Older dated evidence is in `docs/reports/LAYER_BOUNDARY_HISTORY_20261115.md`. It is history: its "bounded" readers and adapters are being removed, not kept.

## Where things stand (2026-11-15)

- Phases 0–3 are done. Phase 4 units 4a–4h are done, except one set of config keys held for a user decision (status doc, "Held for the user").
- The v3 Hub releases are prepared in `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`), local and unpublished.
- 122 stale tests fail on `main` as well; they are deferred by the user.

## Work to do next, in order

1. **The held config keys**, once the user decides: either add a generic schema for top-level keys and delete the named readers, or keep them. The `tuning` block already has one validated shape; move the user's flat `tuning.n_trials`/`n_jobs` under `tuning.options.optuna` only with their go-ahead.
2. **Phase 5.** Write the phase 5 plan document first (bundle manifest, application/workspace resolution, inclusions and exclusions), add it to `docs/README.md`, then implement: all results paths from the workspace, results moved to `src/usr/results` (moving the user's `results/` needs a dry run shown to the user and their go-ahead, as in 4f), explicit bundles, declared requirements, a versioned bundle manifest.
3. **Phase 6**, then **phase 7**, as the plan describes.
4. **Publication, only when the user asks:** follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`. Remove `ai_assistant`, `theta_mcp` and the five MIMIC packages from the v3 feed, push `theta-hub-v3` and `theta-plugins-v3`, verify the client feed from a clean workspace, set `published_at`, merge to `main` together with the Hub change, and remove the "pre-v3 Hub releases" row from the exception register.

## Hard constraints

- No legacy support. Old patterns are rejected by ordinary validation; no code recognizes, detects or names them, and there are no compatibility adapters, forwarders, deprecation paths or migration hints. The single exception is the `src.app.api` forwarding package at phase 6. A compatibility path missing from the exception register is a bug.
- Never pick a winner between conflicting files. Back up, hash-check and show a dry run before moving the user's files.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through the SDK.
- Do not run anything on a cluster.
- Do not publish to the Hub, push the `-v3` worktrees or merge to `main` until the user asks. Pushing `implement/layer-boundaries` itself is allowed.
- Configuration UI changes follow `docs/reference/CONFIG_PARAMETER_FORMS.md`.
- Do not modify `../theta-plugins` or `../theta-hub`; work in the `-v3` worktrees and commit there with `git -c user.name=Tacho2`.
- Leave `prepare_tabular.py` alone.

## Testing

- Run tests per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. A whole-suite run in one process can die silently.
- Before calling a failure a regression, compare with and without your change (`git stash push -u`, rerun, `git stash pop`), or compare fresh worktrees by test ID.
- Tests select their workspace through fixtures (`fixture_workspace`, `optuna_workspace`, `sweep_workspace`, `themes_fixtures`); never `monkeypatch.chdir` alone. Tests must not depend on what the user has installed.
- For the installed-distribution probe, build in a fresh environment with a fresh setuptools build base and follow `scripts/verify_installed_distribution.py`.

## When you stop

Commit each unit separately with the attribution trailer the session gives you. Replace the status doc's contents to match the new state (it is not an append-only log; record evidence worth keeping in the commit messages or a dated report). Add new documents to `docs/README.md`. Report what landed (with commits), what was verified and how, what failed or was skipped, and the next item. Then update this prompt.
