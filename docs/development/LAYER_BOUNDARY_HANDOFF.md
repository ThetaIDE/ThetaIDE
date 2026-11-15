# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. It lists everything that remains, in the order to do it. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch exists on `origin` (ThetaIDE/ThetaIDE), but the phase 4 commits are local and unpushed. It has no PR. It must not be merged to `main` until the v3 Hub releases are published together with it.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`, especially "Confirmed decisions", which win over anything else, and phases 5–7.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: the progress outline and dated sections.
4. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` and `docs/reference/PLUGIN_STORAGE.md`.

## Where things stand (2026-11-15)

- **Phases 0–3** are complete locally. The only open items are external gates (see the last section).
- **Phase 4 (units 4a–4g) has landed locally**, with its close-out: the IDE suite is hermetic, the phase 4 leftovers are fixed, and authored packages import only SDK 2.3. See "Hermetic tests and phase 4 leftovers" in the status doc. All branch commits are local and unpushed.
- **The v3 Hub releases are prepared, not published.** Worktrees `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`) hold the sources, archives, registry entries and feeds. `mimic`, `mimic-early-prediction`, `reward-shaping`, `mimic-env` and `mimic-clinical-suite` are held back at their published storage-v2 releases (user decision). The user's own `../theta-plugins` and `../theta-hub` checkouts have uncommitted work of their own; never modify them.
- **The repository workspace runs the prepared releases** (20 packages). Optuna and Slurm are not installed there.
- **Tests:** 122 failures in 27 files remain. Every one also fails on `main` and comes from a stale test, not from user state. The list is in the status doc.

## User decisions (2026-11-14)

- **`ai_assistant` and `theta_mcp`** stay as they are. The user tells their author. Do not port, drop or rebuild them.
- **SDK namespace at phase 6:** the user accepted the current wording. `src.app.api` moves to `src.ide.api` with a forwarding `src.app.api` package, as `SDK.md` and `LAYER_BOUNDARIES.md` say. That is the single exception to "no forwarding modules in phase 6": it exists so the published packages keep working without another republish. Every other module moves without forwarders.
- **MIMIC packages are held back** from v3 rather than ported; their experiments select a `cql` agent no package provides.
- **`ncshare-site` owns the NCShare site profiles**; Slurm ships only the generic template.
- **No publishing yet.** Do not publish to Theta Hub, push any repository or merge to `main` until the user asks.
- **Tests must not depend on what the user has installed** (see "Hermetic tests").

## Work to do next, in order

### 1. Remaining phase 4 cleanup

- **v1 `episode_outcomes` reader.** `src/app/core/execution_features.py` still reads a capability's top-level `episode_outcomes`. Move the flag into `execution_features` in `src/usr/algorithms/sb3-dev` and `../theta-plugins-v3/methods/sb3`, rebuild that unpublished archive (`scripts/package_plugins.py`; in `theta-hub-v3`, replace it with `scripts/add_releases.py`, then `scripts/build_index.py` and `scripts/validate_manifest.py --check-archives`), reinstall `sb3-suite` in the repository workspace, and delete the reader.
- **Stale tests (122, optional before phase 5).** The largest are `test_results_panel` (32; the results explorer moved to current-item selection on `main`), `test_job_manager` (14; the runner now launches `APPLICATION_ROOT/run_pipeline.py`, so its fake pipeline needs a redesign) and `test_queue_panel` (11). `test_trial_seed_visibility`, `test_objective_components` and `test_tuner_components` use the removed `theta_execution.script` and `theta_objective_runner.script`; rewrite them against a Theta-owned bootstrap with fixture packages, or move real-package coverage to `theta-plugins-v3`. Use the fixture conventions in the status doc. Compare against `main` by test ID before calling anything a regression.

### 2. Publication (only when the user asks)

Follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`: push `theta-hub-v3` and `theta-plugins-v3`, verify the client feed from a clean workspace, set `published_at` in the registry entries, merge `implement/layer-boundaries` to `main` together with the Hub change, and remove the last exception-register row ("Direct host imports in Hub package releases published before package format v3") from `docs/reference/LAYER_BOUNDARIES.md`. `ai_assistant` and `theta_mcp` stay in the feed at storage v2 until their author updates them.

### 3. Phase 5: remote bundles (delivery G)

Tuner continuation artifacts and declared site dependencies already landed with delivery D. Remaining, per the plan:
- Replace broad checkout copying with an explicit bundle: the application build plus the selected providers and their dependencies, and snapshots of the authored sources and configs used. Record IDs, versions, digests, contract versions and input mappings. Unselected experiments and providers are not required.
- Resolve application files from the application context and research files from the workspace context; do not assume the workspace is an editable IDE checkout. Keep site transport, incremental hashes and data publication semantics.
- Generic cluster setup installs the requirements the selected application, providers and tuner declare, instead of always selecting the Optuna extra.
- Version the bundle manifest. **No readers for old jobs or snapshots** (confirmed decision); confirm with the user that no site job is running before the cutover.
- Exit: an external workspace prepares and runs a reproducible bundle with the installed IDE, selected providers and declared data mappings, and continuation verifies provenance. The real cluster smoke run is a separate rollout gate.

### 4. Phase 6: consolidate core into `src/ide` (delivery H)

- Move `src/app` and the desktop host `frontend` into `src/ide`. Extension contracts live in `src.ide.api`; `src.app.api` stays as the one forwarding package (user decision above). Generic plotting infrastructure moves with the host; research plotting stays in packages.
- Add installed CLI entry points. Keep the small root launchers and Hydra discovery adapters already in the exception register. Runtime services resolve resources from the installed application, not a repository root.
- Update bundled providers, inventory, package data, docs, tests and tooling in the same change; there must not be two core trees. No other forwarders: saved Hydra targets and checkpoints that name old module paths stop working by decision.
- Build and inspect the wheel and sdist, install them in an isolated environment, and launch the desktop and headless CLI against a clean external workspace with `scripts/verify_installed_distribution.py`.
- Replace the remaining working-directory `results/` paths in the pipeline (listed in the status doc) with the selected workspace.
- Exit: substantial core code lives under `src/ide`; fresh-process checkpoint reload and workspace relocation pass; installed entry points do not need the repository as working directory.

### 5. Phase 7: enforcement (delivery I)

- Extend `tests/test_sdk_boundary.py` from the optional components and bundled packages to installed and authored packages and the whole desktop and core tree. Cover `import`, `from` and dynamic imports. Check package imports against the SDK and the package's declared dependencies; a broad `src.ide` prefix is not permission.
- Keep a finite, reviewed exception list (the register in `LAYER_BOUNDARIES.md`).
- Add behavior tests with differently named fixture providers, since static checks miss backend-name branches, configuration assumptions and undeclared file access.
- Adapter retirement already happened in phase 4; only the SDK forwarding package remains, with its retirement boundary documented.
- Archive obsolete architecture guidance and keep the docs index current.
- Exit: a compatible extension can be added without editing core policy, downloaded packages do not depend on particular `src/usr` files, user edits affect only the intended selection, and CI enforces this through imports and behavior.

### External gates (need the user)

- A real cluster smoke run (phase 2, again after phase 5).
- An interactive check of the installed desktop: editor, figure and export matrix, and the migrated app-plugin panels (phases 1 and 3, again after phase 6).

## Hard constraints

- No legacy support: no compatibility adapters, deprecation paths, readers for old formats or messages about old formats. The one exception is the `src.app.api` forwarding package in phase 6.
- Never pick a winner between conflicting files. Leave historical results in place.
- Core must not branch on package IDs, backend names or institution names. Packages import host code only through the SDK.
- Configuration UI changes follow `docs/reference/CONFIG_PARAMETER_FORMS.md`.
- Do not modify `../theta-plugins` or `../theta-hub`; work in the `-v3` worktrees. Commit there with `git -c user.name=Tacho2`.

## Testing

- Run tests per file: `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q -p no:cacheprovider tests/<file>.py`. A whole-suite run in one process can die silently.
- Before calling a failure a regression, compare with and without your change (`git stash push -u`, rerun, `git stash pop`), or compare fresh worktrees by test ID.
- Package integration: in `../theta-plugins-v3`, `../Theta-IDE/venv/bin/python -m pytest -q tests/test_native_integration.py` (71 passed).
- Tests select their workspace through fixtures (`fixture_workspace`, `optuna_workspace`, `sweep_workspace`, `themes_fixtures`); never `monkeypatch.chdir` alone. A run fails if tests write results into the checkout.
- For the installed-distribution probe, build with a fresh setuptools build base and follow `scripts/verify_installed_distribution.py`.

## When you stop

Commit each unit separately with the attribution trailer the session gives you. Keep the status doc's progress outline and a dated section current. Add new documents to `docs/README.md`. Report what landed (with commits), what was verified and how, what failed or was skipped, and the next item. Then update this prompt.
