# Layer boundary handoff prompt

Paste the prompt below into a new agent session in this repository to continue the layer boundary refactor. It lists everything that remains, in the order to do it. Update it when the state it describes changes.

---

You are continuing the layer boundary refactor of Theta-IDE on the branch `implement/layer-boundaries`. The branch exists on `origin` (ThetaIDE/ThetaIDE), but the phase 4 commits are local and unpushed. It has no PR. It must not be merged to `main` until the v3 Hub releases are published together with it.

**Read first, in this order:**
1. `AGENTS.md` and `docs/README.md`.
2. `docs/plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md`, especially "Confirmed decisions", which win over anything else, and phases 5–7.
3. `docs/development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md`: the progress outline and dated sections.
4. `docs/reference/LAYER_BOUNDARIES.md` (contract and exception register), `docs/reference/SDK.md` and `docs/reference/PLUGIN_STORAGE.md`.

## Where things stand (2026-11-14)

- **Phases 0–3** are complete locally. The only open items are external gates (see the last section).
- **Phase 4 (units 4a–4g) has landed locally.** Package format v3 only, `src/usr` layout, every compatibility adapter retired, SDK 2.2 (`src.app.api`), the user's workspace moved.
- **The v3 Hub releases are prepared, not published.** Worktrees `../theta-plugins-v3` and `../theta-hub-v3` (branch `v3-republish`) hold the sources and 39 release archives with registry entries and feeds. The user's own `../theta-plugins` and `../theta-hub` checkouts have uncommitted work of their own; never modify them.
- **The repository workspace runs the prepared releases.** All 20 installed packages were updated from `../theta-hub-v3` (for example `sb3-suite` and `d3rlpy-suite` 2.0.0, `blendrl` 1.3.0). Optuna and Slurm are deliberately not installed there.

## User decisions (2026-11-14)

- **`ai_assistant` and `theta_mcp`** stay as they are. The user tells their author. Do not port, drop or rebuild them.
- **SDK namespace at phase 6:** the user accepted the current wording. `src.app.api` moves to `src.ide.api` with a forwarding `src.app.api` package, as `SDK.md` and `LAYER_BOUNDARIES.md` say. That is the single exception to "no forwarding modules in phase 6": it exists so the published packages keep working without another republish. Every other module moves without forwarders.
- **No publishing yet.** Do not publish to Theta Hub, push any repository or merge to `main` until the user asks.
- **Tests must not depend on what the user has installed** (see "Hermetic tests").

## Work to do next, in order

### 1. Phase 4 close-out: MIMIC configurations in `theta-plugins-v3`

The only CQL agent is `d3rlpy/cql` (in `d3rlpy-suite`). Some MIMIC files in `theta-plugins-v3` still select an agent named plain `cql`, which no package provides, so they fail. Fix them before publishing `mimic`, `mimic-early-prediction` and `reward-shaping`:
- `experiments/mimic/configs/experiment/mimic/*.yaml` and `methods/cql_*.yaml`: methods with `agent: cql` and parameters `epochs`, `cql_alpha`, `pos_action_weight`, `bellman_loss`, combined with the `blendrl`, `dueling_resnet`, `transformer` and `cew` models.
- `get_agent_class("cql")` in `eval/mimic-early-prediction/data_module.py`, `eval/mimic-early-prediction/eval_logic.py` and `eval/reward-shaping/reward_shaping.py`.
- A comment in `models/cew/cew_model.py`.

`d3rlpy/cql` uses d3rlpy's own `d3rlpy/vector` encoder, so it cannot take those models or those parameters. Ask the user how each experiment should change: port it to `d3rlpy/cql` with d3rlpy parameters, or remove it. Do not invent an agent. Then rebuild the affected archives with `scripts/package_plugins.py` in `theta-plugins-v3`. The prepared versions are unpublished, so remove the old copies from `../theta-hub-v3/packages`, add the new ones with `../theta-hub-v3/scripts/add_releases.py`, rebuild the feeds with `scripts/build_index.py`, and check with `scripts/validate_manifest.py --check-archives`. Confirm the archives are deterministic (two builds, identical bytes) and rerun `tests/test_native_integration.py` in `theta-plugins-v3`.

### 2. Hermetic tests

The IDE suite must pass on a fresh checkout without the user's installed packages, ignored recipes or rules, or sibling checkouts. Today a fresh worktree fails about 289 tests and the populated workspace about 49, so about 240 tests depend on ignored workspace content. The work:
- **Inventory.** Run each test file in a fresh worktree (`git worktree add`), one file at a time. Classify every failure by what it reads: installed packages, `src/usr/config` recipes, `src/usr/datasets`, authored `src/usr` packages, `../theta-plugins`, `../theta-hub`, the environment.
- **Fixture packages.** Replace those dependencies with small fixture packages and configs created in `tmp_path` (follow `tests/test_provider_exports.py` and `tests/test_app_plugin_sdk.py`). Tracked fixtures go under `tests/fixtures`. Behavior that genuinely needs the real research packages (training SB3, BlendRL or d3rlpy end to end) belongs in `theta-plugins` `tests/test_native_integration.py`, which already builds and installs every archive.
- **`tests/component_fixtures.LocalHub`** must not fall back to `../theta-hub` or `../theta-plugins`. Build the needed releases from tracked fixtures, or skip when the explicit `THETA_HUB_ROOT` / `THETA_PLUGINS_ROOT` is unset.
- **Known cases:**
  - `test_ncshare_hardening`: two real-pipeline cases need Optuna and Slurm installed in the root workspace. Install them into a `tmp_path` workspace with `install_optional` instead.
  - `test_blendrl_rich_monitors::test_recipe_preserves_dnn...` reads the user's ignored `src/usr/config/experiment/newMonitors/onlineRL.yaml`, whose PPO method is disabled. Use `tests/fixtures/workspace_config/experiment/newMonitors/onlineRL.yaml`.
  - `test_rulesets::test_authored_nsfr...` reads `../theta-plugins`, which is still storage v2.
  - `test_sb3_theta_policy` (13 failures) imports `src.usr.models.blendrl`, decorator-registered models and `in.envs.cartpole.valuation`, none of which exist. Rewrite it against fixture packages or move its real-package coverage to `theta-plugins`.
- **Exit:** the full suite, run per file in a fresh worktree, has no failures that come from missing user state. Record the before and after counts in the status doc.

### 3. Bugs and leftovers found in phase 4

- **Lease host identity.** `src/app/components/transactions.py` (`assert_idle`, lines ~67 and ~77) stores `socket.gethostname()` and treats a lease from any other host name as live. On macOS the host name changes with the network (`MacBookPro.lan`, `MacBook-Pro-8.local`, `MacBookPro.attlocal.net`), so dead leases block updates forever. 83 such leases were moved to `.thetaide/stale-provider-leases-20261114T200025/`. Use a stable machine identity (for example a machine ID persisted in the user's state directory), and add a test with a changed host name.
- **Slurm vs `ncshare-site`.** Installing Slurm 1.1.0 into this repository's workspace is refused because the authored `ncshare-site` package owns `site/ncshare/*` and Slurm ships the same profiles. Ask the user which should own them, then remove the duplicate from the other.
- **`src/app/core/loader_migration.py`** is used only by `tests/test_data_loader_bindings.py` and `scripts/migrate_data_loaders.py`. Under the no-legacy decision, check whether it is migration code that should go, and delete it with its script and tests if so.
- **Authored packages still import private host modules.** `src/usr/models/blendrl-dev` (`src.app.monitoring.api`, `src.app.monitoring.measurements`, `src.app.core.types`, `src.app.core.protocols`, `src.app.core.provider_loader`, `src.app.core.data_loaders`) and `src/usr/packages/themes-dev` (`src.app.core.provider_catalog`, `src.app.monitoring.api`, `src.app.services.cancellation`). Port them to SDK 2.2 (`selected_capability`, `src.app.api.monitoring`, `src.app.api.execution.TrainingCancelled`), adding SDK exports only when they are general. This is a prerequisite for phase 7.

### 4. Publication (only when the user asks)

Follow `docs/development/THETA_HUB_RELEASE_WORKFLOW.md`: push `theta-hub-v3` and `theta-plugins-v3`, verify the client feed from a clean workspace, set `published_at` in the registry entries, merge `implement/layer-boundaries` to `main` together with the Hub change, and remove the last exception-register row ("Direct host imports in Hub package releases published before package format v3") from `docs/reference/LAYER_BOUNDARIES.md`. `ai_assistant` and `theta_mcp` stay in the feed at storage v2 until their author updates them.

### 5. Phase 5: remote bundles (delivery G)

Tuner continuation artifacts and declared site dependencies already landed with delivery D. Remaining, per the plan:
- Replace broad checkout copying with an explicit bundle: the application build plus the selected providers and their dependencies, and snapshots of the authored sources and configs used. Record IDs, versions, digests, contract versions and input mappings. Unselected experiments and providers are not required.
- Resolve application files from the application context and research files from the workspace context; do not assume the workspace is an editable IDE checkout. Keep site transport, incremental hashes and data publication semantics.
- Generic cluster setup installs the requirements the selected application, providers and tuner declare, instead of always selecting the Optuna extra.
- Version the bundle manifest. **No readers for old jobs or snapshots** (confirmed decision); confirm with the user that no site job is running before the cutover.
- Exit: an external workspace prepares and runs a reproducible bundle with the installed IDE, selected providers and declared data mappings, and continuation verifies provenance. The real cluster smoke run is a separate rollout gate.

### 6. Phase 6: consolidate core into `src/ide` (delivery H)

- Move `src/app` and the desktop host `frontend` into `src/ide`. Extension contracts live in `src.ide.api`; `src.app.api` stays as the one forwarding package (user decision above). Generic plotting infrastructure moves with the host; research plotting stays in packages.
- Add installed CLI entry points. Keep the small root launchers and Hydra discovery adapters already in the exception register. Runtime services resolve resources from the installed application, not a repository root.
- Update bundled providers, inventory, package data, docs, tests and tooling in the same change; there must not be two core trees. No other forwarders: saved Hydra targets and checkpoints that name old module paths stop working by decision.
- Build and inspect the wheel and sdist, install them in an isolated environment, and launch the desktop and headless CLI against a clean external workspace with `scripts/verify_installed_distribution.py`.
- Exit: substantial core code lives under `src/ide`; fresh-process checkpoint reload and workspace relocation pass; installed entry points do not need the repository as working directory.

### 7. Phase 7: enforcement (delivery I)

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
- Package integration: in `../theta-plugins-v3`, `../Theta-IDE/venv/bin/python -m pytest -q tests/test_native_integration.py` (57 passed at 4g).
- For the installed-distribution probe, build with a fresh setuptools build base and follow `scripts/verify_installed_distribution.py`.

## When you stop

Commit each unit separately with the attribution trailer the session gives you. Keep the status doc's progress outline and a dated section current. Add new documents to `docs/README.md`. Report what landed (with commits), what was verified and how, what failed or was skipped, and the next item. Then update this prompt.
