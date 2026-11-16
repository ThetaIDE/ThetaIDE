# IDE, plugin, and user layer implementation plan

Status: in implementation on `implement/layer-boundaries`. Phases 0–4 are implemented; phase 5 is next. The [implementation status](../development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md) tracks progress and the [handoff prompt](../development/LAYER_BOUNDARY_HANDOFF.md) lists the work in order. Plan written 2026-11-13 from the [boundary audit](../reports/LAYER_BOUNDARY_AUDIT_20261113.md); rewritten 2026-11-15 so every phase matches the confirmed decisions below. The [storage contract](../reference/PLUGIN_STORAGE.md) and the [ownership contract](../reference/LAYER_BOUNDARIES.md) describe supported behavior as it lands.

## Confirmed decisions

The user is the only user. These decisions win over anything else in this plan or in older documents.

### Ownership (2026-11-13)

- User datasets belong in `src/usr/datasets`. Deliberate external data references stay references, not an alternative default location.
- Package defaults stay inside their owning packages. Editable copies and overrides belong under `src/usr`, normally `src/usr/config` or an authored package's `configs/` directory.
- Slurm site support and tuning are standalone installable components, never mandatory core functionality or an implicit default. Core owns general extension, execution and artifact APIs; components own scheduler and tuner behavior, dependencies, settings, UI contributions and integration hooks.
- Without the relevant component installed, ordinary IDE flows expose no Slurm, site-specific or tuning-specific controls, settings or dependency requirements. An installed but unconfigured component may contribute its setup flow. A saved recipe that names a missing component produces an actionable error, not a silent fallback.
- Lightning and the four base paradigms stay in core. Extracting them is deferred.
- Bundled and downloaded plugins use the same public API. No plugin-specific host branches, private privileges or dedicated exceptions. New extension points are general contracts proven with independent fixture implementations.

### No legacy support (2026-11-14, refined 2026-11-15)

- Old patterns are unsupported and rejected. Everything that existed before a change becomes unsupported when the change lands; nothing is kept for behavior that no longer exists, and partially migrated states are not wanted.
- **Rejection is generic.** An old input fails through ordinary validation: an unknown key, an unsupported version, a missing file or module. No code recognizes, detects or names an old pattern, and there are no migration hints. Old inputs must still fail; they are never silently accepted or reinterpreted.
- Package format v3 only, on the IDE and on the Hub. Config destinations are logical selectors.
- The user's own files are moved completely when a layout changes (backup, hash check, dry run shown to the user first); migration code does not ship.
- Every compatibility adapter is retired, including the import aliases that let old checkpoints load. Old checkpoints, receipts, snapshots, jobs and saved Hydra targets stop working. Historical results are not rewritten and have no replay support.
- **No forwarding modules.** The 2026-11-14 exception for an `src.app.api` → `src.ide.api` forwarder lapsed when the `src/ide` move was dropped (below).

### Scope and gates (2026-11-15)

- **Results move into the user layer.** Results, including checkpoints, logs and tuning artifacts, live under `src/usr/results` in the selected workspace (phase 5). Checkpoints are saved there with the rest of the results.
- **Experiment export is removed** (exporter, export dialog, `run_pipeline.py export` and their tests). Component package building (`scripts/package_plugins.py`, the scaffolder) stays.
- **Workflow presets are rewritten in generic terms** and name no research package or experiment.
- **`from_study` is a generic feature**: it turns a configuration recorded in the results into experiment parameters. It names no tuner and resolves inside the selected workspace.
- **The tabular generator** (`prepare_tabular.py`) stays where it is; the user will revisit it.
- **Third-party and held-back packages leave the v3 feed.** At publication, `ai_assistant`, `theta_mcp` and the five MIMIC packages (`mimic`, `mimic-early-prediction`, `reward-shaping`, `mimic-env`, `mimic-clinical-suite`) are not in the feed. The work is ThetaIDE, not other authors' plugins.
- **No cluster testing.** The real cluster smoke run is not a gate and is not to be run. Local fixtures cover remote behavior.
- **Core stays at `src/app` and `frontend`** (2026-11-15). There is no move into `src/ide`; phase 6 is only installed entry points and distribution verification.
- **The desktop check is accepted as working** (the user opens the app and launches experiments). The remaining interactive items are listed in the status doc; the installed-wheel desktop launch belongs to the phase 6 gate.

## End state

```text
<application distribution>/
  src/
    ide/                         # current src/app plus desktop host
      api/                       # supported provider and app-plugin contracts
      config_defaults/           # generic framework defaults
      desktop/                   # current frontend host
      ...                        # execution, services, telemetry, CLI
    plugins/
      inventory.json
      components/bundled/<id>/   # distribution-owned defaults and implementation
      app/bundled/<id>/

<selected workspace>/
  src/
    plugins/
      components/installed/<id>/ # downloaded/imported managed packages
      app/installed/<id>/
    usr/
      config/<group>/            # editable YAML, same Hydra selectors as today
      datasets/                  # user dataset bytes
      results/                   # outputs: runs, checkpoints, logs, tuning artifacts
      models/ algorithms/ ...    # authored/copied complete packages
      app_plugins/ sites/ rules/ packages/ ...
  .thetaide/                     # state, journals, retained snapshots, cache
```

The application and workspace may share a root during development. Core must still distinguish them logically. Caches, credentials, virtual environments, repository tooling and test fixtures stay outside the source roots.

An authored package's `configs/` holds its owned defaults. `src/usr/config` holds editable recipes and overrides. Never maintain two writable authorities for the same user edit. Keep package IDs, capability selectors, Hydra group names and Python module paths stable.

## Phase 0 — Contract and baseline (done)

Reference ownership contract, read-only workspace inventory (`python -m src.app.workspace_inventory`), baseline captures and the exception register.

## Phase 1 — Workspace and configuration ownership (done)

`WorkspacePaths` carries explicit context through composition, validation, catalogs, plotting, datasets, scaffolding and workers. `ConfigResolver` reports selector, owner, origin, effective file and override status for the GUI, CLI and workers. Precedence: a `src/usr/config` file, then the selected provider profile, then the framework fallback; ambiguous owners fail. Editing a managed profile creates a user override; updates and uninstall never touch it. Institution defaults moved into packages.

Unit 4h replaced the research workflow examples in core (`src/app/pipeline/workflow/presets.py`) with generic presets.

## Phase 2 — Runner features and standalone site/tuner components (done)

Execution features and preflight hooks replace backend-name branches. Resource hints replace SB3 checks. The portable tuner contract moves Optuna storage and trial state into its provider. Contribution and readiness contracts drive tuning and site UI. Slurm and Optuna are optional components installed through the ordinary lifecycle. Declared cross-package exports (`provider_export`) replace private file-path helpers.

Unit 4h removed the historical readers it added (top-level `episode_outcomes`, unmarked tuning summaries, implicit Lightning features, the Slurm `legacy` publication state) and the dead Slurm entry points left in `src/app/pipeline`. The flat tuner fields wait with the held config keys.

## Phase 3 — App-plugin SDK (done)

The versioned `src.app.api` facade (now SDK 2.3) and `PluginContext` services. Bundled panels, both optional components and an independent fixture plugin use only the SDK. SDK requirements are checked before activation.

## Phase 4 — Retire `in/` and every adapter (done)

Units 4a–4g are described in the [phase 4 plan](LAYER_BOUNDARY_PHASE4_PLAN.md): package format v3, the `src/usr` cutover, adapter retirement, documentation, hermetic tests, the move of the user's workspace and the prepared v3 Hub releases.

### 4h — Close-out

Done in `2898c2d`..`e3c98cd`, with three exceptions. Item 1's config keys that no generic validator would reject (old method keys and `task:`, the `tensorboard` flag, the `sweep` flag, the Hydra direction and the flat tuner fields) were held for the user and removed afterwards (`0f373a7`, `4f990b0` and the TensorBoard commit). The theme colour roles and the hotkey pane list turned out to be current behavior. Item 8's user-settings and `test_ncshare_hardening` tests are deferred (user decision).

1. **Remove the remaining legacy code** (see the triage list in the status doc): flat tuner fields, the Slurm `legacy` publication state, Optuna's `version_N` and old sweep layouts, sibling-run grouping in `result_paths.py`, unmarked tuning summaries, forwarding imports, `.plugins.json` migration and deprecated plugin-manager aliases, widget attribute aliases, the old `tensorboard` key and old method-key fallbacks, old theme colour names, the legacy hotkey list, the `in.envs.` hot-reload path and the `run_pipeline.py` legacy-key message. Old inputs must still fail through ordinary validation.
2. **Retire the `episode_outcomes` reader.** Drop the duplicate top-level key from `sb3-dev` and `sb3-suite` (the flag already lives in `execution_features`), rebuild the unpublished archive in `theta-hub-v3`, reinstall it, delete the reader.
3. **Require declared execution features.** A Lightning provider without `execution_features` no longer gets Lightning's features implicitly.
4. **Remove experiment export.**
5. **Rewrite the workflow presets** generically.
6. **Make `from_study` generic and workspace-relative.**
7. **Delete dead core code:** `src/app/pipeline/slurm_runner.py`, `tuner_finalize.py`.
8. **Guards and tests:** extend `test_layout_guard` to dotted module paths (`in.envs`); give `test_config_viewer_composite` fixtures instead of the user's recipes; stop tests from reading user-level settings (`~/.config/thetaide/settings.toml`, QSettings); `test_ncshare_hardening` loads Slurm through the installed component.
9. **Exception register:** remove the "authored `src/usr` packages" row (they import only the SDK).

Exit gate: no tracked source outside tests recognizes an old pattern; the exception register lists only Lightning and the base paradigms, the Hydra namespace, the root launchers and (until publication) the pre-v3 Hub releases; the affected test files fail no test that passes on `main`.

## Phase 5 — Results in the user layer and remote bundles

Start with a short phase 5 plan document (bundle manifest fields, application/workspace resolution, what a bundle includes and excludes), as phase 4 did. Units:

1. **Every results path from the workspace.** Pipeline, telemetry, callbacks, Lightning builder, contract logger, `from_study`, the tuning monitor and runners stop using working-directory `results/` paths (the list is in the status doc). All go through `WorkspacePaths`.
2. **Move results to `src/usr/results`.** Change the single workspace definition; update monitors, the results browser, site publication, receipts, ignores and the test guard that fails runs writing into the checkout. Move the user's existing `results/` completely with the 4f method (backup, hash check, dry run shown first). Remote bundles never include `src/usr/results`.
3. **Explicit bundles.** Replace broad checkout copying with the application build plus the selected providers and their dependencies, and snapshots of the authored sources and configs used. Record IDs, versions, digests, contract versions and input mappings. Unselected experiments and providers are not required. Resolve application files from the application context and research files from the workspace context; keep site transport, incremental hashes and publication semantics.
4. **Declared requirements.** Generic cluster setup installs what the selected application, providers and tuner declare, instead of always selecting the Optuna extra. (Tuner continuation artifacts and site dependencies landed in phase 2.)
5. **Versioned bundle manifest.** Exactly one supported version. Older jobs and snapshots fail like any unsupported version.

Exit gate (local): an external workspace prepares and runs a reproducible bundle through the local and fixture schedulers with the installed IDE, selected providers and declared data mappings; continuation verifies provenance; nothing writes results outside `src/usr/results`. No cluster run.

## Phase 6 — Installed distribution

Core stays at `src/app` and `frontend` (user decision, 2026-11-15); nothing moves.

1. Add installed CLI entry points. Keep the small root launchers and Hydra discovery adapters listed in the exception register. Runtime services resolve resources from the installed application, not a repository root.
2. Build and inspect the wheel and sdist in a freshly built environment, install them in isolation, and launch the desktop and headless CLI against a clean external workspace with `scripts/verify_installed_distribution.py`.

Exit gate: installed entry points do not need the repository as working directory; the installed desktop opens and launches an experiment against an external workspace.

Done 2026-11-16: `theta = run_pipeline:cli` and `theta-ide = frontend.__main__:launch`; an installed application uses the working directory (or `THETA_WORKSPACE`) as its workspace; launchers that started `run_pipeline.py` relative to the working directory start the application's copy. Bundles take the application's identity and requirements from the installed distribution when there is no source tree. Evidence: [phase 6 distribution report](../reports/LAYER_BOUNDARY_PHASE6_DISTRIBUTION_20261116.json).

## Phase 7 — Enforcement

- Extend `tests/test_sdk_boundary.py` from the optional components and bundled packages to installed and authored packages and the whole desktop and core tree. Cover `import`, `from` and dynamic imports. Check package imports against the SDK and the package's declared dependencies; a broad `src.ide` prefix is not permission.
- Keep a finite, reviewed exception list (the register in `LAYER_BOUNDARIES.md`). Every exception is in the register; a compatibility path that is not in the register is a bug.
- Add behavior tests with differently named fixture providers, since static checks miss backend-name branches, configuration assumptions and undeclared file access.
- Archive obsolete architecture guidance and keep the docs index current.

Exit gate: a compatible extension can be added without editing core policy, downloaded packages do not depend on particular `src/usr` files, user edits affect only the intended selection, and CI enforces this through imports and behavior.

## Delivery units

| Delivery | Scope | State |
| --- | --- | --- |
| A | Ownership contract, inventory, baseline | Done |
| B | Explicit workspace context and source-aware config resolution | Done |
| C | Runner features and resource hints | Done |
| D | Standalone Slurm and tuning components, contribution and readiness contracts | Done |
| E | App SDK and bundled-panel migration | Done |
| F | Package format v3, `in/` retirement, adapter retirement (phase 4 including 4h) | Done |
| G | Results in `src/usr/results` and explicit remote bundles (phase 5) | Next after F |
| H | Installed entry points and distribution validation (phase 6) | After G |
| I | Boundary enforcement (phase 7) | After H |

Each unit is reviewable on its own and lands with its guardrails. No Hub publication, push to `main` or destructive workspace move is authorized by this plan; each needs the user's go-ahead.
