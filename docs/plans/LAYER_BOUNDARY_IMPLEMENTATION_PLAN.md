# IDE, plugin, and user layer implementation plan

Status: in implementation on `implement/layer-boundaries`. Phases 0–3 have met their local exit gates; phase 4 is next. See the [progress outline](../development/LAYER_BOUNDARY_IMPLEMENTATION_STATUS.md#progress-outline) for per-phase state and remaining external gates. Plan written 2026-11-13 with user decisions incorporated; this document itself makes no runtime changes. Based on the [boundary audit](../reports/LAYER_BOUNDARY_AUDIT_20261113.md) and the clarified user architecture. This plan replaces the older organization proposal's target layout for this work; update the current [storage contract](../reference/PLUGIN_STORAGE.md) as implementation lands.

## Confirmed decisions

- User datasets belong in `src/usr/datasets`. Migrate workspace-owned data there; deliberate external data references can remain references, not an alternative default location.
- Package defaults remain inside their owning packages. Editable copies and overrides belong under `src/usr`, normally `src/usr/config` or an authored package's `configs/` directory.
- Slurm site support and tuning/sweeping become proper standalone installable components. Neither is mandatory core functionality or an implicitly installed default. Core owns general extension, execution and artifact APIs; components own scheduler/tuner behavior, dependencies, settings, UI contributions and integration hooks.
- Without the relevant component installed, ordinary IDE flows expose no Slurm/site-specific or tuning-specific controls, settings or dependency requirements. The Hub can advertise available components. An installed but unconfigured component may contribute its setup flow; operational controls require the relevant configuration/capabilities. Missing components in an explicitly opened saved recipe produce actionable diagnostics, not silent fallback.
- Lightning and the four base paradigms remain in core for this implementation. Extracting them for a lighter distribution is deferred.
- Bundled and downloaded plugins use the same general public API. Alpha APIs may change to support required capabilities, but no plugin-specific host branches, private privileges or dedicated exceptions are permitted. Express extensions through reusable contracts and prove them with independent fixture implementations.
- `in/` migration scope (decided 2026-11-14; supersedes conflicting phase 4 steps below):
  - **Hard cutover.** Once the migration command ships, the IDE reads only `src/usr`. An unmigrated workspace fails with one actionable error that names the migrate command. There is no read-only transition period and no dual-root reading.
  - **No released-archive adapter.** Storage-v2 archives whose config destinations encode `in/config` are rejected with a "needs a newer release" error. Every published Hub component is rebuilt and re-released in the new format through the [production release workflow](../development/THETA_HUB_RELEASE_WORKFLOW.md), and only when a release is requested. The IDE cutover and the republished feed must be coordinated.
  - **Backup, then manual restore.** Before any move, the command copies `in/` and the affected state records into a timestamped backup under `.thetaide/`. Symlinks are copied as links; external targets are never copied. Undo is a documented manual restore, not a journaled `--rollback`. The dry-run still reports the full inventory and collisions and stops on any conflicting changed file, and an interrupted run must be safe to rerun.
  - **Historical results are records only.** Saved result configs, receipts and run records keep their `in/` paths unchanged. Replaying or resuming one stops with an actionable error naming the old path; no automatic path mapping.

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
      config/<group>/           # editable YAML, same Hydra selectors as today
      datasets/                 # canonical home for user dataset bytes
      models/ algorithms/ ...   # authored/copied complete packages
      app_plugins/ sites/ rules/ packages/ ...
  results/                       # output lifecycle
  .thetaide/                     # state, journals, retained snapshots, cache
```

The application and workspace may physically share a root during development. Core must still distinguish them logically. No requirement to put results, caches, credentials, virtual environments, repository tooling or test fixtures under the three source roots.

An authored package's `configs/` holds its owned defaults. `src/usr/config` holds editable project recipes and overrides. These have different roles; never maintain two synchronized writable authorities for the same user edit. Keep full-profile overrides initially unless an explicitly versioned merge contract is added. Core root defaults must compose with user settings through a tested, explicit mechanism rather than silently changing Hydra search semantics.

Use `src/ide`, not `src/src`: it names the responsibility and accommodates desktop and headless code. Keep package IDs, capability selectors and Hydra group names stable during storage changes. Directory naming is the last structural milestone, not the first.

## Phase 0 — Freeze the architecture contract and baseline (G1–G11)

1. Add a reference contract defining layer ownership, public API modules, allowed import directions, configuration ownership/precedence, and compatibility policy. Record this plan as the active proposal in the documentation index; leave current guides accurate until implementation lands.
2. Inventory tracked **and ignored** input/config/source files, package manifests, state ownership records and external dataset references. Record origin, hashes, effective selector and intended destination. Include local edits and the tracked `.npz`, not just `git ls-files` output.
3. Capture representative CLI/GUI config resolution, package inventories, checkpoint loading, and remote snapshot/continuation behavior. Reuse existing tests and small fixture providers. Record clean checkout versus this research workspace separately.
4. Create an explicit exception register for saved imports, required third-party namespace adapters, and any core-owned Lightning/base paradigm implementation. Every exception needs a reason and retirement or long-term support policy.

Exit gate: agreed ownership and compatibility contract; reproducible inventory; no assumption that ignored local files can be discarded.

## Phase 1 — Make workspace and configuration ownership explicit (G1, G6, G7)

1. Extend `WorkspacePaths` with named user-config, user-input, framework-default and provider-config concepts. Pass context through composition, validation, catalogs, plotting, dataset discovery, exports, scaffolding and workers. Remove module-import workspace selection from reusable services.
2. Add a source-aware config resolver that returns selector, owning package/origin, effective file and override status. Use the same resolver for GUI lists, CLI composition, schema selection and export. Report conflicting package profile owners instead of depending on scan order.
3. Define stable precedence: explicit user override, explicitly selected provider profile, framework fallback. Ambiguous providers fail before execution. Define how explicit capability overrides also affect profile/schema ownership; do not let those systems choose different owners.
4. Make editing a provider default an explicit operation that creates a user-owned override. Ordinary package updates manage provider roots; they must not rewrite user overrides. Preserve existing edited release recovery and transaction guarantees.
5. Move institution defaults and research workflow examples out of core policy into packages. Core provides general configuration/authoring APIs; the standalone site component contributes site-specific authoring/setup and scheduler behavior. Project connection settings stay in usr.

Exit gate: the same config resolves to the same owner/value in GUI, CLI and workers in two separate workspaces from an unrelated CWD. Missing/ambiguous providers produce clear diagnostics. No source migration required yet.

## Phase 2 — Extract standalone site/tuner components and complete general contracts (G2, G3, G6, G9)

1. Add runner capability fields for execution contract, supported input types, evaluation ownership, resume/reload modes, checkpoint format, tuning/winner promotion and episode outcomes. Provide a selected-provider preflight hook for constraints that cannot be expressed declaratively.
2. Replace backend-name branches in execution with contract/feature checks. Move current SB3/d3rlpy restrictions into their adapters without weakening rejection of unsupported evaluator, resume, data-module or tuning combinations.
3. Introduce resource/device hint hooks with conservative core defaults. Replace SB3 identity/substrings in scheduling decisions. Core remains responsible for enforcing the user's resource policy.
4. Extend the tuner API for portable storage/artifact declarations, launch bindings, continuation validation/repair, snapshot publication and monitor data. Move Optuna SQLite/trial-state logic into its provider. Keep remote file transport, process supervision and publication mechanics generic.
5. Express pruning/objective compatibility as capabilities. Remove implicit Optuna installation/selection from core services and default startup. Make tuning controls and provider-specific monitors contributed extensions; use the shared parameter controls for ordinary config boxes, preserving the specialized Methods layout.
6. Add public cross-package exports and dependency-aware resolution. Validate caller dependency/version requirements; migrate private file-path helper calls incrementally. Define compatibility checks for app plugins as well as component runners. Do not broaden this into a dependency solver or code sandbox.

7. Extract Slurm submission, allocation validation, scheduler polling, cancellation, continuation, site templates/settings and scheduler-specific environment/setup logic into a standalone site component. Audit `src/app/sites`, pipeline Slurm runners, desktop controls and bootstrap dependencies. Move reusable transport/process/artifact primitives into general APIs only when their contract is independent of Slurm. Remote orchestration may live in the component where no general core responsibility remains.
8. Define general capability-driven contribution and readiness contracts for settings, config sections, commands and monitors. Remove Slurm/tuning controls from unconditional host construction. Do not implement visibility by hardcoding checks for package IDs. Support absent, installed/unconfigured, configured, disabled and removed states, including missing providers referenced by saved recipes.

Exit gate: a clean IDE has no Slurm/tuning UI or runtime dependency requirement; installed components expose setup and configured functionality appropriately. Removing/disabling them removes their contributions and cleans up lifecycle resources. An independent site/scheduler fixture and non-Optuna tuner integrate without core edits. Existing unsupported operations still fail with actionable errors, and Slurm/Optuna behavior is preserved through their components.

## Phase 3 — Stabilize the app-plugin SDK (G5, G9)

1. Publish a versioned host facade for workspace/config access, changes, settings, themes, process tracking, commands, editor actions and UI slots. Mark supported widget helpers as public API.
2. Migrate bundled text editor, model builder, TensorBoard, documentation and arXiv panels to the facade. Replace config-viewer imports and `context.window` implementation access with focused services/events.
3. Validate host API requirements before activation. Cover activation/deactivation, settings, edits, workspace context and cleanup with a small independent app-plugin fixture.

Exit gate: bundled plugins and the fixture operate without imports of private host modules or reliance on concrete `Window` attributes. Headless catalogs do not require Qt or execute panel code.

## Phase 4 — Version and execute the `in/` migration (G1, G6–G8, G11)

1. Separate native package **format/API versions** from workspace **layout version**. Define a new config destination representation (logical group/selector or an explicitly versioned destination contract), so newly exported archives do not encode `in/config` as permanent ownership policy.
2. Continue reading released storage-v2 archives through a bounded adapter that translates their config destination metadata. Do not silently reinterpret their implementation paths, rewrite signed/downloaded archives, or extend generic legacy discovery. Coordinate future package releases with the existing Theta Hub production workflow when releases are actually requested.
3. Build a migration command with dry-run, inventory, collision diagnostics, recovery journal, idempotent resume and rollback. Acquire workspace locks and require idle implementations. Never choose between conflicting changed files automatically.
4. Move editable profiles to `src/usr/config` with selectors preserved. Deduplicate unchanged materialized defaults only when provenance and effective resolution prove it is safe. Preserve intentional full-profile overrides. Handle root `config.yaml`, absolute Hydra defaults, interpolations and nested group references explicitly.
5. Move all workspace-owned user dataset bytes to `src/usr/datasets`; preserve deliberate external references. Migrate known dataset references and site/input mappings structurally; surface unresolved arbitrary strings for review. Historical result configs remain historical records, with a deliberate replay path policy.
6. Relocate the tabular generator to a provider/example package with explicit output context. Remove unused `in/envs` only after checking ignored local content. Move apprenticeship and plotting defaults into their owners.
7. Translate state/config ownership records, retained backups, exporter mappings, fork provenance, CLI help and editor links. Inventory old-root consumers and provide read-only compatibility diagnostics during transition; do not leave two active write roots or rely on symlinks that violate managed-root protections.
8. Update ignores, tracked fixtures and documentation; remove `in/` after all active readers/writers have migrated. Keep external datasets outside distribution payloads.

Exit gate: dry-run followed by migration and rerun is stable; rollback restores bytes, modes and ownership. User edits survive update/uninstall. Existing released archives install through the adapter. Config composition and dataset selection match the baseline. Both clean and populated workspaces work.

## Phase 5 — Make remote bundles reflect layer ownership (G3, G7, G10)

Design the bundle format alongside phases 1–2; land its storage-path cutover with phase 4 before removing old paths.

1. Replace broad checkout copying with an explicit application build plus selected provider/dependency and authored source/config snapshots. Record IDs, versions, digests, contract versions and input mappings. Unselected user experiments/providers should not be implicitly required.
2. Resolve application files from application context and research files from workspace context. Avoid assuming the workspace contains a complete editable IDE checkout. Preserve site transport, incremental hashes and data publication semantics.
3. Have the selected tuner declare its continuation artifacts and any environment requirements. Generic cluster setup installs the declared application/provider requirements instead of always selecting the Optuna extra.
4. Version manifests and retain an explicit compatibility reader for already-created jobs/snapshots. Running or resumable jobs must not be broken by source-path changes.

Exit gate: an external workspace can prepare and run a reproducible bundle with the installed IDE, selected providers and declared data mappings; continuation verifies provenance. Tests cover old snapshots and the new format. A real cluster smoke run is a rollout gate, separate from local implementation tests.

## Phase 6 — Consolidate core into `src/ide` (G4, G8, G11)

1. Move `src/app` and the desktop host from `frontend` into `src/ide`, keeping extension contracts in a clearly supported API namespace. Move generic plotting infrastructure with the host; keep research plotting implementations in plugins.
2. Add installed CLI entry points. Retain small root launchers and required Hydra discovery adapters as documented compatibility entry points. Runtime services should resolve resources from the installed application, not a presumed repository root.
3. Update bundled provider SDK imports, inventory, package data, documentation, tests and tooling together. Avoid a permanent pair of core package trees. Maintain fixed adapters for old saved Python targets and checkpoints until their declared retirement boundary.
4. Build and inspect wheel/sdist, install them in an isolated environment, then launch the desktop and headless CLI against a clean external workspace. Verify bundled manifests/assets/defaults are present and installed/user payloads/data/state are absent.

Exit gate: all substantial core implementation lives under `src/ide`; compatibility modules forward to identical canonical objects. Fresh-process checkpoint reload and workspace relocation pass. Repository CWD is unnecessary for installed entry points.

## Phase 7 — Enforce boundaries and retire transitional paths (all gaps)

- Expand architecture checks to `import`, `from`, dynamic imports and the full desktop/core tree. Maintain a finite reviewed exception list. Check plugin imports against the supported SDK and declared package dependencies; don't treat a broad `src.ide` prefix as permission to import internals.
- Add behavior tests using differently named fixture providers. Static checks cannot detect all backend-name branches, configuration assumptions or undeclared file access.
- Run selected compatibility, transaction/recovery, config provenance, execution, app-plugin, packaging and snapshot suites. Reuse existing fixtures rather than running expensive training for path-only changes.
- Retire old `in/`, `frontend.plugins.core`, `src.app.tuners`, `src.usr` compatibility targets only at a documented boundary with saved-target/checkpoint fixtures and downstream caller migration. Keep legitimate long-term third-party namespace adapters explicit.
- Update current storage/API guides, built-in documentation and the documentation index as each phase changes supported behavior. Archive obsolete architectural guidance after the replacement is supported.

Exit gate: a compatible extension can be added without editing core policy, downloaded packages do not depend on particular usr files, and user edits affect only the intended project/provider selection. CI enforces those claims through both imports and behavior.

## Suggested delivery units and dependencies

| Delivery | Scope | Depends on |
| --- | --- | --- |
| A | Reference ownership contract, inventory tooling, baseline tests | None |
| B | Explicit workspace context and source-aware config resolution | A |
| C | Runner features and resource hints | A, context portions of B |
| D | Standalone Slurm site and tuning components, general lifecycle/UI contracts, absence/readiness gates | C and B |
| E | App SDK and bundled-panel migration | A and B |
| F | Versioned package config destinations and recoverable `in/` migration | B, relevant C/D/E consumers ready |
| G | Remote bundle cutover and old-job readers | B, C, D; coordinate cutover with F |
| H | `src/ide` consolidation and installed-distribution validation | C–G |
| I | Adapter retirement and strict boundary CI | H plus compatibility retirement conditions |

Each delivery should be reviewable independently. New guardrails should land with the relevant contract changes, rather than waiting until delivery I. No component publication, remote rollout or destructive workspace migration is authorized merely by writing this plan.
