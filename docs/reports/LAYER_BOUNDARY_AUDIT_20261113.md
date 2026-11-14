# IDE, plugin, and user layer boundary audit

Date: 2026-11-13. Status: source audit and proposed architecture, not an implemented migration.

User decisions following this audit are recorded in the [implementation plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md#confirmed-decisions): canonical datasets under `src/usr/datasets`, editable defaults under usr, standalone optional Slurm and tuning components with capability-driven UI, shared public APIs for all plugins, and Lightning/base paradigms retained in core. Those decisions resolve the options discussed below.

## Finding

`in/` is not the last breach. It is a visible split in user ownership, but the more consequential gaps are provider-specific decisions in core execution, Optuna-specific cluster orchestration, desktop plugins depending on host internals, and application code outside the proposed core root. Moving directories alone would leave those dependencies intact.

The current three roots are **`src/app`, `src/plugins`, and `src/usr`**, not `src/src`. Recommend **`src/ide`** for the eventual core root, including both desktop and headless framework code. Establish contracts and path policy before renaming it.

This report applies the clarified target below. It deliberately differs from the deferred [workspace separation proposal](../plans/PROJECT_ORGANIZATION_PLAN.md), which proposed retaining `in/` and moving managed packages into state storage. Current supported behavior remains documented in [Plugin storage](../reference/PLUGIN_STORAGE.md). The [implementation plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) describes the new proposal's sequence.

## Target and interpretation

| Layer | Owns | Allowed dependencies |
| --- | --- | --- |
| `src/ide` | Host shell, headless services, contracts, configuration composition, extension loading, generic execution and artifacts | Standard/third-party libraries and other core modules; extensions only through declared contracts |
| `src/plugins` | Bundled and downloaded component/app packages, their implementations, defaults, resources and tests | Public core API, package-relative code, explicitly declared compatible package dependencies |
| `src/usr` | Authored/copied components and app plugins, editable YAML, user resources and input data | Public core API and explicitly selected compatible providers/dependencies |

Static dependencies point from user code toward plugin/core APIs and from plugins toward core APIs. At runtime, the core necessarily invokes plugin and user implementations. That inversion is valid when the host owns the contract, checks compatibility, and loads the explicitly selected implementation. Generic discovery of `src/usr` is therefore not itself a breach. A core branch on `sb3`, or a bundled default that assumes a particular user's site directory, is a breach of the proposed rule.

“Plugins agnostic to usr” means no dependency on a particular user's source layout, package ID, or mutable settings. Plugins may consume configuration and data passed by the host, and resolve deliberately declared user overrides. A user-authored provider can satisfy a dependency only through explicit identity/version/contract selection; its physical origin must not accidentally change behavior.

The three roots classify source and research inputs. Results, durable application state, caches, build output and development tooling retain separate lifecycles. They do not need to be forced into a source layer.

## Scope and evidence limits

Inspected workspace path policy, tracked and ignored `in/` content, bundled and locally installed manifests/source, authored packages, execution/composition services, desktop extension interfaces, packaging, remote snapshots, and existing boundary tests. This is a static audit of the working tree, including concurrent uncommitted configuration-panel work. No plugin code, training, remote jobs, or package releases were executed.

At inspection, `in/` contained 95 YAML files and one Python file, excluding cache files. `git ls-files in` listed 26 tracked files, including placeholders and one `.npz` dataset. Installed component roots contained 20 manifests, bundled component roots six, bundled app roots five, and authored roots twelve. These are local snapshot counts, not distribution guarantees. No direct Python `import`/`from` references to `src.usr` or `usr.*` were found in inspected bundled/installed plugin Python sources; that result does not rule out dynamic lookup, strings or undeclared runtime coupling.

## What already fits

- [WorkspacePaths](../../src/app/workspace.py) separates application and selected workspace, centralizes authored/managed/bundled roots, and rejects installation into protected core, bundled and user locations.
- [ProviderCatalog](../../src/app/core/provider_catalog.py) reads manifest metadata without importing implementations, reports duplicate identities/capabilities, distinguishes origin, and supports explicit ownership overrides.
- [Native package validation](../../src/app/components/archive.py) checks storage/API versions and package containment. [Lifecycle operations](../../src/app/components/lifecycle.py) check declared component versions/dependency cycles and use shared transactions/locks. These mechanisms should be extended, not replaced by folder scanning.
- [Provider loading](../../src/app/core/provider_loader.py) supports package-relative code/resources and stable workspace-specific checkpoint namespaces. Physical authored code can load into a plugin namespace without being moved into managed storage.
- [Tuner contracts](../../src/app/pipeline/tuners.py), execution contexts/results, monitor contracts, and app plugin lifecycle interfaces already provide parts of the intended integration boundary.
- [Existing organization tests](../../tests/test_provider_organization.py) cover protected destinations, dependency ownership, import identity, relocation, distribution policy and framework dispatch. The repo already rejects several former legacy roots.

## `in/`: disposition by content

| Content | Observed role | Proposed destination/action |
| --- | --- | --- |
| `in/config/experiment`, user model/agent/env/data/eval/plot profiles | Editable research recipes and overrides; many local files are ignored by Git | `src/usr/config/<same Hydra group and relative selector>` |
| `in/config/config.yaml` | Tracked workspace copy of framework root config | Core owns root defaults; preserve intentional user changes as an explicit user override under `src/usr/config`, with tested Hydra composition semantics |
| Four base paradigm YAMLs | Core defaults copied into user-visible input space | Keep canonical definitions in core `config_defaults/paradigms`; only intentional user overrides under `src/usr/config/paradigms` |
| `in/config/paradigms/apprenticeship.yaml` | Tracked profile selecting `themes-dev/runner` and `themes-dev/trajectories` | Research/provider-owned definition; put its default in the owning THEMES package and editable overrides under usr. A clean IDE must not advertise a runnable provider it lacks |
| `in/config/hydra/sweeper/optuna_*.yaml` | Copies of bundled Optuna defaults; provider-specific Hydra targets | Optuna package `configs/hydra/sweeper`; materialize a user copy only for deliberate editing |
| `in/config/site/*.yaml` | Tracked copies of core site profiles; additional authored site defaults also exist in `src/usr/sites` | Generic local schema/default in core; site templates in provider packages; machine/project settings in usr |
| `in/config/workflow/*.yaml` | Tracked copies of core workflow examples | Provider/example package owns examples referencing research recipes; usr owns edited workflows; core owns the generic workflow engine/schema |
| `in/config/plot/thetalib/tuning.yaml` | Tracked provider-related plotting profile | Owning plotting package; user override under usr |
| `in/datasets/prepare_tabular.py` | Executable scikit-learn dataset generator, tracked despite broad dataset ignore | Example/data-provider package with an explicit output path; not mixed with input bytes |
| `in/datasets/**` | Research inputs, including tracked `pyrenees/pyrenees_gmm_scaler.npz` and ignored local data | `src/usr/datasets` or an explicit external dataset reference; preserve bytes and references; do not put generated data in a wheel |
| `in/envs/.gitkeep` | Abandoned implementation root; current contract says it is ignored | Remove after checking local/ignored content and compatibility references; never reactivate automatic discovery |

Thirteen paths exist both in `in/config` and core `config_defaults`; twelve were byte-identical at inspection, while `online_rl.yaml` differed in bytes (the displayed definitions appeared equivalent). Byte differences alone must not be interpreted as meaningful user edits. Compare parsed YAML and provenance before deduplicating.

## Gap register

Priority describes migration importance, not a claim of a currently failing feature. P1 establishes correctness/extension boundaries; P2 consolidates storage and APIs; P3 retires compatibility and naming debt.

### G1 — User configuration has multiple authorities (P1)

Evidence: [workspace.py](../../src/app/workspace.py) defines `config` as `in/config`, while `research` is `src/usr`; `config_roots()` adds every package's `configs` directory and core defaults. [Archive validation](../../src/app/components/archive.py), [forking](../../src/app/components/forking.py), installer/configuration services and exporter manifests encode `in/config/...` as destinations.

Impact: a component's packaged profile, materialized active copy and authored copy can all exist. Origin and edit ownership are harder to understand. Config lookup uses ordered roots and first matching files; capability duplicate diagnostics do not by themselves define conflict policy for every profile selector. This audit found no duplicate destination among the inspected plugin manifest mappings, but the resolution policy still needs an explicit collision contract.

Required boundary: source-aware config resolution with a documented precedence and owner. Package defaults remain package-owned; user edits live in usr and override deliberately. Any decision to merge partial overrides rather than replace a full profile must be explicit and tested against Hydra semantics.

### G2 — Core execution knows concrete provider restrictions (P1)

Evidence: [execution_backend](../../src/app/core/execution.py) recognizes `sb3`, `d3rlpy`, `lightning`, and `numerical`, rejects backend-specific evaluator/resume/data-module combinations, and recommends `sb3-dev/d3rlpy-dev` in compatibility errors. [Resource estimation](../../src/app/services/resource_budget.py) uses SB3 name checks and `sb3_native_policy` to decide memory/device behavior.

Impact: adding a compatible runner can require modifying core; changing provider support requires synchronized host edits. Restrictions currently prevent unsupported behavior and must survive migration.

Required boundary: execution feature declarations and selected-provider validation hooks for resume, evaluation, input types, tuning promotion, resource/device estimates. Keep conservative core fallbacks and rejection of unsupported combinations. Decide explicitly whether Lightning remains a supported core implementation or becomes a bundled adapter; either option must remove special treatment of unrelated provider identities.

### G3 — Optuna implementation details escape the tuner contract (P1)

Evidence: [site execution](../../src/app/sites/execution.py) rewrites `tuning.options.optuna.*` and paths under `results/optuna`; [continuation](../../src/app/sites/continuation.py) imports Optuna and manipulates trial states; [publication](../../src/app/sites/publication.py) handles its database; [Slurm setup](../../src/app/sites/slurm.py) installs `.[optuna]`. [Objective validation](../../src/app/objectives/api.py) reads Optuna pruning settings. [Sweep UI](../../frontend/monitor/sweep_view.py), [runner manager](../../frontend/runner_manager.py) and configuration controls contain Optuna-specific behavior. Defaults in [tuners.py](../../src/app/pipeline/tuners.py) and [monitor selection](../../src/app/telemetry/monitors.py) also name it.

Impact: the nominally generic tuner interface does not cover continuation, transport, publication and monitoring. Another tuner cannot get equivalent integration simply by implementing today's protocol.

Required boundary: extend tuner lifecycle/artifact/monitor contracts; move database repair, storage rewrites and provider UI behind the selected tuner. A distribution can select Optuna as a default through inventory/configuration; generic services should not assume that identity.

### G4 — Desktop host and contracts sit outside the core root (P2)

Evidence: [frontend](../../frontend) contains the app, settings, UI, plugin host, terminal, monitors and result browser. Root [run_pipeline.py](../../run_pipeline.py), [plot](../../plot) adapters and [hydra_plugins](../../hydra_plugins) also carry runtime entry points. [Packaging](../../pyproject.toml) explicitly assembles these separate package trees.

Impact: filesystem ownership does not match the desired three-layer explanation. Renaming only `src/app` would leave a substantial part of the core outside it.

Required boundary: eventually consolidate headless and desktop host under `src/ide`, with public contract modules and small documented entry-point adapters. Generic third-party discovery entry points are legitimate adapters, even if a tool requires a top-level namespace.

### G5 — Bundled app plugins rely on private host implementation (P1)

Evidence: [PluginContext](../../frontend/plugins/context.py) exposes the whole `Window`; the [text editor](../../src/plugins/app/bundled/text_editor/panel.py) accesses its settings/theme/Hub client and imports config-viewer invalidation; [model builder](../../src/plugins/app/bundled/model_builder/__init__.py) imports `frontend.config_viewer.config_tree`. Bundled panels import host theme/widgets directly. The TensorBoard [server](../../src/plugins/app/bundled/tensorboard/server.py) uses module-level service paths.

Impact: physical placement under plugins does not make these independently compatible extensions. Host refactors can break them outside a versioned API, and workspace selection can depend on ambient state.

Required boundary: expose settings, themes, config documents/change notifications, workspace paths, editor actions and process services through a versioned host facade. Public widget helpers can remain a supported SDK. Access to a particular host window's internals should not be required.

### G6 — Core ships institution/user-specific assumptions (P1)

Evidence: core [NCShare defaults](../../src/app/config_defaults/site/ncshare/_defaults.yaml) and [ARC defaults](../../src/app/config_defaults/site/arc/_defaults.yaml) embed `src/usr/sites/<name>/ssh_config` and shortcut paths. [Site catalog](../../src/app/sites/catalog.py) computes the editable site path directly and creates authored sites. Core [workflow presets](../../src/app/pipeline/workflow/presets.py) reference CartPole, BlendRL and MIMIC recipes.

Impact: generic defaults assume particular higher-layer content. A path to an explicitly selected user site is valid; hardcoded institution names and examples in the generic host are the coupling.

Required boundary: keep generic scheduler/SSH configuration services in core, resolve selected site ownership through metadata, place institution templates and research presets in packages, and create editable copies under usr through generic authoring services.

### G7 — Workspace context is centralized but not consistently used (P1)

Evidence: [dataset discovery](../../src/app/pipeline/datasets.py), [tabular loading](../../src/app/core/paradigm_impls/base/tabular.py), [plot configuration](../../src/app/core/plotting/base.py), and desktop dataset discovery in [config_viewer.py](../../frontend/config_viewer.py) still construct CWD-relative `in/...` paths. [Composition](../../src/app/pipeline/compose.py) initializes `CONFIG_DIR` at import; [catalog services](../../src/app/services/catalog.py) also have a module-level config path. Several root CLI export/scaffold branches construct services with `PROJECT_ROOT`.

Impact: a root rename has a wide surface, and selected-workspace behavior can differ between paths that receive context and paths relying on process state. These are static coupling findings; this audit did not reproduce a specific cross-workspace failure.

Required boundary: explicit `WorkspacePaths`/run context through composition, editing, discovery, exports and workers. Only entry points choose a default workspace. Relative dataset references resolve against the selected workspace, independently of CWD.

### G8 — Compatibility namespaces obscure ownership (P3)

Evidence: [import_aliases.py](../../src/app/core/import_aliases.py) maps old user, plotting, frontend-plugin and tuner paths to bundled implementations. [src/__init__.py](../../src/__init__.py) installs finders at import and its docstring still calls usr an installed-component location. [setup.py](../../setup.py) and [MANIFEST.in](../../MANIFEST.in) intentionally distribute fixed user-namespace shims. [Hot reload](../../src/app/core/hot_reload.py) retains bounded legacy namespace handling.

Impact: old import paths look like current architecture and are still serialized in targets/checkpoints. This is intentional compatibility debt, not evidence that managed installation still writes implementations into usr.

Required boundary: keep a finite, tested adapter list with retirement criteria. Do not extend it for new providers. Renames must preserve canonical class identity and fresh-process checkpoint loading, then retire adapters at a declared compatibility boundary.

### G9 — Compatibility checks do not yet define the whole integration surface (P1)

Evidence: native packages require API 1/storage 2, lifecycle validates dependency versions, tuners check required methods, and execution checks contract versions. However [provider_object](../../src/app/core/provider_loader.py) loads any selected package-relative object without checking the caller's dependency declaration or a public-export contract. Installed NEUMANN/BlendRL packages use helper file paths inside NSFR. App plugins routinely import arbitrary host modules. [Parameter validation](../../src/app/pipeline/component_validation.py) is opt-in and currently focused on agent/model schemas.

Impact: “compatible” partly means a manifest version, partly method checks, and partly knowledge of private filenames. Installation validation alone cannot establish compatibility of every selected combination or direct helper call.

Required boundary: version public APIs, declare callable exports and feature requirements, validate selected dependency closure and composition, and define actionable failure diagnostics. Preserve existing version checks. Python plugins remain trusted in-process code; these are architecture rules and validation, not a security sandbox.

### G10 — Remote execution snapshots merge application and workspace lifecycles (P1)

Evidence: [snapshot.py](../../src/app/sites/snapshot.py) freezes `src`, `frontend`, `hydra_plugins`, `plot`, and `in/config` from a supplied root. [Site execution](../../src/app/sites/execution.py) and [transfer](../../src/app/sites/transfer.py) separately refer to `in/datasets`. Remote setup installs the snapshot as an application checkout.

Impact: changes to source ownership affect execution, hashes, packaging and continuation together. A workspace separate from its application installation needs an explicit way to supply both sources; the current broad root list does not express a selected provider dependency closure.

Required boundary: an execution bundle records application build, selected provider IDs/versions/digests, user config/source snapshots and input mappings. Retain reproducible continuation and existing transfer/publication policies rather than merely changing the copy list.

### G11 — Distribution policy and tests encode the transitional layout (P2)

Evidence: [pyproject.toml](../../pyproject.toml), [setup.py](../../setup.py), [MANIFEST.in](../../MANIFEST.in), and [.gitignore](../../.gitignore) distinguish bundled/installed/user payloads but contain old path lists and special exceptions. A dataset generator and `.npz` remain tracked under an otherwise ignored input tree. [Workspace defaults tests](../../tests/test_workspace_defaults.py) and [organization tests](../../tests/test_provider_organization.py) assert current `in/` and compatibility paths. The core import-boundary test checks direct `ImportFrom` nodes under `src/app`; dynamic forwarding, name-based policy and frontend coupling evade that check.

Impact: existing checks provide valuable protection but do not enforce the clarified architecture. A source rename can pass a narrow import test while preserving semantic breaches.

Required boundary: validate built wheel/sdist contents, full core/SDK imports, clean external-workspace startup and behavior with substitute providers. Explicitly classify packaging adapters, examples and compatibility exceptions.

## Recommended order

Start with public contracts, configuration provenance, provider-neutral execution/tuner behavior and explicit workspace context. Migrate `in/` only with recovery and config-package compatibility. Consolidate host code into `src/ide` after those boundaries stabilize. Finish by retiring adapters and enforcing the architecture in CI. See the [implementation plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md) for milestones and completion gates.
