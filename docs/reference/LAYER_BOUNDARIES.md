# Layer ownership contract

This is the active ownership contract for the [implementation plan](../plans/LAYER_BOUNDARY_IMPLEMENTATION_PLAN.md). The on-disk package format is described in [Plugin storage](PLUGIN_STORAGE.md).

| Layer | Responsibility | Dependencies |
| --- | --- | --- |
| Core (currently `src/app` and `frontend`, eventually `src/ide`) | Configuration, discovery, orchestration, desktop, transport, shared contracts | Core and declared third-party libraries; selected extensions through public contracts |
| Packages (`src/plugins`) | Implementations, immutable defaults, assets, declared dependencies | Public core APIs, package-relative code, declared public package exports |
| User (`src/usr`) | Authored packages, editable config overrides, dataset bytes | Public core APIs and explicitly selected compatible providers |

Core must not branch on extension IDs, backend names or institution names. Lightning and the four base paradigms are retained core implementations. Slurm and tuning are optional components; installing them must use the same APIs as any independent implementation. Contributions require generic compatibility, enabled-state and readiness checks. Saved references to absent providers require actionable errors.

## Configuration authority

Package defaults stay inside their package. Editing a default explicitly creates a complete user override; updates and uninstall must not overwrite that override. `python run_pipeline.py component --root WORKSPACE config override model/package/profile` creates a recoverable copy under `src/usr/config`, retaining comments and Hydra headers. Repeating it preserves existing edits. Resolution precedence is a `src/usr/config` file, then a selected provider profile, then framework fallback. Draft previews temporarily precede these sources. Conflicting package profile owners are errors; capability ownership overrides must also select the corresponding profile and schema. Package IDs, Hydra selectors and serialized class identity remain stable across storage changes. Arbitrary partial YAML merging is not supported.

New scaffolds create editable profiles under `src/usr/config`, with declared provider defaults under the authored package's `configs/`. Export reads effective profiles and writes storage-v3 packages. ARC/NCShare site templates are owned by the optional Slurm component; explicit connection edits to managed templates create usr overrides that survive its update or uninstall. A non-local site profile must declare `site.scheduler` ([scheduler contract](SCHEDULER_CONTRACT.md)).

Workspace context is passed explicitly through reusable services. Entry points (the desktop host, validation worker, `run_pipeline.py`, CLIs) may select a process workspace; `configure_workspace` only records that selection in the process environment and never rebinds module constants. Omitted service context resolves through `WorkspacePaths.for_process()`, never a raw CWD guess. App plugins read the host's selection through `PluginContext.workspace_paths()`. The application root supplies code/assets/defaults; the selected workspace supplies research inputs/configuration/state. Neither is inferred from the other's existence. GUI metadata, method/schema selection, composition and worker Hydra discovery use `ConfigResolver`; the small `hydra_plugins/theta_config_source.py` adapter preserves ordinary Hydra defaults semantics. User dataset bytes live in `src/usr/datasets`; deliberate external references remain external.

## Public integration surface and exceptions

Packages import host code only from the versioned [Theta SDK](SDK.md) (`src.app.api`, SDK 2.0); its exports are the canonical objects behind the contracts below. The existing public contracts are [API](API.md), [execution v1](EXECUTION_CONTRACT.md), [data loaders](DATA_LOADER_CONTRACT.md), [monitors](MONITOR_CONTRACT.md), provider capabilities, [extension contributions/readiness v1](EXTENSION_CONTRIBUTIONS.md), [schedulers v1](SCHEDULER_CONTRACT.md), the local tuner protocol, [portable tuner execution v1](TUNER_EXECUTION_CONTRACT.md) and declared package exports (`provider_export`, see [plugin storage](PLUGIN_STORAGE.md#cross-package-exports)). New APIs must be general and tested with independent providers; this list does not make arbitrary `src.app` or `frontend` modules public. `tests/test_sdk_boundary.py` enforces this for the optional components, bundled component packages and bundled app plugins. At consolidation `src.app.api` becomes `src.ide.api`, with forwarding.

| Exception | Reason | Support/retirement policy |
| --- | --- | --- |
| Lightning execution and four base paradigms | Confirmed product scope | Supported core implementation; extraction deferred |
| `hydra_plugins` discovery namespace | Hydra requires third-party discovery outside core namespace | Small adapters only; `theta_optuna_sweeper.py` loads the installed Optuna component's `hydra_sweeper` export, and implementations belong to their provider |
| Root launchers `run_pipeline.py` and `frontend/__main__.py` | Current CLI and desktop entry points | Supported until phase 6 adds installed entry points |
| Direct `src.app.core.model_registry` and `src.app.objectives` imports in installed Hub packages | Written before SDK 1.1 exported model registration and objective results | Same objects as the SDK exports. Removed when the packages are republished for phase 4 (unit 4g) |
| Direct host imports outside `src.app.api` in authored `src/usr` packages (`blendrl-dev`, `themes-dev`) | Authored before the SDK covered monitoring, data-loader and provider-resource helpers | Not enforced by `tests/test_sdk_boundary.py`; authored packages move to the SDK before phase 7 enforcement covers every package |

Package format/API versions and workspace layout versions are separate contracts. The table lists every remaining exception; there are no compatibility adapters or readers for earlier layouts, receipts or import paths ([phase 4 plan](../plans/LAYER_BOUNDARY_PHASE4_PLAN.md)). Source relocation must preserve canonical objects for checkpoints; remote bundles need versioned manifests. Moving the user's workspace must account for every file, including ignored ones, stop on conflicting files, preserve bytes, modes and symlinks, and keep a timestamped backup. No automatic conflict winner, symlink shortcut, archive rewrite or destructive research-workspace migration is allowed.

## Reproducible inventory

Run `python -m src.app.workspace_inventory --workspace PATH --output FILE`. The read-only inventory includes ignored local source/inputs, file hashes/modes, tracked status, symlinks, package metadata, config selectors, intended destinations, ownership state and structured dataset references. Keep populated-workspace inventories private: paths may identify research inputs. Run the same command against a clean external workspace for the separate distribution baseline. Inventory is evidence, not permission to migrate files.
