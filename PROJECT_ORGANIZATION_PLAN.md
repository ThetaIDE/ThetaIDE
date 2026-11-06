# Project organization and workspace separation plan

Status: deferred architecture proposal. The scoped implementation is documented separately below; the target file trees, `theta` commands, package format, and migration phases in this proposal are not the current product layout or a committed delivery sequence.

Current agreed delivery scope: keep the existing checkout layout, centralize its paths, and share component
development/configuration operations between GUI and terminal. Separate research workspaces, package stores,
and application source renaming are deferred. See [Component development](COMPONENT_DEVELOPMENT.md) for the
implemented workflow; the larger directory trees below remain future design options.

## Recommendation

Separate the Theta application checkout from a research workspace. Make the CLI and desktop app open the same workspace and use the same component, configuration, and execution services.

Keep `in/` for terminal users initially, with a narrower meaning: editable experiment recipes, project configuration, and references to data. Put editable component development projects in visible `components/` and `plugins/` directories. Put downloaded Hub packages in a managed store under `.thetaide/`. Keep results visible under `results/`.

The objective is one understandable workspace containing the user's research, with clear ownership inside it. A single undifferentiated directory for source, installed packages, settings, and caches would hide important distinctions. The application source tree must contain neither workspace installations nor personal research files.

A developer can keep a workspace anywhere, including beside the Theta checkout. Framework contributors can use an ignored `workspaces/` directory inside the checkout, but tests and application startup must never assume that arrangement.

## 1. Baseline findings before the scoped implementation

The following baseline findings motivated the proposal. The scoped implementation has since extracted headless component services and centralized paths for their existing layout; the remaining rows describe deferred architectural work, not a claim that all old coupling is still present.

| Current behavior | Consequence | Change needed |
| --- | --- | --- |
| Installer deploys into `src/usr/`, `in/`, and `plot/`, with optional root-relative assets | User packages and application code share directory trees; ownership and uninstall require distributed tracking | Keep each installed package in its own root and resolve its capabilities from a manifest |
| `run_pipeline.py` builds `sys.path` from its own checkout | CLI execution depends on running from an application source installation | Installable CLI plus an explicit workspace context |
| `compose.py` has a checkout-relative `CONFIG_DIR`; validation also derives workspace paths from that constant | A custom config directory alone does not establish a consistent workspace | Pass one context through composition, validation, launching, and plotting |
| Model discovery scans and imports `src.usr.models.*` | Relocating source folders alone will break runtime discovery | A shared resolver that returns component identities and explicit import roots/entry points |
| Headless services previously imported `frontend.hub` installer/models | This coupling has been removed in the scoped implementation | Implementations now live in `src/app/components/`, with frontend compatibility aliases |
| Local packages are retained separately from editable active installations | There are multiple representations of a component whose authority needs explanation | Local development folders become the active source; exports and retained snapshots are explicitly named artifacts |
| Settings, appearance, queues, caches, run records, and outputs derive paths differently | Workspace switching, backup, and cleanup have inconsistent behavior | Central path policy with a written lifecycle for every directory |
| Root package discovery includes `src`, `plot`, and `hydra_plugins`, but not `frontend` | Development layout and distributable application layout need reconciliation | Verify wheel contents and launchers as part of the source-package migration |

Current implementation anchors: `src/app/workspace.py`, `src/app/components/installer.py`, `src/app/components/development.py`, `src/app/components/configuration.py`, `src/app/services/local_components.py`, `src/app/services/component_exporter.py`, `src/app/services/plugin_scaffolder.py`, `src/app/pipeline/compose.py`, `src/app/core/model_registry.py`, `frontend/settings.py`, `run_pipeline.py`, and `pyproject.toml`.

## 2. Proposed research workspace

```text
my-research/
  theta.toml                       # Workspace marker and portable project settings
  theta.lock                       # Exact selected package versions, hashes, local registrations
  in/
    config/
      experiment/<group>/          # Editable recipes and group defaults
      model/                       # User-authored model profiles, when needed
      agent/                       # User-authored agent profiles, when needed
      env/                         # User-authored environment profiles, when needed
      data/  eval/  plot/           # User-authored profiles, when needed
      site/  workflow/             # Project execution profiles
    datasets/                      # Local input data or reference files; usually ignored
    resources/                     # Project-owned rules and other experiment assets
  components/
    my-model/                      # Complete editable research component project
  plugins/
    my-ide-tool/                   # Complete editable desktop plugin project
  results/
    logs/  plots/  exports/         # Preserve current output conventions initially
    ...                            # Other declared run outputs
  .thetaide/
    packages/<id>/<version>/<hash>/ # Managed Hub package payloads, including IDE plugins
    state/                         # Durable local state: visibility, UI layout, queue, notes
    recovery/                      # Operation journals and migration/recovery backups
    cache/                         # Disposable indexes, downloads, generated config views
```

The config category names shown above retain current selectors, including singular `experiment/`, `model/`, and `agent/`. Do not combine a storage migration with renaming Hydra groups. Create optional directories when needed rather than filling every workspace with empty folders.

`theta.toml` is the portable workspace definition: schema version, display name, enabled component roots, supported external paths, and project execution defaults. Keep UI geometry, timestamps, absolute paths to a person's interpreter, and personal preferences out of it. Keep declarative experiment execution parameters in YAML rather than duplicating their authority in TOML.

`theta.lock` records the selected installation for each component identity, package digests, dependency relationships, and registered local paths relative to the workspace where possible. It is portable and suitable for version control. Download URLs are provenance, never identity by themselves. A local source digest is evidence of a revision, not a claim that its mutable working directory is frozen.

There is one managed store per workspace initially. An optional machine-wide download cache can deduplicate archive bytes later; it must not become a second authoritative installation catalog. Existing user-global preferences at `~/.config/thetaide/settings.toml` can remain in place during this work.

### Terminal use stays direct

Users should be able to open `in/config/experiment/<group>/<experiment>.yaml`, edit ordinary YAML, and run an experiment without starting the GUI. Hub defaults remain visible through inspection commands and the editor, without copying them into the editable input tree.

Proposed commands below are design targets, not commands available today:

```bash
theta init my-research
cd my-research
theta hub install blendrl
theta run benchmarks/cartpole seed=7
theta run benchmarks/cartpole dry_run=true
theta config show benchmarks/cartpole --resolved --sources
theta component copy blendrl my_blendrl
theta component develop ./components/my_blendrl
theta component inspect my_blendrl
theta component export my_blendrl --out results/exports/my_blendrl.zip
theta gui .
```

Use nearest-ancestor workspace discovery from the current directory, with `--workspace <path>` taking precedence. Read the workspace marker; do not infer the workspace from the application's installed location. Preserve Hydra override syntax. Define argument forwarding so CLI options cannot consume experiment overrides accidentally. Batch jobs receive an explicit workspace path and resolved execution specification.

Keep `python run_pipeline.py <group>/<experiment> ...` as a compatibility wrapper during migration, with its current legacy workspace behavior until a workspace is explicitly selected or migrated. Do not quietly change the meaning of existing shell scripts.

Keeping `in/` avoids unnecessary disruption for terminal users. Renaming it to `inputs/` or flattening it is optional future work; the important changes are ownership and resolution, not spelling.

## 3. One editable root per local component or plugin

```text
components/my-model/
  .theta_component.json            # Portable identity, capabilities, config roots, dependencies
  pyproject.toml                  # Python package metadata and optional dependencies
  src/my_model/                   # Unique importable Python package
  config/model/my_model/          # Owned default profiles
  assets/                         # Owned rules, templates, static resources
  tests/
  README.md
  COMPONENT_CONTEXT.md            # Generated development guidance; no authority over runtime
```

Use the same package-root principle for every supported research kind and for GUI plugin projects, with their appropriate contracts. A package can provide multiple components and profiles. An IDE plugin has a distinct GUI capability/entry point and lifecycle; it does not need a separate installation-storage system. Built-in plugins ship with Theta's desktop package.

Local development registration points directly to this root. Editing its source or default profiles affects the next fresh run. Do not mirror that source into `src/usr/` or synchronize two writable trees. Export produces a snapshot of the folder's saved contents; a retained recovery snapshot is not another active development source. Arbitrary external Git repositories can be explicitly registered rather than copied into the workspace.

Deep Copy collects owned source, profiles, schemas, and assets into a new folder, assigns a distinct ID and Python namespace, and records parent ID/version/digest. Shared dependencies remain references. Extend current model-specific copying deliberately by kind; until a kind is supported, reject unsupported identity transformations with actionable locations rather than applying broad substitutions.

An ordinary YAML copy is a new experiment/profile referring to shared component code. A component copy creates an independent implementation. Preserve this distinction in the CLI, context documents, and UI.

Use component-relative resource references for packaged assets and workspace-relative references for project assets. Define both forms in the resolver. Explicitly registered external datasets are allowed; a package must not acquire arbitrary filesystem write destinations through its manifest. Avoid process-wide `chdir()` as a resource-resolution mechanism.

## 4. Hub packages, configuration, identity, and protection

### Package boundaries

Hub installation downloads, verifies, validates, and stages an entire package before committing its managed-store entry. Published payloads remain read-only; installation state, generated context, caches, bytecode, and plugin preferences live outside the payload. Packages never scatter writable files into application source or the user's input folders.

Version the manifest/package format. Separate publisher metadata from installation metadata: release manifests describe capabilities and package-relative files; a local ledger records install paths, provenance, ownership, and activation. New packages cannot specify arbitrary workspace `target_path` destinations. Adapt legacy root-relative archives in isolated staging rather than copying their payloads directly over workspace files.

Select one installation per component ID within a workspace. Several versions may be stored, but the lock selects the active one. IDs and aliases must be validated for collisions; duplicate providers are errors with their source locations, never winners chosen by scan order. A local fork gets a new identity and never silently shadows its Hub parent.

### Configuration resolution

Keep portable selectors such as `model=blendrl/default` independent of physical storage paths. Resolve through a provider catalog shared by the GUI, CLI, training subprocesses, sweeps, plotting, and export.

Configuration comes from three sources:

1. Packaged framework defaults.
2. The explicitly selected Hub or local component provider for each namespaced profile.
3. Workspace profiles and explicit experiment/CLI overrides.

Workspace overrides must be intentional and inspectable. Record override relationships or explicit references when a workspace profile replaces a provider profile; reject accidental same-name collisions. Do not silently merge files because their basenames match. Preserve current Hydra defaults ordering and override behavior with composition fixtures before replacing the config search implementation.

Implement provider-backed Hydra search paths where practical. If existing Hydra integration requires a combined tree, build a disposable config view under `.thetaide/cache/`, keyed by provider digests and local input changes. That view is generated, carries file-level provenance, and is never opened as an editable source. It must not introduce another copy users are expected to maintain.

`theta config show --sources` should identify the contributing file and package for each resolved value where possible. The GUI opens the same source files and represents Hub files as read-only. Existing shared parameter-form conventions still apply.

### Lifecycle and reproducibility

| Action | Required behavior |
| --- | --- |
| Install Hub package | Verify archive digest, install under its version/hash root, update ledger and lock transactionally |
| Update | Stage a new version alongside the old one, validate dependencies, then change the selection; preserve rollback |
| Copy for development | Create and register a complete editable project with new identity; retain the original |
| Hide | Update user visibility state only; existing experiments still resolve |
| Uninstall | Remove the active installation/registration without deleting local development source; diagnose affected references |
| Delete local project | Explicit separate operation with affected paths/references; use Trash where supported |
| Clear cache | Remove only disposable data; never notes, queues, local source, results, or unique recovery data |

Each run records resolved config, package digests, framework version, dependency/environment details, and local-source revision/digest/dirty status. Support a source snapshot for reproducible local runs; at minimum, clearly mark runs using mutable local working trees. Existing running processes retain their captured resolution when installs or selections change. Use operation locks and package leases so uninstall/update does not remove files still needed by active runs or loaded GUI plugins; defer deletion/unload where needed.

File modes alone cannot prevent the account that owns a package from changing permissions or replacing files. Baseline mode uses read-only payloads plus digest validation against locked provenance before loading; treat this as accidental-edit protection and tamper detection. If protection from that same account is required, define a separate hardened deployment with an OS-protected store owned by a different privileged identity and a narrowly scoped installer service. That is a separate security deliverable, not a guarantee supplied by this directory layout. In-process third-party plugins still execute with their process privileges; package organization is not a plugin sandbox.

## 5. Proposed application source layout

Use a standard `src` packaging layout with one actual application namespace. `theta` below is the proposed import namespace; check compatibility and existing distribution/import collisions before committing to the name. Keep the distribution name and user-facing product identity stable.

```text
Theta-IDE/
  pyproject.toml
  src/theta/
    sdk/                          # Supported extension contracts and registration APIs
    workspace/                    # WorkspaceContext, path policy, settings scopes
    components/                   # Headless catalog, resolution, package lifecycle, manifests
    hub/                          # Headless registry client, downloads, feed/cache handling
    config/                       # Composition, schemas, config providers and provenance
    runtime/                      # Execution plans, training, sweeps, telemetry, site adapters
    plotting/                     # Plotting framework and genuine built-in plot support
    services/                     # Run catalog, exports, datasets, jobs; split by responsibility
    cli/                          # Argument parsing and terminal presentation
    desktop/
      app.py                      # Application composition and window wiring
      features/                   # Experiments, components, results, editor, settings, etc.
      widgets/                    # Shared controls and parameter form primitives
      plugins/                    # GUI plugin host and bundled feature plugins
      resources/                  # Icons and packaged desktop assets
    resources/config/             # Shipped framework defaults, independent of workspace
    compat/                       # Temporary import/path bridges with removal criteria
  tests/
    unit/  integration/  gui/  migration/
    fixtures/                     # Tiny packages/workspaces; no installed research suite required
  examples/                       # Small documented workspaces/component templates
  docs/
  scripts/                        # Maintainer tooling, not public runtime APIs
  workspaces/                     # Optional ignored developer research workspaces
  run_pipeline.py                 # Temporary compatibility launcher
```

Make the boundaries functional before moving files into this tree:

- `sdk` exposes intentional component contracts and registration APIs. Components import these rather than application internals. Move framework registries currently under `src/usr/`, including the agent registry, into application-owned modules.
- Workspace, config, component management, and runtime services have no Qt dependency and never import desktop modules. Keep import-time initialization light so CLI inspection does not load GUI or ML backends unnecessarily.
- The desktop and CLI call the same use cases. Neither adapter duplicates copying, config composition, ownership checks, or package lifecycle rules.
- Runtime imports a selected component through an explicit resolved provider. Discovery inspects manifests before executing component code; stop recursively importing arbitrary directories just to populate lists.
- Migrate existing entry-point groups through explicit capability adapters. Test real loading for each advertised group rather than assuming declarations imply complete runtime support.
- Centralize legacy kind aliases at the component boundary; do not rename stored identities as an incidental refactor.
- Keep built-in model-builder infrastructure, such as the sequential-spec interpreter, with application-owned runtime support instead of installed user-model directories.
- Split large UI modules by feature/use case after their services are extracted. For example, separate composition/schema logic from `config_viewer.py`, and window wiring from feature behavior in `frontend/app.py`. Keep one shared implementation of ordinary parameter form controls.

Avoid a single large move/rename commit. Preserve temporary forwarding imports such as `src.app.core.*` where published components need them, with tests and a documented compatibility window. Test class identity, registries, serialization, and multiprocessing: loading the same implementation under two module names must not produce duplicate registrations or incompatible types.

## 6. Shared workspace and path services

Introduce a `WorkspaceContext` carrying the workspace root, input/config roots, local development registrations, managed store, state/cache/recovery roots, output paths, and effective settings. Application resources have a separate immutable location supplied through package-resource APIs.

All services receive this context or the relevant narrower interface. Remove assumptions based on `__file__.parents`, `PROJECT_ROOT`, process current directory, and `.thetaide/runs` directory names. Resolve paths once at the boundary; include the context in subprocess and Slurm execution specifications.

Assign settings scopes explicitly:

| Scope | Contents | Source control / backup |
| --- | --- | --- |
| Application distribution | Framework code, built-in plugins, schemas/defaults | Released application |
| Workspace definition | Project settings, local registrations, selected package lock | Commit and back up |
| Research source | `in/config/`, project resources, `components/`, `plugins/` | Commit and back up; allow independent component repos |
| Input data | Datasets or external data references | Usually ignored; user-defined data backup policy |
| Results | Logs, metrics, checkpoints, plots, exported packages | Usually ignored; intentional research retention policy |
| Durable personal state | UI settings, notes, visibility, run history/index, queued jobs | Ignore in Git; back up unique content |
| Managed packages | Verified installed versions, reproducible from lock when available | Do not require Git; include unavailable/private packages in offline backups |
| Cache | Download/index cache and generated views | Ignore; safe to regenerate |
| Recovery | Interrupted-operation journals and unique pre-migration backups | Ignore; retain until operation/migration is verified |

Provide an inspectable path report and a workspace backup/export policy. A backup claiming to be complete must account for external development roots/data, unique state, and offline packages; report omitted external content explicitly.

## 7. Migration and delivery sequence

### Phase 0 — Characterize and settle contracts

Deliver an inventory of path consumers, config groups, discovery mechanisms, package layouts, settings stores, import contracts, and subprocess launchers. Capture representative legacy workspaces as fixtures. Identify edited Hub files, shared assets, nested component manifests, local-only source, and custom target paths before migration.

Acceptance: a written ownership map and baseline tests demonstrate current GUI/CLI composition parity, package install/update/uninstall, multi-profile copying, and CLI/Slurm path behavior. Reconcile docs that currently promise direct editing of Hub originals with the copy-first policy in this plan.

### Phase 1 — Introduce workspace context without moving user files

Add versioned workspace definitions, explicit workspace selection, a centralized path service, and adapters for the existing layout. Thread context through composition/validation, exports, runtime discovery, queues, plotting, and launchers. Move headless Hub/component operations out of `frontend` behind compatibility imports. Add installed CLI entry points while preserving the old launcher.

Acceptance: launch two independent workspaces from arbitrary directories, including paths with spaces; GUI and CLI resolve identical configs and components. A temporary read-only application checkout supports both without receiving user writes. Creating a wheel and using it outside the checkout works.

### Phase 2 — Implement provider-based loading and managed packages

Add a single installation ledger/resolver, versioned package format, isolated staging, hash-selected package roots, config providers, resource references, and dependency/identity conflict diagnostics. Load packaged components directly from their roots. Redirect bytecode and generated files outside protected payloads. Support old releases through a legacy package adapter.

Acceptance: install, run, update, and uninstall each supported package kind without writing into application source or `in/config/`. Cover multi-component suites, shared dependencies, interrupted installs, failed updates, active runs, GUI-plugin restart/unload behavior, and offline reopens.

### Phase 3 — Make local development folders authoritative

Scaffold self-contained projects and register their source roots directly. Extend supported component copying, folder editor routing, context generation, schemas, and exporters. Keep package defaults with their component and experiment overrides in `in/`. All CLI and GUI development actions use the same services.

Acceptance: install A, copy to B, edit/test B with terminal tools, run both independently, restart the IDE, export B, and run its exported package in a clean workspace. Editing B never changes A. Unsupported copy kinds fail clearly. Uninstall preserves B's development folder; Hide preserves runtime references.

### Phase 4 — Explicitly migrate existing workspaces

Add a dry-run migration command that outputs a concrete source-to-destination map, identities, config-provider mappings, conflicts, data volume, and backup locations. Running migration requires the user to have chosen it; application startup must not silently reorganize their research.

Use this sequence:

1. Stop or refuse migration while affected experiments/plugins are active; obtain a workspace migration lock.
2. Inventory and hash the old source/config/assets/state. Detect symlinks and overlapping ownership. Refuse ambiguous ownership pending a resolution.
3. Stage local development projects and protected package roots on the destination filesystem. Copy first; never assume cross-filesystem renames are atomic.
4. Preserve modified Hub installations as explicitly identified local development projects. Show new identities and proposed recipe reference changes before applying them; do not relabel edits as the original trusted Hub release.
5. Translate legacy root-relative asset and config references through an explicit mapping. Preserve multi-profile relationships, nested dependencies, schemas, aliases, custom resources, and recorded parent provenance. Support old import layouts through adapters until compatible releases exist.
6. Compare recipe resolution, selected providers, resource paths, and representative fresh-process executions against the baseline. Preserve old run records and original configs alongside any translated versions.
7. Commit the new workspace definition/lock and migrated state only after validation. Journal every operation. Retain the legacy tree and recovery manifest for rollback; offer cleanup as a separate action after verification.

Acceptance: interrupted migrations recover; rerunning migration is idempotent; rollback restores the old layout. Modified Hub source, private components, edited YAMLs, notes, queues, and historical results survive. An unavailable Hub/network does not strand locally available components.

### Phase 5 — Finish application package and feature organization

Move implementation modules into the standard source package by subsystem. Bundle resources through package metadata, add compatibility imports, restructure tests, and update CI coverage/lint/type paths. Extract feature modules from large desktop files without changing UX behavior. Remove app-relative runtime paths once their tests prove replacement.

Acceptance: wheel install includes desktop resources, built-in plugins, framework defaults, and CLI entry points. CLI works without Qt installed. GUI optional dependencies are sufficient for the desktop. Tests use disposable workspaces; a clean checkout needs no private components or researcher data.

### Phase 6 — Document and retire compatibility deliberately

Update README, installation/contribution guides, component SDK/authoring docs, terminal workflows, built-in help, and Hub release tooling. Publish compatible Hub package revisions only through `THETA_HUB_RELEASE_WORKFLOW.md` when publication is separately authorized. List supported legacy imports, archives, config selectors, and launchers with explicit retirement criteria.

Acceptance: both a terminal researcher and a component author can follow the docs from a clean environment. Only remove a bridge after supported packages and migration tests no longer require it. Do not make users republish every component just to adopt the initial path/context abstraction.

## 8. Cross-cutting verification and decisions

The critical integration matrix covers GUI and CLI; legacy and new layout; Hub and local sources; all supported kinds; local and Slurm execution; custom/external paths; online/offline use; and multiple workspaces in the same desktop session. Include Linux, macOS, and Windows path/permissions coverage. Test Hydra composition in the actual subprocess path, including sweeps, instead of testing only a parent-process search path.

Prioritize a small end-to-end vertical slice before broad module moves: explicit workspace context, one managed Hub model, one editable fork, configuration composition, a fresh-process run, export, update, and uninstall. Then expand package kinds and perform source-tree cleanup.

Recommended decisions for the first implementation:

- Keep `in/config/` and current output conventions during migration.
- Default to workspace-local managed packages and visible local development roots.
- Treat local development roots as authoritative; keep no second writable runtime copy.
- Preserve existing stable component IDs/selectors; reject ambiguous providers.
- Move lifecycle services below the UI before moving the desktop package.
- Keep user-wide library sharing, per-component dependency environments, sandboxing, and privileged storage as separate follow-up designs with explicit guarantees.

If independent research workspaces are revisited, use the vertical slice in Phases 2–3 to evaluate that design before committing to broad moves. The current delivery stops at shared paths and headless component development on the existing layout; it does not promise an independent-workspace milestone.
