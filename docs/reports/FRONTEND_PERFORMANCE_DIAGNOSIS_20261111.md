# Frontend performance diagnosis — 2026-11-11

## Fix handoff

This report combines the initial provider-discovery diagnosis with the hotkey,
settings, and startup follow-up. The scope is frontend responsiveness and
startup architecture, not an isolated hotkey tweak. No production fixes had
been applied when the diagnosis was recorded; implementation progress follows below. The implementation should address both the cost of rebuilding
forms and the unnecessary events that request those rebuilds.

### Implementation progress — stage 1 (2026-11-11)

Completed the first settings-cascade improvement:

- Config viewers compare the four form display preferences before rebuilding.
  Unrelated settings leave existing editors and per-tab display choices intact.
  Relevant display changes still refresh forms, including external file edits.
- `SettingsManager.set_many` writes related preferences atomically and emits
  once after persistence succeeds. Equal effective values skip writes/signals;
  failed writes do not publish in-memory changes. The existing no-argument
  `changed` signal remains compatible with other subscribers.
- Pane shortcut edits/reset and plugin enabled/disabled state use batched writes.
  The first pane edit migrates the legacy list into explicit bindings; once
  present, that mapping is authoritative for runtime navigation. Rebinding or
  clearing cannot reactivate an old digit through the legacy fallback. Digit
  slots are mirrored in the same commit for older clients. Conflicts keep the
  existing first-matching-binding behavior.
- Removed Window's duplicate SettingsView synchronization, consolidated conflict
  validation within synchronization, and skipped unchanged menu conflict styles.

Verification: 14 focused checks passed across `test_settings_invalidation.py`,
`TestHotkeySettings`, `TestKeyParsing`, and `TestShellHelpers`. Coverage includes
multiple viewers, no-op writes, failed persistence, external atomic replacement,
legacy migration, rebinding, clearing, and runtime letter/conflict resolution.
A separate offscreen diagnostic loaded the actual THEMES synthetic experiment:
one hotkey edit took **1.28 ms**, emitted **one notification**, and performed
**zero form rebuilds**. This measures that edit path, not overall startup or form
construction. The earlier diagnosis reproduced two rebuilds per binding edit.

Full-window terminal-settings tests encountered the same offscreen Qt WebEngine
abort described below; desktop startup and full-window interaction remain
unverified. Stage 1 included no provider discovery or startup restructuring.

### Implementation progress — stage 2 (2026-11-11)

Completed discovery reuse during form construction:

- `provider_discovery_snapshot` gives one synchronous form operation a shared
  manifest inventory and catalog freshness check. `load_file`, `_render_boxes`,
  and local sweep-box refreshes use it; nested operations share the snapshot.
  Equal `WorkspacePaths` objects reuse results, keyed by workspace, state,
  application, and extra provider roots.
- The snapshot ends on return or exception. The next operation rescans, retaining
  external edits, atomic replacements, additions, and removals without a timeout
  or a new persistent filesystem cache. Unscoped callers retain freshness checks
  on each lookup. Explicit provider lifecycle invalidation clears an active
  snapshot as well as the existing catalog/import caches.
- Catalog reconstruction consumes the manifests already discovered for its
  freshness check, avoiding a second traversal. The stamp also tracks
  `plugin.json`, whose metadata is merged into plugin capabilities.
- Loader role filtering skips metadata fallback when a profile already declares
  its role. Selector construction still loads runtime classes; metadata-only
  selectors remain a separate next stage.

Verification: **134 focused checks passed** across
`test_provider_discovery_snapshot.py`, `test_provider_organization.py`,
`test_native_contract.py`, `test_parameter_forms.py`, and
`test_settings_invalidation.py`. Coverage includes one traversal per provider
root during actual form loads/rebuilds, preserved dirty values, external/atomic
manifest edits, nested provider additions/removals, extra roots, distinct
workspaces/state/application roots, plugin metadata, inventory/override changes,
explicit lifecycle invalidation, and exception cleanup.

An offscreen diagnostic loaded the actual THEMES synthetic experiment, warmed
imports, and compared complete rebuilds in the same process. With the operation
scope active, rebuilds took **0.108 s** and **0.148 s**, each with **13 root walks**
(one per existing provider root). Bypassing only the rebuild's scope took
**3.185 s**, with **2,769 root walks** across those same 13 roots (213 discovery
passes). These are warm form-construction measurements, not desktop startup
timings. The cold load still imported training dependencies and encountered the
restricted-environment Matplotlib font-cache setup described in the diagnosis.

### Implementation progress — stage 3 (2026-11-11)

Completed metadata-only selector population:

- Data-module, evaluator, and plotter choices no longer resolve runtime classes,
  inspect their source files, or invoke class compatibility hooks. Choices use
  catalog capabilities, aliases, legacy framework built-in names, and YAML
  profiles from workspace/package/framework config roots. Workspace profile
  defaults take precedence over packaged defaults.
- Compatibility comes from profile metadata, then capability metadata, then
  package metadata. Undeclared compatibility remains unrestricted. Training and
  symbolic-state roles stay separate, state-provider aliases are included, and
  profiles can inherit their provider's role. Existing explicit selections
  remain visible even when unavailable or incompatible; executable loading and
  runtime validation are retained in execution paths.
- Choice lists, capability metadata, and parsed profile YAML share the existing
  operation-scoped discovery lifetime. Cache keys include workspace and
  paradigm; returned copies isolate caller edits. Lifecycle invalidation clears
  these values too. The next build reads external YAML/manifest changes again.

Verification: **172 checks passed**: 110 across selector metadata, shared
parameter forms, discovery snapshots, settings invalidation, and the applicable
component-override checks, plus 62 loader-binding, provider-organization, and
native-contract checks. A fresh-process regression guard renders the actual
THEMES synthetic form and enumerates all four selectors while rejecting any
attempt to import Torch, Lightning, Matplotlib, SB3, or Optuna; no such imports
were attempted. Additional coverage exercises aliases, capability/package/profile
compatibility, loader roles, profile defaults, duplicate-owner overrides,
per-build read reuse, copy isolation, YAML edits, and lifecycle invalidation.

Three broader `test_experiment_component_overrides.py` checks fail in existing
runtime registry paths: `test_registered_evaluator_receives_recipe_parameters`,
`test_evaluation_hook_honors_frequency`, and
`test_plot_manager_preserves_parameter_override_precedence`. The same three
failures were reproduced in a separate process using the original registry
sources from `HEAD` (loaded in memory without changing the working tree).
The focused pass above excludes those three failures.

An offscreen fresh-process diagnostic measured **0.275 s** for importing the
viewer and creating its widget, **0.948 s** for the first THEMES form load, and
**0.082–0.142 s** for three subsequent complete rebuilds. No training/plotting
stack modules were loaded. These timings measure the config widget in this
checkout, not full desktop startup, and are not a controlled cold-time ratio
against the earlier font-cache-affected profile.

Next stages retain lazy startup, scoped subscriber updates, deferred hidden-form
refresh, runner IPC, and telemetry/persistence work.

### Priority and source map

| Priority | Problem | Main entry points |
| --- | --- | --- |
| P0 | Repeated recursive discovery inside supposedly cached catalog reads | `src/app/core/provider_catalog.py::get_catalog`, `src/app/workspace.py::manifests` |
| P0 | Unrelated settings changes rebuild every subscribed config form | `frontend/config_viewer.py::_on_settings_changed`, `frontend/settings.py::SettingsManager.set` |
| P1 | Selector construction loads executable providers and training dependencies | `frontend/config_viewer.py::_component_choices`, `src/app/core/datamodule_registry.py::get_datamodule_class` |
| P1 | Startup eagerly builds hidden UI and restores all config forms before showing the window | `frontend/app.py::Window.__init__`, `make_center`, `restore_session`, `frontend/settings_view.py::build_pages` |
| P1 | Runner IPC blocks the GUI while waiting for replies | `frontend/job_manager.py::_rpc`, `_refresh`, `__init__` |
| P2 | Duplicate settings synchronization, writes, and unconditional restyling | `frontend/app.py::_on_settings_changed`, `frontend/settings_view.py::sync_from_settings`, `_validate_hotkey_conflicts` |
| P2 | History/telemetry work grows with accumulated runs and trials | `frontend/model.py::Store`, `frontend/monitor/panel.py::refresh_live`, `frontend/live_run.py::LiveRunCache` |

## Confirmed bottleneck

Experiment form construction repeatedly discovers providers on the Qt UI thread.
The provider catalog caches parsed metadata, but every `get_catalog()` call first
calls `WorkspacePaths.manifests()`, recursively walks component roots, reads
manifests, and computes a freshness stamp. The cache therefore does not avoid
the expensive discovery work.

Measured against this working tree with `venv/bin/python`, Qt's offscreen
platform, and the actual
`src/usr/packages/themes-dev/configs/experiment/AL/themes_synthetic.yaml` form:

| Operation | Observed elapsed time |
| --- | --- |
| One warm manifest discovery | 14–17 ms |
| One warm catalog lookup | About 14–15 ms |
| Warm ConfigTree construction | About 49–50 ms |
| Warm config fingerprint | About 17–18 ms |
| Warm complete experiment form rebuild, without cProfile | 3.2–4.7 s |
| Same rebuild, with discovery results reused in memory | 0.19–0.22 s |

The last measurement used a temporary `unittest.mock.patch` of
`WorkspacePaths.manifests`, caching its return value per paths object in the
diagnostic process. No application source was changed. Catalog stamp checks and
widget construction still ran. This is a causality experiment, not a safe final
cache implementation: production invalidation must still detect additions,
removals, edits, installs, and workspace switches.

A warm rebuild profiled with cProfile took 6.53 s (profiling adds overhead):

- 217 `get_catalog()` calls: 6.29 s cumulative.
- 217 `manifests()` calls: 6.11 s cumulative.
- Four `_component_choices()` calls: 5.04 s cumulative.
- 102 `get_datamodule_class()` calls and 75 `loader_metadata()` calls.
- About 80,000 `scandir` calls and 127,000 `stat` calls.

These times overlap; they must not be added together. Approximately 96% of
profiled rebuild time was beneath catalog lookup. The 3.2–4.7 s unprofiled
measurements are the more useful indication of user-visible blocking.

## Why the cost multiplies

`frontend/config_viewer.py::_render_boxes` rebuilds the form synchronously.
`_component_choices` enumerates runtime classes for available data modules,
evaluators, and plotters. Each class resolution calls the provider catalog.
Loader bindings, metadata, config-root resolution, and validation add further
catalog lookups. The role filter also uses
`values.get("role", loader_metadata(...))`: Python evaluates that default even
when `role` is already present.

On a fresh process, populating dropdowns imports executable provider code and
the training stack, including Lightning and its transitive dependencies. The
first profiled form load took 22–23 s; about 17.6 s in the first profile was
under imports. That absolute cold time includes a restricted-environment
Matplotlib font-cache build, so it should not be presented as a normal desktop
startup benchmark. The repeated multi-second warm stall is independent evidence.

The old dropdown class-resolution loop predates the recent changes. Catalog
freshness discovery was introduced in `6b766b32` on November 8; loader-role
filtering and bindings were expanded in `5728e136` on November 10. Their
interaction is a supported explanation for a recent regression, although no
historical checkout was benchmarked to isolate a single introducing commit.

## Hotkey binding exposes broad settings invalidation

Confirmed call chain for changing one pane binding:

1. `SettingsView._on_pane_hotkey_changed` calls `SettingsManager.set` once for
   `hotkeys.bindings`, then again for the legacy `hotkeys.panes` mapping.
2. Each `set` reads/parses and writes the settings file and synchronously emits
   the unqualified `changed` signal. It has no equality guard, so writing an
   unchanged value also emits.
3. Each config viewer constructed with that settings manager subscribes to
   `changed`. `_on_settings_changed` calls `_render_boxes()` unconditionally,
   including when the viewer's display mode has not changed and the edited
   setting has no relevance to the form.
4. Hidden open experiment tabs remain subscribed. One binding edit therefore
   requests two full rebuilds per subscribed viewer. This multiplies the
   catalog bottleneck above by the number of open experiment tabs.

A diagnostic using the real settings handler, a temporary settings file, and
one loaded THEMES config viewer reproduced two signals and two full rebuilds
for a single binding edit. An unrelated notification preference also rebuilt
that viewer. This is not merely slow key recording: unrelated UI state is
coupled through a global invalidation signal.

Additional amplification confirmed in the code:

- `Window._on_settings_changed` explicitly calls
  `settings_view.sync_from_settings`, while `SettingsView` independently
  connects that same method to `SettingsManager.changed`. One emission normally
  synchronizes the settings view twice.
- Each synchronization validates hotkey conflicts separately after pane edits
  and menu edits. The binding handler validates again after both writes.
  Conflict validation reapplies styles to every shortcut editor even when
  their conflict state has not changed.
- `Window._on_settings_changed` reapplies all menu action shortcuts for every
  settings change. The text editor plugin also calls `apply_theme` on every
  general settings change, rebuilding styles/icons and sending web-view
  configuration even for unrelated preferences.
- `PluginManager.save_states` writes enabled and disabled lists with separate
  `set` calls, exposing the same global cascade during plugin lifecycle work.

Fix direction: attach changed key paths or change categories to notifications;
batch related updates into one atomic commit/notification; skip equal writes;
make subscribers act only on relevant changes. Keep file-watcher reloads
compatible by computing a diff or coalescing a single broad reload. Hidden
forms should mark relevant state stale and update when activated rather than
rebuilding immediately. Coalesce structural work and preserve edits/focus.
Removing only the duplicate hotkey write would leave the larger problem.

There is also a related correctness issue worth covering in that refactor:
moving a pane from a digit binding to a letter leaves its old entry in
`hotkeys.panes`, because the old entry is only cleared when the new binding is
empty. Runtime navigation falls back to that list, so the old digit can remain
active despite the editor showing the new binding. Use one authoritative
mapping, with an explicit migration for the legacy representation, and check
rebinding, clearing, and conflicts against actual runtime behavior.

## Startup work to restructure

The following are code-confirmed startup paths, not a completed timing
breakdown of the user's running desktop:

- `Window.__init__` completes `make_center`, plugin initialization, default
  experiment loading, session restoration, theme refresh, and initial run
  handling before `main` calls `window.show`. Expensive work in any of these
  phases delays the first usable window. A splash repaint does not make that
  work asynchronous.
- `make_center` constructs all built-in panes up front, including hidden
  Workflows, Monitor and its views, Results, Plot Viewer, Terminal, and the
  entire Settings view. `SettingsView.build_pages` builds every category,
  including all shortcut rows, before Settings is opened.
- Terminal construction creates a `QWebEngineView` and starts loading its page
  immediately. Enabled plugins initialize before the main window appears; the
  enabled text editor also owns a web view. Defer heavy pane content and plugin
  UI until first activation where their contracts permit it. Retain cheap
  registrations for commands, menus, and services that must exist immediately.
- `init_default_experiment` can build a fallback experiment form that
  `restore_session` immediately closes and replaces. Restore the saved active
  experiment first and use a fallback only when no saved entry is valid.
- `restore_session` calls `open_file` for every valid saved config, building
  each form synchronously, including background tabs. Restore tab identities
  immediately and materialize inactive forms on demand. The inspected saved
  session listed five config tabs; the exact startup cost depends on which
  paths remain valid.
- Each restored form can trigger the provider scans/imports described above.
  Dropdown population can import Lightning/Matplotlib before the explicitly
  lazy Figure Workspace ever opens, defeating that intended import deferral.
- `JobManager.__init__` spawns/connects the runner and then calls `_refresh`,
  which uses the synchronous `_rpc` reply wait. Runner readiness belongs in an
  asynchronous initialization state, not the window constructor.
- `Store.load` reads all saved `record.json` files, including their metric
  histories, synchronously during startup. Load run summaries first and defer
  full metric histories until a view needs them.
- Startup applies an application stylesheet, initializes the theme, calls
  `theme_changed`, then applies initial zoom styling after widgets have been
  constructed. Consolidate initial theme/zoom setup and avoid redundant
  restyling of the whole widget tree. This is a candidate contributor, not an
  isolated proven dominant startup cost.

A full offscreen Window startup attempt encountered a Qt WebEngine abort, so
there is no valid end-to-end startup timing from that attempt. Do not interpret
that harness failure as the cause of the user's normal desktop startup delay.
No further timing work is required before implementing the known fixes.

## Additional scaling and repeated-work issues

- `ConfigViewer._get_resolved_defaults` repeatedly reads and parses root/group
  YAML during a build; `_component_profiles` likewise reparses profile YAML.
  Resolve immutable defaults/profiles once per build or config generation and
  reuse the result, with edits and external changes invalidating that snapshot.
- `config_fingerprint` traverses config/manifests and reads their bytes during
  tab switching; `tab_changed` can compute it for both the window and the
  destination viewer. Share one freshness result per activation rather than
  independent scans. Preserve external-edit detection.
- `ConfigTabManager` polls all open config files every 750 ms even with a
  filesystem watcher. It also reconciles all watched paths each pass. Retain
  atomic-replacement support, but coalesce notifications, reduce idle polling,
  and avoid repeated watch-set/file-content work where nothing changed.
- `MonitorPanel._tick_status` invokes full `refresh_live` each second while
  visible. That path reads study/execution state, loads live data, and rebuilds
  navigation menus. Separate clock/status text from data refresh, reuse
  unchanged snapshots, and avoid rebuilding menus/plots without a data change.
- `LiveRunCache.load` and `load_version` create a new `AgentFeed` and replay a
  whole telemetry file whenever its stamp changes. Growing remote telemetry
  can therefore be replayed in full repeatedly. Use incremental tailing with
  explicit reset/truncation/version-switch handling and bounded display data.
- `Store.save` serializes the full run plus its metric history and rewrites the
  derived CSV. `Window.on_job_metrics` can persist active runs every five
  seconds on the GUI thread. Move persistence off that thread and avoid full
  history rewrites as runs grow; preserve durable final saves and ordering.
- Settings' resource-budget timer runs every two seconds even when its page is
  hidden. Gate optional page work by activation and pause it while hidden.

These are concrete work paths to address or review during the same pass; apart
from the catalog/settings cascade, their relative impact has not been isolated.

## Other findings and limits

- `frontend/job_manager.py::_rpc` waits synchronously, sleeping in 20 ms
  increments for up to 30 s. Launch/cancel/queue actions can freeze the UI while
  waiting for the detached runner. This is a code-confirmed blocking path; a
  slow-runner timeout was not induced against the user's workspace.
- Monitor refresh reads sweep summaries and execution records on the UI
  thread. Existing local examples measured 1–61 ms per `study_summaries` call,
  up to 100 trials. This can contribute periodic work, but was not the dominant
  reproduced stall.
- Config fingerprints and ConfigTree construction were tens of milliseconds,
  not seconds in this checkout. They do not explain the measured form stalls.
- This diagnosis exercises real config widgets headlessly; it does not profile
  the user's running desktop process, GPU painting, terminal, or an active
  training session. Existing uncommitted work was retained.

## Recommended implementation order

1. Stop unrelated settings changes from rebuilding config forms; batch settings
   writes and remove duplicate subscriptions/synchronizations. Cover hotkey
   rebinding and legacy fallback correctness together.
2. Reuse one provider-discovery/catalog snapshot throughout a form build and
   give discovery explicit, correct invalidation rather than recursively
   rescanning on every lookup.
3. Populate selectors from manifest/schema metadata. Load executable training
   providers only when execution actually requires them. Avoid eager metadata
   defaults and reuse choice/loader calculations within a build.
4. Restructure startup around a usable shell and the saved active pane. Defer
   inactive forms, heavy panels, and historical metric loading. Move runner
   waits and large persistence/telemetry reads off the UI thread, with queued
   completion/error delivery and appropriate pending-action state.
5. Reduce duplicate refreshes, styles, watcher/poll work, and full-history
   telemetry replay. Validate behavior with multiple open tabs, saved sessions,
   provider additions/removals, workspace switches, external/atomic file edits,
   unavailable runners, and long-running/remote training. Preserve unsaved
   edits, selection, focus, viewport, and shutdown durability.
