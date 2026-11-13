# Desktop startup investigation — 2026-11-12

## Correction: normal launcher and real runner history

The earlier copied-state checks were not representative of the user's launch.
They omitted the real runner's historical jobs and treated the first paint as
an editable/usable startup milestone. The user reproduced an approximately
18-second launch followed by another ten-second freeze. Those earlier usability
claims are withdrawn; the prior timings below are historical diagnostics.

The normal `python -m frontend` macOS app-bundle path was then profiled against
the real state. Its runner had **128 started, finished jobs** (76 completed,
37 failed, 15 cancelled). `JobManager._poll` replayed their complete telemetry on
the GUI thread, including reentrant polling during splash repaint. History
restoration reset the feeds and caused a second replay. The profiled process
performed **256 archive scans**, processed about **891,000 metric records**, and
spent **73.9 seconds under telemetry scanning**, with a **41.9-second post-paint
event-loop gap**. Profiling adds substantial overhead; these numbers establish
the cause rather than a normal-launch speed ratio.

The fix:

- Startup never streams completed archives automatically. Results and explicit
  run selection load historical data on demand, including complete metrics for
  jobs that finished while the desktop was closed.
- Runner readiness remains asynchronous, while telemetry waits until after the
  window has been shown. Only jobs observed live by this desktop are streamed,
  including their final tail.
- Desktop stream/CSV catch-up reads at most 64 records or 256 KiB per batch,
  preserves complete-record boundaries and offsets, and yields between batches.
  Final live jobs continue draining before being marked scanned. Log reads and
  per-turn job scheduling are bounded too.
- Live metric processing checks event-producing prefixes before computing
  metric display descriptions, avoiding millions of irrelevant descriptions.
- The first-paint marker is now `active_editor_first_paint`, explicitly distinct
  from responsiveness. Optional diagnostics capture main-thread profiles,
  sampled Python stacks, and 50 ms event-loop heartbeats through the normal
  bundle handoff.

A subsequent **unprofiled normal app-bundle launch with the real 128-job runner**
painted the active experiment at **5.818 seconds in the GUI process**. Including
its **0.769-second bundle handoff**, the outer command-to-paint time was
**6.587 seconds**. Across **294 heartbeat
samples over the following 15 seconds**, the longest event-loop gap was
**314.8 ms**, the window stayed enabled with no modal blocker, and validation
completed at **13.483 seconds**. The ten-second freeze did not recur in this
measurement. This is evidence for this workspace/run, not a universal startup
latency guarantee. Logs are in `/tmp/theta-actual-startup-safe/43340/` (profiled
reproduction) and `/tmp/theta-actual-startup-fixed/92463/` (unprofiled verification).

Verification for this correction: **48 focused checks passed** across startup
services, monitor streams, and telemetry, including a 128-finished-job startup
fixture, bounded stream/CSV reads, partial-line preservation, and draining the
final live batch. OS process probes required execution outside the sandbox.
Broader monitor tests retain older completed-run/queue expectations; the two
lifecycle assertion failures were reproduced with the original runner source.

## Finding

The largest confirmed avoidable startup cost is executable provider loading from
the Run button's missing-component check. Session restoration amplifies it by
activating a fallback experiment and every restored tab before the window is
shown. Background composition also imports training dependencies during this
construction. Faster developer reloads need changes to these paths, rather than
only a quicker launcher or splash screen.

The initial investigation used process-local diagnostic experiments. The first
production implementation is recorded below. Existing working-tree changes,
including the reload-button work, were retained.

The [earlier frontend diagnosis](FRONTEND_PERFORMANCE_DIAGNOSIS_20261111.md)
records the implemented discovery-snapshot and metadata-only selector fixes.
Those fixes do not cover the Run button's separate validation path.

## Implementation progress — first startup pass

- The Run button now uses `find_unavailable_components`, which resolves declared
  provider metadata within one discovery snapshot. Runtime preflight still uses
  `find_missing_components` and proves selected provider imports work. Algorithm
  aliases and variant suffixes share the same metadata resolver as runtime loading.
- Session restoration creates lightweight viewer shells for saved tabs, retaining
  order, identities, and pinned state. Only the final active form reads YAML and
  builds its controls. Inactive forms read current disk contents on selection;
  refreshes, polling, and plugin changes do not materialize them. Existing
  rename/delete handling and dirty-form protection remain in use.
- A fallback experiment opens only when no saved valid config remains. Initial
  pane selection no longer reloads the active form. Composition is suppressed
  during construction and starts through the existing debounce after the first
  window show. Run remains disabled until the matching validation result arrives.
- Startup and successful composition no longer request the unrelated recipe-search
  health check. The explicit Settings connection check remains available.

The same temporary-data harness, with the real runner and real saved two-tab
session, measured **4.60 s** for the implemented constructor versus **11.37 s**
before this pass with a populated Matplotlib cache (about 60% less elapsed time).
The active BlendRL file loaded once, session restoration took **0.93 s**, and no
Torch, Lightning, or Matplotlib modules were imported by constructor return.
The terminal/editor WebEngine exclusions and native-launch limits below still
apply. This is a constructor measurement, not validation-ready time.

Tests cover fresh-process window construction with training imports rejected,
saved-session and missing-session fallback paths, initial pending validation,
successful readiness, stale replies, failure handling, lazy activation, atomic
external replacement, dirty edits, deferred-tab rename/close, metadata aliases,
variant suffixes, discovery reuse, and runtime import failures. The older
selector import guard was updated to the recipe's current `AL/codex/` path.

Verification: **175 passed, 2 skipped, 2 deselected** across startup, config tabs,
validation debounce, native contracts, provider discovery/metadata/organization,
parameter forms, settings invalidation, and applicable suite checks. The two
deselected suite-installation tests use local external archives rejected by the
existing native-contract requirement. A separate broader validation run hit
`test_eval_episodes_offline_rl` and
`test_supervised_paradigm_passes_without_component_plugins`; both failures were
reproduced with the original `HEAD` validation source loaded in memory (their
`MockCfg` lacks `items`). Mypy reports the same **16 errors in 8 files** with the
implementation and with the original validation source supplied via
`--shadow-file`; the outputs match exactly. Ruff is not installed in this venv.
Changed Python files pass syntax checks, and `git diff --check` passes.

## Implementation progress — completed startup follow-up

- Components defers its default form; Settings builds categories on first use
  and samples optional resource information only while its backend page is visible.
  Terminal shells and the text editor renderer initialize on demand. Editor
  commands/services register immediately. Explicit file opens and SSH/upload
  shells still initialize their requested content directly.
- Run summaries load after the first paint on a background worker. New saves
  create atomic summary caches; older records migrate in that worker. Full
  metrics load on selection/plotting, and metadata saves preserve deferred
  histories. External and atomic replacements invalidate cached summaries.
  Late history loading preserves new runs and removes duplicate transient entries.
- Runner construction no longer waits for a snapshot reply. Readiness,
  timeouts, recovery, and replay use a Qt timer. Missing/stale/corrupt snapshots
  preserve last-known jobs; saved jobs are not interrupted before readiness.
  Telemetry read failures do not disconnect an otherwise healthy runner.
- The desktop now uses a persistent private validation interpreter. It starts on
  demand, stays warm, invalidates providers/config caches between requests, and
  returns JSON reports. Obsolete queued compositions skip execution; late results
  cannot enable Run. Results/catalog operations have a separate queue, so they
  can finish during validation. Shutdown terminates validation without stopping
  the detached training runner.
- Optional JSONL startup milestones distinguish imports, construction, first
  paint, editable experiment, validation, history, runner, and renderer readiness.
  See the [measurement and lifecycle guide](../development/STARTUP_PERFORMANCE.md).

A follow-up headless constructor probe measured **3.06 s**, compared with the
first pass's **4.60 s** and original warm-cache **11.37 s**. Those measurements
retain the original headless terminal/editor exclusions and are not native
process-to-window comparisons.

Native macOS QA then used the real main entry point, real terminal and editor,
the saved two-tab session, and copies of all **137 desktop records**. It exercised
both initial summary migration and cached summaries. The final source was checked
again after the terminal reentrancy guard and separate validation queue:

| Final native milestone | Elapsed from process entry |
| --- | ---: |
| Imports complete | 0.57 s |
| Window constructor starts / finishes | 2.08 s / 5.02 s |
| Active experiment first paint | **5.44 s** |
| Runner ready | 3.40 s |
| Run summaries and deferred services ready, initial migration | 7.11 s |
| Matching experiment validation succeeds | 13.81 s |

The preceding cached-summary native check loaded summaries about **0.16 s after
first paint**. Native startup samples with isolated validation ranged from
5.44–5.75 s to the editable experiment. These are local samples with normal OS
and cache variability, not a controlled speedup ratio against the headless
baseline. The earlier two-second engineering target has not been demonstrated.

The final check measured first Settings activation at **0.21 s**, terminal
construction at **0.39 s** plus **0.66 s** to its renderer handshake, and editor
opening at **0.34 s** plus **0.07 s** to its handshake. It verified exactly one
initial shell, successful CodeMirror opening, successful isolated validation,
no Torch/Lightning/Matplotlib/SB3/Optuna imports in the UI process, and clean
shutdown. The native trace exposed a reentrant terminal-initialization path;
its guard and regression test prevent duplicate shells and creation after close.

Verification: the combined regression run passed **263 tests**, with four
headless native-WebEngine cases excluded. Follow-up startup/lifecycle checks
passed **26 tests**; the final service-only run passed **19 tests**, including
telemetry-error handling. These runs overlap. Native QA exercised the real web
surfaces separately. Mypy still reports the original **16 errors in 8 files**;
Ruff is unavailable in this venv. Syntax, documentation links, and diff whitespace
checks pass. Existing unrelated working-tree changes were preserved.

## Measurements and limits

Used this checkout's `venv/bin/python`, fresh Python processes, Qt offscreen,
Fusion style, a temporary copy of workspace settings, the real configuration and
provider roots, and copies of the saved desktop `record.json` files. There were
two saved experiment tabs, with the BlendRL sweep active. The real detached runner
was started against temporary job storage; the workspace's live runner/database
was not used. A lightweight widget replaced TerminalPanel and the text editor
plugin was disabled in the temporary settings to avoid headless WebEngine
failures. Documentation remained enabled.

Timings below measure `Window(...)` construction, before `show()`, excluding
frontend import and native launch/painting. They are indicative local probes,
not a controlled native desktop benchmark or guaranteed production speedups.
Background worker timing, filesystem caches, concurrent workspace activity, and
profiling overhead affect the results. Rows contain nested work and must not be
added together.

| Probe | Constructor elapsed | Training imports by constructor return |
| --- | ---: | --- |
| Existing code, initial probe with temporary Matplotlib font-cache build | 22.57 s | Torch, Lightning, Matplotlib |
| Existing code, writable populated Matplotlib cache | 11.37 s | Torch, Matplotlib |
| Replace only `_provider_exists` with catalog resolution in memory | 8.47–9.80 s | Torch, from background composition |
| Metadata lookup plus suppress background backend submissions during construction | 7.48 s | None of Torch, Lightning, Matplotlib, SB3, Optuna |
| Above, plus skip fallback experiment and restore only the saved active experiment | 5.12 s | None of those imports |

The final row models deferred work; it does not implement inactive-tab placeholders
or production validation. It still loads the active form twice and constructs
the hidden Components panel. Suppressing backend submissions is a diagnostic,
not a shippable implementation. Real execution validation must remain available.

Representative existing-code measurements with a populated font cache:

| Work | Elapsed |
| --- | ---: |
| Frontend import (separate fresh processes) | 0.39–0.63 s |
| Plugin discovery/Hub setup interval | 0.74 s |
| Store.load: 137 records, 179.5 MiB of JSON | 0.92 s |
| JobManager constructor, isolated healthy runner | 0.30 s |
| make_center, excluding real terminal/editor WebEngine | 1.27 s |
| ComponentsPanel, including its default form | 0.88 s |
| SettingsView, all pages | 0.11 s |
| Fallback experiment open, subsequently discarded | 2.54 s |
| restore_session | 4.13 s |
| Initial set_zoom | 0.64 s |

A separate cProfile run attributed **16.9 s of a 27.4 s constructor** to eight
`check_missing_components` calls, including provider imports and font-cache
initialization. This establishes the call path; use the unprofiled probes above
for elapsed-time estimates. It also recorded 84 manifest discoveries, totaling
2.9 s under profiling, so operation-scoped discovery has more callers to cover.

Scratch harnesses and logs were written under `/tmp/theta_startup_*` and
`/tmp/theta-startup-*`. The initial investigation did not run GUI integration
tests. The implementation adds the headless window regression described above;
native launch timing is recorded in the completed follow-up above.

## Confirmed call paths

### Run button loads executable training providers

`ConfigTabManager.open_file` changes the current tab, emitting
`current_tab_changed`. `Window.on_config_tab_switched` calls
`update_launch_state` → `check_missing_components` →
`validation.find_missing_components` → `_provider_exists` →
`provider_loader.resolve_provider` → `load_capability`.

An import trace captured the THEMES adapter importing Torch on this path. The
profile captured the BlendRL model importing Captum, which imports Matplotlib.
This work runs synchronously on the GUI thread merely to decide whether a
referenced component exists. A broken provider import is also caught by the
window's broad exception handler, so the cost does not reliably produce a useful
diagnostic at this point.

### Background validation overlaps form restoration

Tab activation calls `request_compose` directly. A second import trace, after
replacing the existence lookup, showed Torch loading from
`LocalBackend._Call.run` → `compose_report` → `validate_composed` →
`validate_experiment_config` → `load_paradigm_definition` → provider resolution.
Another trace captured paradigm auto-discovery importing Lightning.

Running this on a worker prevents a direct GUI-thread import, but Python work
still competes with construction. Multiple restored tabs queue composition on
the single backend worker; superseded callbacks are discarded, but the queued
work still executes. Startup also calls `request_schema`; `config_schema` scans
recipes until one composes and validates successfully, rather than checking only
the saved active experiment.

### Restoration constructs and activates more than the visible form

`Window.__init__` calls `init_default_experiment` before `restore_session`.
Restoration closes that fallback, calls `open_file` for each saved config, and
then selects the saved active config. Each open makes its tab current. Switching
to the restored Experiment pane runs `tab_changed`; its initial fingerprint
check causes the active viewer to reload again. The probes recorded two
`ConfigViewer.load_file` calls for the active BlendRL file, including the
active-only diagnostic.

All of this finishes before `main` calls `window.show()`.

## Recommended implementation order

1. **Make UI availability checks metadata-only.** Add a separate installed-provider
   lookup for editor/Run-button availability, using catalog aliases and resolved
   profiles. Share one discovery snapshot per check. Preserve executable loading
   and error reporting in explicit validation and launch paths; do not globally
   weaken `_provider_exists` in pipeline validation. Add a fresh-process guard
   that opens the actual window/session while rejecting training imports from
   availability checks. The existing widget-only selector guard misses this path.
2. **Restore the active UI once.** Check saved valid entries before creating a
   fallback. Register inactive tab identities without constructing forms; load
   them when selected. Suppress intermediate tab activation/compose work during
   restoration, then validate the final active recipe once. Initialize the active
   viewer's freshness baseline after loading to avoid its immediate second load.
   Preserve external/atomic file changes, dirty edits, tab order, and watchers.
3. **Keep initial validation off the construction path.** Show the active editor
   with a pending validation state; keep Run disabled until its matching result
   arrives. Coalesce obsolete queued compositions and replace the unrelated
   recipe-search health check with validation of the active recipe. Evaluate a
   separate validation process if heavy imports still compete with UI updates.
   Merely using `singleShot(0, expensive_work)` moves the stall after paint; it
   does not make the UI usable during that work.
4. **Defer hidden content and full histories.** Start with the Components default
   form and terminal/editor web views. Keep plugin command/menu/service
   registration immediate and make heavy view construction lazy. Load run
   summaries first and metric histories on selection; the current 179.5 MiB JSON
   load grows with usage. Settings is measurable but smaller here, so defer its
   pages after the larger costs.
5. **Remove runner readiness from the window constructor.** The healthy temporary
   runner cost about 0.3 s, but `_rpc` sleeps on the GUI thread for up to 30 s.
   Use asynchronous readiness with a pending/unavailable state. Test reconnects
   to running jobs and unavailable runners without changing queue ownership.

Add opt-in startup timings for process entry, imports complete, shell painted,
active editor usable, validation ready, and deferred services ready. Compare
fresh-process reloads with the same saved session and warm caches, and measure
first activation of deferred panes so the work is not simply hidden. A reasonable
initial engineering target is an editable active pane within two seconds on this
machine, but the probes do not demonstrate that target yet.

## Developer workflow

`THETAIDE_DIRECT=1 venv/bin/python -m frontend` bypasses the macOS app-bundle
handoff and keeps the process attached to its launching terminal. Ctrl+R there
re-executes Python; it is a convenient full restart, not live Python-widget
replacement. The reload-button changes already present in the working tree use
that same restart path. The launcher itself is not the dominant measured cost.

Prefer making ordinary startup fast first. Consider an opt-in UI preview mode
afterward for isolated panels, themes, and layouts, with explicit fixture state
and no initial training validation. In-place reloading of arbitrary Qt classes,
signals, plugins, and web views adds lifecycle complexity and is a separate
project; the existing developer restart can benefit immediately from the startup
changes above.
