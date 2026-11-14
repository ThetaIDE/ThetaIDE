# Desktop startup regression and post-launch crash — 2026-11-14

Branch `implement/layer-boundaries`, after the phase 3 SDK work. Before the
layer-boundary refactor, the [2026-11-12 investigation](STARTUP_PERFORMANCE_20261112.md)
had brought the active experiment's first paint to **5.8 s** with validation
ready at **13.5 s**. On this branch the desktop took about a minute to open and
then aborted shortly after showing the window.

## Symptoms

Normal launch, real workspace, `THETAIDE_DIRECT=1 THETAIDE_STARTUP_TRACE=…`:

| Milestone | Before the fix | After the fix | 2026-11-12 baseline |
| --- | --- | --- | --- |
| Building the workspace (phase) | 16.2 s | 0.7 s | — |
| `window_constructed` | 59.6 s | 4.9 s | — |
| `first_paint` | 66.8 s | **5.3 s** | 5.8 s |
| `validation_ready` | not reached (abort) | **13.3 s** | 13.5 s |

The abort printed:

```
File "frontend/site_sync.py", line 84, in _refresh_availability
    self.menu.close()
RuntimeError: wrapped C/C++ object of type QMenu has been deleted
Fatal Python error: Aborted
```

## Causes

Both came from site-sync availability checks added on this branch, compounded by
an existing per-call cost in configuration resolution.

1. **Crash.** `SiteSyncControl` created its menu as a child of its default
   button. `ExperimentSiteSyncControl` replaces that button with the tree
   toolbar's button and `deleteLater()`s the original, deleting the menu with it.
   A new one-second availability timer then called `self.menu.close()` whenever no
   ready scheduler served the site. The exception escaped a Qt slot, and PyQt
   aborted the process.
2. **Slow startup and a busy GUI thread.** The same timer ran every second on the
   GUI thread in all three sync controls (experiment tree, Results, Monitor).
   Each tick listed every site through `transfer.connections()`, which took
   **1.9 s** on the real workspace. `sites.catalog.all_profiles()` called
   `WorkspacePaths.resolve_config()` once per site selector, and each call
   constructed a new `ConfigResolver`. Each resolver rescans every installed
   manifest through `get_catalog()` and maps every package profile.
   `connections()` then called `load()` once per site, and each `load()` listed
   all profiles again.

A cProfile of one launch attributed **140 s of 157 s** to 3,591 resolver
constructions. These included 3,530 manifest scans (92 s) and 316,000 package
resource lookups (45 s). `_refresh_availability` accounted for 132 s, and the
config tree's site nodes (`_config_relative_path`, 308 calls) for another 9 s.

## Fixes

- The menu belongs to the control (`QMenu(self)`), so replacing the button never
  strands it.
- Sync availability is checked on events instead of polling. The triggers are
  the control being shown, the application regaining focus, the menu opening,
  `set_config`, a change of the monitored run, and the window's Hub
  install/uninstall handler, which refreshes every `SiteSyncControl`. A
  scheduler removed mid-transfer still cancels the transfer, through that
  handler.
- `all_profiles()` uses one resolver per listing. `load()` and `profiles()`
  accept a precomputed listing, so `connections()` resolves once. That brought
  it from 1.9 s to about 0.15 s warm.
- Within a `provider_discovery_snapshot()`, `WorkspacePaths.config_resolver()`
  returns one shared resolver, and `all_profiles` is `discovery_cached`. The
  snapshot's invalidation clears both. `ConfigTreeWidget.populate` and the sync
  controls' availability and menu reads run inside a snapshot, matching the
  config viewer's existing pattern.
- `ConfigTree.refresh_groups()` also invalidates its resolver's cached selector
  index (`ConfigResolver.invalidate_selectors()`). Without this, a group folder
  created after the tree was built stayed invisible to group inheritance.

## Verification

- Plain launch: first paint at 5.3 s, validation at 13.3 s, no tracebacks.
- A 150-second launch with `--startup-diagnostics` (which profiles the main
  thread, so it runs slower) exited cleanly. Its longest event-loop gap was
  295 ms, against 315 ms in the 2026-11-12 baseline.
- Regression tests in `tests/test_experiment_site_sync.py`:
  - The control's menu survives replacement of its button, and no timer polls.
    With the old menu owner, this test reproduces the exact `RuntimeError`.
  - Listing three sites builds one `ConfigResolver`, and a snapshot shares one
    resolver across `resolve_config` calls.

## Follow-up: an unlogged abort after launching a job

Later the same night, the desktop aborted two seconds after starting a
`NewMonitors/onlineRL` job. The training job kept running in the runner, and a
relaunched desktop showed its monitors. The macOS crash report shows the same
mechanism: a `QTimer` slot raised and PyQt's default handler called
`qFatal`. The desktop had been launched from the app bundle, which discards
stderr, so the Python traceback was lost.

Three offscreen reproductions did not trigger it: attaching to the live job, a
fresh launch in a cloned workspace, and a fresh launch with the real run
history. The specific cause is still unknown.

The desktop now installs `frontend/error_log.py` before creating the
application. An uncaught exception in any slot is appended to
`.thetaide/logs/desktop-errors.log`, written to stderr when there is one, and
reported in the console and status bar, and the window keeps running. Without
a terminal, native faults go to `.thetaide/logs/desktop-faults.log`.
`tests/test_desktop_error_log.py` covers both the hook and PyQt's abort
without it.

## Remaining costs

- `get_catalog()` still scans the component roots for manifests on every call
  outside a snapshot (about 15 ms on this workspace). Code that resolves many
  profiles in one synchronous operation should run inside
  `provider_discovery_snapshot()`.
- `ExtensionSettings` reconciles its contributions once a second, but only while
  its settings page is visible (about 15 ms per tick).
- Plugin discovery (about 0.8 s) and session restore (about 0.9 s) are the
  largest remaining construction phases. They match the pre-refactor profile.
