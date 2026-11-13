# Startup performance and lifecycle

The desktop restores the active experiment before showing the window. Saved
background config tabs retain their identities and load forms on selection.
The Components default form, Settings categories, terminal shells, and text editor
renderer initialize when opened. Commands and editor-service registration remain
available immediately; opening a file or an explicit SSH/upload shell initializes
the required view directly.

## Measuring a developer launch

From the checkout root on macOS:

```bash
THETAIDE_DIRECT=1 THETAIDE_STARTUP_TRACE=/tmp/theta-startup.jsonl venv/bin/python -m frontend
```

On Windows/Linux, set `THETAIDE_STARTUP_TRACE` in the shell and run the normal
`python -m frontend` command. Tracing is optional, appends JSON lines, and cannot
prevent startup if its destination is unavailable. Each launch/reload gets a new
`startup_id`; the PID can remain the same when the developer reload uses `exec`.
Elapsed times are milliseconds from that interpreter's frontend entry point.

| Milestone | Meaning |
| --- | --- |
| `process_entry`, `app_import_start`, `imports_complete` | Frontend entry/import boundaries |
| `phase` | Constructor progress and its current message |
| `window_construct_start`, `window_constructed` | Synchronous window construction |
| `window_shown`, `first_paint` | Native show events and the first main-window paint |
| `active_editor_first_paint` | First paint with the restored Experiment pane visible; does not establish responsiveness |
| `runner_ready`, `runner_unavailable` | Detached runner connection transitions |
| `history_ready` | Run summaries loaded, with record/error counts |
| `deferred_services_ready` | Both history and runner readiness are available |
| `validation_requested`, `validation_ready` | Current configuration request/result; result includes `valid` |
| `terminal_renderer_ready`, `editor_renderer_ready` | Deferred web surfaces completed their bridge handshake |

Compare the same workspace, saved tabs, active pane, plugins, and cache state.
Record process-to-first-paint, post-show event-loop latency, and validation-ready
time separately. First paint alone does not establish that the app accepts input. Include first
activation of deferred views in manual QA. A headless constructor time excludes
native application setup, painting, and renderer readiness. Current measurements
and the original diagnosis are in the [startup report](../reports/STARTUP_PERFORMANCE_20261112.md).

## Validation and cancellation

The production window uses `LocalBackend(isolate_validation=True)`. A private
interpreter starts on the first validation request, remains warm, and handles
composition and explicit pipeline checks serially. It receives JSON arguments
and returns JSON reports through files in a private temporary directory. Provider
modules and compose caches invalidate between requests so source edits and
installs/removals remain visible. Training dependencies stay in that interpreter.

Validation has its own work queue. Run catalog/file operations keep their
separate queue and can complete while validation runs. Editing cancels pending
composition work immediately; superseded queued calls skip execution and late
results cannot enable Run. A validation already executing can finish, and the
latest requested configuration follows it. Run waits for matching successful
validation and runner readiness. Worker failure/timeout produces an ordinary
validation error; the next request can start a replacement worker.

Closing the desktop shuts down its validation interpreter and drops late
callbacks. The detached training runner and its jobs continue independently.
`LocalBackend` retains its default in-process mode for headless adapters/tests
that intentionally share runtime state; the desktop opts into isolation.

## Runner connection

`JobManager` construction starts/connects the detached runner and writes a
snapshot request without waiting for a reply. Its Qt timer completes the
handshake asynchronously. `ready`, `readiness_error`, and `readiness_changed`
expose pending, available, and unavailable states. Actions before readiness fail
immediately with a connection message. Reconciliation never marks saved jobs
interrupted while readiness is unknown.

A missing, stale, or invalid snapshot retains the last valid jobs/queue while
reconnecting. Once the handshake succeeds, saved jobs reconcile and telemetry
can replay. Loading history after an early telemetry poll resets feeds only for
those restored records, preserving new jobs created in the current session.
Existing synchronous action RPCs remain separate from this startup handshake.

## Saved run summaries

`record.json` remains authoritative. Each successful save also attempts an
atomic `summary.json` cache containing metadata, latest metric values, progress,
best reward, and a bounded sparkline. Its source stamp includes size, modification
time, change time, and inode. External/atomic record changes invalidate it;
missing or malformed caches rebuild from the record.

Run summaries load after the first paint on a background worker. An older
installation migrates records there on its first launch. Warm launches read the
small summaries. Full metrics materialize on run selection or plotting.
Persistence hydrates a deferred history before saving notes/metadata, and fresh
metric replies clear the old deferred reference before persistence. Missing
backing records cannot be overwritten with an empty deferred metric list.
Summary-cache write failures do not invalidate the authoritative record.

## Verification

`tests/test_startup_services.py` covers summary migration/invalidation and save
safety, coalescing and independent queues, worker isolation/lifetime, runner
readiness/timeout/recovery, and lazy renderer lifecycle. The fresh-process window
guards are in `tests/test_startup_performance.py`; existing tab, provider,
parameter-form, Settings, layout, and shutdown tests cover integration.

Native WebEngine checks require a working desktop. Validate terminal readiness,
exactly one initial shell, editor file opening, clean shutdown, and absence of
training imports in the UI process. Use copied application state for repeatable
manual measurements and leave the live runner/database alone.

## Reproducing a reported startup freeze

Use the user's normal launcher and full runner state. The macOS bundle handoff
passes these diagnostic flags into the GUI process:

```bash
venv/bin/python -m frontend --startup-diagnostics /tmp/theta-startup-profile
```

This writes per-PID main-thread profiles, sampled Python stacks, and event-loop
heartbeat events. For a low-overhead verification after identifying the cause:

```bash
venv/bin/python -m frontend --startup-diagnostics /tmp/theta-startup-latency --startup-latency-only
```

Heartbeat gaps show post-paint stalls; events also record whether the window is
enabled and whether a modal widget is active. Profiles change absolute timings,
so compare responsiveness using the latency-only mode. An empty runner fixture
cannot reproduce startup replay costs: include a snapshot with many completed
jobs in regression coverage. Desktop startup skips archive streaming entirely;
live catch-up reads bounded batches and drains final tails across UI turns.
