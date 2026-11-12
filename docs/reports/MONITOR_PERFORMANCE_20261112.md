# Monitor responsiveness investigation — 2026-11-12

The monitor performs substantial telemetry and chart preparation on the Qt UI
thread. Repeated work grows with accumulated history, so a long run can make
ordinary updates stall the whole app. This pass fixes incremental stream reads,
metric indexing, chart sampling, and hidden-pane scheduling. It does not establish
end-to-end desktop latency under an active training workload.

## Reproduced costs and changes

Synthetic offscreen measurements in this checkout, using 50,000 metric rows,
12 metrics, one run, and three single-row updates per scenario:

| Synchronous operation | Before (Git HEAD) | After |
| --- | --- | --- |
| Initial chart history | 388 ms | 258 ms |
| Append reporting one metric | 418–457 ms | 1.7–9.0 ms |
| Append reporting all 12 metrics | 415–430 ms | 4.0–4.7 ms ordinarily; 80.5 ms on bucket-width rebuild |
| Refresh unchanged charts | 2.8 ms | 1.3 ms |
| Initial disk stream history | 305 ms | 301 ms |
| Append one disk stream record | 293–327 ms | 0.15–0.29 ms |

These are sequential before/after runs on the same machine, not controlled
production benchmarks. Chart timings cover `ChartsView.show_lives`, including
widget creation/update calls, but exclude actual painting and event-loop latency.
Disk timing covers `LiveRunCache.load`; initial version discovery is included.
Do not add the chart and disk numbers as an estimate of a normal app refresh.

Reproduce from the repository root:

```sh
venv/bin/python scripts/benchmark_monitor_refresh.py --baseline
venv/bin/python scripts/benchmark_monitor_refresh.py
```

The baseline option loads the relevant source files from Git HEAD in memory;
it does not change the checkout. Its meaning changes after committing these fixes.
The script uses temporary synthetic telemetry and does not touch user runs.

### Stream replay

`frontend/live_run.py::LiveRunCache` previously recreated an `AgentFeed` and
`LiveRun` whenever either telemetry file's stamp changed. One appended record
therefore decoded and reprocessed the complete stream.

The cache now retains the feed and live state for a growing append-only stream
on the same device/inode/version. It replays replacements, non-growing changes,
truncations, version changes, and CSV fallback. CSV replay is intentional:
training tools can rewrite earlier rows even when the header stays the same.
Partial lines remain pending in the existing tail reader. A stream becoming
available after CSV fallback resets the source.

The stream contract is append-only. Arbitrary in-place overwrites that also grow
the same inode cannot be distinguished from appends by metadata alone. Atomic
remote sync replacements deliberately replay in full; this optimization primarily
helps appended local/cached streams, not repeated atomic replacement of whole
remote histories. It does not accelerate the first load.

### Chart indexing and sampling

`frontend/monitor/live_view.py::ChartsView` previously rebuilt every metric's
points and sorted full histories whenever any row arrived. It also constructed a
chart dictionary for every historical point before thinning to about 2,000 points.

It now indexes only appended rows, invalidates only affected metric samples, and
sorts only when out-of-order points arrive. Source replacement, truncation,
addition/removal, or a change between step and transitions axes rebuilds the index.
Full-resolution indexed points remain available to existing consumers.

`IncrementalXYThinner` in `frontend/results/data.py` samples tuples before chart
dictionary allocation and reuses completed extrema buckets. Its output matches
full thinning, preserving bucket minima/maxima, stable x order, and the latest
point. A change in bucket width still rebuilds that metric's sample. Large batches
or many late/out-of-order points can therefore still cause a noticeable refresh.
No recorded history is deleted or sampled on disk.

### Hidden monitor

`MonitorPanel.live_changed` previously scheduled updates regardless of pane
visibility. It now schedules only while visible; the timer checks visibility again
before refreshing, and showing the pane schedules a fresh update. LiveRun ingestion
continues, so opening the monitor catches up with the current state.

Explicit calls to `refresh_live` remain synchronous. This is not complete lazy
construction or removal of all work related to monitoring while its pane is hidden.

## Remaining priorities

1. **Move history loading and persistence off the GUI thread.** First stream loads,
   CSV replay, and atomic remote replacements still decode full files synchronously.
   `Window.on_job_metrics` also persists full metric histories periodically.
   Use a worker owning each feed/cache, immutable batches delivered to Qt, and
   generation checks when runs/selections change. Bound outstanding work and
   coalesce stale display snapshots; preserve final-save durability and ordering.
2. **Separate clock ticks from data refresh.** `_tick_status` still calls full
   `refresh_live` once a second while visible. That repeats study/execution reads,
   source selection, menu construction, and chart updates without new data. Give
   status text its own cached input; preserve a separate source-discovery poll for
   remote/external runs and sweep lifecycle changes. Snapshot study summaries once
   per refresh because panel and `Window.live_for_run` can both request them.
3. **Reduce painting and repeated widget updates.** Charts still copy up to about
   2,000 points per series in `Chart.set_series`, and all metric cards are updated,
   including those below the scroll viewport. Reuse unchanged chart inputs, update
   visible cards first, and materialize large metric grids progressively. Preserve
   hover, zoom, theme/color changes, and held y-axis state.
4. **Bound display memory and expensive diagnostic work.** Metric rows, indexed
   points, events and cached runs remain unbounded. Keep durable full telemetry,
   with explicit cache eviction and a multiresolution display index for long runs.
   System history is already bounded. Profile the Visuals, Feed, System and sweep
   views independently with the user's actual diagnostics; the chart benchmark
   makes no claim about those views or training-side monitor overhead.
5. **Address adjacent runner stalls.** The detached-runner frontend's synchronous
   RPC reply waits can block actions independently of monitor rendering. Treat
   those as a separate asynchronous UI task, as detailed in the prior frontend
   performance diagnosis.

## Verification and limitations

- **55 distinct focused checks passed** across cache resets, monitor refresh cost,
  sweep monitoring, synced sweeps, stream routing, Results data and interactive
  charts. Coverage includes incremental record counts, pinned versions, atomic
  replacement, truncation, CSV rewrites, partial lines, axis/source changes,
  out-of-order points, bucket boundaries, and hidden-pane catch-up scheduling.
- The broader widget run passed **83 checks**, with three failures in unchanged
  app paths: `test_last_finished_run_opens_results_only_after_all_jobs_finish`
  (test host lacks `runs`), and the two capacity cases of
  `test_queued_experiments_wait_for_slots_and_start_in_order` (all jobs start).
  These tests exercise `Window`/runner scheduling, outside the changed refresh paths.
- Telemetry checks passed **23/23**, with the two process-sampling checks rerun
  outside the sandbox because macOS denied process enumeration inside it.
- Full-window monitor tests abort during the offscreen terminal's Qt WebEngine
  initialization, matching the prior diagnosis. Twelve full-window cases were
  excluded from the broader widget run. Full desktop interaction, GPU painting,
  concurrent training contention, and real remote-sync throughput remain unverified.
- Existing unrelated working-tree changes were retained. No component release or
  Hub publication was performed.
