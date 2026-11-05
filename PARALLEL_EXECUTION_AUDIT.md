# Parallel execution and compute safety audit

Date: 2026-11-03. Scope: current working tree, including the detached runner and ongoing queue/monitor changes. This is an audit and design proposal; no runtime behavior was changed.

Follow-up implementation (updated): Settings → Training Pipeline now exposes Theta's CPU/RAM share of system capacity, a maximum number of training tasks, learned capacity estimates, and pause/resume actions. Manual per-job memory requests were removed. Cold estimates use engine overhead, model size, batch size, replay storage, and dataset size; sampled process memory peaks and CPU consumption refine subsequent estimates. Unknown tasks start a few seconds apart while usage becomes measurable. Methods, replicates, and fresh-process Optuna trials share an atomic workspace resource pool; pause applies to internal trial/method starts as well as experiment dispatch. Individual trial completions update Optuna and the monitor immediately. Targets permit five percentage points of tolerance; admission separately preserves 5% physical RAM headroom. Crossing a soft target does not cancel running work. Sustained free RAM below 5% pauses the pool and stops the newest experiment. Accelerator tasks remain exclusive, and local distributed-device launches are rejected. GPU VRAM budgets, cross-workspace coordination, and hard OS isolation remain outstanding. Findings below describe the original pre-implementation state.

Follow-up validation: 33 resource-budget and detached-runner tests passed, 2 optional real-training tests skipped. A broader pipeline run passed 71 tests, skipped 2 and deselected the previously failing Ruleset2 fixture test. The budget settings card was rendered and inspected separately. Existing queue-panel tests still pass removed constructor arguments (`on_move`); the full settings-window suite aborts in embedded-terminal construction in this environment. No full-GUI-suite success is claimed.

Adaptive scheduling validation: a broader regression run passed 79 tests with 2 skipped; a subsequent resource/tuning run passed all 44 selected tests. Real trial workers had distinct PIDs, overlapping start/finish times, separate artifacts, and completed Optuna objectives. Tiny SB3 and Lightning tasks also completed through the fresh-worker path and promoted their winning checkpoints. The automatic settings card was rendered and inspected. One previously failing Ruleset2 fixture test remains deselected. The live user's existing sweep was not stopped or migrated; already-running coordinators retain their loaded scheduler until restart.

Rolling-dispatch correction: the managed sweeper now supplies jobs lazily to a continuous dispatcher instead of waiting for a batch to finish. Completion is recorded before a replacement is sampled. Resource availability, pause, failure limits, and time budgets still gate new starts. Portable summaries distinguish pending sampled trials from running workers and include unsampled future/pending counts. The monitor table and queue show a grouped future backlog without inventing parameter values or trial identities. The desktop tuner now labels scheduling as automatic. A real three-trial/two-slot test verifies that trial 3 starts after trial 1 completes and before trial 2 finishes. Sixteen rolling/cancellation/monitor/training checks and twenty-one UI/filter checks passed; SB3 and Lightning worker checks required process-inspection permission outside the macOS sandbox. The pending-trials UI was rendered and inspected separately.

Resource-only concurrency correction: removed the separate maximum-task setting and all count-based admission checks from managed scheduling. Saved `training.max_parallel`/`queue.max_parallel` values are ignored. Settings now expose CPU and RAM budgets only. Trial dispatch fills capacity until resource admission blocks; managed methods and seed runs use the same pool. Their lightweight coordinator-thread bound derives from physical CPU capacity and the estimator's CPU floor, so it does not restrict resource-admissible work or prevent a higher soft target during an execution. Coordinator startup remains staggered for measurement. Legacy CLI launcher/seed settings continue to apply outside managed desktop execution. Tests demonstrate five small tasks fitting despite a saved limit of one, while two large concurrent requests compete atomically for a budget that can hold only one. The rolling-worker test now uses resource estimates to allow two workers, also with a saved limit of one.

## Conclusion

Parallel training is possible today across methods, repeated seeds, and separate queued experiments. Parallel Optuna trials require a parallel Hydra launcher; increasing Theta's trial batch size alone does not provide concurrency. The installed environment has Hydra's serial BasicLauncher and no Joblib launcher.

The current queue is an experiment-count limiter, not a compute scheduler. It cannot guarantee that a user's machine will remain responsive or avoid out-of-memory failures. Do not advertise its width as a machine safety limit.

## What actually runs

| Layer | Current control | Actual behavior |
| --- | --- | --- |
| Desktop queue | `max_parallel`, default 1 | Counts active pipeline parents; the queued dispatch path respects the count. |
| Methods | `run_in_parallel` | Starts one training subprocess per method, without a separate method cap. |
| Repeated seeds | `variability.parallel_runs`, default 1 | Limits concurrent method/seed tasks within one variability execution. Does not multiply by method count: each worker receives one method. |
| Optuna | `tuning.n_jobs` / tuner options | Asks for a batch, then submits it to Hydra. BasicLauncher executes that batch serially. |
| Parallel Hydra launcher | Launcher-specific worker count | A separately installed/configured process launcher could run a trial batch concurrently. Not installed, packaged, or verified here. |
| Training internals | BLAS/PyTorch threads, data workers, vector environments, distributed processes | Additional resource demand, outside the desktop queue's accounting. |
| Slurm | Site/experiment `resources` | Allocations are generated separately; the desktop tracks the submission process rather than the remote job lifetime. |

For homogeneous experiments, approximate active training tasks as:

`experiment width × concurrent methods × min(trial batch size, parallel launcher workers)`

With BasicLauncher the final factor is 1. With variability use `experiment width × min(parallel_runs, planned method/seed tasks)` instead. Variability currently rejects tuning, recovery, workflow launches, and Slurm; these are separate modes, not supported multiplicative combinations. Threads, workers, coordinator overhead, datasets, and GPU contexts add demand beyond these task counts.

Example: 2 experiments × 3 methods × 4 parallel trials would mean up to 24 training tasks, despite the queue showing a width of 2. This is a hypothetical configured parallel launcher; the current default executes trials serially.

## Findings and evidence

### P1 — Nested concurrency bypasses the intended machine budget

`frontend/runner_manager.py:114–125,584` counts pipeline statuses only. `src/app/pipeline/local_runner.py:63–76` creates a thread per method, each driving its own subprocess. `src/app/pipeline/variability.py:177` has its own independent pool. There is no shared reservation for CPU threads, RAM, GPU memory, or disk space.

One queued experiment may therefore overload a machine. A single training task can also exceed memory, even with every concurrency setting at 1.

### P1 — Immediate launches can exceed the queue limit

`frontend/runner_manager.py:225–259` starts `queue=False` launches without `has_capacity()`. The current F5 flow uses `add_to_queue(start=True)` (`frontend/app.py:2057`), which is safer, but the callable runner API still bypasses the limit.

Reproduced with an isolated headless runner, temporary database, and `_start` replaced by a status update: `max_parallel=1`, two immediate launches, `running_count=2`. The queued path correctly started one and retained two waiting entries. No training was started in this reproduction.

Admission must be authoritative in the runner, not dependent on the caller or UI button state.

### P1 — GPU allocation and native CPU threads are unmanaged locally

The local subprocess environment (`src/app/pipeline/runtime.py:65`) inherits the environment and optional site settings. There is no scheduler assigning GPU devices, enforcing GPU exclusivity, or dividing CPU thread budgets. Lightning defaults to one device (`src/app/core/lightning_builder.py:479`), but independently launched tasks can select the same GPU. Multiple one-device jobs do not automatically spread across GPUs.

There is no automatic local cap on OpenMP, OpenBLAS, MKL, PyTorch threads, data-loader workers, or environment workers. Site-specific settings are not a general admission policy.

### P1 — Restart and cancellation do not establish durable ownership of all work

The detached runner's lock is scoped to its IPC directory (`frontend/job_runner.py`). It prevents duplicate owners for that store, not competing runners in other workspaces, CLI launches, other applications, or another OS user.

`JobStore.recover_active_jobs()` detects live recorded PIDs, but `RunnerManager._recover()` subsequently marks remaining active records interrupted instead of supervising them. After an abnormal runner death, a surviving descendant may consume resources outside the new runner's count. This is a code-path risk, not a reproduced runner-crash experiment.

Cancellation attempts to terminate descendants (`src/app/services/processes.py`), but inaccessible enumeration falls back to the known parent. The POSIX fallback assumes a process group whose ID is the PID; `_start()` does not explicitly establish such a group. Cleanup needs real process-tree tests for each supported platform and launcher.

Optuna's exception cleanup handles Python exceptions, including KeyboardInterrupt/SystemExit. Abrupt termination, SIGKILL, host failure, and OOM termination can bypass it, leaving RUNNING trials and a summary marked running. The ask/tell loop has no durable ownership lease or stale-trial reconciliation.

### P1 — Memory telemetry does not regulate launches

`src/app/telemetry/system.py` samples training process trees and some host statistics. Its CUDA measurements describe allocations/reservations visible to that process, not all device consumers or complete GPU usage. The external fallback (`src/app/services/processes.py:process_stats`) samples only the parent and creates a new psutil Process for each CPU sample, which makes CPU percentages unreliable as a historical utilization measure.

None of this telemetry feeds `has_capacity()`. There is no host-pressure circuit breaker, memory reservation, OOM cooldown, or admission pause. Observability alone is not protection.

### P2 — Optuna is synchronous and batch-oriented

`src/app/tuners/optuna/sweeper.py:114–190` asks for an entire batch, waits for launcher completion, and only then tells Optuna the results. Fast trials wait for the slowest before another batch is sampled. With BasicLauncher, all batch suggestions are made before the first serial result is available. Increasing batch size can reduce feedback without increasing throughput.

`timeout_seconds` is checked between batches, not enforced on running jobs. Failure limits are evaluated after the batch, so multiple expensive failures can occur before stopping. These controls are not hard runtime or resource limits.

Trial identities and UUID-suffixed log versions provide useful artifact separation. That is necessary groundwork, but does not establish parallel-launcher compatibility, safe GPU isolation, or resource scheduling.

### P2 — SQLite should not become the parallel/distributed default

All studies default to a shared SQLite database. Preparation enables WAL and a busy timeout, which reduces some contention but does not create concurrent writers or make SQLite appropriate on network filesystems. Sweeper and monitor processes reopen storage from URL strings, so preparation's connection settings are not uniformly reused.

Parallel methods or experiments can contend even when each individual sweep runs serially. Optuna advises against SQLite for parallel optimization and specifically warns about NFS. Use a client/server RDB such as PostgreSQL for a supported parallel/distributed mode. Journal storage is an alternative requiring changes to Theta's URL-oriented storage and dashboard integration, not a drop-in supported setting here.

Multiple coordinators resuming one study are also unsafe at the product level: per-run study attributes and summary files can overwrite one another, and each coordinator budgets its own additional trials. Either disallow this or implement study-wide coordination.

### P2 — Queue settings and tests have drifted during the runner/UI refactor

The current `QueuePanel` displays active cards only. It does not expose the older width selector, pause controls, or pending queue management expected by `tests/test_parallel_runs.py`. The application still restores `queue.max_parallel` and exposes a setter. Users therefore lack a clear current control for this policy.

`JobStore` does not persist `run_in_parallel`; reproduced by saving and reloading a job. Reconnect state can consequently lose the detail used to split method cards. Admission-relevant resource plans should be explicit persisted data, not reconstructed from display cards.

Existing parallel-queue tests instantiate the detached desktop client while monkeypatching methods/state formerly owned in-process. They do not validate the new authoritative runner. Failing tests must be repaired before treating this suite as a concurrency guarantee.

### P2 — Slurm needs a distinct lifecycle

`run_pipeline.py` calls `run_slurm_training()` without tracking completion of its returned job IDs. The desktop QProcess finishing therefore means submission finished, not training completed. Local queue slots cannot represent remote active-job limits; local cancellation cannot be assumed to perform `scancel` on submitted jobs.

Support remote submitted/running/terminal states, durable Slurm IDs, polling, remote cancellation, and allocation-aware trial limits separately. Trials within one allocation must share that allocation's budget. Do not use local host free memory to size remote work.

## Recommended regulation

### Separate backlog length from execution width

Queued entries consume no training compute until admitted. Keep a generous bounded backlog for usability and persistence performance; an initial product guardrail such as 100 pending experiments is reasonable, but is not a compute-derived safety number. Warn about large expanded workloads and projected disk use. Plan large sweeps and replicate sets lazily rather than materializing every task/future upfront.

For a desktop, budget the user's compute target across Theta workspaces. Per-project queues can remain separate views of the same target scheduler. For a future shared service, use both per-user quotas and a global node budget; individual quotas alone cannot prevent aggregate overload.

### Use one authoritative admission service

Every training task—ordinary method, replicate, or trial—must obtain a reservation from the same target scheduler. Pipeline parents should not hold all worker slots while waiting for children; that can deadlock nested execution. Start requests from UI, CLI, plugins, and launchers must follow the same admission policy or be explicitly marked unmanaged.

Persist a task resource request: CPU threads, estimated peak host RAM, GPU device/exclusivity and estimated peak VRAM, internal worker counts, and optional disk/runtime limits. Unknown workloads begin alone. Reserve before spawning, release only after confirmed exit, and reconcile leases after restart using PID plus process creation identity rather than PID alone.

Default to one training task per GPU and one unknown task per compute target. Treat Apple unified memory as shared host/GPU capacity; do not add two independent memory pools. For GPU sharing, require measured headroom and an explicit supported mode.

### Help users choose with a measured recommendation

Offer Auto (recommended), Manual, and pause-after-current-work. Auto starts with one task, profiles representative training plus evaluation/checkpoint/data loading, and proposes greater width only after measuring peak demand. A short warmup may miss later peaks; show estimate confidence and invalidate profiles when model, dataset, batch size, hardware, engine, or worker settings change. For sweeps, profile a plausible expensive configuration, not only the cheapest trial.

As initial tunable policy, reserve roughly 25% of host RAM and at least one or two CPU cores for the OS/IDE; apply a 1.3–1.5 multiplier to observed peak memory. These are conservative starting heuristics, not guarantees or universal hardware facts. Respect container/process allocation limits and other applications' current usage.

For homogeneous tasks, derive recommended width from the minimum CPU, RAM, GPU/exclusivity, user-cap, and target-cap bounds. For heterogeneous work, sum reservations per resource/device instead of multiplying one average. Also check current physical headroom before each admission; do not double-subtract reservations already reflected in measured usage.

Show a concrete launch preview:

> 40 trials queued; 2 can train at once. Each is estimated to use 3 GB RAM and 2 CPU threads. Theta will keep 8 GB RAM available for the system. Recommendation confidence: medium.

Explain the limiting resource and why work waits. Show actual active training tasks separately from experiments and submitted trials. If the selected launcher is serial, show effective trial concurrency as 1. A batch created with `study.ask()` is not proof that every trial is executing.

### Add a pressure circuit breaker

Before every new task, check live available RAM, device memory, disk headroom, and external consumption. Under sustained pressure, stop admissions immediately and retain queued work. After an OOM, pause admissions and reduce the recommendation rather than immediately launching the next identical job. Distinguish OOM from an ordinary model error.

Do not repeatedly kill all work based on CPU utilization alone. Offer targeted cancellation/checkpointing where supported; checkpoint support is engine-specific. Use hysteresis and a cooldown before resuming admissions. Reactive monitoring cannot prevent every sudden allocation spike. Hard resource isolation requires platform support, such as Linux cgroups/containers, Windows Job Objects, or Slurm allocations; local macOS admission remains best effort.

### Preserve fairness

Use FIFO initially, with bounded backfilling for smaller tasks that fit. Age waiting tasks to avoid starvation, and limit the share of slots one sweep can occupy. A user's short single run should not sit behind thousands of eagerly scheduled trials. Add per-user weights/quotas only when there is actually a shared service.

## Implementation order and acceptance gates

1. Make all runner launch paths honor the same count, repair runner/UI tests, restore an understandable width control, and persist the execution plan. Keep unverified parallel trial launchers clearly marked unsupported.
2. Bound method/replicate concurrency and establish a shared task budget, process ownership, cancellation, and restart reconciliation. Include CLI and multiple-workspace behavior.
3. Add target discovery, resource requests, conservative defaults, live admission checks, explainable recommendations, and OOM cooldown.
4. Implement and test a supported parallel-trial adapter. Prefer fresh training subprocesses, explicit device assignment, PostgreSQL-backed storage, and completion-driven ask/tell. If using Hydra Joblib, configure an explicit small worker limit; its documented default uses all CPU cores.
5. Add independent Slurm job lifecycle/quotas and cluster allocation validation.

Required verification: real PID/timestamp overlap for parallel trials; global admission under simultaneous callers; independent workspace coordination; cancellation during worker startup; confirmed descendant exit; runner crash with live child; stale trial reconciliation; concurrent storage writes; artifact isolation and resume; simulated low RAM/GPU memory/disk; OOM classification/cooldown; and fair scheduling under a large sweep. GPU behavior needs actual supported GPU hardware; mocks cannot establish memory safety.

Do not claim machine-crash prevention. The achievable promise is conservative admission, explicit limits, pressure response, recoverable state, and stronger isolation where the platform supports it.

## External references

- Hydra 1.3 multi-run behavior: https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/multi-run/
- Hydra 1.3 Joblib launcher and process workers: https://hydra.cc/docs/1.3/plugins/joblib_launcher/
- Optuna 4.9 FAQ, including parallelization, reproducibility, GPU assignment, and SQLite limitations: https://optuna.readthedocs.io/en/v4.9.0/faq.html
- Optuna RDBStorage heartbeat documentation: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.RDBStorage.html — built-in heartbeat is intended for `optimize()`, not Theta's `ask()`/`tell()` loop. Simply adding `heartbeat_interval` will not implement recovery here.

## Validation record

Installed environment inspected: Python 3.11.13, Optuna 4.9.0, Hydra 1.3.7, hydra-optuna-sweeper 1.1.2, psutil 7.2.2; hydra-joblib-launcher absent. Installed BasicLauncher source confirms a serial loop over jobs.

Isolated runner reproduction confirmed immediate-launch bypass, queued-path count enforcement, and loss of `run_in_parallel` on persistence. No production training, settings changes, dependency installations, stress tests, or release actions were performed.

Targeted test command:

```sh
QT_QPA_PLATFORM=offscreen OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
venv/bin/python -m pytest -q tests/test_parallel_runs.py \
  tests/test_tuning_trial_artifacts.py tests/test_optuna_pruning.py \
  tests/test_variability.py tests/test_job_manager.py
```

Result: **52 passed, 10 failed, 2 skipped** in 82 seconds. Six failures are in `test_parallel_runs.py`: three mutate the desktop client's local state instead of the detached runner, and three reference removed UI attributes. One sweep test expects `Ruleset2` in an experiment fixture that lacks it. Three variability failures concern old unique-log-root expectations, a missing evaluation-events path, and an old checkpoint path assertion. These latter artifact/telemetry failures need reconciliation against the current layout before concluding whether all are stale tests. None establishes successful real concurrent Optuna execution. Existing test workloads include small CPU training runs; no GPU or saturation test was performed.
