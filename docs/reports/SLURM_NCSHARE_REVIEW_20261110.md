# Slurm integration and NCShare operational review

Reviewed 2026-11-10. Scope: the current checkout, its site profiles, submission and transfer code, telemetry, mechanistic monitors, and NCShare's published guidance. This is a source review with local checks, not a cluster load test. No SSH connection or scheduler submission was made, and no implementation files were changed.

The normal training path uses Slurm appropriately and has several useful safeguards. The main shared-cluster risks are unbounded work during manual sync, frequent writes to shared storage, and retained source/sync copies. There is also a concrete SQLite WAL incompatibility with NCShare's NFS storage. Default mechanistic monitoring has a snapshot count limit; the concern is aggregate cost across runs and repeated copying, rather than an inherently endless default snapshot loop.

## Integration summary

- Site components own the scheduler, resource profiles, SSH configuration, connection, environment, and shortcuts. NCShare supplies CPU and GPU profiles.
- GUI Run saves and uploads the project, copies a validation/submission command, and opens an SSH terminal. The user pastes the command to submit. The desktop application remains local; this does not deploy a remote IDE server.
- Cluster-side submission freezes source/configuration into a per-execution workspace, assigns unique execution/attempt/result IDs, writes a receipt and batch script, and calls `sbatch` once. Ambiguous responses become `submission_unknown`, without automatic retry.
- One allocation runs the common pipeline: training, evaluation, configured monitors, tuning or variability, and plots. Methods, trials, and seeds run serially. Managed thread limits derive from `SLURM_CPUS_PER_TASK`. Validation ordinarily restricts execution to one node and one GPU/device.
- Sync Metrics and Sync Results are explicit actions, with no automatic remote polling loop. They query scheduler state, enumerate owned outputs, snapshot mutable files, transfer through rsync, verify checksums, and publish locally. Metrics includes monitor snapshot arrays, not just scalar CSVs.
- Cleanup is explicit and requires confirmed termination plus matching, verified local and remote artifacts. Source and datasets remain. Remote cancellation, automatic recovery, dashboards, and workflow DAG execution are unsupported.

Relevant implementations: [GUI launch](../../frontend/site_launch.py), [allocation executor](../../src/app/sites/execution.py), [transfers](../../src/app/sites/transfer.py), and [manual sync UI](../../frontend/site_sync.py).

## Prioritized findings

### 1. High at scale: a metrics refresh performs full remote copying and hashing on the login node

In `transfer.index`, every active execution is traversed. Every selected artifact without a finalized manifest entry is copied to a new `results/slurm/.sync/GENERATION` tree and hashed. This includes unchanged, already committed monitor `snapshot_*.npz` files. CSV/JSONL snapshots are additionally read entirely into memory to truncate an incomplete trailing record. Sync Metrics can also trigger `finalize`, which hashes **all** owned terminal outputs, including checkpoints excluded from the eventual metrics download.

The local ledger skips unchanged downloads only after that remote work has happened. A small or zero transfer count therefore does not establish low cluster I/O. The SSH command runs on the configured host, currently `login.ncshare.org`; full indexing, copying, hashing, and SQLite backup run there.

Evidence: [transfer.py](../../src/app/sites/transfer.py), `ssh`, `index`, and `sync`; [execution.py](../../src/app/sites/execution.py), `finalize`; [NCShare SSH alias](../../src/usr/sites/ncshare/ssh_config).

A local check indexed an unchanged 1 MiB monitor NPZ twice and confirmed two different scratch copies with identical checksums.

NCShare permits lightweight transfers and job monitoring on its shared login node, while reserving computation for scheduled nodes. Large artifact preparation is a scaling risk under that policy; this review did not measure actual login-node load. [NCShare login guidance](https://userguide.ncshare.org/guides/setupssh/).

Recommended changes: have the allocation publish lightweight manifests and committed artifact hashes; reuse immutable monitor artifacts; support execution-specific sync and bounded log tails; prepare large snapshots/final hashes in an allocation. Add a byte/time budget and rate limit. Preserve checksums while moving their expensive generation out of interactive login-node helpers.

### 2. High for tuning reliability: SQLite WAL is enabled on NFS, with cross-host backup access

Submission rewrites SQLite tuning storage to `results/optuna/EXECUTION_ID/study.sqlite`. `get_optuna_storage` explicitly enables `PRAGMA journal_mode=WAL`. The configured project is in home, which NCShare mounts over NFS. While training accesses the database on a compute node, Results sync opens the live database through SQLite on the login node to back it up.

Execution-specific databases prevent unrelated runs from sharing one study file, but do not solve the filesystem or cross-host shared-memory problem. SQLite documents that WAL does not work over network filesystems. This is a study integrity/availability risk, not evidence of a cluster-wide scheduler failure. [SQLite WAL documentation](https://sqlite.org/wal.html), [NCShare architecture](https://userguide.ncshare.org/guides/slurm/architecture/).

Evidence: [execution.py](../../src/app/sites/execution.py), `submit`; [Optuna storage](../../src/plugins/components/bundled/optuna/utils.py), `get_optuna_storage`; [transfer.py](../../src/app/sites/transfer.py), SQLite backup in `index`.

Use an approved external database, which the executor already preserves, or implement node-local study storage with compute-side consistent exports and stage-out. The current rewrite means merely configuring another SQLite path is insufficient. Do not have the login node open a live compute-node WAL database.

### 3. Medium, increasing with workload: the working project and all results default to home

The NCShare connection uses `/hpc/home/cegbert1/ThetaIDE`. Outputs, checkpoints, source copies, sync scratch, logs, and tuning storage all live beneath that project. Stream telemetry flushes every record; CSV logging defaults to flushing each row. System samples default to every two seconds. These are buffered-stream flushes, not an `fsync` for every telemetry record, but they still encourage small writes to shared storage.

NCShare describes home as a 50 GB location for scripts and environments. `/work` is intended for active working data and purges files older than 75 days. Both are shared storage: moving results to `/work` improves placement/capacity but does not eliminate NFS contention. `/scratch` is job-private, node-local temporary storage. [NCShare storage guidance](https://userguide.ncshare.org/storage/).

Evidence: [NCShare defaults](../../src/usr/sites/ncshare/configs/site/ncshare/_defaults.yaml), [stream sink](../../src/app/telemetry/sinks.py), [CSV logger](../../src/app/core/contract_logger.py), and [system sampler](../../src/app/telemetry/system.py).

Keep environments in home, use shared working storage for the project/results, and configure absolute interpreter paths when separating the environment. Add cluster-specific telemetry buffering, sampling limits, retention, and storage-budget checks. For substantial I/O, implement scratch staging with periodic durable exports and final stage-out. **Do not simply change `remote_root` to `/scratch`**: the current submission/sync design requires a shared path visible from login and compute nodes. The generic `/scratch` examples in [the cluster guide](../guides/CLUSTER_JOBS.md) are inappropriate for NCShare's scratch semantics.

### 4. Medium: source and upload history accumulate even after successful cleanup

Every submission physically copies the source tree into `results/slurm/EXECUTION_ID/workspace`. Every distinct push has a directory in `.thetaide/uploads/SOURCE_HASH`. Uploads use `--link-dest` where possible, but activation copies files into the live project and submission freezes another full copy. There is no upload-history collector, and execution cleanup does not remove the frozen workspace. Failed submission preparation can also leave source copies.

The present source selection contains 998 files totaling 11,145,723 bytes, excluding datasets and environments. One thousand submissions would create roughly one million frozen source files and 11.1 GB of logical source payload before results and upload history. This is a scale estimate, not a measured NCShare allocation.

Evidence: [snapshot.py](../../src/app/sites/snapshot.py), `freeze`; [transfer.py](../../src/app/sites/transfer.py), `push`, `activate`, and `clean_remote`; [execution.py](../../src/app/sites/execution.py), `submit` and `owned_files`.

A disposable local cleanup check confirmed that metrics were removed while both workspace and upload directories remained.

Recommended changes: retain source by content hash and share immutable snapshots; add explicit, safe garbage collection for unused uploads and terminal execution workspaces. Preserve receipts, manifests, and source identity. Never remove a snapshot referenced by a pending or running job. Bundle large collections of small files for transfer, consistent with NCShare's storage guidance.

### 5. Medium: failed indexing leaves sync scratch without a usable release token

`index` creates copies before it returns a generation ID. It has no exception cleanup. The client only calls `_release` when it received a complete listing, so an indexing error, malformed/incomplete response, or timeout can strand the directory. The SSH helper has a 180-second timeout. Even a received listing can leak if the best-effort release call fails. There is no implemented next-visit sweep of abandoned generations.

Evidence: [transfer.py](../../src/app/sites/transfer.py), `ssh`, `index`, `sync` finally block, and `release`.

A local injected indexing failure left one scratch generation behind. Add exception cleanup inside `index`, explicit generation ownership/leases, and bounded stale-generation reclamation that protects active transfers. Do not use a blind age-based delete while transfers may still be running.

### 6. Medium: monitors limit snapshot count, but lack aggregate storage budgets and rebuild reports repeatedly

Mechanistic monitors default to `interval_updates: 10`, `max_snapshots: 32`, 128 probes, and 512 probe-collection steps. The manager enforces the count, reserves a final slot, and skips disabled instances. These are meaningful safeguards.

However, each snapshot writes JSON plus compressed arrays, rewrites monitor state, and calls `write_report`. Report generation loads every earlier snapshot and its arrays, then rewrites a self-contained HTML report. Across K snapshots this gives K(K+1)/2 snapshot loads: 528 for the default K=32. Network reports embed all saved arrays as JSON. No aggregate byte budget spans instances and runs; the allowed snapshot count reaches 100,000.

For scale, 40 trials × 4 monitor instances × 32 snapshots × 2 snapshot files gives 10,240 snapshot files, before probes, state, reports, normal logs, checkpoints, or source copies. Trials are serial but retained artifacts accumulate. Snapshot byte sizes depend strongly on model, probe, and monitor settings.

Evidence: [monitor configuration](../../src/app/monitoring/config.py), [manager](../../src/app/monitoring/runtime.py), [snapshot writer](../../src/app/monitoring/api.py), and [HTML report generator](../../src/app/monitoring/report.py).

Generate HTML locally after sync or once at completion; cache immutable snapshot data; impose per-snapshot, per-monitor, and per-execution byte/file budgets. For initial cluster runs, use a few enabled monitors, small probe banks, and 8–16 snapshots. Increasing the update interval alone does not reduce the eventual file cap if a long run still reaches it.

### 7. Medium: `plot_only=true` bypasses scheduler dispatch

`run_pipeline.py` handles `plot_only` and invokes the plotting subprocess before checking the Slurm scheduler branch. Thus running a Slurm-site plot-only command from the login terminal performs plotting there. Large result scans and report generation can become unscheduled computation.

Evidence: [pipeline entry point](../../run_pipeline.py), plot-only branch before scheduler submission; [plot subprocess](../../src/app/pipeline/datasets.py), `run_plotting`.

Move scheduler dispatch ahead of compute-bearing modes or reject remote plot-only execution outside an allocation. Until then, generate plots locally from synced results or run the command inside a scheduled interactive allocation.

### 8. Medium: actual GRES and extra directives can contradict resource validation

`validate_allocation` checks `resources.gpus`, nodes, and trainer devices. Header generation separately accepts a raw `resources.gres` and arbitrary `extra_sbatch_flags`. These can request resources inconsistent with the validated values.

A local check passed validation with `gpus: 1` and `gres: gpu:h200:8`, then generated `#SBATCH --gres=gpu:h200:8`. Extra directives can similarly override node/task/CPU requests. Slurm still applies its own scheduling and account limits; this is an unintended-allocation and utilization issue, not an escape from Slurm.

Evidence: [allocation validation](../../src/app/sites/execution.py), `validate_allocation`; [batch header](../../src/app/pipeline/slurm.py), `get_gres_header` and `generate_sbatch_header`.

Validate a single normalized allocation representation and reject conflicting GRES/extra resource directives. Avoid raw resource overrides until that validation exists.

### 9. Medium compatibility/efficiency: GPU setup, walltime, and preemption need NCShare-specific treatment

- The GPU profile requires `module load cuda`, and the script fails on a missing module command. NCShare documents that it does not use environment modules and does not provide a host CUDA toolkit. Remove the module requirement and install a compatible environment/container. The current executor has no dedicated Apptainer launch wrapper. [NCShare architecture](https://userguide.ncshare.org/guides/slurm/architecture/).
- Both profiles inherit `time: null`; the header then omits walltime. This relies on the partition default, rather than explicitly requesting an appropriate duration. Set a short explicit limit for initial checks and right-size subsequent runs.
- The `gpu` partition can cancel and requeue preempted jobs, while Theta emits `--no-requeue`, rejects recovery, and refuses to run an execution whose receipt already has `started_at`. Periodic model checkpoints do not by themselves provide a supported end-to-end resume path. This can lose progress or waste repeated computation. Do not enable requeue without implementing resumable attempts and validating actual partition behavior. [NCShare GPU guidance](https://userguide.ncshare.org/guides/gpu/).
- The GPU profile reserves 16 CPUs for one GPU. This may be appropriate, but serial trials and small RL models can leave much of the allocation idle. Measure real CPU/GPU utilization and adjust resources; Slurm CPUs are hardware threads on NCShare. [NCShare efficiency guidance](https://userguide.ncshare.org/guides/efficiency/).

Evidence: [GPU profile](../../src/usr/sites/ncshare/configs/site/ncshare/gpu.yaml), [defaults](../../src/usr/sites/ncshare/configs/site/ncshare/_defaults.yaml), and [executor](../../src/app/sites/execution.py).

### 10. Lower priority initially: scheduler queries and listings scale with project history

Each sync/status action scans project receipts. Each unresolved/nonterminal receipt issues its own `squeue`, followed by `sacct` when absent from the queue. Confirmed terminal receipts skip those queries, which helps. There is no batched status query, minimum refresh interval, or execution filter. Many active/unknown jobs can create a scheduler RPC burst and hit the SSH helper timeout. Every sync still lists historical finalized artifacts.

Evidence: [execution.py](../../src/app/sites/execution.py), `refresh_status`; [transfer.py](../../src/app/sites/transfer.py), `index` and `status_local`.

Batch scheduler queries, cache terminal state and immutable manifests, filter executions, and debounce manual refreshes. The current UI does not automatically poll the remote scheduler.

## Recommended order

1. Before tuning: resolve shared-NFS SQLite WAL and live cross-host backup.
2. Before the first GPU job: remove the CUDA module requirement, validate the environment, and set explicit resources/walltime.
3. Before frequent monitoring or large sweeps: bound remote sync work; move heavy preparation into allocations; add scratch/upload/workspace reclamation and storage budgets.
4. Before enabling many monitors: defer HTML generation, constrain probes/snapshot bytes, and measure one representative run's files, bytes, and sync latency.
5. Before supporting automatic retries/requeue: implement resumable attempts and test interruption/preemption on disposable jobs.

Use `/work` with explicit export/backup practices for working data and home for environments. Introduce job-local scratch through an intentional staging design, not by changing the remote project root alone. Keep the explicit human submission and conservative cleanup safeguards.

## Validation and limits

Ran `QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q tests/test_sites.py tests/test_site_workflow.py tests/test_site_ownership.py tests/test_site_sync_ui.py tests/test_slurm.py tests/test_mechanistic_monitors.py tests/test_monitor_lifecycle.py`.

Result: **63 passed, 2 failed**. The failures were `test_reused_experiment_streams_only_current_launch_and_all_parallel_seeds` (extra running entry) and `test_queue_labels_method_without_started_feed_as_pending` (`Training` versus `Running`). All selected site/Slurm and mechanistic-monitor tests passed. These tests use local files/fake cluster commands and do not prove live scheduler, NFS, quota, or preemption behavior.

Additional disposable local checks reproduced unchanged NPZ recopying, scratch retention after indexing failure, workspace/upload retention after cleanup, and conflicting GPU validation/GRES output. Source size was measured from the current upload selection. No production data was deleted, cluster resource usage was not measured, and no claim of actual disruption is made.
