# Slurm hardening and NCShare CPU rollout

Status: implementation plan revised 2026-11-10 after storage, parallelism, publication, and recovery decisions; implementation and live validation remain outstanding. Addresses the [operational review](../reports/SLURM_NCSHARE_REVIEW_20261110.md). A ready-to-use [implementation handoff](../development/SLURM_NCSHARE_IMPLEMENTATION_HANDOFF.md) accompanies this plan.

First supported target: **`ncshare/common`**, including its existing `ncshare_common` compatibility alias. The user's reference to `ncsu_common` is interpreted as this existing NCShare CPU profile, not a new cluster or profile rename. GPU qualification follows CPU acceptance.

## Decisions

1. Keep NCShare addresses, paths, resource defaults, export intervals, retention, and monitoring budgets in the **NCShare site component**. Implement reusable lifecycle mechanisms in the shared Slurm adapter. Avoid NCShare-specific branches in training or monitor providers.
2. Keep application/configuration, compact source snapshots, and Python environments in home. Put job inputs and published results in `/work/<uid>/thetaide/<project_id>`. Execute and write temporary state on job-local `/scratch`. This requires separating control, working, and runtime paths; changing `connection.remote_root` alone is insufficient.
3. Support overnight execution with the desktop disconnected. The allocation publishes periodic consistent exports and a final export to `/work`; later sync reads those exports. The desktop is not required for checkpointing, retention maintenance, or job completion.
4. Use **one exclusively owned scratch-local SQLite database per allocation**, accessed only by that allocation's processes. Multiple allocations may run independent studies on the same node; this is not a one-study-per-node restriction. Export consistent standalone database snapshots from the compute node. Database-server provisioning and new PostgreSQL integration are outside this rollout; retain the backend extension point and existing explicitly configured external storage behavior without silent fallback.
5. Transfer committed artifacts and chunks with precomputed hashes. Do not discover, snapshot, or hash the training tree from the login node during normal sync.
6. Keep explicit submission, no automatic resubmission/requeue, and verified result cleanup. Treat scheduler state, training outcome, publication state, and local transfer state separately.
7. Preserve existing local parallel trials. Current Slurm execution forces serial trials; make that limitation visible and qualify any within-allocation parallelism separately against actual allocated resources. SQLite does not by itself mandate serial training, but concurrency needs validation.
8. Make publication cadence site-configurable and specific to output type. Five minutes is a proposed fallback interval for study backups during long trials, not a cluster requirement or a universal export interval.
9. Include explicit continuation of an exported Optuna study in a new allocation. Distinguish that from resuming an interrupted trial's model/optimizer state, which requires a matching checkpoint and an independently supported runner resume contract.

NCShare documents home for scripts/environments, shared `/work` for working data, temporary job-private scratch, and a 75-day modification-time purge on `/work`. Its storage guidance recommends bundling small files. [Storage guidance](https://userguide.ncshare.org/storage/), [Python environments](https://userguide.ncshare.org/software/python/).

### Database choice

Optuna's `RDBStorage` is an interface, not a guarantee that storage is a database server: the current code already constructs it with SQLite. PostgreSQL/MySQL servers solve the shared-file storage problem because clients talk to a server instead of opening a database file over NFS. Optuna supports server-backed storage for multi-node use. [Optuna storage options](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html).

Review of NCShare's published services/software/storage guidance and targeted searches did **not establish an available user PostgreSQL/MySQL service**. This is not evidence that institutional support cannot supply one. Do not use Slurm's accounting database for application studies or install a database daemon on the login node. Database provisioning is not a prerequisite for the CPU rollout.

For the current Slurm path, scratch-local SQLite removes NFS WAL access without requiring a database service. Keep every live connection and backup operation on the owning compute node; never attach another allocation to that active database. Exported copies are immutable data, not alternate live writers. Local SQLite can contain multiple named studies where the existing per-method design requires them, while the database's allocation owner remains exclusive.

Future PostgreSQL qualification would require compute-node connectivity, credentials, TLS, driver compatibility, schema initialization, pooling, backups, and runtime secret references redacted from all captured configuration. Do not add that infrastructure to the initial implementation. Preserve existing explicit external URLs safely and never silently redirect a failed server connection into local SQLite.

### Trial parallelism

The current code has two different behaviors: `src/plugins/components/bundled/optuna/parallel.py` launches parallel local workers subject to `n_jobs` and the local resource pool, whereas `src/app/sites/execution.py` sets `THETA_MANAGED_SERIAL=1` and removes the workstation resource pool. The sweeper consequently uses batches of one in Slurm allocations. Do not describe parallelism as absent across Theta, or as already supported in Slurm.

Qualify the initial CPU path with one active trial, preserving local behavior. Before honoring Slurm `n_jobs > 1`, add an allocation-scoped admission policy that caps aggregate CPU threads, memory, and devices using the allocation rather than whole-node capacity. Exercise same-host SQLite WAL contention, short transactions, bounded lock waits, pruning reports, backup progress, and cancellation under concurrent trials. This is a separate acceptance gate within the rollout; if it is not delivered, explicitly reject the unsupported Slurm concurrency request instead of silently serializing it. Do not add nested seed/trial concurrency that exceeds the allocation.

Optuna discourages SQLite for parallel optimization because of its concurrency limits; modest same-host concurrency is something to benchmark and qualify, not a promised general scaling solution. Heavy contention is a reason to defer that mode or change the backend later. [Optuna FAQ](https://optuna.readthedocs.io/en/stable/faq.html).

## Ownership and contracts

| Owner | Responsibility |
| --- | --- |
| `src/usr/sites/ncshare` | CPU resource profile, home/work/scratch policy, interpreter location, publication intervals, purge-aware timestamps, retention and monitor defaults |
| `src/app/sites` | Allocation validation, source identity, staging, publication, manifests, status reconciliation, transfer leases and garbage collection |
| Pipeline/runtime boundary | Consistent runtime cwd, interpreter, workspace/data/output roots, relative datasets, and plot-only routing |
| Optuna component | Backend selection, local DB lifecycle/exports, server connections, study summaries and portable export identity |
| Monitoring/telemetry | Buffering, snapshot sampling/budgets, immutable artifacts, explicit skipped-output metadata, deferred reports |
| Frontend/CLI | Manual sync scope, scalar/diagnostic selection, artifact availability, publication errors, cleanup previews |

Define a versioned declarative site execution policy with fields for runtime mode, work/scratch roots, publication schedules by output type, stage-out allowance, storage limits, tuning backend, trial concurrency/admission, and monitor/telemetry defaults. Resolve user/project placeholders on the cluster through an explicit allowlist, not arbitrary shell evaluation. Existing sites without this policy retain compatible shared-workspace behavior, while receiving shared validation fixes.

The initial NCShare CPU profile uses `common`, one node, one task, four Slurm CPUs, 8 GB RAM, zero GPUs, and a 15-minute smoke-test walltime. Longer/overnight work uses an explicit experiment walltime checked against current partition limits. Runtime and helper interpreters use an absolute home environment path. Create/validate this environment separately from scratch staging; do not generate an unused environment under `/work`.

Site UI additions must use the existing [shared parameter controls](../reference/CONFIG_PARAMETER_FORMS.md). This is a narrow execution/storage contract, not implementation of the deferred workspace reorganization proposal.

## Implementation sequence

### 1. Close scheduler and resource-validation gaps

**Change:** normalize the final allocation once, and generate directives from that object. Validate GRES GPU type/count against the declared allocation; reject conflicting raw `extra_sbatch_flags` for managed resources, task count, output paths, working directory, requeue, and signal behavior. Account for inherited `SBATCH_*` submission environment overrides so the actual submitted allocation cannot bypass validation. Keep supported non-resource flags explicitly classified.

Route `plot_only` through the Slurm allocation path before starting plotting; keep offline validation/export commands available. The plot-only worker must operate on explicitly identified prior results rather than a newly stamped empty result ID. Plots get their own owned output/manifest entries and cannot wipe prior inputs. Until that routing is complete, reject Slurm plot-only outside an allocation with a clear error.

**Acceptance:** the reproduced one-versus-eight-GPU conflict is rejected; task/node/CPU/GPU overrides cannot contradict validation; CPU jobs emit no GPU request; no plotting or training subprocess starts on the login node; invalid requests fail before source upload/submission where locally knowable.

**Files:** `run_pipeline.py`, `src/app/pipeline/slurm.py`, `src/app/sites/execution.py`, relevant composition/preflight code.

### 2. Introduce scratch execution and safe study storage

**Change:** separate durable control/source locations from shared working outputs and temporary runtime paths. Stage immutable source and selected inputs into a unique scratch directory per execution. Use the home interpreter, but run the pipeline from scratch with matching workspace, provider state, output root, and dataset mappings. Audit subprocesses that derive paths from module location or assume a project-local `venv`.

Stage only declared inputs; do not copy an entire dataset library by default. Preflight selected input size, scratch free space, and configured byte limits. Large shared read-only datasets need an explicit mapping and an I/O policy. Fail clearly if required scratch is unavailable rather than silently reverting to heavy NFS writes. Keep source/configuration frozen for queued jobs.

Remove unconditional rewriting of every SQLite URL to a shared `results/optuna` path. Resolve the default database to an execution-local scratch path. Coordinators, objective callbacks, pruning, and trial workers must all receive the same resolved backend. Take consistent SQLite backups **on that compute node**; close/convert exported copies to standalone non-WAL snapshots. No remote helper opens the live database. [SQLite backup API](https://sqlite.org/backup.html).

**Acceptance:** a small CPU study with pruning/summary/winner promotion works entirely on scratch; no live database or WAL/SHM files appear on NFS; its exported copy can be opened independently. Separate allocations have separate databases even when sharing a node. A configured server failure is reported without fallback or credential leakage. Training, evaluation, variability, datasets, and plotting resolve the same intended runtime paths. Local parallel trials still work; Slurm requests for unsupported concurrency fail explicitly. Within-allocation parallelism is only enabled after the separate resource/contention gate above passes.

**Files:** `src/app/sites/execution.py`, new shared staging support, `src/app/pipeline/runtime.py`, selected workspace/output boundary code, Optuna component/storage utilities, NCShare profile/setup generation.

### 3. Publish outputs from the allocation, including abrupt-stop handling

**Change:** add an allocation-owned publisher. Hash artifacts when committed; accumulate append-only logs/metrics in bounded immutable segments; bundle many diagnostic files into bounded immutable archives on scratch. Do not repack or recopy old snapshots on each publication. Copy newly committed objects to temporary names on `/work`, verify/record completion, then atomically advance a versioned manifest. Preserve the existing extracted local result layout through safe client-side materialization.

Use one bounded publisher with schedules by output type, rather than a full-results copy every five minutes:

| Output | Initial publication policy |
| --- | --- |
| Small metrics/status and bounded log segments | Every 60 seconds while changed; configurable, with 30–60 seconds as the initial interactive range |
| Consistent study database snapshot | Request at trial completion; coalesce bursts of short trials with a configurable minimum spacing (initially 30 seconds). Also request every five minutes during long active trials |
| Model checkpoints | Publish after a committed checkpoint is available, with priority over optional diagnostic payloads; checkpoint creation follows the experiment's schedule |
| Monitor arrays and other large committed artifacts | Queue once on commit and batch transfers with byte/concurrency limits; do not reload or repackage previously committed artifacts |
| Final outputs | Export all remaining required artifacts and seal the payload before successful stage-out |

These are initial engineering defaults, **not NCShare-mandated limits**. Tune them against measured CPU, byte, and metadata costs. Consistent database exports may use SQLite's backup API while training continues on the same host; only the exported copy must be standalone and closed before publication. Do not copy a live `.sqlite` file without its transactional state.

Coalesce overlapping requests; permit one exporter per execution; bound queued bytes and export concurrency across the project. Transfer large objects in bounded units so status and study durability do not wait behind an entire multi-gigabyte diagnostic bundle. Track the last successful export separately for study state, metrics, and checkpoints, including the represented training position. An interval is a target, not a guarantee under bandwidth pressure. Persistent publication lag stops admission of new trials and surfaces a failure instead of allowing unlimited scratch accumulation.

Reserve walltime for stage-out, based on configured allowance and measured output size/rate. On a pre-timeout signal, stop admitting work, request a supported checkpoint, close streams, and publish within the remaining time. Normal exit, nonzero exit, and catchable cancellation attempt a final export. SIGKILL/node failure can lose unpublished scratch output: report the last confirmed export and incomplete artifacts; never infer completeness from scheduler termination alone.

Seal the **payload** manifest in the allocation after producers close. Reconcile scheduler termination and small late scheduler stdout/stderr separately afterward. The existing rule that finalization requires scheduler termination must become a two-stage contract; do not weaken cleanup by pretending a running job is already terminal. Interrupted/unsealed runs may be inspected and synced, but cannot receive a fabricated complete manifest. Any needed expensive repair runs in a separate explicit CPU allocation, not a login-node scan.

**Acceptance:** closing the desktop does not stop durable exports; small updates and DB backups obey their configured independent schedules; unchanged large artifacts are neither reread nor retransferred; tiny trials do not trigger an unbounded backup burst. Timeout/cancellation retains the last verified publication; an interrupted copy never becomes visible as committed; training-complete/publication-failed remains distinguishable from fully exported; export retry is idempotent and preserves artifact ownership. A database export never implies that an in-progress trial's training state is recoverable.

### 4. Make sync incremental, bounded, and suitable for morning retrieval

**Change:** normal SSH helpers read small execution indexes and committed manifests. Batch `squeue`/`sacct` queries; cache terminal states; offer execution filters. Sync Metrics defaults to scalars, summaries, state, and bounded log previews. Rich monitor arrays require explicit diagnostic selection; Results sync retrieves the complete owned artifact set, including monitor data.

Use existing ledger concepts plus immutable object IDs and checksums to fetch only missing objects. Keep resumable local staging and checksum verification. For large transfers, display bytes/progress and support cancellation without restarting completed objects. Never snapshot the live tree, hash checkpoints, or open live SQLite during sync. Any unavoidable large packaging/hash/repair operation is scheduled computation.

On `/work` publication/upload, give newly staged files current modification times; preserve original scientific/source timestamps in metadata. The purge is mtime-based, and current `rsync -t`/`copy2` behavior can preserve timestamps older than 75 days. Do not periodically touch old results to defeat the cluster purge policy. [NCShare timestamp guidance](https://userguide.ncshare.org/storage/).

**Acceptance:** retrieving completed overnight runs needs no running compute allocation when their payloads were sealed; a second unchanged sync reads only bounded control metadata and transfers zero artifact payload; active metrics sync does not grow remote scratch; runtime cost depends on selected/new artifacts rather than traversing all historical result files.

**Files:** `src/app/sites/transfer.py`, `src/app/sites/execution.py`, `frontend/site_sync.py`, CLI and monitor/results readers.

### 5. Give temporary and retained storage explicit lifecycles

**Change:** use a content-addressed immutable source archive store, so repeated submissions reference one copy per source identity. Submission pins the source before it can queue. Garbage collection uses project ownership, references, and leases under a shared mutation lock; it cannot rely on timestamp age alone.

Automatically reclaim uncommitted upload fragments, released transfer generations, superseded preview generations, and unreferenced cache objects at bounded maintenance points. Implement exception cleanup in every staging/indexing operation. Persist leases sufficiently to distinguish active transfers after client crashes; collect only after expiry plus ownership/reference checks. Show reclaimable bytes in a cleanup preview.

Extend explicit verified Results cleanup to release execution workspaces, obsolete managed source references, and owned artifact objects once no surviving manifest/job/transfer references them. Keep compact receipts, manifests, and source identity. Scientific outputs remain remote until verified local retrieval and explicit cleanup; temporary objects do not need to accumulate until then. Unknown scheduler state conservatively pins data and appears as a reconciliation action. Storage caps can block new submissions instead of silently deleting unresolved data.

Before adopting an existing project, inventory legacy uploads/workspaces/`.sync` copies and produce a cleanup/migration preview. Preserve legacy results and aliases; do not automatically move/delete old home outputs or attempt to read live legacy NFS WAL databases. Export an active legacy study on its owning compute node, or wait for its processes to close before migration. Old unsealed jobs use explicit repair/migration paths. Never apply the new collection policy to unclassified files.

**Acceptance:** repeated identical submissions do not duplicate source payload; failed transfers leave no unowned permanent scratch; a collector cannot delete a running/pending/unknown job's source or an active transfer; cleanup after verified sync removes all reclaimable execution payload. A cleanup failure is retryable without destroying retained proofs.

### 6. Replace expensive monitor behavior with measured budgets

**Change:** remove HTML regeneration from every monitor collection. Render reports locally from synced artifacts, or once after collection when explicitly requested. Store immutable snapshots once; write only changed state needed for supported recovery; avoid rewriting state for every unrelated instance. Keep probe banks shareable within an execution when identities match.

Retain explicit snapshot-count configuration for compatibility, but introduce a budgeted collection policy for new cluster runs. Enforce aggregate diagnostic bytes/files across the entire execution, not just each trial or monitor. Allocate that budget across planned methods/trials/seeds and reserve final snapshots. Use expected update count, measured artifact size, and a minimum collection interval to choose a feasible schedule. Record collection positions, coverage, skipped snapshots, and budget reasons. Explicit exhaustive sampling that cannot fit must fail preflight or require a different budget; it must not be silently downsampled.

Use provider payload estimates before collection and a bounded serializer before publication. Large rich outputs can be reduced only through declared probe/feature sampling with recorded coverage; otherwise skip them with an explicit diagnostic status. Optional diagnostic budget exhaustion must not corrupt objective metrics or checkpoints. Provide an explicit strict failure policy for experiments that require those diagnostics.

Add site-configurable time/size-based telemetry flushing and resource sampling. Pilot defaults: five-second stream flushing and ten-second system samples on scratch, with small metrics published every 60 seconds and large arrays batched after commit as above. Scratch flushing, shared-storage publication, desktop sync, model checkpoint creation, and database backup are separate controls. Lifecycle/errors flush promptly. Set initial byte/file budgets from the representative CPU benchmark before overnight qualification; keep 32 only as a legacy count setting, not a claim of a universal safe limit. Increasing probes/features is subject to those same budgets.

**Acceptance:** snapshot/report processing grows linearly in committed snapshot bytes; no historical NPZ reload on every collection; byte/file caps hold across a multi-trial study; first/final observations fit their reservations or produce an explicit availability error; training output is unchanged by optional monitor throttling. Full Results sync reconstructs readable monitor artifacts and local HTML reports.

**Files:** `src/app/monitoring/{runtime,api,config,report}.py`, `src/app/telemetry`, monitor parameter controls, NCShare policy defaults.

### 7. Support explicit study continuation and honest interruption recovery

**Change:** add an explicit continuation operation referencing a prior execution, exported database generation/hash, and study identity. Acquire a durable study-ownership lease before submission; reconcile any unknown prior allocation and refuse a second continuation owner. Pin the selected export, validate the schema/provider/objective/search-space compatibility policy, and restore it into the new allocation's scratch database. Preserve completed/pruned/failed trial history and allocate new trial numbers/output identities without overwriting old attempts. Persist a lineage pointer and portable references to prior checkpoints/results; pin those dependencies against cleanup.

Mark trials that were running in the selected export as interrupted/failed under an explicit retry policy; do not count them as completed or accidentally leave them running forever. A retry gets a new attempt identity. Preserve/export sampler state when needed for the selected algorithm, or document and record that suggestion sequences can differ after restart; a DB copy alone is not a promise of bit-for-bit reproducibility. Retain the existing local new/resume study semantics and reject incompatible continuation requests clearly.

Distinguish two capabilities in UI, CLI, and receipts:

- **Continue the study:** reuse the verified historical trials, then schedule additional/new trials in a new allocation. This belongs in the CPU rollout.
- **Resume an interrupted training trial:** restore the matching model, optimizer, scheduler, RNG, data/environment, monitor, and runner-specific state where supported. Do not claim this works merely because a study DB exists. Preserve and advertise checkpoint availability; full runner-specific resume and automatic Slurm requeue remain separately gated follow-ups.

Database and checkpoint exports can represent different training positions. Record both positions and validate compatible committed artifacts before claiming any training resume. For abrupt node loss, report exactly which exports survived; unpublished scratch work may be lost. A training checkpoint is not automatically created by the DB backup interval.

**Acceptance:** continue a finished exported study and a deliberately interrupted study into fresh scratch, preserving trial history and avoiding duplicate ownership or output collisions. Concurrent continuation attempts are refused. Garbage collection retains referenced inputs. Unsupported mid-trial resume fails explicitly while study continuation remains usable. Missing/corrupt/incompatible exports cannot create a falsely successful continuation.

### 8. Qualify CPU behavior, then permit overnight use

Automated checks must cover real lifecycle properties: matching resource requests, plot routing, scratch path separation, DB export integrity, allocation-exclusive database ownership, local parallelism regression, explicit Slurm concurrency handling, publication scheduling/coalescing, atomic publication, forced-stop behavior, study continuation, repeated-sync I/O counts, leases/GC races, archive traversal/link rejection, purge-safe timestamps, legacy migration, and monitor budget behavior. Keep current scheduler mocks; add local filesystem/subprocess lifecycle tests. New database-server integration/tests are outside this rollout. Resolve or isolate the two existing monitor UI lifecycle failures before declaring a green supported workflow; do not silently drop them from the report.

Live validation proceeds on `ncshare/common` in this order:

1. Check home/work permissions, selected inputs, interpreter, and partition limits without launching training on the login node.
2. Run a short CPU experiment; verify requested resources and scratch writes, then complete a Results sync.
3. Run a small CPU tuning study and a separate two-seed experiment. Verify studies, objectives, checkpoints, plots, and monitors after local reconstruction.
   If within-allocation parallel trials are enabled, separately verify at least two concurrent trials against an explicitly sufficient CPU/memory allocation, including lock contention, pruning, simultaneous completions/exports, cancellation, and aggregate resource usage. Otherwise verify a clear error for that request.
4. Close/disconnect the desktop during a run. Confirm exports continue, then retrieve results after completion.
5. Exercise a short walltime, user cancellation, interrupted transfer, disk-budget exhaustion, and failed publication on disposable runs. Verify partial-result semantics and the durable export boundary.
6. Repeat sync unchanged; record remote bytes read/written, file/object counts, scheduler query counts, and login-helper CPU/RSS. Confirm these do not scale with historical artifact payload.
7. Preview and perform verified cleanup on disposable outputs, then confirm shared source and other executions remain intact.
8. Run one representative overnight CPU workload within an explicit walltime and measured storage budget; sync the next morning and compare final manifests/checksums and expected outputs.
9. Continue a completed and an interrupted exported study in new allocations. Confirm history, ownership, trial/output identity, and the distinction between study continuation and training-checkpoint resume.

Record measurements and chosen monitor/publication limits in a development status document, update the cluster guide with the actual NCShare layout, and link both from the documentation index. Real-cluster acceptance is required; local test success alone is not sufficient.

## GPU follow-up and completion criteria

After the CPU acceptance gate, remove NCShare's invalid `module load cuda` requirement, verify a compatible home environment/container, right-size CPU/RAM/GPU requests, and test one short GPU job. Requeue/resume requires a separate implemented attempt/checkpoint protocol before enabling it; the present no-requeue stance remains explicit until then. Do not make GPU testing or expensive GPU runs a dependency of CPU hardening.

This plan is complete when the CPU sequence passes, overnight runs remain retrievable without a desktop connection, no normal sync opens a live NFS database or prepares large artifacts on the login node, temporary storage has bounded collection, scientific output cleanup remains verified, monitor overhead/storage meet measured execution budgets, and explicit study continuation works. Parallel local trials must remain functional; Slurm parallelism is either separately qualified within allocation limits or explicitly unavailable. PostgreSQL integration, full mid-trial training recovery, and automatic preemption/requeue recovery are follow-ups, not hidden requirements of that milestone.
