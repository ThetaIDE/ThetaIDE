# NCShare CPU implementation and validation status

Updated 2026-11-10. Implements the revised [plan](../plans/SLURM_NCSHARE_HARDENING_PLAN.md) and [handoff](SLURM_NCSHARE_IMPLEMENTATION_HANDOFF.md). The original handoff/plan/review and existing index edits were uncommitted user work and have been preserved. No SSH, real Slurm submission/cancellation, remote cleanup, component release, commit, or push was performed.

## Implementation stages

| Stage | Implemented locally | Acceptance boundary |
| --- | --- | --- |
| Allocation validation | shared normalized resource object, GRES consistency, allowlisted extras, SBATCH environment rejection; Slurm plot-only rejected | one CPU allocation; parallel Slurm trials/seeds explicitly rejected |
| Scratch/storage | versioned site-owned policy; home control/source/interpreter, declared work inputs, fresh local scratch; shared subprocess roots and allocation SQLite guard | actual NCShare mounts/free-space/provider environment still need live verification |
| Publication | same-node bounded SQLite backup, closed standalone exports, independently scheduled scalar/DB/checkpoint work, immutable chunk/bundle objects, atomic manifests, final sealing and signal handling | SIGKILL/node failure loses unpublished work; no model-state resume or auto-requeue |
| Incremental sync | execution filter; manifest-only managed payload discovery; checksum/object cache; safe ranges; local materialization/report generation; progress/cancel for managed transfers; bounded scheduler log previews; batched/cached scheduler queries | helper metadata still scales with selected manifest size; use `--execution` for historical projects |
| Storage lifecycle | source archives deduplicated by identity; mutation-locked references and ownership; expiring/renewed transfer/upload leases; temporary maintenance; obsolete object collection; verified cleanup releases publication/inputs/workspace | legacy/unclassified data remains inventory-only; active/unknown jobs, transfers, and continuation dependencies pin data |
| Monitor cost | no per-collection HTML rebuild; measured execution/run byte/file shares, raw-array/serialized bounds, first/final capacity reservation, skip coverage, strict policy; buffered telemetry | pilot limits are engineering choices; representative NCShare overhead/storage benchmark remains a gate |
| Continuation | verified selected DB export into fresh scratch; exclusive latest-lineage owner; source/config/runtime dependency compatibility; interrupted trials fail explicitly; new attempt/result/trial identity; checkpoint dependency pins | basic completed/interrupted studies covered locally; complex repeated-seed artifact references require further qualification |

The CPU profile preserves `ncshare/common` and `ncshare_common`, and uses 4 CPUs/8 GB/0 GPUs/15 minutes initially. GPU profile qualification is deferred. Local parallel Optuna workers remain available; Slurm `n_jobs > 1` is rejected until aggregate CPU/memory/device admission and contention/cancellation tests are qualified.

Managed scratch SQLite is never opened by normal login helpers. Legacy active SQLite sync is rejected with migration instructions; old shared-workspace execution is not retroactively converted. A final manifest requires both a compute-sealed payload and separately confirmed scheduler termination. Database and checkpoint durability positions remain distinct.

## Local evidence

The initial requested audit suites reproduced **63 passed, 2 failed**. The failures were the two already recorded in the handoff:

- `test_monitor_lifecycle.py::test_reused_experiment_streams_only_current_launch_and_all_parallel_seeds`: the current queue supplies a method parent in addition to the two seed leaves.
- `test_monitor_lifecycle.py::test_queue_labels_method_without_started_feed_as_pending`: the current UI labels the active training phase `Training`, while this older test expects `Running`.

These failures are isolated from the managed storage tests and remain reported; their assertions have not been weakened or removed. The supported end-to-end desktop workflow is **not declared wholly green**. A broader exploratory run also found 11 existing `test_queue_panel.py` failures targeting removed constructor callbacks/window queue APIs. Queue implementation was not changed by this work. Current hierarchy coverage is retained in `test_queue_hierarchy.py`.

New `test_ncshare_hardening.py` uses real files, SQLite WAL connections, fake scheduler/transfer commands, and real subprocesses. It exercises resource overrides; archive traversal/link refusal; purge-safe input mtimes; allocation-exclusive scratch; standalone WAL backups; DB cadence/coalescing; atomic failed copies; immutable bundles and repeat-sync zero payload; manifest-only login work; shared diagnostic budgets; continuation compatibility/ownership/interruption reconciliation; scheduler batching; signal stage-out; runtime limits; transfer cancellation; and verified cleanup/lease exclusion.

Two integrations execute the **actual pipeline from archived source and providers in fresh local scratch**: a 64-transition CartPole CPU run and a three-trial CPU Optuna study. The latter is continued in a new attempt and its four completed trials verified. This validates local path/interpreter/storage integration, not NCShare Slurm or NFS behavior.

The original adaptive-trials regression needed its component config roots added to the direct Hydra test harness. Its serial and two-worker subprocess checks then passed, including measured worker overlap, distinct output identities, and slot refill. macOS sandbox process enumeration was denied; the same local regression ran successfully outside the sandbox: **3 passed** (Optuna distribution deprecation warnings only).

A core checkpoint before final expansion passed **77 tests**, including real CPU study continuation. A later 150-test run caught a transient WAL-disappearance race during continuation: the publisher now tolerates disappearing producer sidecars, and dedicated regression coverage was added. The broad final run passed **150 tests with the same 2 baseline monitor-lifecycle failures** (152 collected, 101.20 seconds). It ran outside the macOS sandbox for process inspection. An additional bounded-metadata serializer regression was added afterward. Final focused checks passed **118 tests, with only the two already-passing full CPU integration cases deselected** (22.77 seconds); these covered the final serializer, cleanup-preview, and scheduler/training-outcome changes. The suite does not establish live acceptance.

```bash
QT_QPA_PLATFORM=offscreen venv/bin/python -m pytest -q \
  tests/test_ncshare_hardening.py tests/test_sites.py tests/test_site_workflow.py \
  tests/test_site_ownership.py tests/test_site_sync_ui.py tests/test_slurm.py \
  tests/test_mechanistic_monitors.py tests/test_monitor_lifecycle.py \
  tests/test_queue_hierarchy.py tests/test_optuna_pruning.py tests/test_sweep_monitoring.py \
  tests/test_telemetry.py tests/test_monitor_streams.py tests/test_monitor_refresh_cost.py \
  tests/test_adaptive_trials.py
```

`git diff --check` and Python compilation checks passed. No mocked result is live cluster acceptance.

## Prepared live CPU commands

These commands are prepared for a session in which you authorize and execute cluster work. They were **not run against NCShare**. Keep source unchanged between the first study and its continuation; changing trial counts/walltime through CLI overrides preserves the source pin.

On the desktop, create disposable pilot recipes, then upload:

```bash
mkdir -p in/config/experiment/ncshare
cp docs/development/examples/ncshare_cpu_smoke.yaml in/config/experiment/ncshare/cpu_smoke.yaml
cp docs/development/examples/ncshare_cpu_study.yaml in/config/experiment/ncshare/cpu_study.yaml
venv/bin/python run_pipeline.py ncshare/cpu_smoke dry_run=true
venv/bin/python run_pipeline.py ncshare/cpu_study dry_run=true
venv/bin/python run_pipeline.py site check ncshare/common
venv/bin/python run_pipeline.py site push ncshare/common
```

The example recipes compose and pass local preflight with this checkout's installed SB3/DNN/CartPole components. Install their matching dependencies on the cluster first. They intentionally start with no monitors and plotting disabled; qualify the training/storage path before adding representative diagnostics/plots.

In your NCShare login terminal (lightweight preflight only):

```bash
cd /hpc/home/cegbert1/ThetaIDE
venv/bin/python -m pip check
scontrol show partition common
id -un
df -h /hpc/home/cegbert1 /work
venv/bin/python run_pipeline.py site inventory
venv/bin/python run_pipeline.py ncshare/cpu_smoke site=ncshare/common dry_run=true
venv/bin/python run_pipeline.py ncshare/cpu_smoke site=ncshare/common
```

Record the printed `EXECUTION_ID`/`JOB_ID` and inspect the exact script and scheduler allocation:

```bash
cat results/slurm/EXECUTION_ID/job.sbatch
squeue -j JOB_ID -o '%.18i %.9P %.8T %.10M %.6D %R'
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID,State,Elapsed,AllocCPUS,ReqMem,MaxRSS,ExitCode
```

Do not invoke `_run` on login. The worker requires `SLURM_JOB_ID`. Confirm from job logs/runtime artifacts that cwd/output/database are the unique scratch directory, Python is the home environment, and requested resources match the header. After completion, from the desktop:

```bash
venv/bin/python run_pipeline.py site sync ncshare/common --mode metrics --execution EXECUTION_ID
venv/bin/python run_pipeline.py site sync ncshare/common --mode results --execution EXECUTION_ID
venv/bin/python run_pipeline.py site sync ncshare/common --mode results --execution EXECUTION_ID
```

The repeat pull must report zero payload. Open the **local exported copy**, never the live scratch database, to check integrity. For the study, submit on login:

```bash
venv/bin/python run_pipeline.py ncshare/cpu_study site=ncshare/common
# Explicit unsupported-concurrency check: must fail before sbatch.
venv/bin/python run_pipeline.py ncshare/cpu_study site=ncshare/common tuning.options.optuna.n_jobs=2 dry_run=true
```

Disconnect the desktop before the study finishes. Later sync its results and inspect the local receipt's `last_durable` and `publication_state`. Find the database hash in local `results/slurm/STUDY_EXECUTION_ID/publication.json` (the `category: database` entry). Continue from login:

```bash
venv/bin/python run_pipeline.py ncshare/cpu_study site=ncshare/common \
  ++study_continuation.execution_id=STUDY_EXECUTION_ID \
  ++study_continuation.export_sha256=DATABASE_EXPORT_SHA256 \
  ++study_continuation.study_name=ncshare_cpu_pilot \
  tuning.options.optuna.n_trials=2
```

The new database should retain the original three trials and add two. Repeat with an interrupted study after accounting confirms termination; exported RUNNING trials should become FAIL, and new trials receive new numbers. An unsealed study can continue from its last verified DB export; this does not restore its interrupted model. A simultaneous second owner must be refused. Check that a historical winner still promotes through pinned checkpoint dependencies.

Next run a **separate** serial two-seed pilot (on login):

```bash
venv/bin/python run_pipeline.py ncshare/cpu_smoke site=ncshare/common \
  '++variability.training_seeds=[42,43]' ++variability.parallel_runs=1
```

Retrieve and verify both seed manifests, objectives, checkpoints, and runtime paths. Add a small explicit monitor configuration and locally supported plots for a separate representative run; record actual file/byte counts and diagnostic overhead before increasing budgets.

For disposable failure tests, increase the training budget so the job remains active. For example:

```bash
venv/bin/python run_pipeline.py ncshare/cpu_study site=ncshare/common \
  total_timesteps=1000000 resources.time=00:05:00
# After at least one durable export exists, explicitly cancel that disposable job:
scancel DISPOSABLE_JOB_ID
```

Also let a separate short-walltime job expire, interrupt a desktop transfer with **Cancel transfer** and retry, and inject a small runtime/publication cap through `++site.execution.max_runtime_bytes=...` or a disposable unwritable work destination. Inspect publication errors, last durable DB/checkpoint positions, incomplete status, and scheduler accounting independently. Do not use shared production output paths for failure injection.

On disposable fully retrieved runs only:

```bash
# On the cluster: preview temporary/source collection; inspect candidates first.
venv/bin/python run_pipeline.py site maintenance
# On the desktop after independent backup and full Results sync:
venv/bin/python run_pipeline.py site clean ncshare/common EXECUTION_ID --preview
venv/bin/python run_pipeline.py site clean ncshare/common EXECUTION_ID
```

Check that only the selected payload/inputs/workspace disappear and that receipts, manifests, other executions, source identity, active leases, and continuation inputs remain protected. Cleanup of a parent with a surviving continuation is expected to fail.

## Live gate and measurements still required

Complete the plan's ordered CPU sequence: permissions/interpreter/partition limits; CPU smoke; tuning and serial two-seed run; disconnected publication; timeout/cancellation/transfer/disk/publication failures; unchanged-sync measurements; verified disposable cleanup; representative overnight CPU workload; completed/interrupted continuation.

Capture Slurm allocation/usage, exported positions, source/manifest identities, scratch/work file and byte counts, publisher counters and duration, helper CPU/RSS, scheduler RPC counts, repeat-sync bytes, and monitor collection/storage overhead. Initial limits are provisional: metrics 60 s; DB 30 s minimum/300 s fallback; chunks 8 MiB/turn 32 MiB; stage-out 120 s; runtime 8 GiB/100k files; queued files 128; inputs 2 GiB/4,096 files; retained objects 16 GiB; source archives 2 GiB; diagnostics 128 MiB/1,024 files/8 MiB snapshot. Defaults also cap source archives and retained publication objects; active leases can deliberately block reclamation and admission.

Remaining limitations: Slurm parallelism, GPU acceptance, server provisioning, full runner checkpoint resume/requeue, automatic repair, large shared read-only dataset mappings, complex seed-panel continuation references, and fairness/timing under real shared-storage pressure are unqualified or deferred. The local chunk publisher bounds work between turns; a blocked filesystem operation is still subject to Slurm's hard deadline. Publication uses producer commit hooks for active large artifacts; third-party producers without that contract are exported only after they close at final stage-out.
