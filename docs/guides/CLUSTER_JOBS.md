# Cluster jobs

Theta submits one experiment per Slurm allocation. Project upload, submission, sync, and cleanup are explicit operations. Closing the desktop does not cancel a submitted job. OpenSSH owns authentication; Theta does not store passwords or silently retry ambiguous submissions.

The hardened CPU pilot is `site=ncshare/common`; `ncshare_common` remains an alias. Its code has local filesystem/subprocess coverage, including actual CPU training and tuning. **Live NCShare acceptance and the overnight gate are still pending.** See the [implementation status and validation commands](../development/SLURM_NCSHARE_IMPLEMENTATION_STATUS.md). GPU qualification, automatic requeue, and full training-checkpoint resume are separate work.

## NCShare storage and setup

Keep the control checkout, compact immutable source archives, and Python environment in home. `connection.remote_root` is this durable control checkout, currently `/hpc/home/cegbert1/ThetaIDE`. Helpers resolve its interpreter to an absolute path. Do not configure `/scratch` as the remote project directory or create the environment under `/work`.

The CPU site's versioned `execution` policy resolves `{user}`, `{uid}`, `{project_id}`, and `{control_root}` through an allowlist. No shell expansion is used:

| Root | Purpose |
| --- | --- |
| home control checkout | configuration, interpreter, source archive store, submission receipts, scheduler stdout/stderr |
| `/work/{user}/thetaide/{project_id}` | declared inputs and immutable published objects/manifests |
| unique `theta-EXECUTION-…` directory under job-local `/scratch` | source, provider state, cwd, datasets, all training outputs, exclusively owned live SQLite |

Scratch must exist, be writable, and be local storage. Missing/network scratch fails explicitly. There is no NFS execution or database fallback. Several allocations can share a node, each with its own scratch tree/database.

On the cluster, provision the home environment first (commands below are examples to execute yourself):

```bash
mkdir -p /hpc/home/cegbert1/ThetaIDE
python3 -m venv /hpc/home/cegbert1/ThetaIDE/venv
```

Locally, configure the existing site component, check transport, and upload:

```bash
venv/bin/python run_pipeline.py site configure ncshare/common \
  --host ncshare --remote-root /hpc/home/cegbert1/ThetaIDE \
  --python /hpc/home/cegbert1/ThetaIDE/venv/bin/python3
venv/bin/python run_pipeline.py site check ncshare/common
venv/bin/python run_pipeline.py site push ncshare/common
```

Both ends need rsync 3 or newer. Authenticate/MFA in a separate terminal first; transfers use SSH BatchMode. Edit SSH aliases in the site component's `ssh_config`. No operation edits home SSH configuration.

Install dependencies in the cluster's home environment after upload:

```bash
cd /hpc/home/cegbert1/ThetaIDE
venv/bin/python -m pip install -e '.[optuna]'
venv/bin/python -m pip check
```

Install the selected research components' dependencies too. Settings → Terminal → Cluster sites generates a setup script from their native manifests; inspect it before running. Setup generation itself makes no connection. Environments are never copied from the desktop.

Project push builds the source archive locally, verifies activation remotely, and retains one archive per source identity. Submission pins that archive; later uploads cannot change queued/running source. Scratch execution requires a new-format archive from `site push`.

Datasets are excluded by default. Declare only required input copies in the site's policy, for example:

```yaml
execution:
  inputs:
    - source: in/datasets/pilot.csv
      target: in/datasets/pilot.csv
```

Sources are control-project-relative, links/traversal are rejected, and targets must be below `in/datasets/` or `in/inputs/`. `site push SITE --include-datasets` uploads datasets to control storage explicitly; submission stages only declared inputs to work, then the allocation copies them to scratch. Input byte/file/free-space limits apply. Relative runtime paths resolve from scratch. Large shared read-only dataset mappings are not qualified in this pilot; declare bounded copies or defer the experiment.

New work uploads/publications get current mtimes, while source archive/input metadata retains original timestamps. Do not touch older results to defeat `/work`'s documented 75-day mtime purge. Scientific results need independent durable backups.

## Submission and concurrency

From the cluster control checkout:

```bash
venv/bin/python run_pipeline.py GROUP/EXPERIMENT site=ncshare/common dry_run=true
venv/bin/python run_pipeline.py GROUP/EXPERIMENT site=ncshare/common
```

The receipt prints the execution ID and job ID. Each attempt gets fresh execution/result identities. Initial CPU defaults are one node/task, four Slurm CPUs, 8 GB, no GPU, and 15 minutes. Use an explicit experiment walltime for longer runs after checking `scontrol show partition common`. Resource directives, GRES, and managed behavior cannot be overridden through raw flags or inherited `SBATCH_*` variables. Unset conflicting submission variables explicitly; the error lists their names.

Slurm admits **one active trial and one active training seed per allocation**. Requests for Optuna `n_jobs > 1` or `variability.parallel_runs > 1` fail clearly. Local parallel trials retain the workstation resource pool and remain supported. Within-allocation parallel admission/contention is not qualified. A two-seed experiment can run its seeds serially with `parallel_runs: 1`.

`plot_only` on Slurm is rejected before plotting starts. Plot synced prior results locally with their original result identity and `wipe=false`. Workflow DAGs, remote dashboards, automatic requeue, and `recover=true` are unsupported.

## Allocation-owned publication

The allocation runs with the home interpreter and scratch cwd/workspace/output roots. SQLite defaults and SQLite tuning URLs resolve to one `results/optuna/EXECUTION/study.sqlite` under that scratch root. Workers/callbacks use the same resolved storage. Explicit external backends remain external; errors do not redirect to SQLite. Put credentials in environment references, never literal captured URLs.

Consistent SQLite backups run on the owning compute node. Published copies are closed, integrity-checked, standalone DELETE-mode databases; live DB/WAL/SHM files are never copied or opened by a login helper.

Publication controls are independent of desktop sync and model checkpoint creation:

| Output | Initial CPU policy |
| --- | --- |
| changed scalar metrics/status/log prefixes | 60 seconds |
| study database | trial completion requests, coalesced to at least 30 seconds; 300-second fallback during a long trial |
| committed checkpoints | priority after producer close/commit |
| committed monitor arrays | queued once, small files bundled, bounded chunks |
| final payload | close producers, drain remaining exports, seal manifest |

These are configurable engineering defaults. Five minutes is not a cluster restriction or a whole-results copy timer. One publisher turn per project bounds simultaneous exports; 8 MiB objects and 32 MiB turns bound bulk work before priority is reconsidered. Admission stops on excessive lag/backlog or runtime storage usage. Cadence targets may slip under I/O pressure.

The batch requests a TERM signal 120 seconds before timeout. Catchable stops forward to producers, allow bounded shutdown, and attempt final publication. SIGKILL/node loss can lose unpublished scratch state. Receipts expose `publication_state`, `publication_error`, and `last_durable` positions for database, metrics, and checkpoints separately. A database export does not restore the running model.

Payload sealing happens before scheduler termination. Final cleanup additionally requires confirmed scheduler termination. Scheduler state, training outcome, publication state, and local transfer errors are distinct. An unsealed payload remains inspectable/syncable but cannot receive a fabricated complete manifest. Expensive repair must run in an explicit CPU allocation; no automatic repair/requeue is provided.

## Sync and diagnostic budgets

Locally:

```bash
venv/bin/python run_pipeline.py site sync ncshare/common --mode metrics --execution EXECUTION_ID
venv/bin/python run_pipeline.py site sync ncshare/common --mode metrics --diagnostics --execution EXECUTION_ID
venv/bin/python run_pipeline.py site sync ncshare/common --mode results --execution EXECUTION_ID
venv/bin/python run_pipeline.py site status ncshare/common
```

Sync Metrics defaults to scalars/status/logs. Choose **Pull diagnostics** in its menu or `--diagnostics` for monitor arrays. Results retrieves every owned published artifact. Execution filtering avoids reading unrelated history. Scheduler queries are batched and briefly cached; known terminal states are not queried again.

For the managed CPU path, login helpers read manifests and precomputed hashes. They never walk scratch, hash checkpoints, or back up live SQLite. Missing immutable objects download into resumable local staging, are checksum-verified, and reconstruct the normal `results/…` layout atomically. A second unchanged sync transfers zero payload. Local monitor HTML is generated after newly retrieved complete snapshots; collection no longer reloads all historical arrays to regenerate HTML each time.

Scratch telemetry flushes on a five-second/size boundary and lifecycle events; resource sampling defaults to ten seconds. The pilot diagnostic policy limits measured storage across all methods/trials/seeds to 128 MiB/1,024 files, with an 8 MiB snapshot cap, per-run shares across planned methods/trials/seeds, and a quarter reserved for final observations. Probe/state files count too. Skipped coverage and reasons are explicit; optional exhaustion does not change objectives/checkpoints. `diagnostic_strict: true` makes unavailable required diagnostics fail. These limits are provisional until representative cluster measurement; exhaustive sampling and full model-state recovery are not promised.

## Continuation and cleanup

Continue a selected exported study explicitly, using a confirmed terminal prior execution:

```bash
venv/bin/python run_pipeline.py GROUP/EXPERIMENT site=ncshare/common \
  ++study_continuation.execution_id=PRIOR_EXECUTION_ID \
  ++study_continuation.export_sha256=EXPORTED_DATABASE_SHA256 \
  ++study_continuation.study_name=EXACT_STUDY_NAME \
  tuning.options.optuna.n_trials=3
```

Run this in the cluster control checkout. Select the hash from the prior publication manifest's `category: database` entry, not an unverified live database. This adds new trials to the exported history. The same source/provider identity, runtime dependency versions, objective, and search-space configuration are required; submission/time/trial-count settings may change. Keep the same recipe and supply changes as overrides. One lineage owner is allowed; continue its latest attempt instead of branching. Unknown prior allocations block continuation.

Restore verifies chunks and database integrity on compute, preserves completed/pruned/failed history, marks exported RUNNING trials failed with interruption provenance, and assigns new trial/output identities. Prior checkpoint dependencies remain pinned. Samplers restart from their configured seed; the next suggestions need not match an uninterrupted run. This is **study continuation**, not training model/optimizer/RNG resume. Complex seed-panel checkpoint references still need live qualification before use.

Remote results remain until explicit verified cleanup:

```bash
venv/bin/python run_pipeline.py site clean ncshare/common EXECUTION_ID --preview
venv/bin/python run_pipeline.py site clean ncshare/common EXECUTION_ID
```

Cleanup requires a full reverified local manifest, confirmed termination, no active/unknown dependency jobs, no continuation pin, and no transfer lease. It removes that execution's published objects, staged inputs, and reclaimable workspace; compact receipts/manifests/source identity remain. An interrupted deletion is retryable. Back up scientific results independently first.

On the cluster, inventory legacy data and preview managed temporary collection:

```bash
venv/bin/python run_pipeline.py site inventory
venv/bin/python run_pipeline.py site maintenance
# After reviewing its candidates:
venv/bin/python run_pipeline.py site maintenance --apply
```

Expired managed uploads/transfers are reclaimed at bounded sync maintenance points. Source GC uses ownership and surviving receipt references under the same project mutation lock as submission. Active transfer leases renew during retrieval. Unclassified legacy uploads/workspaces/sync copies are inventory-only and are never moved/deleted automatically.

Other sites without an execution policy retain the shared-workspace compatibility executor. Legacy active SQLite sync is now refused: export on its owning compute node or migrate after its processes close. Old unsealed runs need explicit scheduled repair/migration; they do not gain the new CPU publication guarantees retroactively.
