# Cluster jobs

Theta submits one experiment per Slurm allocation. Slurm support, including the ARC/NCShare site templates and the cluster setup panel, is the optional `slurm` scheduler component; install it into the workspace first (see [the scheduler contract](../reference/SCHEDULER_CONTRACT.md#installing-the-slurm-component)). Without it, no cluster settings or commands appear and saved cluster recipes report the missing scheduler. Project upload, submission, sync, and cleanup are explicit operations. Closing the desktop does not cancel a submitted job. OpenSSH owns authentication; Theta does not store passwords or silently retry ambiguous submissions.

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

Sync, upload, and remote checks allow 60 seconds to establish each SSH connection. For a slower login host, set `connection.ssh_connect_timeout_seconds: 120` (a positive integer) in the site's `_defaults.yaml`. This setting also applies to rsync's SSH transport; it controls connection setup, not the duration of a download. A connection timeout can also indicate an unreachable host or a required VPN; increasing it does not resolve those network problems.

### Sync transport and login-node load

For compute-published results, sync reads the committed manifest, skips verified objects already cached locally, and downloads all missing objects in one sequential rsync batch per remote storage root. A typical NCShare pull has a work-storage batch and, when needed, a small home-control batch for scheduler log previews. It does not start one rsync or remote Python helper per object. An unchanged pull starts no rsync.

On macOS and Linux, each sync uses a private temporary SSH control socket so indexing, downloads, lease renewal, and release can reuse one authenticated connection. The connection closes after cleanup on success or failure, with a 60-second idle expiry as a fallback. Windows keeps separate SSH connections while retaining batched downloads. The socket is scoped to the sync, without editing home SSH settings or sharing another terminal's master connection.

The download lease lasts two hours and is renewed at five-minute intervals while rsync is running, rather than for every object. Each rsync batch retains the existing one-hour transfer limit. Cancellation or renewal failure terminates the transfer child; sync then attempts lease release. Completed objects from failed batches are checksum-verified and cached on the client, while rsync partial files remain available for retry. Objects are verified before local result reconstruction; corrupt or incomplete data is not installed as a result.

Compute jobs continue to own publication hashing, bundling, and database exports. The login node serves published objects and lightweight control operations; the client verifies hashes and reconstructs results. Use metrics sync during training and full results sync when needed to reduce transferred bytes.

On macOS, `/usr/bin/rsync` may be Apple's openrsync (rsync 2.6.9 compatibility), which does not support the protected-argument upload flags. Install modern rsync with `brew install rsync`, then set `connection.rsync_executable` in the site component to its absolute path: typically `/opt/homebrew/bin/rsync` on Apple Silicon or `/usr/local/bin/rsync` on Intel. Check the installed location with `brew --prefix rsync`; its executable is under `bin/rsync`. Pinning the absolute path avoids differences between terminal and desktop-app PATHs. Keep `remote_rsync` separate: it names the executable on the cluster.

Run performs the local version check before staging or connecting for upload. Transfer failures include rsync's captured error output, rather than only a generic exit status. After setting the path, use `site check ncshare/common` to verify both ends before retrying Run.


Install dependencies in the cluster's home environment after upload:

```bash
cd /hpc/home/cegbert1/ThetaIDE
venv/bin/python -m pip install -e .
venv/bin/python -m pip check
```

Install the selected research and tuner components' dependencies too (for example Optuna's `optuna` and `sqlalchemy`). Settings → Terminal → Cluster sites generates a setup script from their native manifests; inspect it before running. Setup generation itself makes no connection. Environments are never copied from the desktop.

Project push compares source files by checksum and reuses unchanged files from the remote checkout. A parameter-only YAML edit transfers that YAML plus small sync manifests, rather than reuploading components or a complete ZIP. New and changed files are uploaded; removed source files are removed during verified activation. The cluster builds and retains one immutable source archive per source identity. Submission pins that archive; later uploads cannot change queued/running source. Scratch execution requires a new-format archive from `site push`.

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

In the desktop, **Run switches to the Terminal pane and creates a new shell** in its existing shell list. That shell shows snapshot preparation, live rsync progress, verification, and then the interactive SSH session. Upload errors remain visible there and prevent the SSH handoff. Existing shells stay available in the list.

Use **Close shell** to terminate that shell's upload/SSH process and close its connection. Switching panes or selecting another shell keeps the session running. Closing the main window also closes its terminal sessions. A late startup callback cannot reopen a closed shell. Jobs already accepted by Slurm continue independently.

The experiment is saved before upload. Once SSH is ready in the configured remote project root, paste the copied command and press Enter:

```bash
venv/bin/python3 run_pipeline.py GROUP/EXPERIMENT
```

The saved experiment supplies the site and parameters. The clipboard command has no `cd`, redundant `site=` override, or chained installation/preflight commands. Experiment validation still runs inside the pipeline. An interpreter configured outside the project keeps its absolute path. Run does not automatically submit a job; using Close shell during upload ends the local handoff, while closing it after a confirmed Slurm job ID does not cancel that job.


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

The batch uses `exec` to make the Python supervisor own the batch PID, and requests a TERM signal 120 seconds before timeout. Catchable stops forward to producers, allow bounded shutdown, and attempt final publication. SIGKILL/node loss can lose unpublished scratch state. Receipts expose `publication_state`, `publication_error`, and `last_durable` positions for database, metrics, and checkpoints separately. A database export does not restore the running model.

Payload sealing happens before scheduler termination. Final cleanup additionally requires confirmed scheduler termination. Scheduler state, training outcome, publication state, and local transfer errors are distinct. An unsealed payload remains inspectable/syncable but cannot receive a fabricated complete manifest. Expensive repair must run in an explicit CPU allocation; no automatic repair/requeue is provided.

## Sync and diagnostic budgets

Monitor’s Queue includes local work and remote executions last confirmed queued or running by the scheduler. Remote cards show their site, Slurm job ID, and last-sync time; clicking opens that experiment in Monitor. Remote cards use locally synced telemetry and cannot cancel through the local job manager. A later sync updates or removes completed allocations from this queue.

In Monitor, **Sync** is a direct button that downloads all published outputs for the displayed remote experiment run: metrics, logs, diagnostics, plots, and checkpoints. It has no menu and uses that run’s site and execution ID. Each pull reads one published snapshot. A sealed publication manifest (or finalized legacy artifacts) moves the run into Results. Confirmed terminal allocations also leave Monitor even if publication failed; Results exposes their downloaded metrics and logs with an **incomplete results** label. Publication completeness and cleanup eligibility remain separate from job termination. Later publications can be pulled manually from the experiment tree, even after Monitor closes.

Cluster Python stderr is retained in `results/slurm/<execution_id>/pipeline.err` and in Slurm stderr; `pipeline.log` remains the combined stdout/stderr history. Nested training workers also retain a sibling `.err` file and forward stderr separately, preserving original dependency tracebacks. Error files are published with other metrics/logs and preview as text in Results → Files. Opening a failed run selects a nonempty Python error log, scheduler error preview, or legacy combined pipeline log without requiring a local job ID. Final publication failures are also written to Slurm stderr so they remain diagnosable when payload publication cannot finish. These changes apply to newly uploaded/submitted code; existing snapshots retain their original files.

**Run uploads project code to the cluster and opens the submission terminal. Monitor Sync, Sync Results, and the experiment toolbar sync icon download published data to your machine.** These sync controls offer only pull actions. Run does not download results, and pulling monitor/results data does not upload project code or launch experiments.

The Experiment panel’s Sync button opens a named log in the Terminal pane, showing SSH/rsync commands, transfer output, and a `Saved <local path> (<bytes> bytes)` line after each verified result is placed locally. Completion reports files saved, unchanged files, downloaded bytes, and the local destination. When experiment sync finishes, its terminal log closes automatically and returns to the Experiment panel, unless you navigated elsewhere during transfer. Monitor and Results sync controls stream to Console and keep you in your current view. Close the sync log to cancel a running transfer. Errors are reported in a dialog, and output remains available in Console. An unchanged sync explicitly reports no new files. Internal manifests and captured experiment configurations are omitted from console output. Scheduler errors remain visible; unknown scheduler status does not finalize a publication or count as confirmed active work. Sync again to resolve scheduler uncertainty. No sync fabricates a seal or repairs missing scratch outputs.

In the Experiment panel, selecting an experiment with a non-local site shows a square **Sync** icon in the file tree toolbar (hover to see the site name). Click it to pull all published results, including monitor diagnostics, for the selected experiment on that site. This is the only sync control in the experiment tree. The run list and Results refresh after transfer; synced running or queued remote jobs also make the Monitor visible. Sync is manual and reflects the last published cluster data.

Locally:

```bash
venv/bin/python run_pipeline.py site sync ncshare/common --mode metrics --execution EXECUTION_ID
venv/bin/python run_pipeline.py site sync ncshare/common --mode metrics --diagnostics --execution EXECUTION_ID
venv/bin/python run_pipeline.py site sync ncshare/common --mode results --execution EXECUTION_ID
venv/bin/python run_pipeline.py site status ncshare/common
```

The CLI metrics mode defaults to scalars/status/logs; add `--diagnostics` for monitor arrays. Monitor Sync and the file-tree icon always request full results. Results retrieves every owned published artifact. Execution filtering avoids reading unrelated history. Scheduler queries are batched and briefly cached; known terminal states are not queried again.

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

### Worker threads and shutdown diagnostics

Allocation cores and worker threads are separate: serial cluster workers default to one
thread, bounded by the allocation. Set `site.threads_per_worker` explicitly when a
workload benefits from more threads. The supervisor logs the chosen count and deadline;
the SB3 development runner logs evaluation durations and progress every 30 seconds.

Set `resources.time` on long sweeps explicitly. Omitting it inherits the partition
default (NCShare common currently defaults to 30 minutes). Sync preserves the raw
accounting state, cancellation attribution when supplied, reason, elapsed time and time
limit. A SIGTERM near `SLURM_JOB_END_TIME` is reported as a likely pre-timeout warning;
other SIGTERMs remain unattributed unless accounting supplies the cause. After producers
stop, unfinished seed records and tuning trials are finalized before the final export.
