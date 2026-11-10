# Cluster jobs through SLURM

Theta keeps Local as the default site. Every experiment and group-default box shows Site above the runtime separator. Changing it writes a Hydra site selection; removing the explicit override restores the inherited site. The + beside Site adds resource overrides using the shared parameter controls.

The initial remote workflow is deliberately manual: upload project source, submit from a cluster terminal, and pull snapshots through Sync Metrics or Sync Results. Theta does not issue remote cancellation, reorder a SLURM queue, launch remote dashboards, or stream telemetry. Run uploads the project and opens a floating SSH terminal with a submission command on the clipboard; local queue controls remain inactive for SLURM experiments. Cancel from your cluster terminal with `scancel JOB_ID`.

## Configure a site

Create an Execution Site through Components → Scaffold, or run locally:

```bash
venv/bin/python3 run_pipeline.py scaffold site mycluster
```

Open Components → System → Sites → mycluster → Defaults. Its editable source is `src/usr/sites/mycluster/configs/site/mycluster/_defaults.yaml`, for example:

```yaml
scheduler: slurm
venv_dir: venv
python_executable: auto
resources:
  partition: gpu
  account: your-allocation
  qos: null
  nodes: 1
  cores: 4
  memory: 16G
  time: '01:00:00'
  gpus: 1
module_loads: []
extra_env_vars: {}
dataset_search_dirs: []
```

Use your cluster's actual partition, account, GPU syntax, modules, and environment. CPU jobs should request `gpus: 0` and a CPU trainer. `extra_sbatch_flags` and `gres_format` remain available for cluster-specific directives. Required modules fail visibly if they cannot load.

Experiment overrides are stored under `resources`; they override non-null site defaults. Older site profiles without `scheduler` retain the previous Local/SLURM inference. New profiles should always declare it.

Sites are native components under `src/usr/sites`, so normal component packaging and installation apply. The site component owns its inherited `connection` and `shortcuts` fields in `_defaults.yaml`, its OpenSSH configuration file, and its named resource profiles. Settings does not store site addresses or maintain a separate registry. Keep private keys outside the project. OpenSSH owns credentials, jump hosts, and host-key checks.

## First upload and environment setup

Use rsync 3 or newer on both machines and an OpenSSH config alias that works from your local terminal. Transfers use BatchMode: establish any interactive MFA session externally first, using your cluster's recommended SSH connection-sharing setup when needed. Theta never stores a password.

On the cluster, create the destination on a shared filesystem visible at the same absolute path on login and compute nodes, plus a Python environment. Use the cluster's required Python module first if applicable:

```bash
mkdir -p /scratch/USER/theta-project
python3 -m venv /scratch/USER/theta-project/venv
```

On your local machine, from the Theta project:

```bash
venv/bin/python3 run_pipeline.py site configure mycluster \
  --host YOUR_SSH_ALIAS \
  --remote-root /scratch/USER/theta-project
venv/bin/python3 run_pipeline.py site check mycluster
venv/bin/python3 run_pipeline.py site push mycluster
```

The same operations are available in the Monitoring and Results sync menus: Check and Sync Project; edit connection fields in Components.

The destination must already exist. Its path must be absolute. `--python /absolute/path/to/python` on `site configure` selects a different bootstrap/helper interpreter. That connection setting runs site helpers; the site's `python_executable` or `venv_dir` selects the batch interpreter. Usually both point to the same environment.

The first upload requires only Python's standard library. Then install Theta and the research components' Python dependencies on the cluster:

```bash
cd /scratch/USER/theta-project
venv/bin/python3 -m pip install -e .
# If this project uses Optuna:
venv/bin/python3 -m pip install -e '.[optuna]'
```

Install the appropriate cluster build of PyTorch and any component dependencies according to your cluster's environment policy. Virtual environments are never copied from the laptop. Editable source components and installed component payloads are uploaded; externally installed Python distributions must also exist in the cluster environment.

Project sync uploads source, configuration, provider selections, and a persistent project ID. It excludes results, virtual environments, caches, and connection settings. A different project ID at the destination causes an error rather than merging projects. A source manifest records exactly what was uploaded; later uploads remove only previously managed source files that were deleted locally. Existing unrelated remote files are retained. Each submission freezes the source independently, so future uploads do not change queued or running jobs.

Datasets are excluded by default. Either configure cluster dataset locations in the site profile/experiment, or explicitly upload `in/datasets`:

```bash
venv/bin/python3 run_pipeline.py site push mycluster --include-datasets
```

Source and dataset symlinks need explicit copies or cluster-side dataset configuration. Project sync does not follow arbitrary symlinks outside the project.

## Submit on the cluster

Select the site in the experiment editor and save, then upload changes. In a cluster terminal:

```bash
cd /scratch/USER/theta-project
venv/bin/python3 run_pipeline.py GROUP/EXPERIMENT site=mycluster dry_run=true
venv/bin/python3 run_pipeline.py GROUP/EXPERIMENT site=mycluster
```

The command prints the execution ID, scheduler job ID, and submission receipt location. Scripts and receipts live under `results/slurm/EXECUTION_ID/`. Inspect `job.sbatch`, `pipeline.log`, and scheduler stdout/stderr there when diagnosing a failure.

Each submission allocates a new execution and attempt ID and stamps its result experiment ID. Original experiment outputs are not purged. A submission timeout is recorded as unknown and is never automatically retried: reconcile it through the sync/status command before manually submitting another attempt. The SLURM job comment contains the project and execution IDs to help reconciliation.

One allocation runs the current pipeline, including evaluation, configured monitors, tuning, variability, and selected plots. This initial executor supports one node and one training device; methods, trials, and training seeds run serially within that allocation. Thread limits use `SLURM_CPUS_PER_TASK`, and workstation resource-pool settings are removed. Workflow-DAG execution, remote recovery, remote dashboards, and automatic SLURM requeue are not supported; submit individual experiments. New attempts retain old results.

Optuna SQLite storage is isolated under `results/optuna/EXECUTION_ID/study.sqlite`, even when the recipe previously used a shared SQLite URL. External database URLs remain external and are not copied by rsync. Telemetry and study summaries remain ordinary run outputs.

## Pull metrics and results

Locally, use Sync Metrics in Monitoring and Sync Results in Results, choosing the connection from the menu. Both run in the background. CLI equivalents:

```bash
venv/bin/python3 run_pipeline.py site sync mycluster --mode metrics
venv/bin/python3 run_pipeline.py site sync mycluster --mode results
venv/bin/python3 run_pipeline.py site status mycluster
```

Metrics sync includes manifests, configuration, metrics, events, logs, and recorded monitor snapshot metadata/arrays. Results sync adds checkpoints, plots, databases, and other owned outputs. The panels reread local files after success. Remote entries show their site and last successful sync; an old snapshot is not treated as proof that the job stalled. Pending submissions appear even before their first metrics exist.

Both modes share one transfer ledger and local output tree. Unchanged finalized artifacts are skipped; growing files need subsequent snapshots. Files download into staging, are checksum-verified, then published atomically. Partial immutable downloads survive a failed transfer for retry. Local outputs with an unrelated owner cause a collision error. A network failure records a sync error without changing the job's status.

Active append-only files are snapshotted remotely, with incomplete CSV/JSONL tails held for the next sync. Active SQLite databases use SQLite's backup API. These transfer snapshots are scratch files, released after the attempt; an interrupted SSH session can leave scratch directories under `results/slurm/.sync/`. They are not original results and can be removed when no transfer is active.

The scheduler state, execution outcome, transfer state, and artifact availability are separate. `squeue` reports active jobs and `sacct` reconciles terminal jobs. If accounting is unavailable and a terminal state cannot be confirmed, status remains uncertain and automatic finalization/cleanup is refused. You can still copy available results. Unexpected termination retains partial outputs.

## Optional remote cleanup

Sync retains remote originals. Cleanup is an explicit CLI operation after a complete Results sync:

```bash
venv/bin/python3 run_pipeline.py site clean mycluster EXECUTION_ID
```

Cleanup requires confirmed scheduler termination, a final artifact manifest, and a reverified local copy of every declared artifact. Missing checkpoints or altered files prevent cleanup. It also refuses while another submitted execution is nonterminal, because outputs may be dependencies. Only that execution's owned result files are removed; shared datasets and source are retained. Submission receipts and artifact manifests remain as provenance/tombstones. Cleanup is not a backup policy: keep your own durable backup before reclaiming cluster storage.

## First-cluster validation checklist

These are manual checks for your real cluster. Development tests use fake scheduler/transfer commands and local files; no SSH connection or cluster submission was attempted during implementation.

1. Create the remote directory/environment and edit the site component, then run Check and Sync Project. Confirm the returned command and inspect the uploaded site profile.
2. Submit a small CPU experiment with a short wall time. Verify `squeue` shows one job, the resources match `job.sbatch`, and the submission command exits while the job stays pending/running.
3. Sync Metrics while it runs. Verify charts/logs update and no checkpoints arrive. Sync again and check transfer counts; unchanged files should be skipped.
4. While the first job is queued/running, edit and upload the project, then submit again. Verify different execution/result IDs and that the first job's frozen source/configuration stays unchanged.
5. After completion, Sync Results twice. Check checkpoint checksums, plots, diagnostics, and telemetry; the second pull should transfer no unchanged artifacts. Remote originals should still exist.
6. Restart Theta, sync again, and confirm jobs/results are rediscovered. If using tuning or variability, repeat with a small study/two seeds and inspect their summaries and execution view.
7. Interrupt a transfer and retry. Confirm previously published artifacts remain readable and unchanged artifacts are skipped. Temporarily break the connection and confirm the app reports a sync error without claiming the job failed.
8. Cancel a job yourself with `scancel JOB_ID`, and separately try a deliberately short time limit. Sync and check terminal state, partial metrics, and logs. Theta's remote cancel/queue controls should do nothing.
9. Only after independently backing up a disposable run, exercise `site clean`. A metrics-only copy must be insufficient; a full verified copy permits cleanup without deleting local results or shared datasets.

## Settings and the Run button

Settings → Terminal → Cluster sites lists the profiles supplied by site components and generates setup commands and shortcuts for the selected profile. Edit address/SSH/rsync settings in the owning component in Components. The script creates the remote environment, uploads the project, and installs Theta plus the pip requirements declared by native component manifests. Review the script for cluster-specific modules/CUDA requirements before running it in a local terminal. Setup generation itself makes no connection and installs nothing.

Reusable scheduler workflow code lives in `src/app/sites/slurm.py`. Site-owned data lives under `src/usr/sites/<site>`:

```text
ncshare/
  .theta_component.json
  site.py
  ssh_config
  configs/site/ncshare/
    _defaults.yaml    # scheduler, shared resources, connection, shortcuts
    default.yaml
    common.yaml       # inherits defaults, overrides CPU resources
    gpu.yaml          # inherits defaults, overrides GPU resources
```

Local and ARC use the same component structure. The Components tree shows one folder per site with Defaults and its named profiles. Select `ncshare/common` or `ncshare/gpu` in experiment and Settings dropdowns. Legacy `site=ncshare_common`, `site=ncshare_gpu`, `site=ncshare`, and `site=local` selectors remain compatibility aliases and are hidden from the profile catalog.

Edit `_defaults.yaml` to share address and transport settings across profiles:

```yaml
connection:
  host: your-ssh-alias
  remote_root: /scratch/USER/theta-project
  python: venv/bin/python3
  ssh_config: src/usr/sites/ncshare/ssh_config
  rsync_executable: rsync
  remote_rsync: rsync
shortcuts:
  directory: src/usr/sites/ncshare/bin
```

A profile can override inherited connection fields if necessary. Shortcut generation writes derived commands to the component's configured directory without rewriting its defaults. The component's `ssh_config` owns OpenSSH aliases and may include `~/.ssh/config`; private keys stay outside the project. No operation rewrites the user's home SSH configuration. Component code and manifests remain part of project upload. External distributions and system modules still need a compatible cluster environment.

For SLURM experiments, Run now uploads the saved project, copies a command that performs `pip check`, experiment preflight, and submission, then opens a floating SSH terminal in the remote project directory. Paste and Enter to run it. Closing the terminal does not cancel submitted batch jobs. Sync Metrics and Sync Results remain explicit actions. Remote cancellation and queue mutation remain unsupported.
