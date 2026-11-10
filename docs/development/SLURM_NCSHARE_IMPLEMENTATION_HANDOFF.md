# Implementation handoff: Slurm hardening for NCShare CPU

Copy the prompt below into a new coding session in this repository. The linked plan defines the current decisions and acceptance criteria; the operational review supplies findings, not proof of actual cluster disruption.

```text
Implement the revised Slurm hardening and NCShare CPU rollout plan in this repository.

Read first:
- AGENTS.md and applicable nested guidance.
- docs/plans/SLURM_NCSHARE_HARDENING_PLAN.md (authoritative implementation scope).
- docs/reports/SLURM_NCSHARE_REVIEW_20261110.md (audit evidence).
- docs/guides/CLUSTER_JOBS.md and docs/README.md.
- docs/reference/CONFIG_PARAMETER_FORMS.md before changing configuration UI.

Inspect the current checkout and git diff before editing. The plan/report/handoff may
still be uncommitted user work; preserve them and all unrelated changes. Reconcile
actual code with the plan rather than assuming the audit is still exhaustive.

User decisions:
1. Qualify ncshare/common first; preserve ncshare_common as an alias. The user once
   called it ncsu_common; do not create a new cluster/profile because of that typo.
   No GPU testing, automatic requeue, or database-server provisioning in this rollout.
2. Use one exclusively owned scratch-local SQLite DB per allocation. Several
   allocations/studies may coexist on a node with separate DBs. All live connections
   and consistent backups stay on the owning compute node. Never open the live DB
   over NFS or copy its bare file while it has unexported WAL state. Publish closed,
   standalone snapshots. Retain an extension point for future server storage.
3. Keep NCShare policy in its site component: home for software/configuration and
   compact source identity, /work for job inputs/published results, job-local /scratch
   for execution. Shared staging, scheduler, transfer, and cleanup mechanisms belong
   in src/app/sites. Do not hardcode NCShare behavior in models or training providers.
4. Overnight execution must work with the desktop disconnected. Publication is
   allocation-owned. Five minutes is our configurable engineering choice for DB
   backup during a long trial, not a cluster restriction or a full-results copy timer.
   Pilot policy: small metrics/status every 60 seconds; DB export requested at trial
   completion with short-trial bursts coalesced (initially 30-second minimum spacing),
   plus every five minutes during long trials; checkpoints after commit with priority;
   monitor arrays queued once after commit and batched; final stage-out on exit.
   Only changed data moves. Bound queues, concurrency, and large-object chunks so
   heavy diagnostics cannot starve status/DB durability. Make schedules site-configurable.
5. Preserve existing local parallel trials. Current Slurm execution forces serial
   trials using THETA_MANAGED_SERIAL=1 and removes the workstation resource pool.
   Do not silently claim parallel Slurm support or disable local parallelism.
   Qualify the serial CPU path first. Only enable within-allocation n_jobs > 1 after
   aggregate allocated CPU/memory/device admission and same-host SQLite contention,
   pruning, backup, and cancellation tests pass. Otherwise reject that request clearly.
6. Periodic DB export preserves study history; it does not restore a running model.
   Implement explicit continuation from a verified prior export into fresh scratch,
   with exclusive study ownership, interrupted-trial reconciliation, unique new
   attempt/output IDs, lineage/dependency pins, and compatibility checks. Full
   model/optimizer/RNG/runner checkpoint resume is separately gated; never promise it
   from a DB copy alone. Report last durable DB and checkpoint positions separately.

Implementation priorities, in dependency order:
- Normalize actual resource requests once; reject conflicting GRES/extra directives/
  SBATCH environment overrides. Fix plot_only scheduler bypass without wiping prior
  results or plotting against a new empty result ID.
- Add explicit control/work/runtime roots, source/input staging, scratch-local DB
  resolution, and consistent compute-side exports. Audit interpreter, cwd, providers,
  relative output/dataset paths, and subprocess propagation.
- Publish committed artifacts atomically with precomputed hashes and bounded segments/
  bundles. Separate payload sealing from later scheduler termination. Handle failures,
  signals, stage-out time, and incomplete exports honestly.
- Make sync consume indexes/manifests and fetch missing immutable objects. No recursive
  result scans, checkpoint hashing, or live SQLite backup on the login node. Preserve
  checksum verification, resumability, local result layout, and explicit diagnostics.
- Deduplicate immutable source snapshots; implement ownership/reference/lease-based
  collection of temporary uploads/transfers/caches. Extend verified result cleanup to
  reclaim execution workspaces. Preserve provenance, running/unknown jobs, active
  transfers, continuation inputs, and unclassified legacy files.
- Defer repeated monitor HTML generation; enforce measured execution-wide byte/file
  budgets and explicit sampling/coverage. Separate scratch flush, publication, DB
  backup, desktop sync, and model-checkpoint schedules. Keep objective/checkpoint
  behavior independent of optional diagnostic throttling.
- Implement and test explicit study continuation, then complete the CPU validation
  sequence in the plan. Keep the stages and documentation synchronized.

NCShare-specific details to preserve:
- /scratch is temporary job-local storage, not a shared remote project root.
- /work is shared NFS and has a 75-day mtime purge. Newly staged files must get current
  mtimes; preserve original timestamps in metadata. Do not touch old files to defeat
  the purge. Do not put the Python environment in /work.
- Keep login helpers lightweight. Schedule expensive packaging/finalization/repair.
- Do not silently fall back to NFS execution, unsafe SQLite, or an alternate DB backend.
- Migrate legacy outputs using an inventory/preview; do not delete or relocate them
  automatically. Preserve existing site aliases and supported local execution behavior.

Validation and delivery:
- Use meaningful filesystem/subprocess tests with fake scheduler/transfer commands
  for ownership, interruptions, budgets, publication timing, DB integrity, continuation,
  resource validation, and repeat-sync I/O behavior. Test archive traversal/link safety.
- Baseline audit run was 63 passed / 2 failed over the site/Slurm/mechanistic-monitor/
  monitor-lifecycle suites; inspect current results rather than assuming that baseline.
  Recorded failures were:
  tests/test_monitor_lifecycle.py::test_reused_experiment_streams_only_current_launch_and_all_parallel_seeds
  tests/test_monitor_lifecycle.py::test_queue_labels_method_without_started_feed_as_pending
- Run relevant checks as implementation proceeds. Do not weaken assertions or omit
  failures merely to claim a green result. Add local parallelism regression coverage.
- Update docs/guides/CLUSTER_JOBS.md and add an implementation/validation status note
  under docs/development, linked in docs/README.md. Label planned versus tested behavior.
- Implement local changes and local validation autonomously. This handoff alone does
  not authorize SSH connections, real job submission/cancellation, destructive remote
  cleanup, publishing components, or paid/external infrastructure changes. Prepare
  exact CPU validation commands; use session-specific authorization for real-cluster
  work. Never report mocked tests as live NCShare acceptance.
- Follow the production release workflow only if a Theta Hub release is actually
  requested. Do not publish or commit/push simply because an archived guide says so.

Carry the work through reviewable code, relevant tests, and accurate documentation.
If a genuine external dependency blocks live acceptance, finish independent local
work and state the exact remaining validation. Report completed phases, checks and
failures, unsupported modes, and any remaining risks without claiming the entire
rollout is qualified prematurely.
```
