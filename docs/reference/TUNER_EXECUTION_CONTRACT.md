# Portable tuner execution contract v1

Local tuners retain `Tuner.api_version = 1` in `src.app.pipeline.tuners`.
Remote execution is an additional, explicit `PortableTuner.remote_api_version = 1`
contract. A local-only tuner remains usable locally; submitting a new remote job
with it fails before allocation with an update-or-run-locally diagnostic.

New launches require an explicit `tuning.component` capability selector. There is
no default tuner, including when the workspace has only one installed provider.
Aliases and case resolve through the ordinary catalog; plans serialize the
canonical capability ID. Local command compilation resolves the tuner in its
explicit workspace, consistent with validation and remote preparation.

The public portable types and helpers are in
`src.app.pipeline.tuner_execution`. This module does not grant access to other
pipeline or site internals. The independent native fixture at
`tests/fixtures/portable_tuner` demonstrates JSON state without Optuna or SQLAlchemy.

## Plan and lifecycle

A portable tuner implements these methods in addition to the local protocol:

| Method | Provider responsibility |
| --- | --- |
| `execution_plan(cfg, *, execution_id)` | Return a `TunerExecutionPlan` describing state artifacts, concurrency, planned trial count, storage kind, dependency distribution names and continuation notes. Must be read-only; preflight can call it more than once. |
| `launch_bindings(cfg, *, plan, runtime, continuation=None)` | Return `TunerLaunchBindings` containing storage/runtime overrides and environment values for the actual allocation directory. |
| `snapshot_artifact(source, target, *, artifact)` | Produce a closed, consistent regular snapshot file at the supplied temporary target. The provider owns the file format and live-state consistency; core owns bounded chunk transport and publication. |
| `continuation_config(tuning)` | Return the scientific portion of the **tuning block only**, normalizing explicitly permitted launch/scheduling differences. Core separately keeps objective, method/search-space and provider source identities exact. |
| `restore_continuation(cfg, *, plan, runtime, lineage)` | Validate restored provider state and repair interrupted trial bookkeeping. Core restores and verifies byte chunks first; this is study continuation, not model/optimizer recovery. |
| `read_artifact_snapshot(artifact, path)` (optional) | Return `(method, study_summary)` pairs from a closed downloaded snapshot for the existing study monitor. Never read or mutate live allocation state. |
| `finalize_interrupted(runtime, record)` (optional existing hook) | Close interrupted provider trials after producers stop and before final publication. |

For example:

```python
from src.app.pipeline.tuner_execution import (
    TunerArtifact, TunerExecutionPlan, TunerLaunchBindings,
)

plan = TunerExecutionPlan(
    component="example", execution_id=execution_id,
    artifacts=(TunerArtifact(
        "search", f"results/example/{execution_id}/state.json", continuation=True,
    ),),
    storage_kind="managed", concurrency=1, trials=20,
    distributions=("example-tuner",),
)
```

Artifact IDs are stable logical names across attempts. Paths must be canonical,
relative, inside `results`, and include the allocation's execution ID as a path
segment. Duplicate IDs/paths and escaping paths are rejected. V1 permits multiple
published state artifacts but one continuation state artifact. External storage
can declare no local artifacts; core does not invent a local database for it.
`storage_kind` is `managed`, `external` or `none`. Trial count and concurrency are
positive integers; the current Slurm adapter independently enforces concurrency 1.
The plan declares dependency *distribution names* for version comparison, not
installation requirements. Installation requirements remain in package metadata.

Launch overrides may assign `tuning.*`, `theta_tuning.*` or `hydra.sweeper.*`
settings. Host workspace, interpreter and CPU/thread policy environment values
cannot be reassigned by a binding. Core applies bindings again after scratch
placement, replacing overrides for the same keys. Bindings should not perform IO.

## Publication and continuation

New site receipts contain `tuner_execution`, holding a serialized
`theta.tuner-execution/v1` plan or explicit `null` for an ordinary run. An ordinary
run never resolves a tuner or gains a SQLite path/environment default. Submission,
allocation supervision, owned-file selection, snapshots, continuation and monitor
reads use the same recorded plan and selected workspace.

Providers create consistent state snapshots; the host handles cadence, queue and
byte limits, immutable object hashes, manifests and final stage-out. The existing
publication category `database` and `database_seconds` policy names also carry
non-database tuner state for compatibility with current result consumers.
Snapshot bytes are preserved even for CSV/JSONL state without trailing newlines.
Final publication drains all declared artifacts even when only one queue slot is
available. A provider hook that fails to create a regular snapshot file is an error.

Continuation requires a verified export of the parent's declared continuation
artifact, the same tuner, matching scientific config and source pins, confirmed
scheduler termination and exclusive lineage ownership. Dependency versions must
match. The combined state/checkpoint input budget is checked before restoration.
Core verifies state and checkpoint object hashes; the provider validates its state
format and repairs interrupted trials. The historical `sampler_restart` lineage
field now contains the provider's continuation notes.

Optuna supplies the managed SQLite path, storage overrides, SQLite guard environment,
trial count/concurrency, Optuna/SQLAlchemy version names, scheduling normalization,
SQLite integrity checking and RUNNING-to-FAIL reconciliation. Its package metadata
now declares its pip dependencies. Generic site setup uses package requirements
instead of an application extra (the `[optuna]` extra was removed). The Optuna tuner is
an optional component maintained in `optional_components/optuna` and installed into a
workspace like any package; saved `src.app.tuners.optuna.*` targets load from that
installation. Configuration and live study monitors
now use the [contribution/readiness contract](EXTENSION_CONTRIBUTIONS.md).

## Study summary readers

`src.app.pipeline.tuning_monitor` provides headless, read-only
`local_study_summaries(project_root, group, experiment_id, started=None)` (pass
`started` by keyword) and `study_summaries(project_root, run, paths=None)` (pass
`paths` by keyword). Local summary reads import neither provider code nor Qt;
they do not need an installed tuner. Queues use the same reader for every provider,
including after a provider is removed, so process supervision remains available.
The desktop's historical `frontend.monitor.sweep_view.study_summaries` name forwards
to the headless function.

Providers export `tuning_summary.json` under a method checkpoint directory or its
`sweeps/<study>/` archive. A current method export wins over its archived copy of
the same study. `started` excludes earlier-launch exports for active queues.
Summaries contain `schema_version: 1`, a nonempty `component` capability selector,
a nonempty `study`, and `trials`. Each trial has a unique nonnegative integer `id`,
a nonempty string `state`, optional finite `value`/`duration_seconds`, and optional
`parameters`/`intermediate_values` mappings. Canonical states are `complete`,
`running`, `pending`, `pruned` and `fail`; unknown provider states remain visible
without being assumed active or completed. Optional trial-count/best-trial fields
are nonnegative integers or null. Parameter importance is a mapping of finite
scores. Direction, when present, is `minimize` or `maximize`.

Markerless historical summaries remain readable. Partial JSON, malformed summaries
and unknown schema versions are skipped until a valid export replaces them. Files
are never rewritten by readers. The host enriches summaries with generic execution
records to distinguish sampled trials from running workers and retain newer synced
trial navigation.

If no summary is present for a synced run, `read_remote_snapshots(root, record,
paths=paths)` can ask its selected portable tuner to decode closed downloaded
artifacts. This explicit operation checks readiness against the saved configuration
before importing the decoder. It validates returned summaries and stamps provider
identity from the execution plan. Missing/unavailable decoders do not require
Optuna or prevent reading existing normalized summaries. Live tuner stores are
never opened by the generic summary reader.

## Shared publication helpers

Providers may use these exports from `src.app.pipeline.tuner_execution`:

- `commit_artifact(path, position=None)` marks a closed/atomically saved artifact.
- `request_artifact_snapshot(position=None)` requests a coalesced state snapshot.
- `check_artifact_admission()` stops new trials when publication budgets are exhausted.
- `sqlite_snapshot(source, target, seconds=20)` provides a bounded same-node SQLite
  backup with a closed, integrity-checked result. Using this helper does not require
  Optuna. The scheduler implementation remains an internal detail.

## Existing receipts

`src.app.pipeline.tuner_execution_compat` is the bounded, metadata-only reader for
site schema-v1 receipts **without** `tuner_execution`. It preserves the historical
Optuna path reservation, including fixed jobs, and honors the recorded external
storage kind. Old fixed jobs retain their environment guard without enabling
tuning. Legacy local-only non-Optuna jobs retain their original overrides and do
not gain managed state or new remote hooks. Reading old receipt metadata executes
no provider code.

New jobs never use this fallback. Explicit `null` does not activate legacy defaults.
`record_config(record)` returns a copy of saved configuration; only old receipts
without the `tuner_execution` marker inherit the historical Optuna selector for an
existing tuning mapping. It never adds tuning to old fixed jobs or changes explicit
other-tuner selections. Receipts carrying a plan or explicit null never inherit a
selector. Unknown old receipt schemas fail before adaptation.

`record_overrides(record)` applies that bounded selection to replay arguments in a
copy. Matching historical overrides are normalized; a contradictory provider
override is rejected before allocation-worker preparation, requiring explicit
resolution. Allocation bindings, interrupted-trial finalization, remote snapshot
readiness and continuation comparison use the adapted configuration. Receipt bytes
and scientific options are preserved, and continuation still checks objective,
search space and source/dependency identity. Ordinary recipes do not enter this
compatibility reader; older recipes must explicitly select their intended provider
before starting a new job.

Unknown portable plan schemas are rejected. The legacy reader remains until a
separately gated old-job retirement; no archived job, installed package or research
input is rewritten by introducing this contract.
