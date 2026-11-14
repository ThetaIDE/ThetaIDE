# Method execution contract v1

Normal training is coordinator → `provider_worker` → Hydra composition in
`train.py` → `execute_method` → the paradigm's registered `runner.run(cfg, context)`.
The runner is called once per resolved method, including each trial and replicate.
The coordinator retains method expansion, output preparation, resources, queues,
site submission and plotting. Runners never call `run_methods` or purge experiments.

`BaseParadigmRunner.run` is abstract. Implementations declare `execution_contract = 1`
and return `RunResult`; exceptions propagate as nonzero worker status. A cancelled
execution remains distinct from failure. The worker persists an `execution.json`
with resolved config, runner, interpreter, provider snapshot, selected backend,
result and lifecycle status. Context cleanup runs even when fitting fails.

`RunContext` supplies identity, workspace, seed, assigned artifact directory,
checkpoint selection, cancellation check and cleanup registration. An explicit
agent seed overrides the run seed; repeats set both to their assigned seed. `RunResult`
contains status, metrics, definitions, objective, checkpoint references/formats and
progress units. Store backend-native files inside `context.artifacts`.

The paradigm chooses the lifecycle runner. The selected algorithm capability owns
`backend` and integer `execution_contract` metadata. Metadata decides the backend;
an explicit config `backend`, where supplied, must agree. Backend labels are provider-defined; core does not maintain an adapter allowlist.
Lightning is the retained core executor. All other labels invoke the selected
algorithm adapter. Current adapters include `sb3-dev`, `d3rlpy-dev` and the
numerical THEMES adapter in `themes-dev`. Algorithm-free neural paradigms
use Lightning. Unknown algorithms, unresolved runners and incompatible selected
algorithm capabilities fail before fitting. Unselected historical packages remain
browsable. Capability `entry_point` is import metadata and remains supported.

Execution `entrypoint`, method `runner`, `script`, `engine`, `venv` and `venv_python`
are rejected. Interpreter selection uses `python` at method/resource launch level;
site-level interpreter resolution remains in the site contract. Nested scientific
parameters with coincident names are not recursively rejected. Historical configs
can be opened for inspection but cannot rely on removed execution behavior.

Both coordinator preflight and workers validate the composed method config. The
preflight composes the same method override list, with representative sweep values;
workers validate each actual sampled config. Profile selection does not redirect
production IDs. Use `sb3-dev/*` or `d3rlpy-dev/*` explicitly.

For migration, explicitly select files, review preview, then apply:

```sh
python -m src.app.pipeline.migrate_execution in/config/experiment/example.yaml
python -m src.app.pipeline.migrate_execution --apply in/config/experiment/example.yaml
```

The tool only writes explicitly selected editable YAML under `src/usr` or
`in/config`. Conflicts and unknown scripts require manual resolution and a
registered implementation. Installed components and snapshots are never edited.

## Evaluation and checkpoints

Lightning algorithm modules own training and native validation metrics. Explicit
`evaluator` selection runs through `ConfiguredEvaluatorCallback` once at its owned
lifecycle event. Merely resolving a paradigm eval protocol does not invoke it.
Online/offline RL registry placeholders produce no additional evaluation.
SB3 owns its matched-seed greedy/sampled rollout evaluation and calls the native
callback; it does not additionally invoke a paradigm eval placeholder. The execution feature declaration determines whether configured evaluator
providers are accepted. Lightning invokes configured evaluators; native RL
adapters currently own evaluation, while the numerical adapter selects and
invokes its own evaluator provider.

SB3 uses `.zip`; d3rlpy uses `.d3`; Lightning uses `.ckpt`. SB3/d3rlpy training
resume is explicitly rejected; inference reload is supported. Lightning recovery
requires explicit `resume_from`, passed to native `trainer.fit(ckpt_path=...)`.
Worker checkpoints/configs remain within the assigned method directory. Trial
winner promotion remains coordinator-owned.

The d3rlpy dev backend has an independently selected encoder model
`d3rlpy-dev/vector`. It records training losses from a static transition dataset;
these are not held-out scores or simulator performance. Its initial budget must
be divisible by `n_steps_per_epoch`. IQL and TD3+BC require continuous actions.
The current Optuna winner promotion supports `.ckpt`/`.zip`; `.d3` promotion is a
remaining integration gate, so a d3rlpy tuning run is not yet supported.

Explicit preprocessing resources retain package loading/leases in `resource_worker`.
They are not selectable normal training entrypoints. General workflows remain deferred.

## Numerical apprenticeship adapter

`themes-dev` declares backend `numerical` and execution contract 1. The
`apprenticeship` paradigm selects its registered runner, which resolves the
selected trajectory/model/evaluator capabilities and invokes the importable
algorithm adapter. It schedules load, stages, segments, policies, evaluation,
and export within one worker; it never launches another experiment coordinator.
Optional runner `validate_config(cfg, paths=...)` performs read-only validation
of each fully composed method during preflight; the numerical runner also checks
its contract before executing. Algorithm/model schemas retain shared form controls.
Package resource dataset URIs resolve through the existing provider resource loader.

The numerical adapter owns evaluation once, writes backend-native inference
bundles and plots in the context directory, and uses standard telemetry and
worker cancellation/finalization. It supports direct runs and training repeats.
Exact resume and tuning/promotion of its multi-file bundles are explicitly
rejected. This does not extend Lightning/SB3/d3rlpy resume guarantees.

## Provider execution features and resource hints

Algorithm capabilities may declare `execution_features` alongside
`execution_contract: 1` and their `backend`. These are additive v1 metadata,
validated by the manifest schema and at selection time:

```json
{
  "execution_features": {
    "input_types": ["dataset_path"],
    "evaluation": "provider",
    "resume_modes": ["inference"],
    "checkpoint_formats": ["example.native/v1"],
    "tuning": false,
    "winner_promotion": false,
    "episode_outcomes": false,
    "intermediate_values": false
  },
  "execution_preflight": "execution_hooks.py:validate",
  "resource_hints": {"base_memory_gib": 0.8, "cpu": 1, "accelerator": false},
  "resource_hint_hook": "execution_hooks.py:resources"
}
```

`input_types` advertises supported inputs; explicit `data_module` and
`dataset_path` selections require the corresponding entry. `evaluation` is
`provider` (native ownership), `configured` or `both`. Exact training recovery
requires the `training` resume mode; inference reload is separately advertised.
Tuning requires both `tuning` and `winner_promotion`. Checkpoint formats describe
native artifacts; the declaration does not add a converter or promotion adapter.
Episode outcome consumers read the feature flag without executing provider code.
The original top-level v1 `episode_outcomes` flag remains a compatibility reader.

Absent features default conservatively: no explicit data module/dataset input,
configured evaluator, training resume, tuning/promotion or episode outcomes.
Retained Lightning providers inherit the core Lightning features when no block
is supplied. Other providers must declare supported operations explicitly;
rejections identify the owning package and version. The editable SB3, d3rlpy and
THEMES manifests now declare their existing behavior. Installed releases are not
rewritten and may need an updated release for newly requested operations.

Optional native-package hooks name a package-relative `file.py:callable` and
receive `(cfg, *, paths)`. Include hook files in the declared implementation
payload; package validation checks containment and existence without importing
them. `execution_preflight` raises an actionable error for
constraints that declarations cannot express. It runs only for the selected
algorithm, in both coordinator and worker execution validation; it may run
multiple times and must be read-only and idempotent. Metadata/catalog browsing
never runs it. Keep hook modules lightweight so preflight does not need to import
a training backend. Python distribution entry-point providers do not currently
support package-relative hooks.

`resource_hint_hook` returns a mapping extending the static `resource_hints`.
Only `base_memory_gib`, `cpu` (positive finite numbers) and `accelerator`
(boolean) are accepted. Without hints, core uses its conservative generic
estimate and assumes accelerator use; an explicit `trainer.accelerator=cpu`
always disables accelerator demand. Model/data/replay/activation estimates and
learned usage still apply. Estimates use the selected workspace, including for
relative dataset paths and nested method device choices. Missing explicitly
selected providers fail with installation guidance and catalog diagnostics.
Hints never bypass the user's CPU/RAM policy or worker limits. SB3's lightweight
hook describes its existing CPU-only Theta policy behavior; core has no SB3 name
or model-flag checks.

## Tuner objective and pruning capabilities

Tuner capabilities may declare additive v1 `tuning_features` metadata:

```json
{
  "tuning_features": {
    "objective_modes": ["metric", "component"],
    "pruning": true,
    "pruning_objectives": false,
    "pruning_repeated_runs": false
  }
}
```

`metric` scores a recorded metric; `component` scores through an objective
provider. Missing declarations support metric objectives without pruning.
Pruning-capable tuners must implement a read-only, idempotent
`pruning_enabled(cfg) -> bool` query. The provider interprets its own options,
including worker runtime bindings and historical settings; core never interprets
a provider's pruner names. Catalog inspection reads metadata without running this
query. Execution readiness and enablement checks precede query execution.

Active pruning with a component objective or repeated runs requires the respective
compatibility flag. Declaring support means the provider implements compatible
intermediate scoring; declarations do not add that behavior automatically.
Active pruning also requires the selected algorithm's
`execution_features.intermediate_values: true`. Coordinator validation checks each
method, and worker dispatch repeats the compatibility checks before importing the
runner. This flag promises reports through the selected tuner's callbacks/monitor;
provider preflight still enforces metric, evaluation-cadence and process constraints.
Core Lightning and editable SB3 adapters report intermediate values. Other runners
default to false. Existing installed manifests are preserved.

Optuna advertises metric/component scoring and pruning, while rejecting pruning
with component objectives or repeated runs. Its query owns namespaced, flat,
Hydra and worker-runtime option precedence. Ordinary objective scoring with no
tuning configuration or tuner runtime does not load a tuner. Objective validation
also accepts explicit workspace paths for method and feature resolution.
