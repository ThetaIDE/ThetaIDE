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
algorithm adapter. Current adapters are the Hub packages `sb3-suite` 2.1 (`sb3/*`) and
`d3rlpy-suite` 2.1 (`d3rlpy/*`, encoder model `d3rlpy/vector`), and the numerical THEMES adapter in `themes-dev`. Algorithm-free neural paradigms
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
production IDs, and adapters declare no bare aliases such as `ppo` or `cql`: select
`sb3/*` or `d3rlpy/*` explicitly.

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

The d3rlpy backend has an independently selected encoder model
`d3rlpy/vector`. It records training losses from a static transition dataset;
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
    "evaluation": "provider",
    "resume_modes": ["inference"],
    "checkpoint_formats": ["example.native/v1"],
    "tuning": false,
    "winner_promotion": false,
    "episode_outcomes": false,
    "intermediate_values": false
  },
  "resource_hints": {"base_memory_gib": 0.8, "cpu": 1, "accelerator": false},
  "resource_hint_hook": "execution_hooks.py:resources"
}
```

`evaluation` is `provider` (native ownership), `configured` or `both`. Exact training recovery
requires the `training` resume mode; inference reload is separately advertised.
Tuning requires both `tuning` and `winner_promotion`. Checkpoint formats describe
native artifacts; the declaration does not add a converter or promotion adapter.
Episode outcome consumers read the feature flag without executing provider code.

Which inputs an algorithm accepts (`data_module`, `dataset_path`, paradigms) is
declared with [compatibility requirements](COMPATIBILITY_REQUIREMENTS.md), not
execution features. Absent features default conservatively: configured evaluation,
and no training resume, tuning/promotion or episode outcomes.
Retained Lightning providers inherit the core Lightning features when no block
is supplied. Other providers must declare supported operations explicitly;
rejections identify the owning package and version. The editable SB3, d3rlpy and
THEMES manifests now declare their existing behavior. Installed releases are not
rewritten and may need an updated release for newly requested operations.

An optional `resource_hint_hook` names a package-relative `file.py:callable`
and receives `(cfg, *, paths)`. Include hook files in the declared implementation
payload; package validation checks containment and existence without importing
them. Checks that need code belong in the capability's
[`preflight`](COMPATIBILITY_REQUIREMENTS.md#preflight) hook, which runs during
validation. Python distribution entry-point providers do not currently support
package-relative hooks.

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

A trial with a component objective or repeated runs trains its seeds in separate
workers, so pruning happens in the trial coordinator: it scores the partial seed
panel with the objective's `intermediate_function` (see
[custom objectives](../guides/CUSTOM_OBJECTIVES.md#pruning)), reports each new
evaluation checkpoint through the tuner's `trial_monitor`, and cancels the trial's
cancellation target when the monitor raises a pruned exception. Pruning requires
an objective with an intermediate function; metric tuning with repeated runs
scores through the bundled `final_metric`, which has one.

Optuna 1.3.0 advertises metric/component scoring and pruning, including component
objectives and repeated runs. Its query owns namespaced, flat,
Hydra and worker-runtime option precedence. Ordinary objective scoring with no
tuning configuration or tuner runtime does not load a tuner. Objective validation
also accepts explicit workspace paths for method and feature resolution.

## Dynamic configuration v1

A provider that declares `dynamic_configuration.v1` can expose safe mutation
boundaries through `src.app.api.dynamic`. Its adapter declares the mutable
paths and constraints, calls the controller with the completed step, available
metrics and effective values, validates the response, and applies the changes.
The adapter records each applied mapping at its effective step. A step-zero
entry sets initial values as part of replay.

The SB3 development PPO provider supports native PPO and a constant learning
rate. It accepts `theta_dynamic.api_version: 1` and a `schedule` of
`{step, parameters}` entries. Mutable paths are `agent.learning_rate`
(positive) and `agent.ent_coef` (nonnegative). After each completed rollout
optimizer call the adapter can change both before the next rollout starts.
Setting a learning rate updates the SB3 schedule and all optimizer groups.

A new segment can set `start` to a positive completed step and provide a
`source` pointing to a compatible v1 PPO training checkpoint. The checkpoint
contains policy and optimizer state, effective values, and progress counters.
The current native runner constructs unnormalized environments. The adapter
checks the algorithm runtime, configuration, and spaces and loads the donor
state before applying the recipient schedule. The fresh environment resets
using the recipient seed plus the completed transition count. See the
[HPO examples](../guides/LOCAL_RL_TUNING.md) for the local recipes.

### Runner profiles for transition-budget tuning

An algorithm can declare a `tuning_profile_hook` on its capability. This is a
read-only package hook with signature `(cfg, *, agent, paths)`, where `cfg` is
the composed method and `agent` includes the selected agent profile and
overrides. The v1 profile declares:

- `api_version: 1`, the progress `unit`, the positive integral budget
  `quantum`, and the `budget_path` written by the tuner.
- `metrics` supported by a final evaluation. On a budgeted job,
  `theta_final_evaluation: true` requests evaluation after the final update
  with the assigned evaluation seed.
- Optional `protected_paths` whose values must stay fixed for comparable
  evaluations, such as the rollout size.
- Optional `mutable_parameters` mapping paths to SDK constraints and a
  `transfer_format` for a checkpoint usable as the source of a v1 dynamic
  segment.

During execution, a runner reports work with
`context.report_progress(unit, completed)` and returns the final measured work
through `RunResult.progress`. The worker retains the last report in its
execution record on cancellation or failure. Progress is work done by the
current job. An adapter can retain cumulative progress in its training-state
checkpoint. Artifact roles `training_state` and `final` identify transferable
training state and the final inference model. The runner writes both into
`context.artifacts`.
