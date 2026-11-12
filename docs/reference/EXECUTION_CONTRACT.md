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
an explicit config `backend`, where supplied, must agree. The initial concrete
backends are Lightning, `sb3-dev`, `d3rlpy-dev`, and the registered numerical
THEMES adapter in `themes-dev`. Algorithm-free neural paradigms
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
callback; it does not additionally invoke a paradigm eval placeholder. External
evaluator providers are currently accepted only by Lightning.

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
